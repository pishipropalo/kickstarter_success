from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score,
                           matthews_corrcoef, confusion_matrix,
                           classification_report)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna
import joblib
import warnings
import pandas as pd
import numpy as np
import multiprocessing as mp
import time
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Настройка параллелизации
N_JOBS = max(1, mp.cpu_count() - 1)
print(f"Используется {N_JOBS} ядер для параллелизации")

# Загрузка данных
print("Загрузка данных...")
try:
    X_train = pd.read_csv('X_cl_train.csv')
    X_val = pd.read_csv('X_cl_val.csv')
    X_test = pd.read_csv('X_cl_test.csv')
    y_train = pd.read_csv('y_cl_train.csv').iloc[:, 0]
    y_val = pd.read_csv('y_cl_val.csv').iloc[:, 0]
    y_test = pd.read_csv('y_cl_test.csv').iloc[:, 0]
except Exception as e:
    print(f"Критическая ошибка: {str(e)}")
    raise

X_combined = pd.concat([X_train, X_val], axis=0)
y_combined = pd.concat([y_train, y_val], axis=0)

print(f"Размер объединенной выборки: {X_combined.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Распределение классов: {dict(zip(*np.unique(y_combined, return_counts=True)))}")

class_weights = compute_class_weight('balanced', classes=np.unique(y_combined), y=y_combined)
class_weight_dict = {i: float(class_weights[i]) for i in range(len(class_weights))}
print(f"Веса классов: {class_weight_dict}")

cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def calculate_metrics(y_true, y_pred, y_scores):
    """Вычисление метрик с учетом типа модели"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    if y_scores.ndim > 1:
        metrics.update({
            'auc_roc': roc_auc_score(y_true, y_scores[:, 1]),
            'auc_pr': average_precision_score(y_true, y_scores[:, 1])
        })
    else:
        # Для моделей с decision function
        metrics.update({
            'auc_roc': roc_auc_score(y_true, y_scores),
            'auc_pr': average_precision_score(y_true, y_scores)
        })
    
    return metrics

def print_results(model_name, metrics):
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

param_distributions = {
    'SVM_RBF': {
        'model__C': [0.1, 1, 10, 100],
        'model__gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'model__class_weight': [class_weight_dict, 'balanced', None],
        'model__probability': [True]
    },
    'SVM_Poly': {
        'model__C': [0.1, 1, 10, 100],
        'model__gamma': ['scale', 'auto'],
        'model__degree': [2, 3],
        'model__coef0': [0.0, 0.1, 1.0],
        'model__class_weight': [class_weight_dict, 'balanced', None],
        'model__probability': [True]
    },
    'SVM_Sigmoid': {
        'model__C': [0.1, 1, 10, 100],
        'model__gamma': ['scale', 'auto'],
        'model__coef0': [0.0, 0.1, 1.0],
        'model__class_weight': [class_weight_dict, 'balanced', None],
        'model__probability': [True]
    },
    'LinearSVM': {
        'model__C': [0.1, 1, 10, 100],
        'model__penalty': ['l2'],
        'model__loss': ['squared_hinge'],
        'model__dual': [True],
        'model__class_weight': [class_weight_dict, 'balanced', None],
        'model__max_iter': [2000]
    },
    'NuSVM': {
        'model__nu': [0.3, 0.5, 0.7],
        'model__gamma': ['scale', 'auto'],
        'model__class_weight': [class_weight_dict, 'balanced', None],
        'model__probability': [True]
    }
}

def create_pipeline(model_name, **params):
    """Создание пайплайна с нормализацией данных"""
    scaler = StandardScaler()
    
    if model_name == 'SVM_RBF':
        model = SVC(kernel='rbf', random_state=42, **params)
    elif model_name == 'SVM_Poly':
        model = SVC(kernel='poly', random_state=42, **params)
    elif model_name == 'SVM_Sigmoid':
        model = SVC(kernel='sigmoid', random_state=42, **params)
    elif model_name == 'LinearSVM':
        model = LinearSVC(random_state=42, **params)
    elif model_name == 'NuSVM':
        model = NuSVC(random_state=42, **params)
    
    return Pipeline([
        ('scaler', scaler),
        ('model', model)
    ])

def fast_randomized_search(model_name, param_dist, n_iter=20):
    """Быстрый randomized search для предварительной оптимизации"""
    print(f"\nБыстрая оптимизация {model_name}...")
    
    pipeline = create_pipeline(model_name)
    
    if model_name == 'NuSVM':
        filtered_params = []
        for _ in range(n_iter * 2):
            params = {}
            for key, values in param_dist.items():
                if hasattr(values, '__iter__') and not isinstance(values, str):
                    params[key] = np.random.choice(values)
                else:
                    params[key] = values
            filtered_params.append(params)
        
        valid_params = []
        for params in filtered_params:
            try:
                model = NuSVC(nu=params['model__nu'], kernel='rbf')
                model.fit(X_train[:100], y_train[:100])
                valid_params.append(params)
                if len(valid_params) >= n_iter:
                    break
            except:
                continue
        
        if not valid_params:
            print(f"Не найдено допустимых параметров для {model_name}")
            return {}, 0.5
        
        search = RandomizedSearchCV(
            pipeline, param_distributions=valid_params,
            n_iter=len(valid_params), cv=cv_strategy, 
            scoring='roc_auc', n_jobs=N_JOBS, random_state=42
        )
    else:
        search = RandomizedSearchCV(
            pipeline, param_dist, n_iter=n_iter, 
            cv=cv_strategy, scoring='roc_auc',
            n_jobs=N_JOBS, random_state=42
        )
    
    try:
        search.fit(X_combined, y_combined)
        return search.best_params_, search.best_score_
    except Exception as e:
        print(f"Ошибка в {model_name}: {e}")
        return {}, 0.5

def optuna_fine_tune(model_name, base_params, n_trials=30):
    """Тонкая настройка лучших параметров с помощью Optuna"""
    print(f"Тонкая настройка {model_name} с Optuna...")
    
    def objective(trial):
        params = {}
        
        if model_name in ['SVM_RBF', 'SVM_Poly', 'SVM_Sigmoid']:
            params['C'] = trial.suggest_float('C', 0.01, 100, log=True)
            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto', 0.01, 0.1, 1])
            params['class_weight'] = trial.suggest_categorical('class_weight', [class_weight_dict, 'balanced', None])
            params['probability'] = True
            
            if model_name == 'SVM_Poly':
                params['degree'] = trial.suggest_int('degree', 2, 4)
                params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)
            elif model_name == 'SVM_Sigmoid':
                params['coef0'] = trial.suggest_float('coef0', 0.0, 1.0)
                
        elif model_name == 'LinearSVM':
            params['C'] = trial.suggest_float('C', 0.01, 100, log=True)
            params['class_weight'] = trial.suggest_categorical('class_weight', [class_weight_dict, 'balanced', None])
            params['max_iter'] = trial.suggest_int('max_iter', 1000, 5000)
            
        elif model_name == 'NuSVM':
            params['nu'] = trial.suggest_float('nu', 0.1, 0.9)
            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto', 0.01, 0.1, 1])
            params['class_weight'] = trial.suggest_categorical('class_weight', [class_weight_dict, 'balanced', None])
            params['probability'] = True
        
        try:
            model = create_pipeline(model_name, **params)
            model.fit(X_train, y_train)
            
            if model_name == 'LinearSVM':
                y_scores = model.decision_function(X_val)
            else:
                y_scores = model.predict_proba(X_val)[:, 1]
            
            return roc_auc_score(y_val, y_scores)
                
        except Exception as e:
            print(f"Ошибка в trial для {model_name}: {e}")
            return 0.5
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params, study.best_value

def train_and_evaluate_model(model_name):
    """Обучение и оценка одной модели"""
    start_time = time.time()
    result = {'model': None, 'params': {}, 'metrics': {}, 'time': 0, 'error': None}
    
    try:
        # Быстрая оптимизация
        base_params, base_score = fast_randomized_search(model_name, param_distributions[model_name])
        print(f"{model_name} - Базовый результат: {base_score:.4f}")
        
        # Тонкая настройка
        final_params, final_score = optuna_fine_tune(model_name, base_params)
        print(f"{model_name} - Финальный результат: {final_score:.4f}")
        
        # Обучение финальной модели
        clean_params = {k.replace('model__', ''): v for k, v in {**base_params, **final_params}.items()}
        final_model = create_pipeline(model_name, **clean_params)
        final_model.fit(X_combined, y_combined)
        
        # Оценка
        y_pred = final_model.predict(X_test)
        
        if model_name == 'LinearSVM':
            y_scores = final_model.decision_function(X_test)
        else:
            y_scores = final_model.predict_proba(X_test)[:, 1]
        
        metrics = calculate_metrics(y_test, y_pred, y_scores)
        
        result.update({
            'model': final_model,
            'params': clean_params,
            'metrics': metrics,
            'time': time.time() - start_time
        })
        
    except Exception as e:
        result.update({
            'error': str(e),
            'time': time.time() - start_time
        })
        print(f"Критическая ошибка в {model_name}: {e}")
    
    return result

def parallel_model_training():
    """Параллельное обучение всех моделей"""
    model_names = ['SVM_RBF', 'SVM_Poly', 'SVM_Sigmoid', 'LinearSVM', 'NuSVM']
    
    print("Начинаем параллельное обучение всех SVM моделей...")
    total_start = time.time()
    
    results = {}
    for model_name in model_names:
        print(f"\nОбучение {model_name}...")
        results[model_name] = train_and_evaluate_model(model_name)
    
    print(f"\nВсе модели обучены за {time.time() - total_start:.1f} секунд")
    return results

if __name__ == "__main__":
    results = parallel_model_training()
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ВСЕХ SVM МОДЕЛЕЙ")
    print("="*60)
    
    successful_models = {}
    for name, res in results.items():
        if res['model'] is not None:
            print_results(name, res['metrics'])
            successful_models[name] = res
        else:
            print(f"\n{name}: ОШИБКА - {res.get('error', 'Неизвестная ошибка')}")
    
    if successful_models:
        # Сравнительная таблица
        comparison = []
        for name, res in successful_models.items():
            row = {'Model': name, 'Time': res['time']}
            row.update(res['metrics'])
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        print(f"\n{df.round(4)}")
        
        best_idx = df['auc_roc'].idxmax()
        best_model = df.loc[best_idx, 'Model']
        best_auc = df.loc[best_idx, 'auc_roc']
        print(f"\nЛучшая модель: {best_model} (AUC-ROC: {best_auc:.4f})")
        
        print("\n" + "="*60)
        print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("="*60)
        
        for name, res in successful_models.items():
            joblib.dump(res['model'], f'optimized_{name.lower()}_model.joblib')
            joblib.dump(res['params'], f'optimized_{name.lower()}_params.joblib')
            print(f"{name}: модель -> optimized_{name.lower()}_model.joblib, параметры -> optimized_{name.lower()}_params.joblib")
        
        df.to_csv('optimized_svm_models_comparison.csv', index=False)
        print("Сравнительная таблица сохранена в optimized_svm_models_comparison.csv")
        
        print(f"\nДетальный отчет для лучшей модели ({best_model}):")
        best_res = successful_models[best_model]
        y_pred = best_res['model'].predict(X_test)
        print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
        
        print(f"\nОбщее время выполнения: {sum(r['time'] for r in successful_models.values()):.1f} секунд")
        print("Все операции завершены успешно!")
    else:
        print("Все модели завершились с ошибками!")
