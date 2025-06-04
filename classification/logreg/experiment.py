from sklearn.linear_model import (LogisticRegression, RidgeClassifier, 
                                 SGDClassifier, LogisticRegressionCV)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score,
                           matthews_corrcoef, cohen_kappa_score, confusion_matrix,
                           classification_report)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # Для работы с RidgeClassifier
import optuna
import joblib
import warnings
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import time
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_JOBS = mp.cpu_count() - 1 
print(f"Используется {N_JOBS} ядер для параллелизации")

print("Загрузка данных...")
X_train = pd.read_csv('X_cl_train.csv')
X_val = pd.read_csv('X_cl_val.csv')
X_test = pd.read_csv('X_cl_test.csv')
y_train = pd.read_csv('y_cl_train.csv').iloc[:, 0]
y_val = pd.read_csv('y_cl_val.csv').iloc[:, 0]
y_test = pd.read_csv('y_cl_test.csv').iloc[:, 0]

X_combined = pd.concat([X_train, X_val], axis=0, ignore_index=True)
y_combined = pd.concat([y_train, y_val], axis=0, ignore_index=True)

print(f"Размер объединенной выборки: {X_combined.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Распределение классов: {y_combined.value_counts().to_dict()}")

print("Масштабирование данных...")
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

class_weights = compute_class_weight('balanced', classes=np.unique(y_combined), y=y_combined)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
scale_pos_weight = (y_combined == 0).sum() / (y_combined == 1).sum()

cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Быстрое вычисление основных метрик"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'auc_roc': roc_auc_score(y_true, y_pred_proba[:, 1]),
        'auc_pr': average_precision_score(y_true, y_pred_proba[:, 1]),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }

def calculate_metrics_without_proba(y_true, y_pred):
    """Вычисление метрик без вероятностей (для RidgeClassifier)"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }

def print_results(model_name, metrics):
    """Быстрая печать результатов"""
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.4f}")

param_distributions = {
    'LogisticRegression': {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg'],
        'max_iter': [100, 200, 500, 1000, 2000],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9], 
        'fit_intercept': [True, False],
        'tol': [1e-4, 1e-5, 1e-6]
    },
    'RidgeClassifier': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        'fit_intercept': [True, False],
        'max_iter': [100, 200, 500, 1000, 2000],
        'tol': [1e-3, 1e-4, 1e-5],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    },
    'SGDClassifier': {
        'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'],  
        'penalty': ['l1', 'l2', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'fit_intercept': [True, False],
        'max_iter': [500, 1000, 2000, 5000],
        'tol': [1e-3, 1e-4, 1e-5],
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'eta0': [0.01, 0.1, 1.0]
    },
    'LogisticRegressionL1': { 
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200, 500, 1000, 2000],
        'fit_intercept': [True, False],
        'tol': [1e-4, 1e-5, 1e-6]
    },
    'LogisticRegressionElasticNet': {  
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'solver': ['saga'],
        'max_iter': [100, 200, 500, 1000, 2000],
        'fit_intercept': [True, False],
        'tol': [1e-4, 1e-5, 1e-6]
    }
}

def fast_randomized_search(model_name, model_class, param_dist, n_iter=30):
    """Быстрый randomized search для предварительной оптимизации"""
    print(f"\nБыстрая оптимизация {model_name}...")
    

    if model_name == 'LogisticRegression':
        simplified_params = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l2'], 
            'solver': ['lbfgs'],  
            'max_iter': [500, 1000]
        }
        model = LogisticRegression(random_state=42, class_weight='balanced')
        
    elif model_name == 'RidgeClassifier':
        simplified_params = {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['auto'],
            'max_iter': [500, 1000]
        }
        model = RidgeClassifier(class_weight='balanced')
        
    elif model_name == 'SGDClassifier':
        simplified_params = {
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'penalty': ['l2'],
            'loss': ['log_loss'], 
            'max_iter': [1000, 2000]
        }
        model = SGDClassifier(random_state=42, class_weight='balanced')
        
    elif model_name == 'LogisticRegressionL1':
        simplified_params = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['liblinear'], 
            'max_iter': [500, 1000]
        }
        model = LogisticRegression(penalty='l1', random_state=42, class_weight='balanced')
        
    elif model_name == 'LogisticRegressionElasticNet':
        simplified_params = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'l1_ratio': [0.1, 0.5, 0.9],
            'solver': ['saga'],
            'max_iter': [500, 1000]
        }
        model = LogisticRegression(penalty='elasticnet', random_state=42, class_weight='balanced')
    
й
    search = RandomizedSearchCV(
        model, simplified_params, 
        n_iter=min(n_iter, 15), 
        cv=3,  
        scoring='roc_auc' if model_name != 'RidgeClassifier' else 'accuracy',  
        n_jobs=1, 
        random_state=42,
        verbose=0
    )
    
    try:
        search.fit(X_combined_scaled, y_combined)
        return search.best_params_, search.best_score_
    except Exception as e:
        print(f"Ошибка в {model_name}: {e}")
        if model_name == 'LogisticRegression':
            return {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 1000}, 0.5
        elif model_name == 'RidgeClassifier':
            return {'alpha': 1.0, 'solver': 'auto', 'max_iter': 1000}, 0.5
        elif model_name == 'SGDClassifier':
            return {'alpha': 0.0001, 'penalty': 'l2', 'loss': 'log_loss', 'max_iter': 1000}, 0.5
        elif model_name == 'LogisticRegressionL1':
            return {'C': 1.0, 'solver': 'liblinear', 'max_iter': 1000}, 0.5
        elif model_name == 'LogisticRegressionElasticNet':
            return {'C': 1.0, 'l1_ratio': 0.5, 'solver': 'saga', 'max_iter': 1000}, 0.5

def optuna_fine_tune(model_name, base_params, n_trials=30):
    """Тонкая настройка лучших параметров с помощью Optuna"""
    print(f"Тонкая настройка {model_name} с Optuna...")
    
    def objective(trial):
        params = base_params.copy()
        
        if model_name == 'LogisticRegression':
            # Тонкая настройка основных параметров
            current_C = base_params.get('C', 1.0)
            params['C'] = trial.suggest_float('C', current_C * 0.1, current_C * 10.0, log=True)
            
            current_tol = base_params.get('tol', 1e-4)
            params['tol'] = trial.suggest_float('tol', current_tol * 0.1, current_tol * 10.0, log=True)
            
            if base_params.get('penalty') == 'elasticnet':
                current_l1_ratio = base_params.get('l1_ratio', 0.5)
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 
                                                       max(0.01, current_l1_ratio - 0.3),
                                                       min(0.99, current_l1_ratio + 0.3))
            
            model = LogisticRegression(**params, random_state=42, class_weight='balanced')
            
        elif model_name == 'RidgeClassifier':
            current_alpha = base_params.get('alpha', 1.0)
            params['alpha'] = trial.suggest_float('alpha', current_alpha * 0.1, current_alpha * 10.0, log=True)
            
            current_tol = base_params.get('tol', 1e-3)
            params['tol'] = trial.suggest_float('tol', current_tol * 0.1, current_tol * 10.0, log=True)
            
            model = RidgeClassifier(**params, class_weight='balanced')
            
        elif model_name == 'SGDClassifier':
            current_alpha = base_params.get('alpha', 0.0001)
            params['alpha'] = trial.suggest_float('alpha', current_alpha * 0.1, current_alpha * 10.0, log=True)
            
            current_tol = base_params.get('tol', 1e-3)
            params['tol'] = trial.suggest_float('tol', current_tol * 0.1, current_tol * 10.0, log=True)
            
            if base_params.get('penalty') == 'elasticnet':
                current_l1_ratio = base_params.get('l1_ratio', 0.15)
                params['l1_ratio'] = trial.suggest_float('l1_ratio',
                                                       max(0.01, current_l1_ratio - 0.1),
                                                       min(0.99, current_l1_ratio + 0.1))
            
            if base_params.get('learning_rate') in ['constant', 'invscaling']:
                current_eta0 = base_params.get('eta0', 0.01)
                params['eta0'] = trial.suggest_float('eta0', current_eta0 * 0.1, current_eta0 * 10.0, log=True)
            
            model = SGDClassifier(**params, random_state=42, class_weight='balanced')
            
        elif model_name in ['LogisticRegressionL1', 'LogisticRegressionElasticNet']:
            current_C = base_params.get('C', 1.0)
            params['C'] = trial.suggest_float('C', current_C * 0.1, current_C * 10.0, log=True)
            
            current_tol = base_params.get('tol', 1e-4)
            params['tol'] = trial.suggest_float('tol', current_tol * 0.1, current_tol * 10.0, log=True)
            
            if model_name == 'LogisticRegressionL1':
                model = LogisticRegression(penalty='l1', **params, random_state=42, class_weight='balanced')
            else:  
                current_l1_ratio = base_params.get('l1_ratio', 0.5)
                params['l1_ratio'] = trial.suggest_float('l1_ratio',
                                                       max(0.01, current_l1_ratio - 0.3),
                                                       min(0.99, current_l1_ratio + 0.3))
                model = LogisticRegression(penalty='elasticnet', **params, random_state=42, class_weight='balanced')
        

        scores = []
        for train_idx, val_idx in cv_strategy.split(X_combined_scaled, y_combined):
            X_train_cv = X_combined_scaled[train_idx]
            X_val_cv = X_combined_scaled[val_idx]
            y_train_cv = y_combined.iloc[train_idx]
            y_val_cv = y_combined.iloc[val_idx]
            
            try:
                model.fit(X_train_cv, y_train_cv)
                

                if model_name == 'RidgeClassifier':
                    y_pred = model.predict(X_val_cv)
                    scores.append(accuracy_score(y_val_cv, y_pred))
                else:
                    y_pred_proba = model.predict_proba(X_val_cv)
                    scores.append(roc_auc_score(y_val_cv, y_pred_proba[:, 1]))
            except:
                return 0.0
        
        return np.mean(scores)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params, study.best_value

def train_and_evaluate_model(model_name):
    """Обучение и оценка одной модели"""
    start_time = time.time()
    
    try:
        if model_name == 'LogisticRegression':
            model_class = LogisticRegression
        elif model_name == 'RidgeClassifier':
            model_class = RidgeClassifier
        elif model_name == 'SGDClassifier':
            model_class = SGDClassifier
        elif model_name in ['LogisticRegressionL1', 'LogisticRegressionElasticNet']:
            model_class = LogisticRegression
        
        base_params, base_score = fast_randomized_search(
            model_name, model_class, param_distributions[model_name], n_iter=15
        )
        
        print(f"{model_name} - Базовый результат: {base_score:.4f}")
        
        final_params, final_score = optuna_fine_tune(model_name, base_params, n_trials=20)
        

        final_params.update(base_params)
        
        print(f"{model_name} - Финальный результат: {final_score:.4f}")
        

        if model_name == 'LogisticRegression':
            final_model = LogisticRegression(**final_params, random_state=42, class_weight='balanced')
        elif model_name == 'RidgeClassifier':
            final_model = RidgeClassifier(**final_params, class_weight='balanced')
        elif model_name == 'SGDClassifier':
            final_model = SGDClassifier(**final_params, random_state=42, class_weight='balanced')
        elif model_name == 'LogisticRegressionL1':
            final_model = LogisticRegression(penalty='l1', **final_params, random_state=42, class_weight='balanced')
        elif model_name == 'LogisticRegressionElasticNet':
            final_model = LogisticRegression(penalty='elasticnet', **final_params, random_state=42, class_weight='balanced')
        
        final_model.fit(X_combined_scaled, y_combined)
        

        y_pred = final_model.predict(X_test_scaled)
        

        if model_name == 'RidgeClassifier':
            metrics = calculate_metrics_without_proba(y_test, y_pred)
        else:
            y_pred_proba = final_model.predict_proba(X_test_scaled)
            metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        elapsed_time = time.time() - start_time
        print(f"{model_name} завершен за {elapsed_time:.1f} секунд")
        
        return {
            'model': final_model,
            'params': final_params,
            'metrics': metrics,
            'time': elapsed_time
        }
    
    except Exception as e:
        print(f"Ошибка в обучении {model_name}: {e}")
        return None


def parallel_model_training():
    """Параллельное обучение всех моделей"""
    model_names = ['LogisticRegression', 'RidgeClassifier', 'SGDClassifier', 
                   'LogisticRegressionL1', 'LogisticRegressionElasticNet']
    
    print("Начинаем параллельное обучение всех моделей логистической регрессии...")
    total_start_time = time.time()
    

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_model = {executor.submit(train_and_evaluate_model, name): name 
                          for name in model_names}
        
        results = {}
        for future in future_to_model:
            model_name = future_to_model[future]
            try:
                result = future.result()
                if result is not None:
                    results[model_name] = result
            except Exception as exc:
                print(f'{model_name} generated an exception: {exc}')
    
    total_time = time.time() - total_start_time
    print(f"\nВсе модели обучены за {total_time:.1f} секунд")
    
    return results


if __name__ == "__main__":
    results = parallel_model_training()
    
    if not results:
        print("Не удалось обучить ни одной модели!")
        exit(1)
    

    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ВСЕХ МОДЕЛЕЙ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ")
    print("="*60)
    
    for model_name, result in results.items():
        print_results(model_name, result['metrics'])
        print(f"  Время обучения: {result['time']:.1f}с")
    
    comparison_data = []
    for model_name, result in results.items():
        row = {'Model': model_name, 'Time': result['time']}
        row.update(result['metrics'])
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    print(f"\n{comparison_df.round(4)}")

    best_model_idx = comparison_df['f1'].idxmax()
    best_model_name = comparison_df.loc[best_model_idx, 'Model']
    best_f1 = comparison_df.loc[best_model_idx, 'f1']
    
    print(f"\nЛучшая модель: {best_model_name} (F1-score: {best_f1:.4f})")
  
    print("\n" + "="*60)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    joblib.dump(scaler, 'logistic_scaler.joblib')
    print("Скейлер сохранен в logistic_scaler.joblib")
    
    for model_name, result in results.items():
        model_filename = f'optimized_{model_name.lower()}_model.joblib'
        joblib.dump(result['model'], model_filename)
        
        params_filename = f'optimized_{model_name.lower()}_params.joblib'
        joblib.dump(result['params'], params_filename)
        
        print(f"{model_name}: модель -> {model_filename}, параметры -> {params_filename}")
    

    comparison_df.to_csv('optimized_logistic_models_comparison.csv', index=False)
    print("Сравнительная таблица сохранена в optimized_logistic_models_comparison.csv")
    
    print(f"\nДетальный отчет для лучшей модели ({best_model_name}):")
    best_model = results[best_model_name]['model']
    y_test_pred_best = best_model.predict(X_test_scaled)
    print(classification_report(y_test, y_test_pred_best, target_names=['Class 0', 'Class 1']))
    
    if hasattr(best_model, 'coef_'):
        feature_coefficients = pd.DataFrame({
            'feature': X_combined.columns,
            'coefficient': best_model.coef_[0],
            'abs_coefficient': np.abs(best_model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        print(f"\nТоп-10 важных признаков ({best_model_name}) по коэффициентам:")
        print(feature_coefficients.head(10)[['feature', 'coefficient']])
        
        feature_coefficients.to_csv(f'feature_coefficients_{best_model_name.lower()}.csv', index=False)

    print(f"\nСравнение по метрикам:")
    print("=" * 40)
    common_metrics = ['accuracy', 'f1', 'mcc']
    for metric in common_metrics:
        if metric in comparison_df.columns:
            best_metric_idx = comparison_df[metric].idxmax()
            best_metric_model = comparison_df.loc[best_metric_idx, 'Model']
            best_metric_value = comparison_df.loc[best_metric_idx, metric]
            print(f"Лучший {metric}: {best_metric_model} ({best_metric_value:.4f})")

    models_with_auc = comparison_df[comparison_df['auc_roc'].notna()]
    if not models_with_auc.empty:
        best_auc_idx = models_with_auc['auc_roc'].idxmax()
        best_auc_model = models_with_auc.loc[best_auc_idx, 'Model']
        best_auc_value = models_with_auc.loc[best_auc_idx, 'auc_roc']
        print(f"Лучший auc_roc: {best_auc_model} ({best_auc_value:.4f})")
    
    print(f"\nОбщее время выполнения: {sum(r['time'] for r in results.values()):.1f} секунд")
    print("Все операции завершены успешно!")
