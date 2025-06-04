from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score,
                           matthews_corrcoef, cohen_kappa_score, confusion_matrix,
                           classification_report)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import optuna
import joblib
import warnings
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import time
import traceback
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


N_JOBS = max(1, mp.cpu_count() - 2) 
print(f"Используется {N_JOBS} ядер для параллелизации")

print("Загрузка данных...")
try:
    X_train = pd.read_csv('X_cl_train.csv')
    X_val = pd.read_csv('X_cl_val.csv')
    X_test = pd.read_csv('X_cl_test.csv')
    y_train = pd.read_csv('y_cl_train.csv').iloc[:, 0]
    y_val = pd.read_csv('y_cl_val.csv').iloc[:, 0]
    y_test = pd.read_csv('y_cl_test.csv').iloc[:, 0]
except Exception as e:
    print(f"Ошибка загрузки данных: {e}")
    exit(1)


X_combined = pd.concat([X_train, X_val], axis=0, ignore_index=True)
y_combined = pd.concat([y_train, y_val], axis=0, ignore_index=True)

print(f"Размер объединенной выборки: {X_combined.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Распределение классов: {y_combined.value_counts().to_dict()}")

class_weights = compute_class_weight('balanced', classes=np.unique(y_combined), y=y_combined)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Быстрое вычисление основных метрик"""
    try:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'auc_roc': roc_auc_score(y_true, y_pred_proba[:, 1]),
            'auc_pr': average_precision_score(y_true, y_pred_proba[:, 1]),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'kappa': cohen_kappa_score(y_true, y_pred)
        }
    except Exception as e:
        print(f"Ошибка вычисления метрик: {e}")
        return {}

def print_results(model_name, metrics):
    """Быстрая печать результатов"""
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.4f}")

param_distributions = {
    'RandomForest': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 15, 20, 25],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.5],
        'bootstrap': [True],
        'criterion': ['gini', 'entropy']
    },
    'ExtraTrees': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 15, 20, 25],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.5],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    },
    'BaggingClassifier': {
        'n_estimators': [50, 100, 200, 300],
        'max_samples': [0.7, 0.8, 0.9, 1.0],
        'max_features': [0.7, 0.8, 0.9, 1.0],
        'bootstrap': [True],
        'bootstrap_features': [True, False]
    }
}

def fast_randomized_search(model_name, model_class, param_dist, n_iter=20):
    """Быстрый randomized search для предварительной оптимизации"""
    print(f"\nБыстрая оптимизация {model_name}...")
    
    try:
        if model_name == 'RandomForest':
            model = model_class(
                random_state=42,
                class_weight='balanced',
                n_jobs=1
            )
        elif model_name == 'ExtraTrees':
            model = model_class(
                random_state=42,
                class_weight='balanced',
                n_jobs=1
            )
        elif model_name == 'BaggingClassifier':
            try:
                base_estimator = DecisionTreeClassifier(
                    random_state=42,
                    class_weight='balanced'
                )
                model = model_class(
                    estimator=base_estimator, 
                    random_state=42,
                    n_jobs=1
                )
            except TypeError:
                base_estimator = DecisionTreeClassifier(
                    random_state=42,
                    class_weight='balanced'
                )
                model = model_class(
                    base_estimator=base_estimator,
                    random_state=42,
                    n_jobs=1
                )

        search = RandomizedSearchCV(
            model, param_dist, 
            n_iter=n_iter, 
            cv=cv_strategy, 
            scoring='roc_auc',
            n_jobs=min(2, N_JOBS),  
            random_state=42,
            verbose=0
        )
        
        search.fit(X_combined, y_combined)
        
        return search.best_params_, search.best_score_
        
    except Exception as e:
        print(f"Ошибка в fast_randomized_search для {model_name}: {e}")
        return {}, 0.0

def optuna_fine_tune(model_name, base_params, n_trials=30):
    """Тонкая настройка лучших параметров с помощью Optuna"""
    print(f"Тонкая настройка {model_name} с Optuna...")
    
    def objective(trial):
        try:
            if model_name == 'RandomForest':
                params = base_params.copy()

                if 'n_estimators' in base_params:
                    params['n_estimators'] = trial.suggest_int('n_estimators', 
                                                             max(50, base_params['n_estimators'] - 50),
                                                             base_params['n_estimators'] + 100)
                
                if 'max_depth' in base_params and base_params['max_depth'] is not None:
                    params['max_depth'] = trial.suggest_int('max_depth',
                                                          max(5, base_params['max_depth'] - 5),
                                                          base_params['max_depth'] + 5)
                
                params['min_impurity_decrease'] = trial.suggest_float('min_impurity_decrease', 0.0, 0.05)
                
                model = RandomForestClassifier(**params, random_state=42, class_weight='balanced', n_jobs=1)
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_val)
                return roc_auc_score(y_val, y_pred_proba[:, 1])
                
            elif model_name == 'ExtraTrees':
                params = base_params.copy()
                
                if 'n_estimators' in base_params:
                    params['n_estimators'] = trial.suggest_int('n_estimators',
                                                             max(50, base_params['n_estimators'] - 50),
                                                             base_params['n_estimators'] + 100)
                
                params['min_impurity_decrease'] = trial.suggest_float('min_impurity_decrease', 0.0, 0.05)
                
                model = ExtraTreesClassifier(**params, random_state=42, class_weight='balanced', n_jobs=1)
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_val)
                return roc_auc_score(y_val, y_pred_proba[:, 1])
                
            elif model_name == 'BaggingClassifier':
                params = base_params.copy()
                
                if 'n_estimators' in base_params:
                    params['n_estimators'] = trial.suggest_int('n_estimators',
                                                             max(10, base_params['n_estimators'] - 25),
                                                             base_params['n_estimators'] + 50)
                
                try:
                    base_estimator = DecisionTreeClassifier(random_state=42, class_weight='balanced')
                    model = BaggingClassifier(estimator=base_estimator, **params, random_state=42, n_jobs=1)
                except TypeError:
                    base_estimator = DecisionTreeClassifier(random_state=42, class_weight='balanced')
                    model = BaggingClassifier(base_estimator=base_estimator, **params, random_state=42, n_jobs=1)
                
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_val)
                return roc_auc_score(y_val, y_pred_proba[:, 1])
                
        except Exception as e:
            print(f"Ошибка в objective для {model_name}: {e}")
            return 0.0
    
    try:
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(objective, n_trials=n_trials, timeout=300) 
        
        return study.best_params, study.best_value
    except Exception as e:
        print(f"Ошибка в optuna_fine_tune для {model_name}: {e}")
        return {}, 0.0

def train_and_evaluate_model(model_name):
    """Обучение и оценка одной модели"""
    start_time = time.time()
    
    try:
        print(f"\n--- Начинаем обучение {model_name} ---")

        if model_name == 'RandomForest':
            model_class = RandomForestClassifier
        elif model_name == 'ExtraTrees':
            model_class = ExtraTreesClassifier
        elif model_name == 'BaggingClassifier':
            model_class = BaggingClassifier
        else:
            raise ValueError(f"Неизвестная модель: {model_name}")
        
        base_params, base_score = fast_randomized_search(
            model_name, model_class, param_distributions[model_name], n_iter=15
        )
        
        if base_score == 0.0:
            print(f"Не удалось оптимизировать {model_name}")
            return None
        
        print(f"{model_name} - Базовый результат: {base_score:.4f}")

        final_params, final_score = optuna_fine_tune(model_name, base_params, n_trials=20)
        
        final_params.update(base_params)
        
        print(f"{model_name} - Финальный результат: {final_score:.4f}")

        if model_name == 'RandomForest':
            final_params['class_weight'] = 'balanced'
            final_model = RandomForestClassifier(**final_params, random_state=42, n_jobs=N_JOBS)
            
        elif model_name == 'ExtraTrees':
            final_params['class_weight'] = 'balanced'
            final_model = ExtraTreesClassifier(**final_params, random_state=42, n_jobs=N_JOBS)
            
        elif model_name == 'BaggingClassifier':

            try:
                base_estimator = DecisionTreeClassifier(random_state=42, class_weight='balanced')
                final_model = BaggingClassifier(
                    estimator=base_estimator, 
                    **final_params, 
                    random_state=42, 
                    n_jobs=N_JOBS
                )
            except TypeError:
                base_estimator = DecisionTreeClassifier(random_state=42, class_weight='balanced')
                final_model = BaggingClassifier(
                    base_estimator=base_estimator, 
                    **final_params, 
                    random_state=42, 
                    n_jobs=N_JOBS
                )
        
        final_model.fit(X_combined, y_combined)

        y_pred = final_model.predict(X_test)
        y_pred_proba = final_model.predict_proba(X_test)
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
        elapsed_time = time.time() - start_time
        print(f"Ошибка при обучении {model_name}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return {
            'model': None,
            'params': {},
            'metrics': {},
            'time': elapsed_time,
            'error': str(e)
        }

def sequential_model_training():
    """Последовательное обучение всех моделей"""
    model_names = ['RandomForest', 'ExtraTrees', 'BaggingClassifier']
    
    print("Начинаем последовательное обучение всех моделей...")
    total_start_time = time.time()
    
    results = {}
    
    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Обучение модели: {model_name}")
        print(f"{'='*50}")
        
        result = train_and_evaluate_model(model_name)
        if result is not None:
            results[model_name] = result
        else:
            print(f"Пропускаем {model_name} из-за ошибки")
    
    total_time = time.time() - total_start_time
    print(f"\nВсе модели обучены за {total_time:.1f} секунд")
    
    return results

if __name__ == "__main__":
    results = sequential_model_training()
    
    if not results:
        print("Не удалось обучить ни одной модели!")
        exit(1)

    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ ВСЕХ МОДЕЛЕЙ")
    print("="*70)
    
    for model_name, result in results.items():
        if 'error' in result:
            print(f"\n{model_name}: ОШИБКА - {result['error']}")
        else:
            print_results(model_name, result['metrics'])
            print(f"  Время обучения: {result['time']:.1f}с")
    

    comparison_data = []
    for model_name, result in results.items():
        if 'error' not in result and result['metrics']:
            row = {'Model': model_name, 'Time': result['time']}
            row.update(result['metrics'])
            comparison_data.append(row)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        print(f"\n{comparison_df.round(4)}")

        if len(comparison_df) > 0:
            best_auc_idx = comparison_df['auc_roc'].idxmax()
            best_model_name = comparison_df.loc[best_auc_idx, 'Model']
            best_auc_score = comparison_df.loc[best_auc_idx, 'auc_roc']
            
            print(f"\nЛучшая модель по AUC-ROC: {best_model_name} ({best_auc_score:.4f})")

            if best_model_name in results and results[best_model_name]['model'] is not None:
                print(f"\nДетальный отчет для лучшей модели ({best_model_name}):")
                best_model = results[best_model_name]['model']
                y_test_pred_best = best_model.predict(X_test)
                print(classification_report(y_test, y_test_pred_best, target_names=['Class 0', 'Class 1']))

        print("\n" + "="*70)
        print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("="*70)
        
        for model_name, result in results.items():
            if 'error' not in result and result['model'] is not None:

                model_filename = f'optimized_{model_name.lower()}_model.joblib'
                joblib.dump(result['model'], model_filename)
                
                params_filename = f'optimized_{model_name.lower()}_params.joblib'
                joblib.dump(result['params'], params_filename)
                
                print(f"{model_name}: модель -> {model_filename}, параметры -> {params_filename}")
        
        comparison_df.to_csv('optimized_models_comparison.csv', index=False)
        print("Сравнительная таблица сохранена в optimized_models_comparison.csv")
    
    print(f"\nОбщее время выполнения: {sum(r['time'] for r in results.values()):.1f} секунд")
    print("Операции завершены!")
