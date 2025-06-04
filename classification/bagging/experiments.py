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

class_weights = compute_class_weight('balanced', classes=np.unique(y_combined), y=y_combined)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}


cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Быстрое вычисление основных метрик"""
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

def print_results(model_name, metrics):
    """Быстрая печать результатов"""
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.4f}")

param_distributions = {
    'RandomForest': {
        'n_estimators': [50, 100, 200, 300, 500, 700, 1000],
        'max_depth': [None, 5, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8, 12],
        'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'max_leaf_nodes': [None, 10, 20, 50, 100],
        'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1],
        'oob_score': [True, False]
    },
    'ExtraTrees': {
        'n_estimators': [50, 100, 200, 300, 500, 700, 1000],
        'max_depth': [None, 5, 10, 15, 20, 25, 30],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8, 12],
        'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy'],
        'max_leaf_nodes': [None, 10, 20, 50, 100],
        'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1]
    },
    'BaggingClassifier': {
        'n_estimators': [10, 50, 100, 200, 300, 500],
        'max_samples': [0.5, 0.7, 0.8, 0.9, 1.0],
        'max_features': [0.5, 0.7, 0.8, 0.9, 1.0],
        'bootstrap': [True, False],
        'bootstrap_features': [True, False],
        'oob_score': [True, False],
        'warm_start': [True, False]
    },
    'BalancedRandomForest': {
        'n_estimators': [50, 100, 200, 300, 500, 700],
        'max_depth': [None, 5, 10, 15, 20, 25],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None, 0.5],
        'criterion': ['gini', 'entropy'],
        'sampling_strategy': ['auto', 'balanced', 'balanced_subsample']
    }
}

def fast_randomized_search(model_name, model_class, param_dist, n_iter=50):
    """Быстрый randomized search для предварительной оптимизации"""
    print(f"\nБыстрая оптимизация {model_name}...")
    
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
        base_estimator = DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced'
        )
        model = model_class(
            base_estimator=base_estimator,
            random_state=42,
            n_jobs=1
        )
    elif model_name == 'BalancedRandomForest':
        model = RandomForestClassifier(
            random_state=42,
            class_weight='balanced_subsample',
            n_jobs=1
        )
        param_dist = param_distributions['RandomForest'].copy()
        param_dist['class_weight'] = ['balanced', 'balanced_subsample']

    search = RandomizedSearchCV(
        model, param_dist, 
        n_iter=n_iter, 
        cv=cv_strategy, 
        scoring='roc_auc',
        n_jobs=N_JOBS,
        random_state=42,
        verbose=0
    )
    
    search.fit(X_combined, y_combined)
    
    return search.best_params_, search.best_score_

def optuna_fine_tune(model_name, base_params, n_trials=100):
    """Тонкая настройка лучших параметров с помощью Optuna"""
    print(f"Тонкая настройка {model_name} с Optuna...")
    
    def objective(trial):
        if model_name == 'RandomForest':
            params = base_params.copy()

            if 'n_estimators' in base_params:
                params['n_estimators'] = trial.suggest_int('n_estimators', 
                                                         max(50, base_params['n_estimators'] - 100),
                                                         base_params['n_estimators'] + 200)
            
            if 'max_depth' in base_params and base_params['max_depth'] is not None:
                params['max_depth'] = trial.suggest_int('max_depth',
                                                      max(5, base_params['max_depth'] - 5),
                                                      base_params['max_depth'] + 10)
            
            if 'min_samples_split' in base_params:
                params['min_samples_split'] = trial.suggest_int('min_samples_split',
                                                              max(2, base_params['min_samples_split'] - 3),
                                                              base_params['min_samples_split'] + 5)
            
            if 'min_samples_leaf' in base_params:
                params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf',
                                                             max(1, base_params['min_samples_leaf'] - 2),
                                                             base_params['min_samples_leaf'] + 4)
            
            params['min_impurity_decrease'] = trial.suggest_float('min_impurity_decrease', 0.0, 0.1)
            
            model = RandomForestClassifier(**params, random_state=42, class_weight='balanced', n_jobs=1)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)
            return roc_auc_score(y_val, y_pred_proba[:, 1])
            
        elif model_name == 'ExtraTrees':
            params = base_params.copy()
            
            if 'n_estimators' in base_params:
                params['n_estimators'] = trial.suggest_int('n_estimators',
                                                         max(50, base_params['n_estimators'] - 100),
                                                         base_params['n_estimators'] + 200)
            
            if 'max_depth' in base_params and base_params['max_depth'] is not None:
                params['max_depth'] = trial.suggest_int('max_depth',
                                                      max(5, base_params['max_depth'] - 5),
                                                      base_params['max_depth'] + 10)
            
            if 'min_samples_split' in base_params:
                params['min_samples_split'] = trial.suggest_int('min_samples_split',
                                                              max(2, base_params['min_samples_split'] - 3),
                                                              base_params['min_samples_split'] + 5)
            
            params['min_impurity_decrease'] = trial.suggest_float('min_impurity_decrease', 0.0, 0.1)
            
            model = ExtraTreesClassifier(**params, random_state=42, class_weight='balanced', n_jobs=1)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)
            return roc_auc_score(y_val, y_pred_proba[:, 1])
            
        elif model_name == 'BaggingClassifier':
            params = base_params.copy()
            
            if 'n_estimators' in base_params:
                params['n_estimators'] = trial.suggest_int('n_estimators',
                                                         max(10, base_params['n_estimators'] - 50),
                                                         base_params['n_estimators'] + 100)
            
            if 'max_samples' in base_params:
                params['max_samples'] = trial.suggest_float('max_samples',
                                                          max(0.3, base_params['max_samples'] - 0.2),
                                                          min(1.0, base_params['max_samples'] + 0.2))
            
            if 'max_features' in base_params:
                params['max_features'] = trial.suggest_float('max_features',
                                                           max(0.3, base_params['max_features'] - 0.2),
                                                           min(1.0, base_params['max_features'] + 0.2))
            
            base_estimator = DecisionTreeClassifier(random_state=42, class_weight='balanced')
            model = BaggingClassifier(base_estimator=base_estimator, **params, random_state=42, n_jobs=1)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)
            return roc_auc_score(y_val, y_pred_proba[:, 1])
            
        elif model_name == 'BalancedRandomForest':
            params = base_params.copy()
            
            if 'n_estimators' in base_params:
                params['n_estimators'] = trial.suggest_int('n_estimators',
                                                         max(50, base_params['n_estimators'] - 100),
                                                         base_params['n_estimators'] + 200)
            
            if 'max_depth' in base_params and base_params['max_depth'] is not None:
                params['max_depth'] = trial.suggest_int('max_depth',
                                                      max(5, base_params['max_depth'] - 5),
                                                      base_params['max_depth'] + 10)
            
            params['min_impurity_decrease'] = trial.suggest_float('min_impurity_decrease', 0.0, 0.1)
            params['class_weight'] = 'balanced_subsample'
            
            model = RandomForestClassifier(**params, random_state=42, n_jobs=1)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)
            return roc_auc_score(y_val, y_pred_proba[:, 1])
    
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

    if model_name == 'RandomForest':
        model_class = RandomForestClassifier
    elif model_name == 'ExtraTrees':
        model_class = ExtraTreesClassifier
    elif model_name == 'BaggingClassifier':
        model_class = BaggingClassifier
    elif model_name == 'BalancedRandomForest':
        model_class = RandomForestClassifier
    
    base_params, base_score = fast_randomized_search(
        model_name, model_class, param_distributions[model_name], n_iter=30
    )
    
    print(f"{model_name} - Базовый результат: {base_score:.4f}")
    
    final_params, final_score = optuna_fine_tune(model_name, base_params, n_trials=50)
    
    final_params.update(base_params)
    
    print(f"{model_name} - Финальный результат: {final_score:.4f}")
   
    if model_name == 'RandomForest':
        final_params['class_weight'] = 'balanced'
        final_model = RandomForestClassifier(**final_params, random_state=42, n_jobs=N_JOBS)
        final_model.fit(X_combined, y_combined)
        
    elif model_name == 'ExtraTrees':
        final_params['class_weight'] = 'balanced'
        final_model = ExtraTreesClassifier(**final_params, random_state=42, n_jobs=N_JOBS)
        final_model.fit(X_combined, y_combined)
        
    elif model_name == 'BaggingClassifier':
        base_estimator = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        final_model = BaggingClassifier(
            base_estimator=base_estimator, 
            **final_params, 
            random_state=42, 
            n_jobs=N_JOBS
        )
        final_model.fit(X_combined, y_combined)
        
    elif model_name == 'BalancedRandomForest':
        final_params['class_weight'] = 'balanced_subsample'
        final_model = RandomForestClassifier(**final_params, random_state=42, n_jobs=N_JOBS)
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

def parallel_model_training():
    """Параллельное обучение всех моделей"""
    model_names = ['RandomForest', 'ExtraTrees', 'BaggingClassifier', 'BalancedRandomForest']
    
    print("Начинаем параллельное обучение всех моделей случайного леса...")
    total_start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=min(len(model_names), N_JOBS)) as executor:
        future_to_model = {executor.submit(train_and_evaluate_model, name): name 
                          for name in model_names}
        
        results = {}
        for future in future_to_model:
            model_name = future_to_model[future]
            try:
                result = future.result()
                results[model_name] = result
            except Exception as exc:
                print(f'{model_name} generated an exception: {exc}')
    
    total_time = time.time() - total_start_time
    print(f"\nВсе модели обучены за {total_time:.1f} секунд")
    
    return results

# Запуск обучения
if __name__ == "__main__":
    results = parallel_model_training()
    
    # Вывод результатов
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ ВСЕХ МОДЕЛЕЙ СЛУЧАЙНОГО ЛЕСА")
    print("="*70)
    
    for model_name, result in results.items():
        print_results(model_name, result['metrics'])
        print(f"  Время обучения: {result['time']:.1f}с")
    
    # Создание сравнительной таблицы
    comparison_data = []
    for model_name, result in results.items():
        row = {'Model': model_name, 'Time': result['time']}
        row.update(result['metrics'])
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    print(f"\n{comparison_df.round(4)}")
    

    best_auc_idx = comparison_df['auc_roc'].idxmax()
    best_f1_idx = comparison_df['f1'].idxmax()
    best_mcc_idx = comparison_df['mcc'].idxmax()
    
    best_auc_model = comparison_df.loc[best_auc_idx, 'Model']
    best_f1_model = comparison_df.loc[best_f1_idx, 'Model']
    best_mcc_model = comparison_df.loc[best_mcc_idx, 'Model']
    
    print(f"\nЛучшие модели:")
    print(f"  По AUC-ROC: {best_auc_model} ({comparison_df.loc[best_auc_idx, 'auc_roc']:.4f})")
    print(f"  По F1-score: {best_f1_model} ({comparison_df.loc[best_f1_idx, 'f1']:.4f})")
    print(f"  По MCC: {best_mcc_model} ({comparison_df.loc[best_mcc_idx, 'mcc']:.4f})")
    

    print("\n" + "="*70)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*70)
    
    for model_name, result in results.items():

        model_filename = f'optimized_{model_name.lower()}_model.joblib'
        joblib.dump(result['model'], model_filename)
        
        # Сохранение параметров
        params_filename = f'optimized_{model_name.lower()}_params.joblib'
        joblib.dump(result['params'], params_filename)
        
        print(f"{model_name}: модель -> {model_filename}, параметры -> {params_filename}")
    
    comparison_df.to_csv('optimized_random_forest_models_comparison.csv', index=False)
    print("Сравнительная таблица сохранена в optimized_random_forest_models_comparison.csv")
    
    print(f"\nДетальный отчет для лучшей модели по AUC-ROC ({best_auc_model}):")
    best_model = results[best_auc_model]['model']
    y_test_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_test_pred_best, target_names=['Class 0', 'Class 1']))
    
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_combined.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nТоп-15 важных признаков ({best_auc_model}):")
        print(feature_importance.head(15))
        
        feature_importance.to_csv(f'feature_importance_{best_auc_model.lower()}.csv', index=False)
        
        print(f"\nСтатистика важности признаков:")
        print(f"  Среднее: {feature_importance['importance'].mean():.4f}")
        print(f"  Медиана: {feature_importance['importance'].median():.4f}")
        print(f"  Стандартное отклонение: {feature_importance['importance'].std():.4f}")
        print(f"  Количество признаков с важностью > 0.01: {(feature_importance['importance'] > 0.01).sum()}")
    
    print(f"\nДополнительная информация о моделях:")
    for model_name, result in results.items():
        model = result['model']
        print(f"\n{model_name}:")
        
        if hasattr(model, 'n_estimators'):
            print(f"  Количество деревьев: {model.n_estimators}")
        
        if hasattr(model, 'max_depth'):
            print(f"  Максимальная глубина: {model.max_depth}")
        
        if hasattr(model, 'min_samples_split'):
            print(f"  Минимальные образцы для разделения: {model.min_samples_split}")
        
        if hasattr(model, 'min_samples_leaf'):
            print(f"  Минимальные образцы в листе: {model.min_samples_leaf}")
        
        if hasattr(model, 'max_features'):
            print(f"  Максимальные признаки: {model.max_features}")
        
        if hasattr(model, 'oob_score_') and model.oob_score is True:
            print(f"  OOB Score: {model.oob_score_:.4f}")
    
    print(f"\nОбщее время выполнения: {sum(r['time'] for r in results.values()):.1f} секунд")
    print("Все операции завершены успешно!")
    

    print(f"\n" + "="*70)
    print("СРАВНЕНИЕ С БАЗОВЫМИ МОДЕЛЯМИ")
    print("="*70)
    
    baseline_results = {}
    baseline_models = {
        'BaselineRandomForest': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=N_JOBS),
        'BaselineExtraTrees': ExtraTreesClassifier(random_state=42, class_weight='balanced', n_jobs=N_JOBS)
    }
    
    for name, model in baseline_models.items():
        start_time = time.time()
        model.fit(X_combined, y_combined)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        elapsed_time = time.time() - start_time
        
        baseline_results[name] = {
            'metrics': metrics,
            'time': elapsed_time
        }
        
        print_results(name, metrics)
        print(f"  Время обучения: {elapsed_time:.1f}с")
    
    print(f"\nУлучшения после оптимизации:")
    if 'RandomForest' in results and 'BaselineRandomForest' in baseline_results:
        rf_improvement = results['RandomForest']['metrics']['auc_roc'] - baseline_results['BaselineRandomForest']['metrics']['auc_roc']
        print(f"  RandomForest AUC-ROC: +{rf_improvement:.4f}")
    
    if 'ExtraTrees' in results and 'BaselineExtraTrees' in baseline_results:
        et_improvement = results['ExtraTrees']['metrics']['auc_roc'] - baseline_results['BaselineExtraTrees']['metrics']['auc_roc']
        print(f"  ExtraTrees AUC-ROC: +{et_improvement:.4f}")
