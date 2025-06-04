from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score,
                           matthews_corrcoef, cohen_kappa_score, confusion_matrix,
                           classification_report)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
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

def print_results(model_name, metrics):
    """Быстрая печать результатов"""
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.4f}")

param_distributions = {
    'AdaBoost': {
        'n_estimators': [50, 100, 200, 300, 500],
        'learning_rate': [0.01, 0.1, 0.5, 1.0, 1.5, 2.0],
        'algorithm': ['SAMME', 'SAMME.R']
    },
    'GradientBoosting': {
        'n_estimators': [50, 100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 9, 11],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'subsample': [0.6, 0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300, 500, 700],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 6, 9, 12, 15],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300, 500, 700],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'num_leaves': [15, 31, 63, 127, 255],
        'min_child_samples': [5, 10, 20, 30, 50],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0]
    },
    'CatBoost': {
        'iterations': [100, 200, 300, 500, 700],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'depth': [4, 6, 8, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'border_count': [32, 64, 128, 254],
        'bagging_temperature': [0, 0.5, 1.0, 2.0]
    }
}

def fast_randomized_search(model_name, model_class, param_dist, n_iter=50):
    """Быстрый randomized search для предварительной оптимизации"""
    print(f"\nБыстрая оптимизация {model_name}...")
    
    if model_name == 'AdaBoost':
        model = model_class(random_state=42)
        fit_params = {'sample_weight': np.array([class_weight_dict[y] for y in y_combined])}
    elif model_name == 'GradientBoosting':
        model = model_class(random_state=42)
        fit_params = {'sample_weight': np.array([class_weight_dict[y] for y in y_combined])}
    elif model_name == 'XGBoost':
        model = model_class(
            random_state=42, 
            eval_metric='logloss', 
            use_label_encoder=False,
            scale_pos_weight=scale_pos_weight,
            n_jobs=1 
        )
        fit_params = {}
    elif model_name == 'LightGBM':
        model = model_class(
            random_state=42, 
            verbose=-1, 
            class_weight='balanced',
            n_jobs=1
        )
        fit_params = {}
    elif model_name == 'CatBoost':
        model = model_class(
            random_state=42, 
            verbose=False,
            class_weights=[1, scale_pos_weight],
            thread_count=1
        )
        fit_params = {}
    
    search = RandomizedSearchCV(
        model, param_dist, 
        n_iter=n_iter, 
        cv=cv_strategy, 
        scoring='roc_auc',
        n_jobs=N_JOBS if model_name not in ['XGBoost', 'LightGBM', 'CatBoost'] else 1,
        random_state=42,
        verbose=0
    )
    
    search.fit(X_combined, y_combined, **fit_params)
    
    return search.best_params_, search.best_score_

def optuna_fine_tune(model_name, base_params, n_trials=100):
    """Тонкая настройка лучших параметров с помощью Optuna"""
    print(f"Тонкая настройка {model_name} с Optuna...")
    
    def objective(trial):
        if model_name == 'AdaBoost':
            params = base_params.copy()
            params['n_estimators'] = trial.suggest_int('n_estimators', 
                                                     max(50, base_params['n_estimators'] - 100),
                                                     base_params['n_estimators'] + 100)
            params['learning_rate'] = trial.suggest_float('learning_rate',
                                                        max(0.01, base_params['learning_rate'] * 0.5),
                                                        base_params['learning_rate'] * 1.5)
            
            model = AdaBoostClassifier(**params, random_state=42)
            model.fit(X_train, y_train, sample_weight=np.array([class_weight_dict[y] for y in y_train]))
            y_pred_proba = model.predict_proba(X_val)
            return roc_auc_score(y_val, y_pred_proba[:, 1])
            
        elif model_name == 'GradientBoosting':
            params = base_params.copy()
            params['n_estimators'] = trial.suggest_int('n_estimators',
                                                     max(50, base_params['n_estimators'] - 100),
                                                     base_params['n_estimators'] + 100)
            params['learning_rate'] = trial.suggest_float('learning_rate',
                                                        max(0.01, base_params['learning_rate'] * 0.5),
                                                        base_params['learning_rate'] * 1.5)
            params['max_depth'] = trial.suggest_int('max_depth',
                                                  max(3, base_params['max_depth'] - 2),
                                                  base_params['max_depth'] + 2)
            
            model = GradientBoostingClassifier(**params, random_state=42)
            model.fit(X_train, y_train, sample_weight=np.array([class_weight_dict[y] for y in y_train]))
            y_pred_proba = model.predict_proba(X_val)
            return roc_auc_score(y_val, y_pred_proba[:, 1])
            
        elif model_name == 'XGBoost':
            params = base_params.copy()
            params['n_estimators'] = trial.suggest_int('n_estimators',
                                                     max(100, base_params['n_estimators'] - 100),
                                                     base_params['n_estimators'] + 100)
            params['learning_rate'] = trial.suggest_float('learning_rate',
                                                        max(0.01, base_params['learning_rate'] * 0.5),
                                                        base_params['learning_rate'] * 1.5)
            params['max_depth'] = trial.suggest_int('max_depth',
                                                  max(3, base_params['max_depth'] - 2),
                                                  base_params['max_depth'] + 2)
            params['scale_pos_weight'] = scale_pos_weight
            params['eval_metric'] = 'logloss'
            params['use_label_encoder'] = False
            
            model = xgb.XGBClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)
            return roc_auc_score(y_val, y_pred_proba[:, 1])
            
        elif model_name == 'LightGBM':
            params = base_params.copy()
            params['n_estimators'] = trial.suggest_int('n_estimators',
                                                     max(100, base_params['n_estimators'] - 100),
                                                     base_params['n_estimators'] + 100)
            params['learning_rate'] = trial.suggest_float('learning_rate',
                                                        max(0.01, base_params['learning_rate'] * 0.5),
                                                        base_params['learning_rate'] * 1.5)
            params['num_leaves'] = trial.suggest_int('num_leaves',
                                                   max(15, base_params['num_leaves'] - 30),
                                                   base_params['num_leaves'] + 30)
            params['class_weight'] = 'balanced'
            
            model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)
            return roc_auc_score(y_val, y_pred_proba[:, 1])
            
        elif model_name == 'CatBoost':
            params = base_params.copy()
            params['iterations'] = trial.suggest_int('iterations',
                                                   max(100, base_params['iterations'] - 100),
                                                   base_params['iterations'] + 100)
            params['learning_rate'] = trial.suggest_float('learning_rate',
                                                        max(0.01, base_params['learning_rate'] * 0.5),
                                                        base_params['learning_rate'] * 1.5)
            params['depth'] = trial.suggest_int('depth',
                                              max(4, base_params['depth'] - 2),
                                              base_params['depth'] + 2)
            params['class_weights'] = [1, scale_pos_weight]
            
            model = cb.CatBoostClassifier(**params, random_state=42, verbose=False)
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
    
    if model_name == 'AdaBoost':
        model_class = AdaBoostClassifier
    elif model_name == 'GradientBoosting':
        model_class = GradientBoostingClassifier
    elif model_name == 'XGBoost':
        model_class = xgb.XGBClassifier
    elif model_name == 'LightGBM':
        model_class = lgb.LGBMClassifier
    elif model_name == 'CatBoost':
        model_class = cb.CatBoostClassifier
    
    base_params, base_score = fast_randomized_search(
        model_name, model_class, param_distributions[model_name], n_iter=30
    )
    
    print(f"{model_name} - Базовый результат: {base_score:.4f}")
    
    final_params, final_score = optuna_fine_tune(model_name, base_params, n_trials=50)
    
    final_params.update(base_params)
    
    print(f"{model_name} - Финальный результат: {final_score:.4f}")
    
    if model_name == 'AdaBoost':
        final_model = AdaBoostClassifier(**final_params, random_state=42)
        final_model.fit(X_combined, y_combined, 
                       sample_weight=np.array([class_weight_dict[y] for y in y_combined]))
    elif model_name == 'GradientBoosting':
        final_model = GradientBoostingClassifier(**final_params, random_state=42)
        final_model.fit(X_combined, y_combined,
                       sample_weight=np.array([class_weight_dict[y] for y in y_combined]))
    elif model_name == 'XGBoost':
        final_params.update({
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'logloss',
            'use_label_encoder': False
        })
        final_model = xgb.XGBClassifier(**final_params, random_state=42)
        final_model.fit(X_combined, y_combined)
    elif model_name == 'LightGBM':
        final_params['class_weight'] = 'balanced'
        final_model = lgb.LGBMClassifier(**final_params, random_state=42, verbose=-1)
        final_model.fit(X_combined, y_combined)
    elif model_name == 'CatBoost':
        final_params['class_weights'] = [1, scale_pos_weight]
        final_model = cb.CatBoostClassifier(**final_params, random_state=42, verbose=False)
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
    model_names = ['AdaBoost', 'GradientBoosting', 'XGBoost', 'LightGBM', 'CatBoost']
    
    print("Начинаем параллельное обучение всех моделей...")
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

if __name__ == "__main__":
    results = parallel_model_training()
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ВСЕХ МОДЕЛЕЙ")
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
    
    best_model_idx = comparison_df['auc_roc'].idxmax()
    best_model_name = comparison_df.loc[best_model_idx, 'Model']
    best_auc = comparison_df.loc[best_model_idx, 'auc_roc']
    
    print(f"\nЛучшая модель: {best_model_name} (AUC-ROC: {best_auc:.4f})")
    
    print("\n" + "="*60)
    print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("="*60)
    
    for model_name, result in results.items():
        model_filename = f'optimized_{model_name.lower()}_model.joblib'
        joblib.dump(result['model'], model_filename)
        
        params_filename = f'optimized_{model_name.lower()}_params.joblib'
        joblib.dump(result['params'], params_filename)
        
        print(f"{model_name}: модель -> {model_filename}, параметры -> {params_filename}")
    

    comparison_df.to_csv('optimized_models_comparison.csv', index=False)
    print("Сравнительная таблица сохранена в optimized_models_comparison.csv")
    

    print(f"\nДетальный отчет для лучшей модели ({best_model_name}):")
    best_model = results[best_model_name]['model']
    y_test_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_test_pred_best, target_names=['Class 0', 'Class 1']))
    
 
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_combined.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nТоп-10 важных признаков ({best_model_name}):")
        print(feature_importance.head(10))
        
        feature_importance.to_csv(f'feature_importance_{best_model_name.lower()}.csv', index=False)
    
    print(f"\nОбщее время выполнения: {sum(r['time'] for r in results.values()):.1f} секунд")
    print("Все операции завершены успешно!")
