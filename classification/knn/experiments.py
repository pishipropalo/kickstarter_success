from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score,
                           matthews_corrcoef, cohen_kappa_score, confusion_matrix,
                           classification_report)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import optuna
import joblib
import warnings
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
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


def create_knn_models():
    """Создание различных вариантов KNN моделей"""
    models = {}
    

    models['KNN_Euclidean'] = {
        'base_model': KNeighborsClassifier,
        'fixed_params': {'metric': 'euclidean', 'n_jobs': N_JOBS},
        'param_dist': {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 21, 31, 51, 71, 101],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]  
        }
    }
    
    models['KNN_Manhattan'] = {
        'base_model': KNeighborsClassifier,
        'fixed_params': {'metric': 'manhattan', 'n_jobs': N_JOBS},
        'param_dist': {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 21, 31, 51, 71, 101],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    }
    
    models['KNN_Minkowski'] = {
        'base_model': KNeighborsClassifier,
        'fixed_params': {'metric': 'minkowski', 'n_jobs': N_JOBS},
        'param_dist': {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 21, 31, 51, 71, 101],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2, 3, 4, 5]
        }
    }
    
    models['KNN_Cosine'] = {
        'base_model': KNeighborsClassifier,
        'fixed_params': {'metric': 'cosine', 'algorithm': 'brute', 'n_jobs': N_JOBS},
        'param_dist': {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 21, 31, 51, 71, 101],
            'weights': ['uniform', 'distance']
        }
    }
    
    models['KNN_Scaled_Standard'] = {
        'base_model': Pipeline,
        'fixed_params': {},
        'param_dist': {
            'scaler': [StandardScaler()],
            'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21, 31, 51, 71],
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan'],
            'knn__algorithm': ['auto', 'ball_tree', 'kd_tree']
        },
        'pipeline_steps': [('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=N_JOBS))]
    }
    
    models['KNN_Scaled_MinMax'] = {
        'base_model': Pipeline,
        'fixed_params': {},
        'param_dist': {
            'scaler': [MinMaxScaler()],
            'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21, 31, 51, 71],
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan'],
            'knn__algorithm': ['auto', 'ball_tree', 'kd_tree']
        },
        'pipeline_steps': [('scaler', MinMaxScaler()), ('knn', KNeighborsClassifier(n_jobs=N_JOBS))]
    }
    
    models['KNN_Scaled_Robust'] = {
        'base_model': Pipeline,
        'fixed_params': {},
        'param_dist': {
            'scaler': [RobustScaler()],
            'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21, 31, 51, 71],
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan'],
            'knn__algorithm': ['auto', 'ball_tree', 'kd_tree']
        },
        'pipeline_steps': [('scaler', RobustScaler()), ('knn', KNeighborsClassifier(n_jobs=N_JOBS))]
    }
    
    models['KNN_FeatureSelection_F'] = {
        'base_model': Pipeline,
        'fixed_params': {},
        'param_dist': {
            'selector__k': [10, 20, 30, 50, 'all'],
            'scaler': [StandardScaler()],
            'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21, 31],
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan']
        },
        'pipeline_steps': [
            ('selector', SelectKBest(f_classif)),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_jobs=N_JOBS))
        ]
    }
    
    models['KNN_FeatureSelection_MI'] = {
        'base_model': Pipeline,
        'fixed_params': {},
        'param_dist': {
            'selector__k': [10, 20, 30, 50, 'all'],
            'scaler': [StandardScaler()],
            'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21, 31],
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan']
        },
        'pipeline_steps': [
            ('selector', SelectKBest(mutual_info_classif)),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_jobs=N_JOBS))
        ]
    }
    
    models['KNN_PCA'] = {
        'base_model': Pipeline,
        'fixed_params': {},
        'param_dist': {
            'scaler': [StandardScaler()],
            'pca__n_components': [0.8, 0.9, 0.95, 0.99, None],
            'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 21, 31],
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan']
        },
        'pipeline_steps': [
            ('scaler', StandardScaler()),
            ('pca', PCA(random_state=42)),
            ('knn', KNeighborsClassifier(n_jobs=N_JOBS))
        ]
    }
    

    models['Weighted_KNN'] = {
        'base_model': Pipeline,
        'fixed_params': {},
        'param_dist': {
            'scaler': [StandardScaler()],
            'knn__n_neighbors': [5, 7, 9, 11, 15, 21, 31],
            'knn__weights': ['distance'],
            'knn__metric': ['euclidean', 'manhattan'],
        },
        'pipeline_steps': [
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_jobs=N_JOBS))
        ],
        'class_weight': 'balanced'
    }

    models['Bagged_KNN'] = {
        'base_model': BaggingClassifier,
        'fixed_params': {
            'random_state': 42,
            'n_jobs': N_JOBS
        },
        'param_dist': {
            'base_estimator': [KNeighborsClassifier(n_neighbors=k, weights=w, metric=m) 
                              for k in [5, 7, 9, 11, 15]
                              for w in ['uniform', 'distance']
                              for m in ['euclidean', 'manhattan']],
            'n_estimators': [10, 20, 30, 50, 100],
            'max_samples': [0.5, 0.7, 0.8, 1.0],
            'max_features': [0.5, 0.7, 0.8, 1.0]
        }
    }
    
    return models

def fast_randomized_search(model_name, model_config, n_iter=30):
    """Быстрый randomized search для предварительной оптимизации"""
    print(f"\nБыстрая оптимизация {model_name}...")
    
    if 'pipeline_steps' in model_config:

        model = model_config['base_model'](model_config['pipeline_steps'])
    else:

        model = model_config['base_model'](**model_config['fixed_params'])

    fit_params = {}
    if 'class_weight' in model_config:
        if hasattr(model, 'set_params'):
            try:
                if 'knn' in model_config['param_dist']:
                    fit_params = {'knn__sample_weight': np.array([class_weight_dict[y] for y in y_combined])}
            except:
                pass

    search = RandomizedSearchCV(
        model, 
        model_config['param_dist'], 
        n_iter=n_iter, 
        cv=cv_strategy, 
        scoring='roc_auc',
        n_jobs=1, 
        random_state=42,
        verbose=0
    )
    
    try:
        search.fit(X_combined, y_combined, **fit_params)
        return search.best_params_, search.best_score_
    except Exception as e:
        print(f"Ошибка при оптимизации {model_name}: {e}")
        return {}, 0.0

def optuna_fine_tune(model_name, model_config, base_params, n_trials=50):
    """Тонкая настройка лучших параметров с помощью Optuna"""
    print(f"Тонкая настройка {model_name} с Optuna...")
    
    def objective(trial):
        try:
            params = base_params.copy()
            
            if 'knn__n_neighbors' in base_params:
                base_k = base_params.get('knn__n_neighbors', 5)
                params['knn__n_neighbors'] = trial.suggest_int('knn__n_neighbors', 
                                                             max(3, base_k - 10), 
                                                             min(101, base_k + 10))
                if 'knn__weights' in base_params:
                    params['knn__weights'] = trial.suggest_categorical('knn__weights', 
                                                                     ['uniform', 'distance'])
                
            elif 'n_neighbors' in base_params:
                base_k = base_params.get('n_neighbors', 5)
                params['n_neighbors'] = trial.suggest_int('n_neighbors',
                                                        max(3, base_k - 10),
                                                        min(101, base_k + 10))
                if 'weights' in base_params:
                    params['weights'] = trial.suggest_categorical('weights', 
                                                                ['uniform', 'distance'])
            
            elif 'base_estimator' in base_params:
                k = trial.suggest_int('k', 3, 21)
                weight = trial.suggest_categorical('weight', ['uniform', 'distance'])
                metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan'])
                params['base_estimator'] = KNeighborsClassifier(n_neighbors=k, weights=weight, metric=metric)
                params['n_estimators'] = trial.suggest_int('n_estimators', 10, 100)
            
            if 'pipeline_steps' in model_config:
                model = model_config['base_model'](model_config['pipeline_steps'])
            else:
                model = model_config['base_model'](**model_config['fixed_params'])
            
            model.set_params(**params)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)
            
            return roc_auc_score(y_val, y_pred_proba[:, 1])
            
        except Exception as e:
            print(f"Ошибка в objective для {model_name}: {e}")
            return 0.0
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    
    try:
        study.optimize(objective, n_trials=n_trials, timeout=300)
        return study.best_params, study.best_value
    except Exception as e:
        print(f"Ошибка при оптимизации {model_name}: {e}")
        return {}, 0.0

def train_and_evaluate_model(model_info):
    """Обучение и оценка одной модели"""
    model_name, model_config = model_info
    start_time = time.time()
    
    try:я
        base_params, base_score = fast_randomized_search(model_name, model_config, n_iter=20)
        
        if base_score == 0.0:
            print(f"{model_name} - Пропускаем из-за ошибки в базовой оптимизации")
            return None
            
        print(f"{model_name} - Базовый результат: {base_score:.4f}")
        
        final_params, final_score = optuna_fine_tune(model_name, model_config, base_params, n_trials=30)
        
        final_params.update(base_params)
        
        print(f"{model_name} - Финальный результат: {final_score:.4f}")
        
        if 'pipeline_steps' in model_config:
            final_model = model_config['base_model'](model_config['pipeline_steps'])
        else:
            final_model = model_config['base_model'](**model_config['fixed_params'])
        
        final_model.set_params(**final_params)
        

        fit_params = {}
        if 'class_weight' in model_config:
            try:
                if hasattr(final_model, 'named_steps') and 'knn' in final_model.named_steps:
                    fit_params = {'knn__sample_weight': np.array([class_weight_dict[y] for y in y_combined])}
            except:
                pass
        
        final_model.fit(X_combined, y_combined, **fit_params)

        y_pred = final_model.predict(X_test)
        y_pred_proba = final_model.predict_proba(X_test)
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        elapsed_time = time.time() - start_time
        print(f"{model_name} завершен за {elapsed_time:.1f} секунд")
        
        return {
            'model_name': model_name,
            'model': final_model,
            'params': final_params,
            'metrics': metrics,
            'time': elapsed_time
        }
        
    except Exception as e:
        print(f"Ошибка при обучении {model_name}: {e}")
        return None

def parallel_model_training():
    """Параллельное обучение всех моделей"""
    models = create_knn_models()
    model_items = list(models.items())
    
    print(f"Начинаем обучение {len(model_items)} моделей KNN...")
    total_start_time = time.time()
    
    results = {}
    for model_name, model_config in model_items:
        result = train_and_evaluate_model((model_name, model_config))
        if result:
            results[result['model_name']] = result
    
    total_time = time.time() - total_start_time
    print(f"\nВсе модели обучены за {total_time:.1f} секунд")
    
    return results


if __name__ == "__main__":
    results = parallel_model_training()
    
    if not results:
        print("Не удалось обучить ни одной модели!")
        exit()
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ВСЕХ KNN МОДЕЛЕЙ")
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
    

    if len(comparison_df) > 0:
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
        
        comparison_df.to_csv('optimized_knn_models_comparison.csv', index=False)
        print("Сравнительная таблица сохранена в optimized_knn_models_comparison.csv")
        
        print(f"\nДетальный отчет для лучшей модели ({best_model_name}):")
        best_model = results[best_model_name]['model']
        y_test_pred_best = best_model.predict(X_test)
        print(classification_report(y_test, y_test_pred_best, target_names=['Class 0', 'Class 1']))
        
        print(f"\nОбщее время выполнения: {sum(r['time'] for r in results.values()):.1f} секунд")
        print("Все операции завершены успешно!")
    else:
        print("Не удалось успешно обучить ни одной модели!")
