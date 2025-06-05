import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, average_precision_score,
                           matthews_corrcoef, confusion_matrix,
                           classification_report)
import os

print("Загрузка данных...")
X_train = pd.read_csv('X_cl_train.csv')
X_val = pd.read_csv('X_cl_val.csv')
X_test = pd.read_csv('X_cl_test.csv')
y_train = pd.read_csv('y_cl_train.csv').iloc[:, 0]
y_val = pd.read_csv('y_cl_val.csv').iloc[:, 0]
y_test = pd.read_csv('y_cl_test.csv').iloc[:, 0]

y_combined = pd.concat([y_train, y_val], axis=0)

def calculate_baseline_metrics(y_true, y_train):
    """Вычисление метрик для бейзлайн модели (всегда предсказывает наибольший класс)"""
    class_counts = np.bincount(y_train)
    majority_class = np.argmax(class_counts)
    
    y_pred = np.full_like(y_true, majority_class)
    y_proba = np.full(len(y_true), class_counts[majority_class]/len(y_train), dtype=float)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_proba),
        'auc_pr': average_precision_score(y_true, y_proba),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    cm = confusion_matrix(y_true, y_pred)
    
    return metrics, cm, majority_class

baseline_metrics, baseline_cm, majority_class = calculate_baseline_metrics(y_test, y_combined)

results_df = pd.DataFrame({
    'Model': ['Baseline (Majority Class)'],
    'Accuracy': [baseline_metrics['accuracy']],
    'Precision': [baseline_metrics['precision']],
    'Recall': [baseline_metrics['recall']],
    'F1': [baseline_metrics['f1']],
    'AUC_ROC': [baseline_metrics['auc_roc']],
    'AUC_PR': [baseline_metrics['auc_pr']],
    'MCC': [baseline_metrics['mcc']],
    'Majority_Class': [majority_class],
    'Class_0_Count': [baseline_cm[0, 0] + baseline_cm[1, 0]],
    'Class_1_Count': [baseline_cm[0, 1] + baseline_cm[1, 1]]
})

output_file = 'baseline_metrics.csv'
results_df.to_csv(output_file, index=False)

print("\n" + "="*60)
print("МЕТРИКИ БЕЙЗЛАЙН МОДЕЛИ (ВСЕГДА ПРЕДСКАЗЫВАЕТ НАИБОЛЬШИЙ КЛАСС)")
print("="*60)
print(f"Наибольший класс: {majority_class}")
print(f"Доля наибольшего класса: {baseline_metrics['accuracy']:.4f}")
print("\nОсновные метрики:")
for metric, value in baseline_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nМатрица ошибок:")
print(baseline_cm)

print("\nОтчет классификации:")
print(classification_report(y_test, np.full_like(y_test, majority_class), 
                           target_names=['Class 0', 'Class 1']))

print(f"\nРезультаты сохранены в файл: {output_file}")
