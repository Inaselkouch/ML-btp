import pandas as pd
import numpy as np
from typing import Any
import seaborn as sns
import matplotlib.pyplot as plt

def zero_one_loss(y, y_pred):
    return np.mean(y_pred != y)

def accuracy(y, y_pred):
    return np.mean(y_pred == y)

def precision(y, y_pred, true):
    tp = np.sum((y_pred == true) & (y == true))
    fp = np.sum((y_pred == true) & (y != true))
    return tp / (tp + fp)

def recall(y, y_pred, true):
    tp = np.sum((y_pred == true) & (y == true))
    fn = np.sum((y_pred != true) & (y == true))
    return tp / (tp + fn)

def f1_score(y, y_pred, true):
    prec = precision(y, y_pred, true)
    rec = recall(y, y_pred, true)
    return 2 * (prec * rec) / (prec + rec)

def confusion_matrix(y, y_pred, true):
    tp = np.sum((y_pred == true) & (y == true))
    fp = np.sum((y_pred == true) & (y != true))
    fn = np.sum((y_pred != true) & (y == true))
    tn = np.sum((y_pred != true) & (y != true))
    return np.array([[tp, fp], [fn, tn]])

def print_report(y, y_pred, true):
    print("0-1 Loss:", zero_one_loss(y, y_pred))
    print("Accuracy:", accuracy(y, y_pred))
    print("Precision:", precision(y, y_pred, true))
    print("Recall:", recall(y, y_pred, true))
    print("F1 Score:", f1_score(y, y_pred, true))
    print("Confusion Matriy_pred:")
    print(plot_confusion_matrix(y, y_pred, true))

def plot_confusion_matrix (y_true, y_pred, true):
    cm = confusion_matrix(y_true, y_pred, true)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()