# data_visualizer/utils/plot_helpers.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_heatmap(data, title='Heatmap', cmap='coolwarm'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, cmap=cmap)
    plt.title(title)
    plt.show()

def plot_histogram(data, feature, title='Histogram'):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(title)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
