# da_vis/visualizer.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import shap
from itertools import cycle

class DataVisualizer:
    def __init__(self, data, model=None):
        self.data = data
        self.model = model

    def correlation_heatmap(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap')
        plt.show()

    def feature_distribution(self, feature):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.show()

    def confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def roc_curve(self, y_true, y_prob, title='ROC Curve'):
        y_true = label_binarize(y_true, classes=list(set(y_true)))
        n_classes = y_true.shape[1]

        if y_prob.ndim == 1:
            raise ValueError("y_prob must be a 2D array with shape (n_samples, n_classes) for multiclass ROC curve")
        
        if y_prob.shape[1] != n_classes:
            raise ValueError("Number of columns in y_prob must match the number of classes")

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.show()

    def precision_recall_curve(self, y_true, y_prob):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, marker='.')
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()

    def feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(12, 6))
            plt.title('Feature Importance')
            plt.bar(range(self.data.shape[1]), importances[indices])
            plt.xticks(range(self.data.shape[1]), self.data.columns[indices], rotation=90)
            plt.show()
        elif hasattr(self.model, 'coef_'):
            importances = self.model.coef_[0]
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(12, 6))
            plt.title('Feature Importance')
            plt.bar(range(self.data.shape[1]), importances[indices])
            plt.xticks(range(self.data.shape[1]), self.data.columns[indices], rotation=90)
            plt.show()
        else:
            print("Model does not have feature_importances_ or coef_ attribute")

    def tsne_plot(self):
        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(self.data)
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
        plt.title('t-SNE Plot')
        plt.show()

    def create_dashboard(self, output='dashboard.html'):
        import dash
        import dash_core_components as dcc
        import dash_html_components as html
        import plotly.express as px

        app = dash.Dash(__name__)

        correlation_fig = px.imshow(self.data.corr(), text_auto=True, aspect="auto", title="Correlation Heatmap")
        feature_dist_fig = px.histogram(self.data, x=self.data.columns[0], title=f'Distribution of {self.data.columns[0]}')

        app.layout = html.Div([
            html.H1('Data Visualizer Dashboard'),
            dcc.Graph(id='correlation-heatmap', figure=correlation_fig),
            dcc.Graph(id='feature-distribution', figure=feature_dist_fig),
        ])

        app.run_server(debug=True, use_reloader=False)
