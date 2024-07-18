# data_visualizer/plots/feature_importance.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap

def plot_feature_importance(model, feature_names, title='Feature Importance'):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.bar(range(len(feature_names)), importances[indices])
        plt.xticks(range(len(feature_names)), np.array(feature_names)[indices], rotation=90)
        plt.show()
    else:
        print("Model does not have feature_importances_ attribute")

def plot_shap_values(model, data, feature_names, title='SHAP Values'):
    explainer = shap.Explainer(model, data)
    shap_values = explainer(data)
    shap.summary_plot(shap_values, features=data, feature_names=feature_names, show=True)
