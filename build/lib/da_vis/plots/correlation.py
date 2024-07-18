# data_visualizer/plots/correlation.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(data, title='Correlation Heatmap'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.show()
