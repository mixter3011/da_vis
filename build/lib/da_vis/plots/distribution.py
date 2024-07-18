# data_visualizer/plots/distribution.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distribution(data, feature, title=None):
    if title is None:
        title = f'Distribution of {feature}'
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], kde=True)
    plt.title(title)
    plt.show()
