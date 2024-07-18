# data_visualizer/plots/dimensionality_reduction.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_tsne(data, title='t-SNE Plot', perplexity=30, n_iter=300, random_state=None):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    tsne_results = tsne.fit_transform(data)
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()

def plot_pca(data, title='PCA Plot'):
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(data)
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_results[:, 0], pca_results[:, 1])
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()
