# Import libraries
from dataclasses import dataclass

import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from scene import HyperspectralScene
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

# Data class for data exploration
@dataclass(init=False)
class DataExploration(HyperspectralScene):
    model_PCA: PCA
    model_TSNE: TSNE
    X_scale: np.ndarray
    X_PCA: np.ndarray
    X_TSNE: np.ndarray

    # Scales each feature to a given range
    def fit_scaler(self):
        model_scale = MinMaxScaler(feature_range=(-1, 1))
        self.X_scale = model_scale.fit_transform(X=self.X)

    # Fits a PCA model to the data
    def fit_PCA(self, n_components=0.98):
        self.model_PCA = PCA(n_components=n_components, whiten=True)
        self.X_PCA = self.model_PCA.fit_transform(X=self.X_scale)

    # Fits a t-SNE model to the data
    def fit_TSNE(self, perplexity=30, early_exaggeration=12,
                 learning_rate=200, n_iter=1000):
        self.model_TSNE = TSNE(perplexity=perplexity,
                               early_exaggeration=early_exaggeration,
                               learning_rate=learning_rate,
                               n_iter=n_iter,
                               n_jobs=-1,
                               init='pca',
                               verbose=2)
        self.X_TSNE = self.model_TSNE.fit_transform(X=self.X_PCA)

    # Set global plot parameters and create a plot instance
    def __plot_parameters(self):
        sb.set(context='paper',
               style='white',
               font='serif',
               rc={'figure.figsize': (9, 6.5)})
        figure, axes = plt.subplots(constrained_layout=True)
        return figure, axes

    # Adds labels to a discrete colorbar
    def __label_colorbar(self, colorbar):
        colorbar_range = colorbar.vmax - colorbar.vmin
        num_labels = len(self.labels)
        colorbar.set_ticks([colorbar.vmin
                            + 0.5
                            * colorbar_range
                            / num_labels
                            + i
                            * colorbar_range
                            / num_labels
                            for i in range(num_labels)])
        colorbar.set_ticklabels(self.labels)
        colorbar.outline.set_visible(False)
        colorbar.ax.invert_yaxis()

    # Plots PCA results
    def plot_PCA(self, plot_path):
        figure, axes = self.__plot_parameters()
        plot = sb.scatterplot(x=self.X_PCA[:, 0],
                              y=self.X_PCA[:, 1],
                              hue=self.y,
                              palette=self.palette,
                              legend=False,
                              ax=axes,
                              linewidth=0,
                              marker='.',
                              s=50)
        PC1 = self.model_PCA.explained_variance_ratio_[0] * 100
        PC2 = self.model_PCA.explained_variance_ratio_[1] * 100
        plot.set(title=f'{self.name} PCA Projection',
                 xlabel=(f'Principal Component 1 - {PC1:.1f}% '
                         f'Explained Variance'),
                 ylabel=(f'Principal Component 2 - {PC2:.1f}% '
                         f'Explained Variance'))
        colormap = ListedColormap(self.palette)
        colorbar = figure.colorbar(mappable=ScalarMappable(cmap=colormap),
                                   ax=axes)
        self.__label_colorbar(colorbar=colorbar)
        figure.savefig(plot_path, format='svg')
        plt.close(fig=figure)

    # Plots t-SNE results
    def plot_TSNE(self, plot_path):
        figure, axes = self.__plot_parameters()
        plot = sb.scatterplot(x=self.X_TSNE[:, 0],
                              y=self.X_TSNE[:, 1],
                              hue=self.y,
                              palette=self.palette,
                              legend=False,
                              ax=axes,
                              linewidth=0,
                              marker='.',
                              s=50)
        plot.set(title=f'{self.name} t-SNE Projection',
                 xlabel='t-SNE Component 1',
                 ylabel='t-SNE Component 2')
        colormap = ListedColormap(self.palette)
        colorbar = figure.colorbar(mappable=ScalarMappable(cmap=colormap),
                                   ax=axes)
        self.__plot_colorbar(figure=figure, axes=axes)
        figure.savefig(plot_path, format='svg')
        plt.close(fig=figure)

    # Plots ground truth classification map
    def plot_gt(self, plot_path):
        figure, axes = self.__plot_parameters()
        gt_plot = sb.heatmap(data=self.gt,
                             cmap=self.palette,
                             xticklabels=False,
                             yticklabels=False,
                             square=True,
                             ax=axes)
        gt_plot.set(title=f'{self.name} Ground Truth Classification Map')
        colorbar = gt_plot.collections[0].colorbar
        self.__label_colorbar(colorbar=colorbar)
        figure.savefig(plot_path, format='svg')
        plt.close(fig=figure)
