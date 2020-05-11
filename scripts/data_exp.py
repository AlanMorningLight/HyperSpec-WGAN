# Import libraries
from dataclasses import dataclass

import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from scene import HyperspectralScene


# Data class for data exploration
@dataclass(init=False)
class DataExploration(HyperspectralScene):
    X_scale: np.ndarray
    model_PCA: PCA
    X_PCA: np.ndarray
    model_TSNE: TSNE
    X_TSNE: np.ndarray

    # Remove unlabeled data from X and y
    def check_remove_unlabeled(self):
        if self.remove_unlabeled:
            self.X = self.X[self.y != 0, :]
            self.y = self.y[self.y != 0]

    # Scale each feature to a given range
    def fit_scaler(self, feature_range):
        model_scale = MinMaxScaler(feature_range=feature_range)
        self.X_scale = model_scale.fit_transform(X=self.X)

    # Fit a PCA model to the data
    def fit_PCA(self, n_components, whiten):
        self.model_PCA = PCA(n_components=n_components, whiten=whiten)
        self.X_PCA = self.model_PCA.fit_transform(X=self.X_scale)

    # Fit a t-SNE model to the data
    def fit_TSNE(self, perplexity, early_exaggeration, learning_rate, n_iter):
        self.model_TSNE = TSNE(perplexity=perplexity,
                               early_exaggeration=early_exaggeration,
                               learning_rate=learning_rate,
                               n_iter=n_iter,
                               n_jobs=-1,
                               verbose=2)
        self.X_TSNE = self.model_TSNE.fit_transform(X=self.X_scale)

    # Set global plot parameters and create a plot instance
    def __plot_parameters(self):
        sb.set(context='paper',
               style='white',
               font='serif',
               rc={'figure.figsize': (9, 6.5)})
        figure, axes = plt.subplots(constrained_layout=True)
        palette = self.palette[1:] if self.remove_unlabeled else self.palette
        return figure, axes, palette

    # Plot a discrete colorbar
    def __plot_colorbar(self, figure, axes):
        if self.remove_unlabeled:
            colormap = ListedColormap(self.palette[1:])
        else:
            colormap = ListedColormap(self.palette)
        colorbar = figure.colorbar(mappable=ScalarMappable(cmap=colormap),
                                   ax=axes)
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

    # Plot PCA results
    def plot_PCA(self, plot_path):
        figure, axes, palette = self.__plot_parameters()
        plot = sb.scatterplot(x=self.X_PCA[:, 0],
                              y=self.X_PCA[:, 1],
                              hue=self.y,
                              palette=palette,
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
        self.__plot_colorbar(figure=figure, axes=axes)
        figure.savefig(plot_path, format='svg')
        plt.close(fig=figure)

    # Plot t-SNE results
    def plot_TSNE(self, plot_path):
        figure, axes, palette = self.__plot_parameters()
        plot = sb.scatterplot(x=self.X_TSNE[:, 0],
                              y=self.X_TSNE[:, 1],
                              hue=self.y,
                              palette=palette,
                              legend=False,
                              ax=axes,
                              linewidth=0,
                              marker='.',
                              s=50)
        plot.set(title=f'{self.name} t-SNE Projection',
                 xlabel='t-SNE Component 1',
                 ylabel='t-SNE Component 2')
        self.__plot_colorbar(figure=figure, axes=axes)
        figure.savefig(plot_path, format='svg')
        plt.close(fig=figure)

    # Plot ground truth classification map
    def plot_gt(self, plot_path):
        figure, axes, _ = self.__plot_parameters()
        gt_plot = sb.heatmap(data=self.gt,
                             cmap=self.palette,
                             cbar=False,
                             square=True,
                             xticklabels=False,
                             yticklabels=False,
                             ax=axes)
        gt_plot.set(title=f'{self.name} Ground Truth Classification Map')
        self.__plot_colorbar(figure=figure, axes=axes)
        figure.savefig(plot_path, format='svg')
        plt.close(fig=figure)

    # Initialize other class attributes
    def __post_init__(self):
        self.check_remove_unlabeled()
