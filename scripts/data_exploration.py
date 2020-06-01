# Import libraries
import csv
from dataclasses import dataclass

import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from hyperspectral_scene import HyperspectralScene


# Data class for data exploration
@dataclass(init=False)
class DataExploration(HyperspectralScene):
    palette: list
    data_exploration_path: str
    X_scale: np.ndarray
    X_PCA: np.ndarray
    X_PCA_variance: np.ndarray
    X_TSNE: np.ndarray

    def __post_init__(self):
        self.__load_palette()
        if self.remove_unlabeled:
            self.X = self.X[self.y != 0, :]
            self.y = self.y[self.y != 0]
            self.labels = self.labels[1:]
            self.samples = self.X.shape[0]
            self.palette[0] = '#FFFFFF'
            self.data_exploration_path = (f"{self.path}/plots/"
                                          f"without-unlabeled/"
                                          f"data-exploration")
        else:
            self.data_exploration_path = (f"{self.path}/plots/"
                                          f"with-unlabeled/"
                                          f"data-exploration")

    # Load a custom color palette from a *.csv file
    def __load_palette(self):
        with open(file=f"{self.path}/data/palette.csv") as file:
            self.palette = list(csv.reader(file, delimiter=','))[0]

    # Set plot parameters and create a plot instance
    def __plot_parameters(self, figsize):
        sb.set(context='paper',
               style='white',
               font='serif',
               rc={'figure.figsize': figsize})
        figure, axes = plt.subplots(constrained_layout=True)
        palette = self.palette[1:] if self.remove_unlabeled else self.palette
        return figure, axes, palette

    # Create a scatterplot
    def __plot_scatterplot(self, X, palette, axes):
        plot = sb.scatterplot(x=X[:, 0],
                              y=X[:, 1],
                              hue=self.y,
                              palette=palette,
                              legend=False,
                              ax=axes,
                              linewidth=0,
                              marker='.',
                              s=50)
        return plot

    # Attach a discrete colorbar
    def __plot_colorbar(self, figure, axes, palette):
        colormap = ListedColormap(colors=palette)
        colorbar = figure.colorbar(mappable=ScalarMappable(cmap=colormap),
                                   ax=axes)
        colorbar_range = colorbar.vmax - colorbar.vmin
        num_labels = len(self.labels)
        colorbar.set_ticks([
            colorbar.vmin + 0.5 * colorbar_range / num_labels +
            i * colorbar_range / num_labels for i in range(num_labels)
        ])
        colorbar.set_ticklabels(self.labels)
        colorbar.outline.set_visible(False)
        colorbar.ax.invert_yaxis()

    # Scale each feature to a given range
    def fit_scaler(self, feature_range):
        model_scale = MinMaxScaler(feature_range=feature_range)
        self.X_scale = model_scale.fit_transform(X=self.X)

    # Fit a PCA model
    def fit_PCA(self, n_components, whiten):
        model = PCA(n_components=n_components, whiten=whiten)
        self.X_PCA = model.fit_transform(X=self.X_scale)
        self.X_PCA_variance = model.explained_variance_ratio_

    # Fit a t-SNE model
    def fit_TSNE(self, perplexity, early_exaggeration, learning_rate, n_iter):
        model = TSNE(perplexity=perplexity,
                     early_exaggeration=early_exaggeration,
                     learning_rate=learning_rate,
                     n_iter=n_iter,
                     random_state=42,
                     n_jobs=-1,
                     verbose=2)
        self.X_TSNE = model.fit_transform(X=self.X_scale)

    # Plot the ground truth classification map
    def plot_ground_truth(self):
        figure, axes, palette = self.__plot_parameters()
        ground_truth_plot = sb.heatmap(data=self.ground_truth,
                                       cmap=self.palette,
                                       cbar=False,
                                       square=True,
                                       xticklabels=False,
                                       yticklabels=False,
                                       ax=axes)
        ground_truth_plot.set(title=(f'{self.name} Ground Truth '
                                     f'Classification Map'))
        self.__plot_colorbar(figure=figure, axes=axes, palette=palette)
        figure.savefig(fname=(f"{self.data_exploration_path}/"
                              f"ground-truth-map.svg"),
                       format='svg')
        plt.close(fig=figure)

    # Plot PCA results
    def plot_PCA(self):
        figure, axes, palette = self.__plot_parameters()
        plot = self.__plot_scatterplot(X=self.X_PCA,
                                       palette=palette,
                                       axes=axes)
        plot.set(title=f'{self.name} PCA Projection',
                 xlabel=(f'Principal Component 1 - '
                         f'{self.X_PCA_variance[0]*100:.1f}% '
                         f'Explained Variance'),
                 ylabel=(f'Principal Component 2 - '
                         f'{self.X_PCA_variance[1]*100:.1f}% '
                         f'Explained Variance'))
        self.__plot_colorbar(figure=figure, axes=axes, palette=palette)
        figure.savefig(fname=(f"{self.data_exploration_path}/"
                              f"PCA-projection.svg"),
                       format='svg')
        plt.close(fig=figure)

    # Plot t-SNE results
    def plot_TSNE(self, figsize):
        figure, axes, palette = self.__plot_parameters(figsize=figsize)
        plot = self.__plot_scatterplot(X=self.X_TSNE,
                                       palette=palette,
                                       axes=axes)
        plot.set(title=f'{self.name} t-SNE Projection',
                 xlabel='t-SNE Component 1',
                 ylabel='t-SNE Component 2')
        self.__plot_colorbar(figure=figure, axes=axes, palette=palette)
        figure.savefig(fname=(f"{self.data_exploration_path}/"
                              f"t-SNE-projection.svg"),
                       format='svg')
        plt.close(fig=figure)
