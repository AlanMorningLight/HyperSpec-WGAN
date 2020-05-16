# Import libraries
import csv
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sb
from h5py import File
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from pandas import DataFrame
from sklearn.metrics import classification_report, confusion_matrix

from scene import HyperspectralScene


# Data class for evaluating CNNs
@dataclass(init=False)
class EvaluateCNN(HyperspectralScene):
    accuracy: DataFrame
    loss: DataFrame
    model: str
    y_test: np.ndarray
    y_pred: np.ndarray
    y_test_pred: np.ndarray

    # Load a model and its training history from *.hdf5 files
    def load_model(self, model, acc_path, loss_path):
        self.model = model
        self.accuracy = pd.read_hdf(path_or_buf=acc_path)
        self.loss = pd.read_hdf(path_or_buf=loss_path)

    # Load testing data and predictions from *.hdf5 files
    def load_data(self, y_test_path, y_pred_path, y_test_pred_path):
        with File(name=y_test_path, mode='r') as file:
            self.y_test = file['y_test'][:]
        with File(name=y_pred_path, mode='r') as file:
            self.y_pred = file['y_pred'][:]
        with File(name=y_test_pred_path, mode='r') as file:
            self.y_test_pred = file['y_test_pred'][:]

    # Generates classification report in LaTeX format
    def generate_report(self, report_path):
        report = classification_report(y_true=self.y_test,
                                       y_pred=self.y_test_pred,
                                       target_names=self.labels,
                                       output_dict=True)
        DataFrame.to_latex(DataFrame.from_dict(report).transpose(),
                           buf=report_path)

    # Plots accuracy and loss for training and validation data
    def plot_history(self, plot_path):
        sb.set(context='paper',
               style='white',
               palette=sb.color_palette(['#C8102E', '#00B388']),
               font='serif',
               rc={'figure.figsize': (9, 6.5)})
        figure = plt.figure(constrained_layout=True)
        gs = GridSpec(nrows=3,
                      ncols=1,
                      figure=figure,
                      height_ratios=[1, 0.01, 1])
        axes_0 = plt.subplot(gs[0])
        axes_1 = plt.subplot(gs[1])
        axes_2 = plt.subplot(gs[2])
        acc_plot = sb.lineplot(data=self.accuracy,
                               dashes=False,
                               ax=axes_0)
        axes_0.get_legend().remove()
        axes_0.set_xticks(ticks=[])
        acc_plot.set(title=f'{self.name} {self.model} Training History',
                     ylabel='Accuracy')
        handles, labels = axes_0.get_legend_handles_labels()
        axes_1.legend(handles=handles,
                      labels=labels,
                      loc='upper center',
                      frameon=False,
                      borderaxespad=0,
                      ncol=2)
        axes_1.axis('off')
        loss_plot = sb.lineplot(data=self.loss,
                                dashes=False,
                                legend=False,
                                ax=axes_2)
        loss_plot.set(xlabel='Epoch', ylabel='Loss')
        figure.savefig(fname=plot_path, format='svg')
        plt.close(fig=figure)

    # Plots confusion matrix
    def plot_confusion(self, palette_path, plot_path):
        confusion = confusion_matrix(y_true=self.y_test,
                                     y_pred=self.y_test_pred,
                                     normalize='true')
        with open(palette_path) as file:
            palette = list(csv.reader(file))[0]
        sb.set(context='paper',
               style='white',
               font='serif',
               rc={'figure.figsize': (6.5, 6.5)})
        figure, axes = plt.subplots(constrained_layout=True)
        conf_plot = sb.heatmap(data=confusion,
                               cmap=sb.color_palette(palette),
                               annot=True,
                               fmt='.2f',
                               annot_kws={'size': 7},
                               linewidths=0,
                               cbar=False,
                               ax=axes,
                               square=True)
        conf_plot.set(title=f'{self.name} {self.model} Confusion Matrix',
                      xlabel='Predicted',
                      ylabel='Actual')
        conf_plot.set_xticklabels(labels=np.unique(self.y_test))
        conf_plot.set_yticklabels(labels=np.unique(self.y_test),
                                  rotation=0,
                                  horizontalalignment='right')
        figure.savefig(plot_path, format='svg')
        plt.close(fig=figure)

    # Plots predicted classification map
    def plot_pred(self, plot_path):
        if self.remove_unlabeled:
            y_temp = np.zeros(shape=len(self.y))
            y_temp[self.y != 0] = self.y_pred
            self.y_pred = y_temp
        y_pred = np.reshape(a=self.y_pred, newshape=self.ground_truth.shape)
        sb.set(context='paper',
               style='white',
               font='serif',
               rc={'figure.figsize': (9, 6.5)})
        figure, axes = plt.subplots(constrained_layout=True)
        ground_truth_plot = sb.heatmap(data=y_pred,
                                       cmap=self.palette,
                                       cbar=False,
                                       square=True,
                                       xticklabels=False,
                                       yticklabels=False,
                                       ax=axes)
        ground_truth_plot.set(title=f'{self.name} {self.model} '
                                    f'Predicted Classification Map')
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
        figure.savefig(plot_path, format='svg')
        plt.close(fig=figure)
