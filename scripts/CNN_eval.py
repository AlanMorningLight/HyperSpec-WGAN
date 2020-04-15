# Import libraries
import csv
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sb
from h5py import File
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from pandas import DataFrame
from scene import HyperspectralScene
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Model, models


# Data class for evaluating CNNs
@dataclass(init=False)
class EvaluateCNN(HyperspectralScene):
    accuracy: DataFrame
    loss: DataFrame
    network: str
    model: Model
    X_all: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    y_test_pred: np.ndarray

    # Loads accuracy and loss history from *.hdf5 files
    def load_history(self, acc_path, loss_path):
        self.accuracy = pd.read_hdf(path_or_buf=acc_path)
        self.loss = pd.read_hdf(path_or_buf=loss_path)

    # Loads a CNN model from *.hdf5 file
    def load_model(self, network, model_path):
        self.network = network
        self.model = models.load_model(filepath=model_path)

    # Loads testing data from *.hdf5 file
    def load_data(self, X_all_path, X_test_path, y_test_path):
        with File(name=X_all_path, mode='r') as file:
            self.X_all = file['X_all'][:]
        with File(name=X_test_path, mode='r') as file:
            self.X_test = file['X_test'][:]
        with File(name=y_test_path, mode='r') as file:
            self.y_test = file['y_test'][:]

    # Predict testing data
    def predict_data(self):
        y_pred = self.model.predict(self.X_all)
        y_test_pred = self.model.predict(self.X_test)
        self.y_pred = np.argmax(a=y_pred, axis=1)
        self.y_test_pred = np.argmax(a=y_test_pred, axis=1)

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
               rc={'figure.figsize': (11, 8.5)})
        figure = plt.figure(constrained_layout=True)
        gs = GridSpec(nrows=3,
                      ncols=1,
                      figure=figure,
                      height_ratios=[1, 0.1, 1])
        axes_0 = plt.subplot(gs[0])
        axes_1 = plt.subplot(gs[1])
        axes_2 = plt.subplot(gs[2])
        acc_plot = sb.lineplot(data=self.accuracy,
                               dashes=False,
                               ax=axes_0)
        axes_0.get_legend().remove()
        axes_0.set_xticks(ticks=[])
        acc_plot.set(title=f'{self.name} {self.network} Training History',
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
               rc={'figure.figsize': (11, 8.5)})
        figure, axes = plt.subplots(constrained_layout=True)
        conf_plot = sb.heatmap(data=confusion,
                               cmap=sb.color_palette(palette),
                               annot=True,
                               fmt='.1f',
                               annot_kws={'size': 8},
                               linewidths=0,
                               cbar=False,
                               ax=axes,
                               square=True)
        conf_plot.set(title=f'{self.name} {self.network} Confusion Matrix',
                      xlabel='Predicted',
                      ylabel='Actual')
        conf_plot.set_xticklabels(labels=self.labels,
                                  rotation=45,
                                  horizontalalignment='right')
        conf_plot.set_yticklabels(labels=self.labels,
                                  rotation=45,
                                  horizontalalignment='right')
        figure.savefig(plot_path, format='svg')
        plt.close(fig=figure)

    # Plots predicted classification map
    def plot_pred(self, plot_path):
        y_pred = np.reshape(a=self.y_pred, newshape=self.gt.shape)
        sb.set(context='paper',
               style='white',
               font='serif',
               rc={'figure.figsize': (11, 8.5)})
        figure, axes = plt.subplots(constrained_layout=True)
        gt_plot = sb.heatmap(data=y_pred,
                             cmap=self.palette,
                             xticklabels=False,
                             yticklabels=False,
                             square=True,
                             ax=axes)
        gt_plot.set(title=f'{self.name} {self.network} '
                          f'Predicted Classification Map')
        colorbar = gt_plot.collections[0].colorbar
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
        colorbar.ax.invert_yaxis()
        figure.savefig(plot_path, format='svg')
        plt.close(fig=figure)
