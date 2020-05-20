# Import libraries
import csv
from dataclasses import dataclass

import numpy as np
import seaborn as sb
from h5py import File
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from pandas import DataFrame, read_csv
from sklearn.metrics import classification_report, confusion_matrix

from hyperspectral_scene import HyperspectralScene


# Data class for evaluating CNNs
@dataclass(init=False)
class EvaluateCNN(HyperspectralScene):
    model_name: str
    palette: list
    evaluate_CNN_path: str
    accuracy: DataFrame
    loss: DataFrame
    best_model: dict
    y_test: np.ndarray
    y_pred: np.ndarray
    y_test_pred: np.ndarray
    y_pred_temp: np.ndarray

    def __post_init__(self, model_name):
        self.model_name = model_name
        self.palette = self.__load_palette(name='palette')
        if self.remove_unlabeled:
            self.labels = self.labels[1:]
            self.palette[0] = '#FFFFFF'
            self.evaluate_CNN_path = (f"{self.path}/plots/"
                                      f"without-unlabeled/"
                                      f"{self.model_name}")
            self.__load_training_history(history_path=(f"{self.path}/model/"
                                                       f"without-unlabeled/"
                                                       f"{self.model_name}/"
                                                       f"history.csv"))
            self.__load_data(model_path=(f"{self.path}/model/"
                                         f"without-unlabeled/"
                                         f"{self.model_name}"))
            self.y_pred_temp = self.y
            self.y = self.y[self.y != 0]
            self.y_pred_temp[self.y_pred_temp != 0] = self.y_pred

        else:
            self.evaluate_CNN_path = (f"{self.path}/plots/"
                                      f"with-unlabeled/"
                                      f"{self.model_name}")
            self.__load_training_history(history_path=(f"{self.path}/model/"
                                                       f"with-unlabeled/"
                                                       f"{self.model_name}/"
                                                       f"history.csv"))
            self.__load_data(model_path=(f"{self.path}/model/"
                                         f"with-unlabeled/"
                                         f"{self.model_name}"))
            self.y_pred_temp = self.y_pred

    # Load testing data and predictions from *.hdf5 files
    def __load_data(self, model_path):
        with File(name=f"{model_path}/y_test.hdf5", mode='r') as file:
            self.y_test = file['y_test'][:]
        with File(name=f"{model_path}/y_pred.hdf5", mode='r') as file:
            self.y_pred = file['y_pred'][:]
        with File(name=f"{model_path}/y_test_pred.hdf5", mode='r') as file:
            self.y_test_pred = file['y_test_pred'][:]

    def __load_training_history(self, history_path):
        history = read_csv(filepath_or_buffer=history_path)
        self.accuracy = history[['accuracy', 'val_accuracy']]
        self.accuracy.columns = ['Training', 'Validation']
        self.loss = history[['loss', 'val_loss']]
        self.loss.columns = ['Training', 'Validation']
        epoch = self.loss['Validation'].idxmin()
        self.best_model = {'accuracy': [epoch,
                                        self.accuracy.iloc[epoch, 0]],
                           'val_accuracy': [epoch,
                                            self.accuracy.iloc[epoch, 1]],
                           'loss': [epoch, self.loss.iloc[epoch, 0]],
                           'val_loss': [epoch, self.loss.iloc[epoch, 1]]}

    # Load a custom color palette from a *.csv file
    def __load_palette(self, name):
        with open(file=f"{self.path}/data/{name}.csv") as file:
            palette = list(csv.reader(file, delimiter=','))[0]
        return palette

    # Set plot parameters and create a plot instance
    def __plot_parameters(self, figsize):
        sb.set(context='paper',
               style='white',
               font='serif',
               rc={'figure.figsize': figsize})
        figure, axes = plt.subplots(constrained_layout=True)
        return figure, axes

    # Generate classification reports in LaTeX format
    def generate_report(self):
        full_report = classification_report(y_true=self.y,
                                            y_pred=self.y_pred,
                                            target_names=self.labels,
                                            output_dict=True,
                                            zero_division=0)
        test_report = classification_report(y_true=self.y_test,
                                            y_pred=self.y_test_pred,
                                            target_names=self.labels,
                                            output_dict=True,
                                            zero_division=0)
        DataFrame.from_dict(full_report).transpose().to_latex(
            buf=f"{self.evaluate_CNN_path}/full-classification-report.tex")
        DataFrame.from_dict(test_report).transpose().to_latex(
            buf=f"{self.evaluate_CNN_path}/test-classification-report.tex")

    # Plot the accuracy and loss for training and validation data
    def plot_training_history(self, figsize):
        figure, axes = self.__plot_parameters(figsize=figsize)
        gs = GridSpec(nrows=3,
                      ncols=1,
                      figure=figure,
                      height_ratios=[1, 0.01, 1])
        axes_0 = plt.subplot(gs[0])
        axes_1 = plt.subplot(gs[1])
        axes_2 = plt.subplot(gs[2])
        accuracy_plot = sb.lineplot(data=self.accuracy,
                                    palette=sb.color_palette(['#C8102E',
                                                              '#00B388']),
                                    dashes=False,
                                    ax=axes_0)
        handles, _ = axes_0.get_legend_handles_labels()
        acc_point = axes_0.scatter(x=self.best_model['accuracy'][0],
                                   y=self.best_model['accuracy'][1],
                                   s=200,
                                   c='#888B8D',
                                   marker='1',
                                   linewidth=2,
                                   zorder=3)
        val_acc_point = axes_0.scatter(x=self.best_model['val_accuracy'][0],
                                       y=self.best_model['val_accuracy'][1],
                                       s=200,
                                       c='#888B8D',
                                       marker='2',
                                       linewidth=2,
                                       zorder=3)
        axes_0.get_legend().remove()
        axes_0.set_xticks(ticks=[])
        accuracy_plot.set(title=(f'{self.name} {self.model_name} '
                                 f'Training History'),
                          ylabel='Accuracy')
        handles = sum([handles, [(val_acc_point, acc_point)]], [])
        axes_1.legend(handles=handles,
                      labels=['Training', 'Validation', 'Best Model'],
                      loc='center',
                      ncol=3,
                      scatterpoints=2,
                      frameon=False,
                      borderaxespad=0,
                      handler_map={tuple: HandlerTuple(ndivide=None, pad=0.8)})
        axes_1.axis('off')
        loss_plot = sb.lineplot(data=self.loss,
                                palette=sb.color_palette(['#C8102E',
                                                          '#00B388']),
                                dashes=False,
                                legend=False,
                                ax=axes_2)
        axes_2.scatter(x=self.best_model['loss'][0],
                       y=self.best_model['loss'][1],
                       s=200,
                       c='#888B8D',
                       marker='2',
                       linewidth=2,
                       zorder=3)
        axes_2.scatter(x=self.best_model['val_loss'][0],
                       y=self.best_model['val_loss'][1],
                       s=200,
                       c='#888B8D',
                       marker='1',
                       linewidth=2,
                       zorder=3)
        loss_plot.set(xlabel='Epoch', ylabel='Loss')
        figure.savefig(fname=f"{self.evaluate_CNN_path}/training-history.svg",
                       format='svg')
        plt.close(fig=figure)

    # Plot full and test confusion matrix
    def plot_confusion_matrix(self, figsize):
        matrix = {}
        matrix['Full'] = confusion_matrix(y_true=self.y,
                                          y_pred=self.y_pred,
                                          normalize='true')
        matrix['Test'] = confusion_matrix(y_true=self.y_test,
                                          y_pred=self.y_test_pred,
                                          normalize='true')
        palette = self.__load_palette(name='confusion-matrix-palette')
        for key, value in matrix.items():
            figure, axes = self.__plot_parameters(figsize=figsize)
            conf_plot = sb.heatmap(data=value,
                                   cmap=palette,
                                   cbar=False,
                                   fmt='.3f',
                                   annot_kws={'size': 6},
                                   square=True,
                                   annot=True,
                                   linewidths=0,
                                   ax=axes)
            conf_plot.set(title=(f'{key} {self.name} {self.model_name} '
                                 f'Confusion Matrix'),
                          xlabel='Predicted',
                          ylabel='Actual')
            conf_plot.set_xticklabels(labels=np.unique(self.y_test))
            conf_plot.set_yticklabels(labels=np.unique(self.y_test),
                                      rotation=0,
                                      horizontalalignment='right')
            figure.savefig(fname=(f"{self.evaluate_CNN_path}/{key.lower()}-"
                                  f"confusion-matrix.svg"),
                           format='svg')
            plt.close(fig=figure)

    # Plot the predicted classification map
    def plot_predicted(self, figsize):
        y_pred = np.reshape(a=self.y_pred_temp,
                            newshape=self.ground_truth.shape)
        figure, axes = self.__plot_parameters(figsize=figsize)
        ground_truth_plot = sb.heatmap(data=y_pred,
                                       cmap=self.palette,
                                       cbar=False,
                                       square=True,
                                       xticklabels=False,
                                       yticklabels=False,
                                       ax=axes)
        ground_truth_plot.set(title=f'{self.name} {self.model_name} '
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
        figure.savefig(fname=f"{self.evaluate_CNN_path}/predicted-map.svg",
                       format='svg')
        plt.close(fig=figure)
