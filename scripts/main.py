# %%
# Import libraries
import os
from copy import copy

from scene import HyperspectralScene
from data_exp import DataExploration
from CNN_1D import Train1DCNN
from CNN_3D import Train3DCNN
from CNN_eval import EvaluateCNN

# %%
# Set working directory
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

# %%
# Initialize hyperspectral scenes
ip = HyperspectralScene(name='Indian Pines',
                        image_path='indian-pines/data/image.mat',
                        gt_path='indian-pines/data/ground-truth.mat',
                        labels_path='indian-pines/data/labels.csv',
                        palette_path='indian-pines/data/plot-palette.csv')

# %%
ip_DE = copy(ip)
ip_DE.__class__ = DataExploration
ip_DE.fit_scaler()
ip_DE.fit_PCA()
ip_DE.fit_TSNE(perplexity=8,
               early_exaggeration=12,
               learning_rate=100,
               n_iter=1000)
ip_DE.plot_PCA(plot_path="indian-pines/plots/PCA-projection.svg")
ip_DE.plot_TSNE(plot_path="indian-pines/plots/t-SNE-projection.svg")
ip_DE.plot_gt(plot_path="indian-pines/plots/ground-truth-map.svg")

# %%
ip_CNN_1D = copy(ip)
ip_CNN_1D.__class__ = Train1DCNN
ip_CNN_1D.fit_scaler()
ip_CNN_1D.fit_PCA()
ip_CNN_1D.prepare_data()
ip_CNN_1D.design_CNN_1D()
ip_CNN_1D.fit_CNN_1D(model_dir="indian-pines/model/CNN-1D")
ip_CNN_1D.save_history(history_dir="indian-pines/model/CNN-1D")

# %%
ip_CNN_3D = copy(ip)
ip_CNN_3D.__class__ = Train3DCNN
ip_CNN_3D.fit_scaler()
ip_CNN_3D.fit_PCA()
ip_CNN_3D.prepare_data()
ip_CNN_3D.design_CNN_3D()
ip_CNN_3D.fit_CNN_3D(model_dir="indian-pines/model/CNN-3D")
ip_CNN_3D.save_history(history_dir="indian-pines/model/CNN-3D")

# %%
ip_CNN_1D_eval = copy(ip)
ip_CNN_1D_eval.__class__ = EvaluateCNN
ip_CNN_1D_eval.load_history(acc_path="indian-pines/model/CNN-1D/accuracy.hdf5",
                            loss_path="indian-pines/model/CNN-1D/loss.hdf5")
ip_CNN_1D_eval.load_model(network='1D-CNN',
                          model_path="indian-pines/model/CNN-1D/model.hdf5")
ip_CNN_1D_eval.load_data(X_all_path="indian-pines/model/CNN-1D/X_all.hdf5",
                         X_test_path="indian-pines/model/CNN-1D/X_test.hdf5",
                         y_test_path="indian-pines/model/CNN-1D/y_test.hdf5")
ip_CNN_1D_eval.predict_data()
ip_CNN_1D_eval.generate_report(
    report_path="indian-pines/plots/CNN-1D/classification-report.tex")
ip_CNN_1D_eval.plot_history(
    plot_path="indian-pines/plots/CNN-1D/training-history.svg")
ip_CNN_1D_eval.plot_confusion(
    palette_path="indian-pines/data/confusion-matrix-palette.csv",
    plot_path="indian-pines/plots/CNN-1D/confusion-matrix.svg")
ip_CNN_1D_eval.plot_pred(
    plot_path="indian-pines/plots/CNN-1D/predicted-map.svg")

# %%
ip_CNN_3D_eval = copy(ip)
ip_CNN_3D_eval.__class__ = EvaluateCNN
ip_CNN_3D_eval.load_history(acc_path="indian-pines/model/CNN-3D/accuracy.hdf5",
                            loss_path="indian-pines/model/CNN-3D/loss.hdf5")
ip_CNN_3D_eval.load_model(network='3D-CNN',
                          model_path="indian-pines/model/CNN-3D/model.hdf5")
ip_CNN_3D_eval.load_data(X_all_path="indian-pines/model/CNN-3D/X_all.hdf5",
                         X_test_path="indian-pines/model/CNN-3D/X_test.hdf5",
                         y_test_path="indian-pines/model/CNN-3D/y_test.hdf5")
ip_CNN_3D_eval.predict_data()
ip_CNN_3D_eval.generate_report(
    report_path="indian-pines/plots/CNN-3D/classification-report.tex")
ip_CNN_3D_eval.plot_history(
    plot_path="indian-pines/plots/CNN-3D/training-history.svg")
ip_CNN_3D_eval.plot_confusion(
    palette_path="indian-pines/data/confusion-matrix-palette.csv",
    plot_path="indian-pines/plots/CNN-3D/confusion-matrix.svg")
ip_CNN_3D_eval.plot_pred(
    plot_path="indian-pines/plots/CNN-3D/predicted-map.svg")
