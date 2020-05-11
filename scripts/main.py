# %%
# Import libraries
import os
from copy import copy

from CNN_1D import Train1DCNN
from CNN_3D import Train3DCNN
from CNN_eval import EvaluateCNN
from data_exp import DataExploration
from scene import HyperspectralScene

# %%
# Set the working directory
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

# %%
# Initialize Indian Pines without unlabeled data
ip = HyperspectralScene(name='Indian Pines',
                        image_path="indian-pines/data/image.mat",
                        gt_path="indian-pines/data/ground-truth.mat",
                        labels_path="indian-pines/data/labels.csv",
                        palette_path="indian-pines/data/plot-palette.csv",
                        remove_unlabeled=True)

# %%
# Data Exploration for Indian Pines without unlabeled data
ip_DE = copy(ip)
ip_DE.__class__ = DataExploration
ip_DE.__post_init__()
ip_DE.plot_gt(
    plot_path="indian-pines/plots/without-unlabeled/ground-truth-map.svg")
ip_DE.fit_scaler(feature_range=(-1, 1))
ip_DE.fit_PCA(n_components=15, whiten=True)
ip_DE.plot_PCA(
    plot_path="indian-pines/plots/without-unlabeled/PCA-projection.svg")
ip_DE.fit_TSNE(perplexity=8,
               early_exaggeration=12,
               learning_rate=100,
               n_iter=1000)
ip_DE.plot_TSNE(
    plot_path="indian-pines/plots/without-unlabeled/t-SNE-projection.svg")

# %%
# Train 1D-CNN for Indian Pines without unlabeled data
ip_CNN_1D = copy(ip)
ip_CNN_1D.__class__ = Train1DCNN
ip_CNN_1D.__post_init__()
ip_CNN_1D.fit_scaler(feature_range=(-1, 1))
ip_CNN_1D.fit_PCA(n_components=15, whiten=True)
ip_CNN_1D.prepare_data()
ip_CNN_1D.design_CNN_1D()
ip_CNN_1D.fit_CNN_1D(model_dir="indian-pines/model/without-unlabeled/CNN-1D")
ip_CNN_1D.predict_data(
    model_path="indian-pines/model/without-unlabeled/CNN-1D/model.hdf5",
    data_dir="indian-pines/model/without-unlabeled/CNN-1D")
ip_CNN_1D.save_history(
    history_dir="indian-pines/model/without-unlabeled/CNN-1D")

# %%
# Train 3D-CNN for Indian Pines without unlabeled data
ip_CNN_3D = copy(ip)
ip_CNN_3D.__class__ = Train3DCNN
ip_CNN_3D.__post_init__()
ip_CNN_3D.fit_scaler(feature_range=(-1, 1))
ip_CNN_3D.fit_PCA(n_components=15, whiten=True)
ip_CNN_3D.prepare_data()
ip_CNN_3D.design_CNN_3D()
ip_CNN_3D.fit_CNN_3D(
    model_dir="indian-pines/model/without-unlabeled/CNN-3D")
ip_CNN_3D.predict_data(
    model_path="indian-pines/model/without-unlabeled/CNN-3D/model.hdf5",
    data_dir="indian-pines/model/without-unlabeled/CNN-3D")
ip_CNN_3D.save_history(
    history_dir="indian-pines/model/without-unlabeled/CNN-3D")

# %%
# Evaluate 1D-CNN for Indian Pines without unlabeled data
ip_CNN_1D_eval = copy(ip)
ip_CNN_1D_eval.__class__ = EvaluateCNN
ip_CNN_1D_eval.load_model(
    model='1D-CNN',
    acc_path="indian-pines/model/without-unlabeled/CNN-1D/accuracy.hdf5",
    loss_path="indian-pines/model/without-unlabeled/CNN-1D/loss.hdf5")
ip_CNN_1D_eval.load_data(
    y_test_path="indian-pines/model/without-unlabeled/CNN-1D/y_test.hdf5",
    y_pred_path="indian-pines/model/without-unlabeled/CNN-1D/y_pred.hdf5",
    y_test_pred_path=(f"indian-pines/model/without-unlabeled/"
                      f"CNN-1D/y_test_pred.hdf5"))
ip_CNN_1D_eval.generate_report(
    report_path=(f"indian-pines/plots/without-unlabeled/"
                 f"CNN-1D/classification-report.tex"))
ip_CNN_1D_eval.plot_history(
    plot_path=(f"indian-pines/plots/without-unlabeled/"
               f"CNN-1D/training-history.svg"))
ip_CNN_1D_eval.plot_confusion(
    palette_path="indian-pines/data/confusion-matrix-palette.csv",
    plot_path=(f"indian-pines/plots/without-unlabeled/"
               f"CNN-1D/confusion-matrix.svg"))
ip_CNN_1D_eval.plot_pred(
    plot_path="indian-pines/plots/without-unlabeled/CNN-1D/predicted-map.svg")

# %%
# Evaluate 3D-CNN for Indian Pines without unlabeled data
ip_CNN_3D_eval = copy(ip)
ip_CNN_3D_eval.__class__ = EvaluateCNN
ip_CNN_3D_eval.load_model(
    model='3D-CNN',
    acc_path="indian-pines/model/without-unlabeled/CNN-3D/accuracy.hdf5",
    loss_path="indian-pines/model/without-unlabeled/CNN-3D/loss.hdf5")
ip_CNN_3D_eval.load_data(
    y_test_path="indian-pines/model/without-unlabeled/CNN-3D/y_test.hdf5",
    y_pred_path="indian-pines/model/without-unlabeled/CNN-3D/y_pred.hdf5",
    y_test_pred_path=(f"indian-pines/model/without-unlabeled/"
                      f"CNN-3D/y_test_pred.hdf5"))
ip_CNN_3D_eval.generate_report(
    report_path=(f"indian-pines/plots/without-unlabeled/"
                 f"CNN-3D/classification-report.tex"))
ip_CNN_3D_eval.plot_history(
    plot_path=(f"indian-pines/plots/without-unlabeled/"
               f"CNN-3D/training-history.svg"))
ip_CNN_3D_eval.plot_confusion(
    palette_path="indian-pines/data/confusion-matrix-palette.csv",
    plot_path=(f"indian-pines/plots/without-unlabeled/"
               f"CNN-3D/confusion-matrix.svg"))
ip_CNN_3D_eval.plot_pred(
    plot_path="indian-pines/plots/without-unlabeled/CNN-3D/predicted-map.svg")

# %%
# Initialize Indian Pines with unlabeled data
ip = HyperspectralScene(name='Indian Pines',
                        image_path="indian-pines/data/image.mat",
                        gt_path="indian-pines/data/ground-truth.mat",
                        labels_path="indian-pines/data/labels.csv",
                        palette_path="indian-pines/data/plot-palette.csv",
                        remove_unlabeled=False)

# %%
# Data Exploration for Indian Pines with unlabeled data
ip_DE = copy(ip)
ip_DE.__class__ = DataExploration
ip_DE.__post_init__()
ip_DE.plot_gt(
    plot_path="indian-pines/plots/with-unlabeled/ground-truth-map.svg")
ip_DE.fit_scaler(feature_range=(-1, 1))
ip_DE.fit_PCA(n_components=15, whiten=True)
ip_DE.plot_PCA(
    plot_path="indian-pines/plots/with-unlabeled/PCA-projection.svg")
ip_DE.fit_TSNE(perplexity=8,
               early_exaggeration=12,
               learning_rate=100,
               n_iter=1000)
ip_DE.plot_TSNE(
    plot_path="indian-pines/plots/with-unlabeled/t-SNE-projection.svg")

# %%
# Train 1D-CNN for Indian Pines with unlabeled data
ip_CNN_1D = copy(ip)
ip_CNN_1D.__class__ = Train1DCNN
ip_CNN_1D.__post_init__()
ip_CNN_1D.fit_scaler(feature_range=(-1, 1))
ip_CNN_1D.fit_PCA(n_components=15, whiten=True)
ip_CNN_1D.prepare_data()
ip_CNN_1D.design_CNN_1D()
ip_CNN_1D.fit_CNN_1D(model_dir="indian-pines/model/with-unlabeled/CNN-1D")
ip_CNN_1D.predict_data(
    model_path="indian-pines/model/with-unlabeled/CNN-1D/model.hdf5",
    data_dir="indian-pines/model/with-unlabeled/CNN-1D")
ip_CNN_1D.save_history(
    history_dir="indian-pines/model/with-unlabeled/CNN-1D")

# %%
# Train 3D-CNN for Indian Pines with unlabeled data
ip_CNN_3D = copy(ip)
ip_CNN_3D.__class__ = Train3DCNN
ip_CNN_3D.__post_init__()
ip_CNN_3D.fit_scaler(feature_range=(-1, 1))
ip_CNN_3D.fit_PCA(n_components=15, whiten=True)
ip_CNN_3D.prepare_data()
ip_CNN_3D.design_CNN_3D()
ip_CNN_3D.fit_CNN_3D(
    model_dir="indian-pines/model/with-unlabeled/CNN-3D")
ip_CNN_3D.predict_data(
    model_path="indian-pines/model/with-unlabeled/CNN-3D/model.hdf5",
    data_dir="indian-pines/model/with-unlabeled/CNN-3D")
ip_CNN_3D.save_history(
    history_dir="indian-pines/model/with-unlabeled/CNN-3D")

# %%
# Evaluate 1D-CNN for Indian Pines with unlabeled data
ip_CNN_1D_eval = copy(ip)
ip_CNN_1D_eval.__class__ = EvaluateCNN
ip_CNN_1D_eval.load_model(
    model='1D-CNN',
    acc_path="indian-pines/model/with-unlabeled/CNN-1D/accuracy.hdf5",
    loss_path="indian-pines/model/with-unlabeled/CNN-1D/loss.hdf5")
ip_CNN_1D_eval.load_data(
    y_test_path="indian-pines/model/with-unlabeled/CNN-1D/y_test.hdf5",
    y_pred_path="indian-pines/model/with-unlabeled/CNN-1D/y_pred.hdf5",
    y_test_pred_path=(f"indian-pines/model/with-unlabeled/"
                      f"CNN-1D/y_test_pred.hdf5"))
ip_CNN_1D_eval.generate_report(
    report_path=(f"indian-pines/plots/with-unlabeled/"
                 f"CNN-1D/classification-report.tex"))
ip_CNN_1D_eval.plot_history(
    plot_path=(f"indian-pines/plots/with-unlabeled/"
               f"CNN-1D/training-history.svg"))
ip_CNN_1D_eval.plot_confusion(
    palette_path="indian-pines/data/confusion-matrix-palette.csv",
    plot_path=(f"indian-pines/plots/with-unlabeled/"
               f"CNN-1D/confusion-matrix.svg"))
ip_CNN_1D_eval.plot_pred(
    plot_path="indian-pines/plots/with-unlabeled/CNN-1D/predicted-map.svg")

# %%
# Evaluate 3D-CNN for Indian Pines with unlabeled data
ip_CNN_3D_eval = copy(ip)
ip_CNN_3D_eval.__class__ = EvaluateCNN
ip_CNN_3D_eval.load_model(
    model='3D-CNN',
    acc_path="indian-pines/model/with-unlabeled/CNN-3D/accuracy.hdf5",
    loss_path="indian-pines/model/with-unlabeled/CNN-3D/loss.hdf5")
ip_CNN_3D_eval.load_data(
    y_test_path="indian-pines/model/with-unlabeled/CNN-3D/y_test.hdf5",
    y_pred_path="indian-pines/model/with-unlabeled/CNN-3D/y_pred.hdf5",
    y_test_pred_path=(f"indian-pines/model/with-unlabeled/"
                      f"CNN-3D/y_test_pred.hdf5"))
ip_CNN_3D_eval.generate_report(
    report_path=(f"indian-pines/plots/with-unlabeled/"
                 f"CNN-3D/classification-report.tex"))
ip_CNN_3D_eval.plot_history(
    plot_path=(f"indian-pines/plots/with-unlabeled/"
               f"CNN-3D/training-history.svg"))
ip_CNN_3D_eval.plot_confusion(
    palette_path="indian-pines/data/confusion-matrix-palette.csv",
    plot_path=(f"indian-pines/plots/with-unlabeled/"
               f"CNN-3D/confusion-matrix.svg"))
ip_CNN_3D_eval.plot_pred(
    plot_path="indian-pines/plots/with-unlabeled/CNN-3D/predicted-map.svg")
