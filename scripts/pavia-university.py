# Import libraries
import os
from copy import copy

from data_exploration import DataExploration
from evaluate_CNN import EvaluateCNN
from hyperspectral_scene import HyperspectralScene
from train_1D_CNN import Train1DCNN
from train_3D_CNN import Train3DCNN

# Set the working directory
if os.path.basename(os.getcwd()) == 'scripts':
    os.chdir("..")

# Initialize Pavia University with unlabeled data
pu = HyperspectralScene(name='Pavia University',
                        path='scenes/pavia-university',
                        remove_unlabeled=False)

# Data Exploration for Pavia University with unlabeled data
pu_DE = copy(pu)
pu_DE.__class__ = DataExploration
pu_DE.__post_init__()
pu_DE.plot_ground_truth()
pu_DE.fit_scaler(feature_range=(-1, 1))
pu_DE.fit_PCA(n_components=15, whiten=True)
pu_DE.fit_TSNE(perplexity=15,
               early_exaggeration=10,
               learning_rate=250,
               n_iter=500)
pu_DE.plot_PCA()
pu_DE.plot_TSNE()

# Train 1D-CNN for Pavia University with unlabeled data
pu_1D_CNN = copy(pu)
pu_1D_CNN.__class__ = Train1DCNN
pu_1D_CNN.__post_init__()
pu_1D_CNN.fit_scaler(feature_range=(-1, 1))
pu_1D_CNN.fit_PCA(n_components=15, whiten=True)
pu_1D_CNN.prepare_data()
pu_1D_CNN.compile_1D_CNN()
pu_1D_CNN.fit_1D_CNN()
pu_1D_CNN.predict_data()

# Evaluate 1D-CNN for Pavia University with unlabeled data
pu_1D_CNN_eval = copy(pu)
pu_1D_CNN_eval.__class__ = EvaluateCNN
pu_1D_CNN_eval.__post_init__(model_name='1D-CNN')
pu_1D_CNN_eval.plot_training_history(figsize=(9, 6.5))
pu_1D_CNN_eval.generate_report()
pu_1D_CNN_eval.plot_confusion_matrix(figsize=(6.5, 6.5))
pu_1D_CNN_eval.plot_predicted(figsize=(9, 6.5))

# Train 3D-CNN for Pavia University with unlabeled data
pu_3D_CNN = copy(pu)
pu_3D_CNN.__class__ = Train3DCNN
pu_3D_CNN.__post_init__()
pu_3D_CNN.fit_scaler(feature_range=(-1, 1))
pu_3D_CNN.fit_PCA(n_components=15, whiten=True)
pu_3D_CNN.prepare_data()
pu_3D_CNN.compile_3D_CNN()
pu_3D_CNN.fit_3D_CNN()
pu_3D_CNN.predict_data()

# Evaluate 3D-CNN for Pavia University with unlabeled data
pu_3D_CNN_eval = copy(pu)
pu_3D_CNN_eval.__class__ = EvaluateCNN
pu_3D_CNN_eval.__post_init__(model_name='3D-CNN')
pu_3D_CNN_eval.plot_training_history(figsize=(9, 6.5))
pu_3D_CNN_eval.generate_report()
pu_3D_CNN_eval.plot_confusion_matrix(figsize=(6.5, 6.5))
pu_3D_CNN_eval.plot_predicted(figsize=(6.5, 8))

# Initialize Pavia University without unlabeled data
pu = HyperspectralScene(name='Pavia University',
                        path='scenes/pavia-university',
                        remove_unlabeled=True)

# Data Exploration for Pavia University without unlabeled data
pu_DE = copy(pu)
pu_DE.__class__ = DataExploration
pu_DE.__post_init__()
pu_DE.plot_ground_truth()
pu_DE.fit_scaler(feature_range=(-1, 1))
pu_DE.fit_PCA(n_components=15, whiten=True)
pu_DE.fit_TSNE(perplexity=15,
               early_exaggeration=10,
               learning_rate=250,
               n_iter=500)
pu_DE.plot_PCA()
pu_DE.plot_TSNE()

# Train 1D-CNN for Pavia University without unlabeled data
pu_1D_CNN = copy(pu)
pu_1D_CNN.__class__ = Train1DCNN
pu_1D_CNN.__post_init__()
pu_1D_CNN.fit_scaler(feature_range=(-1, 1))
pu_1D_CNN.fit_PCA(n_components=15, whiten=True)
pu_1D_CNN.prepare_data()
pu_1D_CNN.compile_1D_CNN()
pu_1D_CNN.fit_1D_CNN()
pu_1D_CNN.predict_data()

# Evaluate 1D-CNN for Pavia University without unlabeled data
pu_1D_CNN_eval = copy(pu)
pu_1D_CNN_eval.__class__ = EvaluateCNN
pu_1D_CNN_eval.__post_init__(model_name='1D-CNN')
pu_1D_CNN_eval.plot_training_history(figsize=(9, 6.5))
pu_1D_CNN_eval.generate_report()
pu_1D_CNN_eval.plot_confusion_matrix(figsize=(6.5, 6.5))
pu_1D_CNN_eval.plot_predicted(figsize=(6.5, 8))

# Train 3D-CNN for Pavia University without unlabeled data
pu_3D_CNN = copy(pu)
pu_3D_CNN.__class__ = Train3DCNN
pu_3D_CNN.__post_init__()
pu_3D_CNN.fit_scaler(feature_range=(-1, 1))
pu_3D_CNN.fit_PCA(n_components=15, whiten=True)
pu_3D_CNN.prepare_data()
pu_3D_CNN.compile_3D_CNN()
pu_3D_CNN.fit_3D_CNN()
pu_3D_CNN.predict_data()

# Evaluate 3D-CNN for Pavia University without unlabeled data
pu_3D_CNN_eval = copy(pu)
pu_3D_CNN_eval.__class__ = EvaluateCNN
pu_3D_CNN_eval.__post_init__(model_name='3D-CNN')
pu_3D_CNN_eval.plot_training_history(figsize=(9, 6.5))
pu_3D_CNN_eval.generate_report()
pu_3D_CNN_eval.plot_confusion_matrix(figsize=(6.5, 6.5))
pu_3D_CNN_eval.plot_predicted(figsize=(6.5, 8))
