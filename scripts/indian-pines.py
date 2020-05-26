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

# Initialize Indian Pines with unlabeled data
ip = HyperspectralScene(name='Indian Pines',
                        path='scenes/indian-pines',
                        remove_unlabeled=False)

# # Data Exploration for Indian Pines with unlabeled data
# ip_DE = copy(ip)
# ip_DE.__class__ = DataExploration
# ip_DE.__post_init__()
# ip_DE.plot_ground_truth()
# ip_DE.fit_scaler(feature_range=(-1, 1))
# ip_DE.fit_PCA(n_components=15, whiten=True)
# ip_DE.fit_TSNE(perplexity=15,
#                early_exaggeration=10,
#                learning_rate=250,
#                n_iter=500)
# ip_DE.plot_PCA()
# ip_DE.plot_TSNE()

# Train 1D-CNN for Indian Pines with unlabeled data
ip_1D_CNN = copy(ip)
ip_1D_CNN.__class__ = Train1DCNN
ip_1D_CNN.__post_init__()
ip_1D_CNN.fit_scaler(feature_range=(-1, 1))
ip_1D_CNN.fit_PCA(n_components=15, whiten=True)
ip_1D_CNN.prepare_data(train_ratio=0.4, test_ratio=0.5, validation_ratio=0.1)
ip_1D_CNN.compile_1D_CNN()
ip_1D_CNN.fit_1D_CNN()
ip_1D_CNN.predict_data()

# Evaluate 1D-CNN for Indian Pines with unlabeled data
ip_1D_CNN_eval = copy(ip)
ip_1D_CNN_eval.__class__ = EvaluateCNN
ip_1D_CNN_eval.__post_init__(model_name='1D-CNN')
ip_1D_CNN_eval.plot_training_history(figsize=(9, 6.5))
ip_1D_CNN_eval.generate_report()
ip_1D_CNN_eval.plot_confusion_matrix(figsize=(6.5, 6.5))
ip_1D_CNN_eval.plot_predicted(figsize=(9, 6.5))

# Train 3D-CNN for Indian Pines with unlabeled data
ip_3D_CNN = copy(ip)
ip_3D_CNN.__class__ = Train3DCNN
ip_3D_CNN.__post_init__()
ip_3D_CNN.fit_scaler(feature_range=(-1, 1))
ip_3D_CNN.fit_PCA(n_components=15, whiten=False)
ip_3D_CNN.prepare_data(train_ratio=0.4, test_ratio=0.5, validation_ratio=0.1)
ip_3D_CNN.compile_3D_CNN()
ip_3D_CNN.fit_3D_CNN()
ip_3D_CNN.predict_data()

# Evaluate 3D-CNN for Indian Pines with unlabeled data
ip_3D_CNN_eval = copy(ip)
ip_3D_CNN_eval.__class__ = EvaluateCNN
ip_3D_CNN_eval.__post_init__(model_name='3D-CNN')
ip_3D_CNN_eval.plot_training_history(figsize=(9, 6.5))
ip_3D_CNN_eval.generate_report()
ip_3D_CNN_eval.plot_confusion_matrix(figsize=(6.5, 6.5))
ip_3D_CNN_eval.plot_predicted(figsize=(9, 6.5))

# Initialize Indian Pines without unlabeled data
ip = HyperspectralScene(name='Indian Pines',
                        path='scenes/indian-pines',
                        remove_unlabeled=True)

# # Data Exploration for Indian Pines without unlabeled data
# ip_DE = copy(ip)
# ip_DE.__class__ = DataExploration
# ip_DE.__post_init__()
# ip_DE.plot_ground_truth()
# ip_DE.fit_scaler(feature_range=(-1, 1))
# ip_DE.fit_PCA(n_components=15, whiten=True)
# ip_DE.fit_TSNE(perplexity=15,
#                early_exaggeration=10,
#                learning_rate=250,
#                n_iter=500)
# ip_DE.plot_PCA()
# ip_DE.plot_TSNE()

# Train 1D-CNN for Indian Pines without unlabeled data
ip_1D_CNN = copy(ip)
ip_1D_CNN.__class__ = Train1DCNN
ip_1D_CNN.__post_init__()
ip_1D_CNN.fit_scaler(feature_range=(-1, 1))
ip_1D_CNN.fit_PCA(n_components=15, whiten=True)
ip_1D_CNN.prepare_data(train_ratio=0.4, test_ratio=0.5, validation_ratio=0.1)
ip_1D_CNN.compile_1D_CNN()
ip_1D_CNN.fit_1D_CNN()
ip_1D_CNN.predict_data()

# Evaluate 1D-CNN for Indian Pines without unlabeled data
ip_1D_CNN_eval = copy(ip)
ip_1D_CNN_eval.__class__ = EvaluateCNN
ip_1D_CNN_eval.__post_init__(model_name='1D-CNN')
ip_1D_CNN_eval.plot_training_history(figsize=(9, 6.5))
ip_1D_CNN_eval.generate_report()
ip_1D_CNN_eval.plot_confusion_matrix(figsize=(6.5, 6.5))
ip_1D_CNN_eval.plot_predicted(figsize=(9, 6.5))

# # Train 3D-CNN for Indian Pines without unlabeled data
# ip_3D_CNN = copy(ip)
# ip_3D_CNN.__class__ = Train3DCNN
# ip_3D_CNN.__post_init__()
# ip_3D_CNN.fit_scaler(feature_range=(-1, 1))
# ip_3D_CNN.fit_PCA(n_components=15, whiten=False)
# ip_3D_CNN.prepare_data(train_ratio=0.4, test_ratio=0.5, validation_ratio=0.1)
# ip_3D_CNN.compile_3D_CNN()
# ip_3D_CNN.fit_3D_CNN()
# ip_3D_CNN.predict_data()

# # Evaluate 3D-CNN for Indian Pines without unlabeled data
# ip_3D_CNN_eval = copy(ip)
# ip_3D_CNN_eval.__class__ = EvaluateCNN
# ip_3D_CNN_eval.__post_init__(model_name='3D-CNN')
# ip_3D_CNN_eval.plot_training_history(figsize=(9, 6.5))
# ip_3D_CNN_eval.generate_report()
# ip_3D_CNN_eval.plot_confusion_matrix(figsize=(6.5, 6.5))
# ip_3D_CNN_eval.plot_predicted(figsize=(9, 6.5))
