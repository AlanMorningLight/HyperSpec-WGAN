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

# Initialize Salinas with unlabeled data
s = HyperspectralScene(name='Salinas',
                       path='scenes/salinas',
                       remove_unlabeled=False)

# # Data Exploration for Salinas with unlabeled data
# s_DE = copy(s)
# s_DE.__class__ = DataExploration
# s_DE.__post_init__()
# s_DE.plot_ground_truth()
# s_DE.fit_scaler(feature_range=(-1, 1))
# s_DE.fit_PCA(n_components=15, whiten=True)
# s_DE.fit_TSNE(perplexity=15,
#               early_exaggeration=10,
#               learning_rate=250,
#               n_iter=500)
# s_DE.plot_PCA()
# s_DE.plot_TSNE()

# Train 1D-CNN for Salinas with unlabeled data
s_1D_CNN = copy(s)
s_1D_CNN.__class__ = Train1DCNN
s_1D_CNN.__post_init__()
s_1D_CNN.fit_scaler(feature_range=(-1, 1))
s_1D_CNN.fit_PCA(n_components=15, whiten=True)
s_1D_CNN.prepare_data(train_ratio=0.4, test_ratio=0.5, validation_ratio=0.1)
s_1D_CNN.compile_1D_CNN()
s_1D_CNN.fit_1D_CNN()
s_1D_CNN.predict_data()

# Evaluate 1D-CNN for Salinas with unlabeled data
s_1D_CNN_eval = copy(s)
s_1D_CNN_eval.__class__ = EvaluateCNN
s_1D_CNN_eval.__post_init__(model_name='1D-CNN')
s_1D_CNN_eval.plot_training_history(figsize=(9, 6.5))
s_1D_CNN_eval.generate_report()
s_1D_CNN_eval.plot_confusion_matrix(figsize=(6.5, 6.5))
s_1D_CNN_eval.plot_predicted(figsize=(6.5, 9.5))

# Train 3D-CNN for Salinas with unlabeled data
# s_3D_CNN = copy(s)
# s_3D_CNN.__class__ = Train3DCNN
# s_3D_CNN.__post_init__()
# s_3D_CNN.fit_scaler(feature_range=(-1, 1))
# s_3D_CNN.fit_PCA(n_components=15, whiten=False)
# s_3D_CNN.prepare_data(train_ratio=0.4, test_ratio=0.5, validation_ratio=0.1)
# s_3D_CNN.compile_3D_CNN()
# s_3D_CNN.fit_3D_CNN()
# s_3D_CNN.predict_data()

# # Evaluate 3D-CNN for Salinas with unlabeled data
# s_3D_CNN_eval = copy(s)
# s_3D_CNN_eval.__class__ = EvaluateCNN
# s_3D_CNN_eval.__post_init__(model_name='3D-CNN')
# s_3D_CNN_eval.plot_training_history(figsize=(9, 6.5))
# s_3D_CNN_eval.generate_report()
# s_3D_CNN_eval.plot_confusion_matrix(figsize=(6.5, 6.5))
# s_3D_CNN_eval.plot_predicted(figsize=(6.5, 9.5))

# Initialize Salinas without unlabeled data
s = HyperspectralScene(name='Salinas',
                       path='scenes/salinas',
                       remove_unlabeled=True)

# # Data Exploration for Salinas without unlabeled data
# s_DE = copy(s)
# s_DE.__class__ = DataExploration
# s_DE.__post_init__()
# s_DE.plot_ground_truth()
# s_DE.fit_scaler(feature_range=(-1, 1))
# s_DE.fit_PCA(n_components=15, whiten=True)
# s_DE.fit_TSNE(perplexity=15,
#               early_exaggeration=10,
#               learning_rate=250,
#               n_iter=500)
# s_DE.plot_PCA()
# s_DE.plot_TSNE()

# Train 1D-CNN for Salinas without unlabeled data
s_1D_CNN = copy(s)
s_1D_CNN.__class__ = Train1DCNN
s_1D_CNN.__post_init__()
s_1D_CNN.fit_scaler(feature_range=(-1, 1))
s_1D_CNN.fit_PCA(n_components=15, whiten=True)
s_1D_CNN.prepare_data(train_ratio=0.4, test_ratio=0.5, validation_ratio=0.1)
s_1D_CNN.compile_1D_CNN()
s_1D_CNN.fit_1D_CNN()
s_1D_CNN.predict_data()

# Evaluate 1D-CNN for Salinas without unlabeled data
s_1D_CNN_eval = copy(s)
s_1D_CNN_eval.__class__ = EvaluateCNN
s_1D_CNN_eval.__post_init__(model_name='1D-CNN')
s_1D_CNN_eval.plot_training_history(figsize=(9, 6.5))
s_1D_CNN_eval.generate_report()
s_1D_CNN_eval.plot_confusion_matrix(figsize=(6.5, 6.5))
s_1D_CNN_eval.plot_predicted(figsize=(6.5, 9.5))

# # Train 3D-CNN for Salinas without unlabeled data
# s_3D_CNN = copy(s)
# s_3D_CNN.__class__ = Train3DCNN
# s_3D_CNN.__post_init__()
# s_3D_CNN.fit_scaler(feature_range=(-1, 1))
# s_3D_CNN.fit_PCA(n_components=15, whiten=False)
# s_3D_CNN.prepare_data(train_ratio=0.4, test_ratio=0.5, validation_ratio=0.1)
# s_3D_CNN.compile_3D_CNN()
# s_3D_CNN.fit_3D_CNN()
# s_3D_CNN.predict_data()

# # Evaluate 3D-CNN for Salinas without unlabeled data
# s_3D_CNN_eval = copy(s)
# s_3D_CNN_eval.__class__ = EvaluateCNN
# s_3D_CNN_eval.__post_init__(model_name='3D-CNN')
# s_3D_CNN_eval.plot_training_history(figsize=(9, 6.5))
# s_3D_CNN_eval.generate_report()
# s_3D_CNN_eval.plot_confusion_matrix(figsize=(6.5, 6.5))
# s_3D_CNN_eval.plot_predicted(figsize=(6.5, 9.5))
