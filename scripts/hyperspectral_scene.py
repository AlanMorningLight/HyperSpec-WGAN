# Import libraries
import csv
from dataclasses import dataclass, field

import numpy as np
from scipy.io import loadmat


# Data class for a hyperspectral scene
@dataclass
class HyperspectralScene:
    name: str
    path: str
    remove_unlabeled: bool
    image: np.ndarray = field(init=False)
    ground_truth: np.ndarray = field(init=False)
    labels: list = field(init=False)
    X: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    samples: int = field(init=False)
    features: int = field(init=False)

    def __post_init__(self):
        self.load_image()
        self.load_ground_truth()
        self.load_labels()
        self.X = np.reshape(a=self.image, newshape=(-1, self.image.shape[2]))
        self.y = np.reshape(a=self.ground_truth, newshape=-1)
        self.samples, self.features = self.X.shape

    # Load a hyperspectral image from a *.mat file
    def load_image(self):
        load = loadmat(file_name=f"{self.path}/data/image.mat").values()
        self.image = list(load)[-1]

    # Load a ground truth from a *.mat file
    def load_ground_truth(self):
        load = loadmat(file_name=f"{self.path}/data/ground-truth.mat").values()
        self.ground_truth = list(load)[-1]

    # Load class labels from a *.csv file
    def load_labels(self):
        with open(file=f"{self.path}/data/labels.csv") as file:
            self.labels = list(csv.reader(file, delimiter=','))[0]
