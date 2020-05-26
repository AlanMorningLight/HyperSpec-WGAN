# Import libraries
from dataclasses import dataclass

import numpy as np
from h5py import File
from skimage.util.shape import view_as_windows
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import (CSVLogger, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.layers import Conv3D, Dense, Dropout, Flatten, LeakyReLU

from hyperspectral_scene import HyperspectralScene


# Data class for a 3D-CNN
@dataclass(init=False)
class Train3DCNN(HyperspectralScene):
    train_3D_CNN_path: str
    y_temp: np.ndarray
    X_scale: np.ndarray
    X_PCA: np.ndarray
    X_all: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    X_valid: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_valid: np.ndarray
    model: Model
    y_pred: np.ndarray
    y_test_pred: np.ndarray

    def __post_init__(self):
        if self.remove_unlabeled:
            self.y_temp = self.y
            self.X = self.X[self.y != 0, :]
            self.y = self.y[self.y != 0] - 1
            self.labels = self.labels[1:]
            self.samples = self.X.shape[0]
            self.train_3D_CNN_path = (f"{self.path}/model/"
                                      f"without-unlabeled/"
                                      f"3D-CNN")
        else:
            self.train_3D_CNN_path = f"{self.path}/model/with-unlabeled/3D-CNN"

    # Scale each feature to a given range
    def fit_scaler(self, feature_range):
        model_scale = MinMaxScaler(feature_range=feature_range)
        self.X_scale = model_scale.fit_transform(X=self.X)

    # Fit a PCA model
    def fit_PCA(self, n_components, whiten):
        model_PCA = PCA(n_components=n_components, whiten=whiten)
        self.X_PCA = model_PCA.fit_transform(X=self.X_scale)
        self.features = self.X_PCA.shape[1]
        if self.remove_unlabeled:
            temp = np.zeros(shape=(len(self.y_temp), self.features))
            temp[self.y_temp != 0, :] = self.X_PCA
            self.X_PCA = temp

    # Split the data into training, testing, and validation sets
    def prepare_data(self, train_ratio, test_ratio, validation_ratio):
        X = np.reshape(a=self.X_PCA,
                       newshape=(self.image.shape[0],
                                 self.image.shape[1],
                                 self.features))
        side = self.features
        pad = (side - 1)//2
        X_pad = np.pad(array=X, pad_width=((pad,), (pad,), (0,)))
        X_split = view_as_windows(arr_in=X_pad,
                                  window_shape=(side, side, side))
        X_all = np.reshape(a=X_split, newshape=(-1, side, side, side))
        if self.remove_unlabeled:
            X_all = X_all[self.y_temp != 0, :, :, :]
        split_1 = 1 - train_ratio
        split_2 = 1 - (0.1/(test_ratio + validation_ratio))
        X_train, X_rest, y_train, y_rest = train_test_split(X_all,
                                                            self.y,
                                                            test_size=split_1,
                                                            random_state=42,
                                                            stratify=self.y)
        X_valid, X_test, y_valid, y_test = train_test_split(X_rest,
                                                            y_rest,
                                                            test_size=split_2,
                                                            random_state=42,
                                                            stratify=y_rest)
        self.X_all = np.reshape(a=X_all,
                                newshape=(-1, side, side, side, 1))
        self.X_train = np.reshape(a=X_train,
                                  newshape=(-1, side, side, side, 1))
        self.X_test = np.reshape(a=X_test,
                                 newshape=(-1, side, side, side, 1))
        self.X_valid = np.reshape(a=X_valid,
                                  newshape=(-1, side, side, side, 1))
        self.y_train = y_train
        self.y_test = y_test
        self.y_valid = y_valid

    # Compile a 3D-CNN model
    def compile_3D_CNN(self):
        inputs = Input(shape=self.X_train.shape[1:])
        x = Conv3D(filters=64,
                   kernel_size=(3, 3, 11),
                   bias_constraint=unit_norm())(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv3D(filters=32,
                   kernel_size=(2, 2, 5),
                   bias_constraint=unit_norm())(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Flatten()(x)
        x = Dense(units=256, activation='relu')(x)
        x = Dropout(rate=0.4)(x)
        x = Dense(units=128, activation='relu')(x)
        x = Dropout(rate=0.4)(x)
        outputs = Dense(units=len(self.labels), activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='Adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    # Fit the 3D-CNN model
    def fit_3D_CNN(self):
        best_weights = ModelCheckpoint(filepath=(f"{self.train_3D_CNN_path}/"
                                                 f"best-weights.hdf5"),
                                       verbose=2,
                                       save_best_only=True,
                                       save_weights_only=True)
        log = CSVLogger(filename=f"{self.train_3D_CNN_path}/history.csv")
        reduce_LR = ReduceLROnPlateau(factor=0.5,
                                      verbose=2,
                                      min_delta=1e-7,
                                      min_lr=1e-6)
        self.model.fit(x=self.X_train,
                       y=self.y_train,
                       batch_size=128,
                       epochs=200,
                       verbose=2,
                       callbacks=[best_weights, log, reduce_LR],
                       validation_data=(self.X_valid, self.y_valid))

    # Predict data using the 3D-CNN model with the best model weights
    def predict_data(self):
        self.model.load_weights(filepath=(f"{self.train_3D_CNN_path}/"
                                          f"best-weights.hdf5"))
        y_pred = self.model.predict(self.X_all)
        y_test_pred = self.model.predict(self.X_test)
        self.y_pred = np.argmax(a=y_pred, axis=1)
        self.y_test_pred = np.argmax(a=y_test_pred, axis=1)
        if self.remove_unlabeled:
            self.y_test += 1
            self.y_pred += 1
            self.y_test_pred += 1
        with File(name=f"{self.train_3D_CNN_path}/y_test.hdf5",
                  mode='w') as file:
            file.create_dataset(name='y_test', data=self.y_test)
        with File(name=f"{self.train_3D_CNN_path}/y_pred.hdf5",
                  mode='w') as file:
            file.create_dataset(name='y_pred', data=self.y_pred)
        with File(name=f"{self.train_3D_CNN_path}/y_test_pred.hdf5",
                  mode='w') as file:
            file.create_dataset(name='y_test_pred', data=self.y_test_pred)
