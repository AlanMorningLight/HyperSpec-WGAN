# Import libraries
from dataclasses import dataclass

import numpy as np
from h5py import File
from pandas import DataFrame
from skimage.util.shape import view_as_windows
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Model, models
from tensorflow.keras.callbacks import History, ModelCheckpoint
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.layers import Conv3D, Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from scene import HyperspectralScene


# Data class for a 3D-CNN
@dataclass(init=False)
class Train3DCNN(HyperspectralScene):
    X_scale: np.ndarray
    X_PCA: np.ndarray
    X_all: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    X_valid: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_valid: np.ndarray
    y_pred: np.ndarray
    y_test_pred: np.ndarray
    model: Model
    history: History

    # Scales each feature to a given range
    def fit_scaler(self):
        model_scale = MinMaxScaler(feature_range=(-1, 1))
        self.X_scale = model_scale.fit_transform(X=self.X)

    # Fits a PCA model to the data
    def fit_PCA(self, n_components=0.98):
        model_PCA = PCA(n_components=n_components, whiten=True)
        self.X_PCA = model_PCA.fit_transform(X=self.X_scale)

    # Splits data into 60% training, 20% testing, 20% validation for 3D-CNN
    def prepare_data(self):
        X = np.reshape(a=self.X_PCA, newshape=(self.image.shape[0],
                                               self.image.shape[1],
                                               self.X_PCA.shape[1]))
        side = X.shape[2]
        pad = int(side / 2)
        X_pad = np.pad(array=X, pad_width=((pad,), (pad,), (0,)))
        X_split = view_as_windows(arr_in=X_pad[1:, 1:, :],
                                  window_shape=(side, side, side))
        X_all = np.reshape(a=X_split, newshape=(X.shape[0] * X.shape[1],
                                                side,
                                                side,
                                                side))
        X_train, X_test, y_train, y_test = train_test_split(X_all,
                                                            self.y,
                                                            test_size=0.4,
                                                            random_state=42,
                                                            stratify=self.y)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test,
                                                            y_test,
                                                            test_size=0.5,
                                                            random_state=42,
                                                            stratify=y_test)
        self.X_all = np.reshape(a=X_all,
                                newshape=(-1, side, side, side, 1))
        self.X_train = np.reshape(a=X_train,
                                  newshape=(-1, side, side, side, 1))
        self.X_test = np.reshape(a=X_test,
                                 newshape=(-1, side, side, side, 1))
        self.X_valid = np.reshape(a=X_valid,
                                  newshape=(-1, side, side, side, 1))
        self.y_train = to_categorical(y=y_train)
        self.y_test = y_test
        self.y_valid = to_categorical(y=y_valid)

    # Designs a 3D-CNN model
    def design_CNN_3D(self):
        input_layer = Input(shape=self.X_train.shape[1:])
        x = Conv3D(filters=8,
                   kernel_size=(3, 3, 11),
                   padding='valid',
                   bias_constraint=unit_norm())(input_layer)
        x = LeakyReLU()(x)
        x = Conv3D(filters=16,
                   kernel_size=(3, 3, 7),
                   padding='valid',
                   bias_constraint=unit_norm())(input_layer)
        x = LeakyReLU()(x)
        x = Conv3D(filters=32,
                   kernel_size=(3, 3, 3),
                   padding='valid',
                   bias_constraint=unit_norm())(input_layer)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(units=256, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(units=128, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        output_layer = Dense(units=len(self.labels), activation='softmax')(x)
        self.model = Model(inputs=input_layer, outputs=output_layer)

    # Fits a 3D-CNN model to the data
    def fit_CNN_3D(self, model_dir):
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        checkpoint = ModelCheckpoint(filepath=f"{model_dir}/model.hdf5",
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True)
        self.history = self.model.fit(x=self.X_train,
                                      y=self.y_train,
                                      batch_size=256,
                                      epochs=200,
                                      verbose=2,
                                      callbacks=[checkpoint],
                                      validation_data=(self.X_valid,
                                                       self.y_valid))

    # Predicts data using the best model and saves testing data
    def predict_data(self, model_path, data_dir):
        self.model = models.load_model(filepath=model_path)
        y_pred = self.model.predict(self.X_all)
        y_test_pred = self.model.predict(self.X_test)
        self.y_pred = np.argmax(a=y_pred, axis=1)
        self.y_test_pred = np.argmax(a=y_test_pred, axis=1)
        with File(name=f"{data_dir}/y_test.hdf5", mode='w') as file:
            file.create_dataset(name='y_test', data=self.y_test)
        with File(name=f"{data_dir}/y_pred.hdf5", mode='w') as file:
            file.create_dataset(name='y_pred', data=self.y_pred)
        with File(name=f"{data_dir}/y_test_pred.hdf5", mode='w') as file:
            file.create_dataset(name='y_test_pred', data=self.y_test_pred)

    # Saves model training history
    def save_history(self, history_dir):
        accuracy = {'Training': self.history.history['accuracy'],
                    'Validation': self.history.history['val_accuracy']}
        loss = {'Training': self.history.history['loss'],
                'Validation': self.history.history['val_loss']}
        DataFrame.to_hdf(DataFrame.from_dict(accuracy),
                         path_or_buf=f"{history_dir}/accuracy.hdf5",
                         key='history',
                         mode='w')
        DataFrame.to_hdf(DataFrame.from_dict(loss),
                         path_or_buf=f"{history_dir}/loss.hdf5",
                         key='history',
                         mode='w')
