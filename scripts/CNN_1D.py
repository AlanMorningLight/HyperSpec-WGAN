# Import libraries
from dataclasses import dataclass

import numpy as np
from h5py import File
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import History, ModelCheckpoint
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from scene import HyperspectralScene


# Data class for 1D-CNNs
@dataclass(init=False)
class Train1DCNN(HyperspectralScene):
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
    history: History

    # Scales each feature to a given range
    def fit_scaler(self):
        model_scale = MinMaxScaler(feature_range=(-1, 1))
        self.X_scale = model_scale.fit_transform(X=self.X)

    # Fits a PCA model to the data
    def fit_PCA(self, n_components=0.98):
        model_PCA = PCA(n_components=n_components, whiten=True)
        self.X_PCA = model_PCA.fit_transform(X=self.X_scale)

    # Splits data into 60% training, 20% testing, 20% validation for 1D-CNN
    def prepare_data(self):
        X_all = self.X_PCA
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
                                newshape=(-1, self.X_PCA.shape[1], 1))
        self.X_train = np.reshape(a=X_train,
                                  newshape=(-1, self.X_PCA.shape[1], 1))
        self.X_test = np.reshape(a=X_test,
                                 newshape=(-1, self.X_PCA.shape[1], 1))
        self.X_valid = np.reshape(a=X_valid,
                                  newshape=(-1, self.X_PCA.shape[1], 1))
        self.y_train = to_categorical(y=y_train)
        self.y_test = y_test
        self.y_valid = to_categorical(y=y_valid)

    # Designs a 1D-CNN model
    def design_CNN_1D(self):
        input_layer = Input(shape=self.X_train.shape[1:])
        x = Conv1D(filters=32,
                   kernel_size=11,
                   padding='causal',
                   bias_constraint=unit_norm())(input_layer)
        x = LeakyReLU()(x)
        x = Conv1D(filters=32,
                   kernel_size=5,
                   padding='causal',
                   bias_constraint=unit_norm())(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = Dense(units=256, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(units=128, activation='relu')(x)
        x = Dropout(rate=0.5)(x)
        output_layer = Dense(units=len(self.labels), activation='softmax')(x)
        self.model = Model(inputs=input_layer, outputs=output_layer)

    # Fits a 1D-CNN model to the data
    def fit_CNN_1D(self, model_dir):
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

    # Saves model training history and testing data
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
        with File(name=f"{history_dir}/X_all.hdf5", mode='w') as file:
            file.create_dataset(name='X_all', data=self.X_all)
        with File(name=f"{history_dir}/X_test.hdf5", mode='w') as file:
            file.create_dataset(name='X_test', data=self.X_test)
        with File(name=f"{history_dir}/y_test.hdf5", mode='w') as file:
            file.create_dataset(name='y_test', data=self.y_test)
