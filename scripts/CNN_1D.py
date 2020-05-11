# Import libraries
from dataclasses import dataclass

import numpy as np
from h5py import File
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Model, models
from tensorflow.keras.callbacks import History, ModelCheckpoint
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from scene import HyperspectralScene


# Data class for a 3D-CNN
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
    y_pred: np.ndarray
    y_test_pred: np.ndarray
    model: Model
    history: History

    # Remove unlabeled data from X and y
    def check_remove_unlabeled(self):
        if self.remove_unlabeled:
            self.X = self.X[self.y != 0, :]
            self.y = self.y[self.y != 0] - 1

    # Scale each feature to a given range
    def fit_scaler(self, feature_range):
        model_scale = MinMaxScaler(feature_range=feature_range)
        self.X_scale = model_scale.fit_transform(X=self.X)

    # Fit a PCA model
    def fit_PCA(self, n_components, whiten):
        model_PCA = PCA(n_components=n_components, whiten=whiten)
        self.X_PCA = model_PCA.fit_transform(X=self.X_scale)

    # Split data into 60% training, 20% testing, and 20% validation
    def prepare_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_PCA,
                                                            self.y,
                                                            test_size=0.4,
                                                            random_state=42,
                                                            stratify=self.y)
        X_test, X_valid, y_test, y_valid = train_test_split(X_test,
                                                            y_test,
                                                            test_size=0.5,
                                                            random_state=42,
                                                            stratify=y_test)
        self.X_all = np.reshape(a=self.X_PCA,
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

    # Design a 1D-CNN model
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
        x = Dropout(rate=0.4)(x)
        x = Dense(units=128, activation='relu')(x)
        x = Dropout(rate=0.4)(x)
        output_layer = Dense(units=len(self.labels), activation='softmax')(x)
        self.model = Model(inputs=input_layer, outputs=output_layer)

    # Fit a 1D-CNN model and save the best model
    def fit_CNN_1D(self, model_dir):
        self.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        checkpoint = ModelCheckpoint(filepath=f"{model_dir}/model.hdf5",
                                     monitor='val_loss',
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

    # Predict data using the best model and save testing data and predictions
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

    # Save model training history
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

    # Initialize other class attributes
    def __post_init__(self):
        self.check_remove_unlabeled()
