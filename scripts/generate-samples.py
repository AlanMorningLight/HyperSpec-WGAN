# Import libraries
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.backend import clip, mean
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (BatchNormalization, Conv1D, Dense,
                                     Flatten, LeakyReLU, Reshape, UpSampling1D)
from tensorflow.keras.optimizers import RMSprop

from scene import HyperspectralScene


class GradientClipping(Constraint):
    # Set clip value
    def __init__(self, clip_value):
        self.clip_value = clip_value

    # Clip model weights
    def __call__(self, weights):
        clip_weights = clip(x=weights,
                            min_value=-self.clip_value,
                            max_value=self.clip_value)
        return clip_weights


@dataclass(init=False)
class Train1DWGAN(HyperspectralScene):
    X_scale: np.ndarray
    model: Model
    real_disc_loss: list
    fake_disc_loss: list
    wgan_loss: list

    def __post_init__(self, class_num):
        self.X = self.X[self.y == class_num, :]
        self.y = self.y[self.y == class_num]
        self.samples = self.X.shape[0]

    # Calculate Wasserstein loss
    def __wasserstein_loss(self, y_true, y_pred):
        return mean(y_true * y_pred)

    # Compile the generator
    def __compile_generator(self):
        inputs = Input(shape=self.features//8)
        x = Dense(units=self.features,
                  kernel_initializer=RandomNormal())(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape(target_shape=(self.features//4, 4))(x)
        x = UpSampling1D()(x)
        x = Conv1D(filters=32,
                   kernel_size=5,
                   padding='causal',
                   kernel_initializer=RandomNormal())(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = UpSampling1D()(x)
        outputs = Conv1D(filters=1,
                         kernel_size=11,
                         padding='causal',
                         activation='tanh',
                         kernel_initializer=RandomNormal())(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=self.__wasserstein_loss,
                      optimizer=RMSprop(lr=0.00005))
        return model

    # Compile the discriminator
    def __compile_discriminator(self):
        inputs = Input(shape=(self.features, 1))
        x = Conv1D(filters=32,
                   kernel_size=11,
                   padding='causal',
                   kernel_initializer=RandomNormal(),
                   kernel_constraint=GradientClipping(clip_value=0.01))(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv1D(filters=32,
                   kernel_size=5,
                   padding='causal',
                   kernel_initializer=RandomNormal(),
                   kernel_constraint=GradientClipping(clip_value=0.01))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Flatten()(x)
        outputs = Dense(units=1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=self.__wasserstein_loss,
                      optimizer=RMSprop(lr=0.00005))
        return model

    def __get_real_samples(self, num_samples):
        index = np.randint(0, self.samples, num_samples)
        X = np.expand_dims(a=self.X_scale[index], axis=2)
        y = np.ones(shape=(num_samples, 1))
        return X, y

    # Scale each feature to a given range
    def fit_scaler(self, feature_range):
        model_scale = MinMaxScaler(feature_range=feature_range)
        self.X_scale = model_scale.fit_transform(X=self.X)

    # Compile the WGAN
    def compile_WGAN(self):
        generator = self.__compile_generator()
        discriminator = self.__compile_discriminator()
        discriminator.trainable = False
        self.model = discriminator(generator)
        self.model.compile(loss=self.__wasserstein_loss,
                           optimizer=RMSprop(lr=0.00005))
