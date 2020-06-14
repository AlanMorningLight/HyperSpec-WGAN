# Import libraries
from dataclasses import dataclass

import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.backend import clip, mean
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Conv1D, Conv1DTranspose, Dense, Flatten,
                                     LeakyReLU, Reshape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

from hyperspectral_scene import HyperspectralScene


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

    # Get the new clip value
    def get_config(self):
        return {'clip_value': self.clip_value}


@dataclass(init=False)
class GenerateSamples(HyperspectralScene):
    class_num: int
    class_label: str
    X_class: np.ndarray
    y_class: np.ndarray
    class_samples: int
    samples_mean_std: DataFrame
    latent_features: int
    generator: Model
    discriminator: Model
    wgan: Model
    X_scale: np.ndarray
    real_disc_history: np.ndarray
    fake_disc_history: np.ndarray
    wgan_history: np.ndarray

    def __post_init__(self, class_num):
        self.class_num = class_num
        self.class_label = self.labels[class_num]
        self.X_class = self.X[self.y == class_num, :]
        self.y_class = self.y[self.y == class_num]
        self.class_samples = self.X_class.shape[0]
        self.samples_mean_std = self.__samples_mean_std(X=self.X_class)
        self.latent_features = self.features // 2
        self.generator = self.__design_generator()
        self.discriminator = self.__compile_discriminator()
        self.wgan = self.__compile_WGAN()

    # Create a DataFrame with the samples' mean and standard deviation
    def __samples_mean_std(self, X):
        X_mean = np.mean(a=X, axis=0)
        X_std = np.std(a=X, axis=0)
        samples_mean_std = DataFrame(
            data={
                'Mean + Standard Deviation': (X_mean + X_std),
                'Mean': X_mean,
                'Mean - Standard Deviation': (X_mean - X_std)
            })
        return samples_mean_std

    # Calculate Wasserstein loss
    def __wasserstein_loss(self, y_true, y_pred):
        return mean(y_true * y_pred)

    # Design the generator
    def __design_generator(self):
        inputs = Input(shape=self.latent_features)
        x = Dense(units=(self.latent_features * 2),
                  kernel_initializer=RandomNormal(stddev=0.02))(inputs)
        x = LeakyReLU(alpha=0.2)(x)
        x = Reshape(target_shape=(self.latent_features * 2, 1))(x)
        x = Conv1DTranspose(filters=32,
                            kernel_size=3,
                            strides=2,
                            padding='same',
                            kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv1DTranspose(filters=64,
                            kernel_size=7,
                            strides=2,
                            padding='same',
                            kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv1DTranspose(filters=32,
                            kernel_size=11,
                            strides=2,
                            padding='same',
                            kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv1D(filters=1,
                   kernel_size=3,
                   padding='same',
                   kernel_initializer=RandomNormal(stddev=0.02))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Flatten()(x)
        x = Dense(units=self.features)(x)
        outputs = LeakyReLU(alpha=0.2)(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    # Compile the discriminator
    def __compile_discriminator(self):
        inputs = Input(shape=self.features)
        x = Reshape(target_shape=(self.features, 1))(inputs)
        x = Conv1D(filters=32,
                   kernel_size=3,
                   strides=2,
                   padding='same',
                   kernel_initializer=RandomNormal(stddev=0.02),
                   kernel_constraint=GradientClipping(clip_value=0.01))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv1D(filters=64,
                   kernel_size=7,
                   strides=2,
                   padding='same',
                   kernel_initializer=RandomNormal(stddev=0.02),
                   kernel_constraint=GradientClipping(clip_value=0.01))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv1D(filters=32,
                   kernel_size=11,
                   strides=2,
                   padding='same',
                   kernel_initializer=RandomNormal(stddev=0.02),
                   kernel_constraint=GradientClipping(clip_value=0.01))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Flatten()(x)
        x = Dense(units=self.latent_features)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(units=self.latent_features // 2)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(units=1)(x)
        outputs = LeakyReLU(alpha=0.2)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss=self.__wasserstein_loss,
                      optimizer=RMSprop(lr=0.00005))
        return model

    # Compile the WGAN
    def __compile_WGAN(self):
        self.discriminator.trainable = False
        model = Sequential()
        model.add(layer=self.generator)
        model.add(layer=self.discriminator)
        model.compile(loss=self.__wasserstein_loss,
                      optimizer=RMSprop(lr=0.00005))
        return model

    # Retrieve samples from the hyperspectral scene
    def __real_samples(self, num_samples):
        index = np.random.choice(a=self.class_samples, size=num_samples)
        X = np.expand_dims(a=self.X_scale[index], axis=2)
        y = -np.ones(shape=(num_samples, 1))
        return X, y

    # Create fake samples using the generator
    def __fake_samples(self, num_samples):
        X = np.random.standard_normal(size=(num_samples, self.latent_features))
        X_gen = self.generator.predict(X)
        y_gen = np.ones(shape=(num_samples, 1))
        return X_gen, y_gen

    # Set plot parameters and create a plot instance
    def __plot_parameters(self):
        sb.set(context='paper',
               style='white',
               font='serif',
               rc={'figure.figsize': (9, 6.5)})
        figure, axes = plt.subplots(constrained_layout=True)
        return figure, axes

    def __plot_training_samples(self, data, epoch):
        figure, _ = self.__plot_parameters()
        gs = GridSpec(nrows=3,
                      ncols=1,
                      figure=figure,
                      height_ratios=[1, 0.01, 1])
        axes_0 = plt.subplot(gs[0])
        axes_1 = plt.subplot(gs[1])
        axes_2 = plt.subplot(gs[2])
        generated_samples_plot = sb.lineplot(
            data=data,
            palette=sb.color_palette(['#C8102E', '#00B388', '#F6BE00']),
            dashes=False,
            ax=axes_0)
        axes_0.get_legend().remove()
        axes_0.set_xticks(ticks=[])
        generated_samples_plot.set(title=(f'{self.class_label} '
                                          f'Generated Samples'),
                                   ylabel='Scaled Intensity')
        handles, labels = axes_0.get_legend_handles_labels()
        axes_1.legend(handles=handles,
                      labels=labels,
                      loc='center',
                      ncol=3,
                      frameon=False,
                      borderaxespad=0)
        axes_1.axis('off')
        actual_samples_plot = sb.lineplot(
            data=self.samples_mean_std,
            palette=sb.color_palette(['#C8102E', '#00B388', '#F6BE00']),
            dashes=False,
            legend=False,
            ax=axes_2)
        actual_samples_plot.set(xlabel='Spectral Band', ylabel='Intensity')
        figure.savefig(fname=(f"{self.path}/plots/wgan/training-history/"
                              f"{self.class_num:02d}-{self.class_label}/"
                              f"epoch-{epoch:04d}.svg"),
                       format='svg')
        plt.close(fig=figure)

    # Scale each feature to a given range
    def fit_scaler(self, feature_range):
        model_scale = MinMaxScaler(feature_range=feature_range)
        self.X_scale = model_scale.fit_transform(X=self.X_class)

    # Train WGAN
    def train_WGAN(self, batch_size, epochs, update_disc_rate):
        real_disc_loss = np.empty(shape=update_disc_rate)
        fake_disc_loss = np.empty(shape=update_disc_rate)
        self.real_disc_history = np.empty(shape=(batch_size * epochs))
        self.fake_disc_history = np.empty(shape=(batch_size * epochs))
        self.wgan_history = np.empty(shape=(batch_size * epochs))
        for i in range(batch_size * epochs):
            epoch = (i + batch_size) // batch_size
            if i % batch_size == 0:
                print(f'Epoch: {epoch:04d}/{epochs}')
            for j in range(update_disc_rate):
                X_real, y_real = self.__real_samples(num_samples=batch_size //
                                                     2)
                real_disc_loss[j] = self.discriminator.train_on_batch(x=X_real,
                                                                      y=y_real)
                X_fake, y_fake = self.__fake_samples(num_samples=batch_size //
                                                     2)
                fake_disc_loss[j] = self.discriminator.train_on_batch(x=X_fake,
                                                                      y=y_fake)
            self.real_disc_history[i] = np.mean(a=real_disc_loss)
            self.fake_disc_history[i] = np.mean(a=fake_disc_loss)
            X_wgan = np.random.standard_normal(size=(batch_size,
                                                     self.latent_features))
            y_wgan = -np.ones(shape=(batch_size, 1))
            self.wgan_history[i] = self.wgan.train_on_batch(x=X_wgan, y=y_wgan)
            print((f'Step: {i + 1} - '
                   f'Real Loss: {self.real_disc_history[i]:.4f} - '
                   f'Fake Loss: {self.fake_disc_history[i]:.4f} - '
                   f'WGAN Loss: {self.wgan_history[i]:.4f}'))
            if (i + 1) % (batch_size) == 0:
                X_gen, _ = self.__fake_samples(num_samples=self.class_samples)
                X_gen_mean_std = self.__samples_mean_std(X=X_gen)
                self.__plot_training_samples(data=X_gen_mean_std, epoch=epoch)
