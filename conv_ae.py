'''Convolutional AutoEncoder'''
from datetime import datetime
import fire
import importlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import norm
import sys
import yaml

from keras import backend as K
from keras.datasets import mnist
from keras.layers import (Input, Layer)
from keras.metrics import binary_crossentropy
from keras.models import Model
from keras.optimizers import RMSprop

from autoencoder import AutoEncoder


class ConvAE(AutoEncoder):
    def _init_data(self):
        (x_train, _), (x_test, _) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.astype('float32') / 255.
        x_test = x_test.reshape(x_test.shape + (1,))
        return x_train, x_test

    def build_model(self):
        input = Input(self.image_size)
        input_dim = self.image_size[0]

        encoder_path, decoder_path = self._init_path()

        encoded = getattr(sys.modules[encoder_path], 'encoder')(input)
        decoded = getattr(sys.modules[decoder_path], 'decoder')(encoded)

        model = Model(input=input, output=decoded)

        return model

    def get_decoder(self):
        return self._decoder

    def model_compile(self, model):
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def train(self, params_path='conv_params.yml'):
        params = self._load_yml_data(params_path)
        self._set_params(**params)
        x_train, x_test = self._init_data()
        model = self.build_model()
        model = self.model_compile(model)
        model.summary()
        model.fit(x_train,
                  x_train,
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  shuffle=True,
                  validation_data=(x_test, x_test))
        model_path = self._save_path()
        model.save_weights(str(model_path))

    def predict(self, model_dir='', fig_name='test.png'):
        model_path = Path(self.model_root, model_dir)
        params_path = list(model_path.glob('*.yml'))[0]
        params = self._load_yml_data(str(params_path))
        self._set_params(**params)
        _, x_test = self._init_data()
        model = self.build_model()
        model_name = Path(model_path, self.model_name)
        model.load_weights(str(model_name))
        model.summary()
        decoded_imgs = model.predict(x_test)
        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(n):

            ax = plt.subplot(2, n, i+1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, n, i+1+n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(fig_name)
        plt.show()

    def demo(self,
             model_dir='',
             vae_params_path='vae_params.yml',
             fig_name='test.png'):
        params = self._load_yml_data(vae_params_path)
        self._set_params(**params)
        self._set_vae_params(**params)
        n = 15
        batch_size = 32
        digit_size = 28
        figure = np.zeros((digit_size*n, digit_size*n))
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
        model = self.build_model()
        model_path = Path(self.model_root, model_dir)
        model_name = Path(model_path, self.model_name)
        model.load_weights(str(model_name))
        decoder = self.get_decoder()
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = np.array([[xi, yi]])
                z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
                x_decoded = decoder.predict(z_sample, batch_size)
                digit = x_decoded[0].reshape(digit_size, digit_size)
                figure[i*digit_size:(i+1)*digit_size,
                       j*digit_size:(j+1)*digit_size] = digit
        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        plt.savefig(fig_name)
        plt.show()

if __name__=="__main__":
    fire.Fire(ConvAE)
