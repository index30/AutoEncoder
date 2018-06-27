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
from keras.layers import (concatenate, Input, Layer)
from keras.metrics import binary_crossentropy
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import to_categorical

from autoencoder import AutoEncoder

class VariationalLayer(Layer):
    def set_z(self, z):
        self.z_mean, self.z_sigma = z[0], z[1]

    def _calc_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        ## reconstraction error
        reconst_error = K.sum(K.binary_crossentropy(x, z_decoded), axis=-1)
        ## regularization parameter
        regular_parameter = -0.5 * K.sum(1 + self.z_mean - K.square(self.z_mean) - K.exp(self.z_sigma))

        return K.mean(reconst_error + regular_parameter)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self._calc_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x


class VAE(AutoEncoder):
    vae_param_names = ['latent_dim']

    def _set_vae_params(self, **parameter):
        for param, value in parameter.items():
            if param in self.vae_param_names:
                setattr(self, param, value)

    ### fetch parameter
    def _fetch_vae_params(self):
        return {attr_name: getattr(self, attr_name)
                for attr_name in self.vae_param_names}

    def _init_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.astype('float32') / 255.
        x_test = x_test.reshape(x_test.shape + (1,))
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        return x_train, x_test, y_train, y_test

    def _save_path(self):
        model_path = Path(self.model_root)
        if not model_path.exists():
            model_path.mkdir()
        now = datetime.now().strftime("%m%d%H%M%S")
        dir_name = self.encode_clf + "_" + self.decode_clf +"_"+ now
        eval_path = Path(model_path, dir_name)
        eval_path.mkdir()
        ## save parameter as yaml
        param_path = Path(eval_path, "params.yml")
        param_data = self._fetch_params()# + self._fetch_vae_params()
        self._write_to_yaml(param_data, param_path)
        ## make callback
        model_path = Path(eval_path, "model.h5")
        return model_path

    def _sampling(self, args):
        z_mean, z_sigma = args
        epsilon = K.random_normal(shape=(self.latent_dim,),
                                  mean=0.,
                                  stddev=1.)
        return z_mean + z_sigma * epsilon

    def build_model(self):
        input = Input(self.image_size)
        input_dim = self.image_size[0]

        encoder_path, decoder_path = self._init_path()

        z_mean, z_sigma, shape_conv = getattr(sys.modules[encoder_path], 'encoder')(input, self.encode_dim, self.latent_dim)
        z_decoded, self._decoder = getattr(sys.modules[decoder_path], 'decoder')([z_mean, z_sigma], self.latent_dim, self._sampling, shape_conv)

        self._decoder.summary()

        l = VariationalLayer()
        l.set_z([z_mean, z_sigma])

        y = l([input, z_decoded])

        model = Model(input=input, output=y)

        return model

    def get_decoder(self):
        return self._decoder

    def model_compile(self, model):
        model.compile(loss=None, optimizer=RMSprop())
        return model

    def train(self, vae_params_path='vae_params.yml'):
        params = self._load_yml_data(vae_params_path)
        self._set_params(**params)
        self._set_vae_params(**params)
        x_train, x_test, y_train, y_test = self._init_data()

        model = self.build_model()
        model = self.model_compile(model)

        model.fit(x=x_train,
                  y=None,
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  shuffle=True)
        model_path = self._save_path()
        model.save_weights(str(model_path))

    def predict(self, model_dir='', fig_name='test.png'):
        model_path = Path(self.model_root, model_dir)
        params_path = list(model_path.glob('*.yml'))[0]
        params = self._load_yml_data(str(params_path))
        self._set_params(**params)
        self._set_vae_params(**params)
        _, x_test, _, _ = self._init_data()
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
    fire.Fire(VAE)
