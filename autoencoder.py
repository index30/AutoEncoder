"""AutoEncoder"""
from datetime import datetime
import fire
import importlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import yaml

from keras import backend as K
from keras.datasets import mnist
from keras.layers import (Input)
from keras.models import Model

class AutoEncoder(object):
    param_names = ['batch_size', 'decode_clf', 'encode_clf',
                   'encode_dim', 'epochs', 'image_size', 'loss',
                   'model_name', 'model_root', 'optimizer']

    def __init__(self):
        self.batch_size = 256
        self.encode_clf = 'simple'
        self.decode_clf = 'simple'
        self.encode_dim = 32
        self.epochs = 50
        self.image_size = (784,)
        self.loss = 'binary_crossentropy'
        self.model_name = 'model.h5'
        self.model_root = 'model'
        self.optimizer = 'adam'

    ### set parameter
    def _set_params(self, **parameter):
        for param, value in parameter.items():
            if param in self.param_names:
                setattr(self, param, value)

    ### fetch parameter
    def _fetch_params(self):
        return {attr_name: getattr(self, attr_name)
                for attr_name in self.param_names}

    def _init_path(self):
        '''import encoder/decoder from directory
        # Returns
            encoder_path:   the path of encoder you want to use
            decoder_path:               decoder
        '''
        encoder_path = 'encoder.' + self.encode_clf + '_encoder'
        decoder_path = 'decoder.' + self.decode_clf + '_decoder'
        importlib.import_module(encoder_path)
        importlib.import_module(decoder_path)
        return encoder_path, decoder_path

    def _load_yml_data(self, param_file):
        '''load the data from the file with yaml
        # Arguments
            param_file: the file of parameter in train
        '''
        with open(param_file) as f:
            return yaml.load(f)

    def _write_to_yaml(self, data, yml_path="params.yml"):
        '''Write data to yaml
        # Arguments
            data:       the data which save to yml(type: dir)
            yml_path:   save name which has extension '.yml'
        '''
        with open(str(yml_path), 'w') as f:
            f.write(yaml.dump(data))

    def _init_data(self):
        '''initialize data from mnist
        # Returns
            x_train:    the data which uses for training
            x_test:                             test
        '''
        (x_train, _), (x_test, _) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.astype('float32') / 255.
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        return x_train, x_test

    def _save_path(self):
        '''save the model and return model_path
        # Returns
            model_path: the path of model saved
        '''
        model_path = Path(self.model_root)
        if not model_path.exists():
            model_path.mkdir()
        now = datetime.now().strftime("%m%d%H%M%S")
        dir_name = self.encode_clf + "_" + self.decode_clf +"_"+ now
        eval_path = Path(model_path, dir_name)
        eval_path.mkdir()
        ## save parameter as yaml
        param_path = Path(eval_path, "params.yml")
        param_data = self._fetch_params()
        self._write_to_yaml(param_data, param_path)
        ## make callback
        model_path = Path(eval_path, "model.h5")
        return model_path

    def build_model(self):
        '''building model for AutoEncoder
        # Returns
            model:  the model of AutoEncoder
        '''
        input = Input(self.image_size)
        input_dim = self.image_size[0]

        encoder_path, decoder_path = self._init_path()

        ### call encoder and decoder
        x = getattr(sys.modules[encoder_path], 'encoder')(input, self.encode_dim)
        x = getattr(sys.modules[decoder_path], 'decoder')(x, input_dim)
        model = Model(input=input, output=x)

        return model

    def model_compile(self, model):
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def train(self, params_path='params.yml'):
        ### load yaml
        params = self._load_yml_data(params_path)
        self._set_params(**params)

        ### init data
        x_train, x_test = self._init_data()

        ### build model
        model = self.build_model()
        model = self.model_compile(model)

        ### train model
        model.fit(x_train,
                  x_train,
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  shuffle=True,
                  validation_data=(x_test, x_test))

        ### save model
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


if __name__=="__main__":
    fire.Fire(AutoEncoder)
