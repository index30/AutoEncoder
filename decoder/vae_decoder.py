'''Decoder in VAE'''
import numpy as np
from keras import backend as K
from keras.layers import (Activation, Conv2D, Conv2DTranspose,
                          Dense, Lambda, Layer, Input, Reshape)
from keras.models import Model

def _sampling(args):
    epsilon_std = 1.0
    z_mean, z_sigma = args
    epsilon = K.random_normal(shape=(LATENT_DIM,),
                              mean=0.,
                              stddev=epsilon_std)
    return z_mean + z_sigma * epsilon

def decoder(input, latent_dim, func_sampling, shape_conv):
    '''Decoder in VAE
    # Arguments
        input:  the input from previous layer
        latent_dim:  the dimension of space of latent
        func_sampling:  the function
    # Return
        x:  the layer of activation
    '''
    z_mean = input[0]
    z_sigma = input[1]
    z = Lambda(func_sampling,
               output_shape=(latent_dim, ))([z_mean, z_sigma])
    dec_input = Input(K.int_shape(z)[1:])
    x = Dense(np.prod(shape_conv[1:]), activation='relu')(dec_input)
    x = Activation('relu')(x)
    x = Reshape(shape_conv[1:])(x)
    x = Conv2DTranspose(32, (3, 3), padding='same', strides=(2, 2))(x)
    x = Activation('relu')(x)
    x = Conv2D(1, (3, 3), padding='same')(x)
    x = Activation('sigmoid')(x)
    decoder = Model(inputs=dec_input, outputs=x)
    z_decoded = decoder(z)

    return z_decoded, decoder
