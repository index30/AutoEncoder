from keras import backend as K
from keras import regularizers
from keras.layers import (Activation, Conv2D, Dense, Flatten)

def encoder(input, dim, latent_dim):
    '''Encoder in VAE
    # Arguments
        input:  the input from previous layer
        dim:    the dimension of encoder
            UnderComplete
                input_dim > dim
            OverComplete
                input_dim < dim
    # Return
        x:  the layer of activation
    '''

    x = Conv2D(32, (3, 3), padding='same')(input)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', strides=(2, 2))(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    shape_conv = K.int_shape(x)
    x = Flatten()(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    z_mean = Dense(latent_dim)(x)
    z_sigma = Dense(latent_dim)(x)
    return z_mean, z_sigma, shape_conv
