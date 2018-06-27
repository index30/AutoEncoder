'''Convolutional Encoder'''
from keras import regularizers
from keras.layers import (Activation, Conv2D, Dense, MaxPooling2D)
from keras.layers.normalization import BatchNormalization

def encoder(input):
    '''Convolutional Encoder
    When you set the dim which over input_dim,
    get the similar result when UnderComplete
    # Arguments
        input:  the input from previous layer
    # Return
        x:  the layer of activation
    '''

    x = Conv2D(16, (3, 3), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(8, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    return x
