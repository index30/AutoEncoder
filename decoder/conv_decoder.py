'''Convolutional Decoder'''
from keras.layers import (Activation, Conv2D, Conv2DTranspose, Dense)
from keras.layers.normalization import BatchNormalization

def decoder(input):
    '''Convolutional Decoder
    # Arguments
        input:  the input from previous layer
    # Return
        x:  the layer of activation
    '''
    x = Conv2DTranspose(8, (3, 3), strides=(2, 2))(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1, (3, 3), padding='same')(x)
    x = Activation('sigmoid')(x)

    return x
