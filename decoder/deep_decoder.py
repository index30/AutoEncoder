'''Deep Decoder'''
from keras.layers import (Activation, Dense)

def decoder(input, input_dim):
    '''Deep Decoder
    # Arguments
        input:  the input from previous layer
        input_dim:  the dimension of input
                    and the last layer
    # Return
        x:  the layer of activation
    '''
    x = Dense(64)(input)
    x = Activation('relu')(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dense(input_dim)(x)
    x = Activation('sigmoid')(x)
    return x
