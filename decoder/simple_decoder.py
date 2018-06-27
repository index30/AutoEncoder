'''Simple Decoder'''
from keras.layers import (Activation, Dense)

def decoder(input, input_dim):
    '''Simple Decoder
    # Arguments
        input:  the input from previous layer
        input_dim:  the dimension of input
    # Return
        x:  the layer of activation
    '''
    x = Dense(input_dim)(input)
    x = Activation('sigmoid')(x)
    return x
