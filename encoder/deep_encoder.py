from keras import regularizers
from keras.layers import (Activation, Dense)

def encoder(input, dim):
    '''Deep Encoder
    When you set the dim which over input_dim,
    get the similar result when UnderComplete
    # Arguments
        input:  the input from previous layer
        dim:    the dimension of encoder at last layer in encoder
    # Return
        x:  the layer of activation
    '''

    x = Dense(128)(input)
    x = Activation('relu')(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dense(dim)(x)
    x = Activation('relu')(x)
    return x
