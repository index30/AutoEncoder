from keras.layers import (Activation, Dense)

def encoder(input, dim):
    '''Simple Encoder
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
    x = Dense(dim)(input)
    x = Activation('relu')(x)
    return x
