from keras import regularizers
from keras.layers import (Activation, Dense)

def encoder(input, dim):
    '''Sparse Encoder
    When you set the dim which over input_dim,
    get the similar result when UnderComplete
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
    x = Dense(dim,
              activity_regularizer=regularizers.l1(1e-4))(input)
    x = Activation('relu')(x)
    return x
