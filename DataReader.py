import numpy as np
import h5py
import pandas as pd
from tables import *
from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model


data = pd.read_hdf('train.hdf5')
data2 = pd.read_hdf('validation.hdf5')
print(data.shape)
print(data2.shape)


# define a sparse AE
encoding_dim = 55
input_data = Input(shape=(61440,))
# add a Dense layer with a L1 activity regularizer
encoded = Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(input_data)
decoded = Dense(61440, activation='sigmoid')(encoded)
autoencoder = Model(input_data, decoded)

# create an encoder model
encoder = Model(input_data, encoded)

# create a decoder model
encoded_input = Input(shape=(encoding_dim,))  # create a placeholder for an encoded input
decoder_layer = autoencoder.layers[-1]  # retrieve the last layer of the autoencoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# compile autoencoder
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# train autoencoder for 5 epochs
autoencoder.fit(data, data, epochs=5, batch_size=55, validation_data=(data2, data2))


