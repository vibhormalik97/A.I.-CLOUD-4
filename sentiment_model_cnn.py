import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.optimizers import Adam

def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling
    """
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open(config["embeddings_path"], encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.array(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    vocab_size = config["embeddings_dictionary_size"]

    n = len(embeddings_index.keys())
    m = len(embeddings_index['the'])

    embedding_matrix = np.zeros((n,m))
    for index, key in zip(range(0, n), embeddings_index.keys()):
        embedding_matrix[index] = embeddings_index[key]

    # define model
    model = Sequential()

    # Layer 1: Embedding layer
    # This layer should load the embeddings vectors from your dictionary as a numpy array
    # - input_leght should be equal to your padding length
    # - input_dim should be the length of your word list
    # - output_dim should be the size your your embedding vectors
    # - trainable True
    model.add(Embedding(vocab_size, 25, weights=[embedding_matrix], input_length=100, trainable=True, name='embedding'))

    # Layer 2: Convolution1D layer
    # - filters 100
    # - kernel_size 2
    # - strides 1
    # - padding 'valid'
    # - activation 'relu'
    model.add(Conv1D(filters = 100, kernel_size = 2, strides = 1, padding = 'valid', activation = 'relu'))

    # Layer 3: GLobalMaxPool1D layer
    model.add(GlobalMaxPool1D())

    # Layer 4: Dense layer
    # - units 100
    # - activation 'relu'
    model.add(Dense(100, activation = 'relu'))

    # Layer 5: Dense layer
    # - units 1
    # - activation 'sigmoid'
    model.add(Dense(1, activation = 'sigmoid'))

    adam = Adam(lr=0.001)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    cnn_model = model

    print('Defined model')
    return cnn_model

def save_model(model, output):
    """
    Method to save models in SaveModel format with signature to allow for serving
    """

    model.save(output)

    print("Model successfully saved at: {}".format(output))