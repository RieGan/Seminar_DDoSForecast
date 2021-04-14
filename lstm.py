import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from helper import readResult
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import backend as K

np.random.seed(1337)

EMBED_SIZE = 64
HIDDEN_SIZE = 64
BATCH_SIZE = 10
EPOCHS = 12


def lstm_zhang(X_train, y_train, X_test, y_test, vocab_size):
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=128)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=128)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Build model...')
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, EMBED_SIZE, input_length=128))
    model.add(tf.keras.layers.SpatialDropout1D(0.2))

    model.add(tf.keras.layers.LSTM(HIDDEN_SIZE))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    # model.compile(optimizer='adam',
    #               loss='binary_crossentropy')

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop')
    model.fit(X_train, y_train, batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(X_test, y_test), verbose=0)

    X_pred = model.predict(X_test)
    results = [result[0] for result in X_pred]

    return readResult(y_test, results)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))

def lstm_improved(X_train, y_train, X_test, y_test, vocab_size):
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=128)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=128)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Build model...')
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, EMBED_SIZE, input_length=128))
    model.add(tf.keras.layers.SpatialDropout1D(0.2))

    peephole_lstm_cells = tfa.rnn.PeepholeLSTMCell(HIDDEN_SIZE)
    model.add(tf.keras.layers.RNN(peephole_lstm_cells))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(loss=root_mean_squared_error,
                  optimizer='rmsprop')
    # model.compile(loss=tf.keras.metrics.RootMeanSquaredError,
    #               optimizer='rmsprop')
    model.fit(X_train, y_train, batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(X_test, y_test), verbose=0)

    X_pred = model.predict(X_test)
    results = [result[0] for result in X_pred]

    return readResult(y_test, results)



