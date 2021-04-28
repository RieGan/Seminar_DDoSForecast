import numpy as np
import tensorflow as tf
from helper import readResult

np.random.seed(1337)

EMBED_SIZE = 64
HIDDEN_SIZE = 32
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 12

nb_filter = 10
filter_length = 5


def cnn_zhang(X_train, y_train, X_test, y_test, vocab_size):
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=128)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=128)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Build model...')
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, EMBED_SIZE, input_length=128))
    model.add(tf.keras.layers.SpatialDropout1D(0.2))
    model.add(tf.keras.layers.Convolution1D(filters=nb_filter,
                                            kernel_size=filter_length,
                                            padding='valid',
                                            activation='relu'))

    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(HIDDEN_SIZE))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Activation(activation="relu"))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    model.fit(X_train, y_train, batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(X_test, y_test), verbose=0)

    X_pred = model.predict(X_test)
    results = [result[0] for result in X_pred]

    return readResult(y_test, results, name="CNN", form="JSON")


def cnn_improved(X_train, y_train, X_test, y_test, vocab_size):
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=MAX_LEN)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=MAX_LEN)

    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Build model...')
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size, EMBED_SIZE, input_length=MAX_LEN))
    print("#1:", model.output_shape)

    model.add(tf.keras.layers.SpatialDropout1D(0.2))
    model.add(tf.keras.layers.Convolution1D(filters=50,
                                            kernel_size=6, activation='relu'))
    print("#2:", model.output_shape)
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    print("#3:", model.output_shape)

    model.add(tf.keras.layers.Convolution1D(filters=50,
                                            kernel_size=5, activation='relu'))
    print("#4:", model.output_shape)
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    print("#5:", model.output_shape)

    model.add(tf.keras.layers.Convolution1D(filters=50,
                                            kernel_size=4, activation='relu'))
    print("#6:", model.output_shape)
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    print("#7:", model.output_shape)

    model.add(tf.keras.layers.Convolution1D(filters=50,
                                            kernel_size=4, activation='relu'))
    print("#8:", model.output_shape)
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    print("#9:", model.output_shape)
    model.add(tf.keras.layers.Convolution1D(filters=50,
                                            kernel_size=4, activation='relu'))
    print("#10:", model.output_shape)
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    print("#11:", model.output_shape)
    model.add(tf.keras.layers.Flatten())
    print("#12:", model.output_shape)
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    print("#13:", model.output_shape)
    model.add(tf.keras.layers.Dense(8, activation="relu"))
    print("#14:", model.output_shape)
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    model.fit(X_train, y_train, batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(X_test, y_test), verbose=0)

    X_pred = model.predict(X_test)
    results = [result[0] for result in X_pred]

    return readResult(y_test, results, name="CNN+", form="JSON")
