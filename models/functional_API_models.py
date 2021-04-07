import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def fun_LSTM_model(train_X, train_y, test_X, test_y, val_X, val_y):
    inputs = tf.keras.Input(shape=(train_X.shape[1], train_X.shape[2]))
    x = layers.LSTM(90, activation='relu', return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(60, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="fun_lstm_model")

    model.compile(
        loss=keras.losses.MSE,
        optimizer=keras.optimizers.Adam()
    )

    history = model.fit(train_X, train_y, batch_size=128, epochs=150, validation_data=(test_X, test_y))

    test_scores = model.evaluate(val_X, val_y, verbose=2)
    #print("Test loss:", test_scores[0])
    #print("Test accuracy:", test_scores[1])

    return model, history, test_scores