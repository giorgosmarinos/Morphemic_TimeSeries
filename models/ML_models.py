from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, MaxPooling3D, Flatten
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Conv1D, Conv3D
from tensorflow.python.keras.layers import MaxPooling1D


def LSTM_model(train_X, train_y, test_X, test_y):
    # design network
    model = Sequential()
    model.add(LSTM(90, return_sequences = True,  input_shape=(train_X.shape[1], train_X.shape[2])))  # 1 , 2
    model.add(Dropout(0.2))
    model.add(LSTM(60, return_sequences = False ))
    model.add(Dropout(0.2))
    #model.add(LSTM(30, return_sequences = False ))
    #model.add(Dropout(0.2))
    #model.add(LSTM(15, return_sequences = False ))
    #model.add(Dense(50))
    #model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_X, train_y, epochs=150, batch_size=128, validation_data=(test_X, test_y),verbose=2, shuffle=False)
    return model


def CNN_model(n_steps, n_features, X, y, val_x, val_y):
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=100, verbose=2, validation_data=(val_x, val_y))
    return model

def CNN_model_multi_steps(n_steps, n_features, X, y, val_x, val_y, n_steps_out):
    # define model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=200, verbose=1, validation_data=(val_x, val_y))
    return model