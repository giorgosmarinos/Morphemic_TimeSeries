from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, MaxPooling3D, Flatten
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import Conv1D, Conv3D


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
    model.fit(train_X, train_y, epochs=150, batch_size=128, validation_data=(test_X, test_y),
              verbose=2, shuffle=False)
    return model
