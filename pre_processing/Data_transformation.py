import pandas
import pandas as pd
import numpy as np

# convert series to supervised learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	if isinstance(data, list):
		n_vars = 1
	else:
		n_vars = data.shape[1]
	if isinstance(data, pd.DataFrame):
		pass
	else:
		data = pd.DataFrame(data)
	cols, names = list(), list()
	print(n_vars)
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(data.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(data.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	cols_to_use = names[:len(names) - (n_out+1)]  # drop the last ones #TODO it must be checked that (n_out + 1) removes always the correct collumns
	agg = agg[cols_to_use]
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

#resample dataset in different time duration
def resample(data, time, way):
	if way == 'mean':
		y = data.resample(time).mean()
	elif way == 'meadian':
		y = data.resample(time).median()
	return y



def reshape_data_single_lag(reframed, train_percentage, test_percentage, valid_percentage):
	# split into train and test sets
	values = reframed.values
	# Sizes
	train_size = int(len(reframed) * train_percentage)
	test_size = int(len(reframed) * test_percentage)
	valid_size = int(len(reframed) * valid_percentage)

	train = values[:train_size]
	test = values[train_size:train_size + test_size]
	val = values[train_size + test_size:]

	# split into input and outputs
	train_X, train_y = train[:, :-1], train[:, -1]
	test_X, test_y = test[:, :-1], test[:, -1]
	val_X, val_y = val[:, :-1], val[:, -1]
	print(train_X.shape)

	### this reshape below is we using it for univariate timeseries
	# reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
	test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
	val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))

	print(train_X.shape, train_y.shape, test_X.shape, test_y.shape, val_X.shape, val_y.shape)

	return train_X, train_y, test_X, test_y, val_X, val_y #TODO put here the object we need to be returned


def reshape_data_multiple_lag(reframed, train_percentage, test_percentage, valid_percentage, n_steps, n_features):
	# split into train and test sets
	values = reframed.values
	# Sizes
	train_size = int(len(reframed) * train_percentage)
	test_size = int(len(reframed) * test_percentage)
	valid_size = int(len(reframed) * valid_percentage)

	train = values[:train_size]
	test = values[train_size:train_size + test_size]
	val = values[train_size + test_size:]
	# split into input and outputs

	n_obs = n_steps * n_features

	train_X, train_y = train[:, :n_obs], train[:, -1]
	test_X, test_y = test[:, :n_obs], test[:, -1]
	val_X, val_y = val[:, :n_obs], val[:, -1]

	print(train_X.shape, len(train_X), train_y.shape)

	# reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], n_steps, n_features))
	test_X = test_X.reshape((test_X.shape[0], n_steps, n_features))
	val_X = val_X.reshape((val_X.shape[0], n_steps, n_features))

	print(train_X.shape, train_y.shape, test_X.shape, test_y.shape, val_X.shape, val_y.shape)

	return  # TODO put here the object we need to be returned


def put_as_first_column(data,column_of_interest):
	data = data.pop(column_of_interest)
	return data

def Min_max_scal_inverse(scaler, data):
	return scaler.inverse_transform(data)

def Min_max_scal(data):
	array = data.values
	values_ = array.astype('float32')
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled = scaler.fit_transform(values_)
	return scaled, scaler


def predictions_and_scores(model, test_X,test_y):
	# make a prediction
	yhat = model.predict(test_X)
	# test_X_reshaped = test_X.reshape((test_X.shape[0], 3*2))
	yhat_reshaped = yhat.reshape((yhat.shape[0], yhat.shape[1]))

	test_y_reshaped = test_y.reshape((len(test_y), 1))

	# calculate RMSE and R2_score
	rmse = sqrt(mean_squared_error(test_y_reshaped, yhat_reshaped))
	r2score = r2_score(test_y_reshaped, yhat_reshaped)
	print('Test RMSE: %.3f' % rmse)
	print('R2_score: %.3f' % r2score)

# multivariate data preparation
from numpy import array
from numpy import hstack


#This one for CNN
# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.33, random_state=42)
	return array(X_train), array(y_train), array(X_test), array(y_test)

#multivariate, multisteps
def split_sequences_multi_steps(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.33, random_state=42)
	return array(X_train), array(y_train), array(X_test), array(y_test)

def prediction_and_score_for_CNN(n_steps,n_features, x_input, model,test_y):
	#x_input = x_input.reshape((1, n_steps, n_features))
	yhat = model.predict(x_input, verbose=2)
	# calculate RMSE and R2_score
	rmse = sqrt(mean_squared_error(test_y, yhat))
	r2score = r2_score(test_y, yhat)
	print('Test RMSE: %.3f' % rmse)
	print('R2_score: %.3f' % r2score)
	#print(yhat)