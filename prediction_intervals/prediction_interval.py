import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

def prediction_interval(model, X_train, y_train, x0, alpha: float = 0.05):
  ''' Compute a prediction interval around the model's prediction of x0.

  INPUT
    model
      A predictive model with `fit` and `predict` methods
    X_train: numpy array of shape (n_samples, n_features)
      A numpy array containing the training input data
    y_train: numpy array of shape (n_samples,)
      A numpy array containing the training target data
    x0
      A new data point, of shape (n_features,)
    alpha: float = 0.05
      The prediction uncertainty

  OUTPUT
    A triple (`lower`, `pred`, `upper`) with `pred` being the prediction
    of the model and `lower` and `upper` constituting the lower- and upper
    bounds for the prediction interval around `pred`, respectively. '''

  # Number of training samples
  n = X_train.shape[0]

  # The authors choose the number of bootstrap samples as the square root
  # of the number of samples
  nbootstraps = np.sqrt(n).astype(int)

  # Compute the m_i's and the validation residuals
  bootstrap_preds, val_residuals = np.empty(nbootstraps), []
  for b in range(nbootstraps):
    train_idxs = np.random.choice(range(n), size = n, replace = True)
    val_idxs = np.array([idx for idx in range(n) if idx not in train_idxs])
    model.fit(X_train[train_idxs, :], y_train[train_idxs])
    preds = model.predict(X_train[val_idxs])
    val_residuals.append(y_train[val_idxs].reshape(y_train[val_idxs].shape[0],1) - preds)
    bootstrap_preds[b] = model.predict(x0)
  bootstrap_preds -= np.mean(bootstrap_preds)
  val_residuals = np.concatenate(val_residuals)

  # Compute the prediction and the training residuals
  model.fit(X_train, y_train)
  preds = model.predict(X_train)
  train_residuals = y_train - preds

  # Take percentiles of the training- and validation residuals to enable
  # comparisons between them
  val_residuals = np.percentile(val_residuals, q = np.arange(100))
  train_residuals = np.percentile(train_residuals, q = np.arange(100))

  # Compute the .632+ bootstrap estimate for the sample noise and bias
  no_information_error = np.mean(np.abs(np.random.permutation(y_train) - \
    np.random.permutation(preds)))
  generalisation = np.abs(val_residuals - train_residuals)
  no_information_val = np.abs(no_information_error - train_residuals)
  relative_overfitting_rate = np.mean(generalisation / no_information_val)
  weight = .632 / (1 - .368 * relative_overfitting_rate)
  residuals = (1 - weight) * train_residuals + weight * val_residuals

  # Construct the C set and get the percentiles
  C = np.array([m + o for m in bootstrap_preds for o in residuals])
  qs = [100 * alpha / 2, 100 * (1 - alpha / 2)]
  percentiles = np.percentile(C, q = qs)

  print(percentiles[0], model.predict(x0), percentiles[1])

  return percentiles[0], model.predict(x0), percentiles[1]


#here I am putting again the function that I have already developed in Data_transformation.py
#in order to be able to run the script. Once the prediction interval function be used in the main.py
#we no longer keep here the "reshape_data_single_lag" function
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
	#print(train_X.shape)

	### this reshape below is we using it for univariate timeseries
	# reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
	test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
	val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))

	#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape, val_X.shape, val_y.shape)

	return train_X, train_y, test_X, test_y, val_X, val_y #TODO put here the object we need to be returned


#load the model
path_to_model = 'C:\\Users\\geo_m\\PycharmProjects\\Morphemic_TimeSeries\\models_saved\\lstm.h5'
model = load_model(path_to_model)

#save data into a csv for testing the confidence interval
import pandas as pd
data = pd.read_csv('C:\\Users\\geo_m\\PycharmProjects\\Morphemic_TimeSeries\\datasets\\data.csv')

data = data.drop(columns=['Unnamed: 0'])
X_train, y_train, test_X, test_y, X_val, val_y = reshape_data_single_lag(data,  0.6, 0.2, 0.2 )

results = []
i=0
while i < 2:
    print(i)
    x_Zero = X_val[i].reshape(1,1,122)#.shape#.reshape(1,1,123)
    print("x zero is now:", x_Zero)
    #x_Zero.reshape(-1,1).shape
    [lower_bound, prediction, upper_bound] = prediction_interval(model, X_train, y_train, x_Zero, alpha=0.05)
    results.append([lower_bound, prediction, upper_bound])
    i+=1

for i in range(len(results)):
    print("lower_bound:",results[i][0],"prediction:", results[i][1], "upper_bound:", results[i][2])