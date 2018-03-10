import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

def readX(filename):
	data = pd.read_csv(filename)
	#print(data)
	return data.as_matrix().astype('float')

def readY(filename):
	data = pd.read_csv(filename)
	#print(data) 32560
	return data.as_matrix().astype('int')

def sigmoid(z):
	out = 1 / (1.0 + np.exp(-z))
	return out

def nortrain(X_train):
	mean = np.mean(X_train, axis = 0)
	std = np.std(X_train, axis = 0)
	X_train = (X_train - mean) / (std + 1e-15)
	return X_train, mean, std

def nortest(X_test, mean, std):
	X_test = (X_test - mean) / (std + 1e-15)
	return X_test

def mix(X_data):
	Copy_data = X_data
	no = np.array([68, 84, 86, 102, 105])
	full = np.array(range(X_data.shape[1]))
	full = np.setdiff1d(full, no)
	take = np.array([0, 1, 3, 4, 5])
	bias = np.array([0.1] * X_data.shape[0]).reshape(-1, 1)
	X_data = np.concatenate((X_data[:, full], X_data[:, take] ** 0.5, X_data[:, take] ** 1.5, X_data[:, take] ** 1.5, X_data[:, [0, 1, 3, 5]] ** 2, X_data[:, take] ** 4, bias), axis = 1)
	return X_data

def trainandtest(X_train, Y_train, X_test, filename):
	model = XGBClassifier(max_depth=3, n_estimators=1500, learning_rate=0.05)
	model.fit(X_train, Y_train.flatten())
	predict_train = model.predict(X_train)
	acc_train = np.mean(Y_train.flatten() == predict_train)
	print("acc_train = ", acc_train)
	predict_test = model.predict(X_test)
	file_out = open(filename, "w")
	file_out.write("id,label\n")
	for i in range(len(predict_test)):
		outstr = str(i+1) + ',' + str(int(predict_test[i])) + '\n'
		file_out.write(outstr)

X_train = readX(sys.argv[1])
Y_train = readY(sys.argv[2])
X_test = readX(sys.argv[3])
X_train = mix(X_train)
X_test = mix(X_test)
(X_train, mean, std) = nortrain(X_train)
X_test = nortest(X_test, mean, std)
trainandtest(X_train, Y_train, X_test, sys.argv[4])