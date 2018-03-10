import sys
import numpy as np
import pandas as pd

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
	X_train = (X_train - mean) / std
	return X_train, mean, std

def nortest(X_test, mean, std):
	X_test = (X_test - mean) / std
	return X_test

def evaluate(X_train, Y_train, weight, bias, prob):
	loss = -np.mean((Y_train.flatten().dot(np.log(prob + 1e-15))) + (1-Y_train).flatten().dot(np.log(1-prob + 1e-15)))
	#print((Y_train.flatten().dot(np.log(prob))) + (1-Y_train).flatten().dot(np.log(1-prob)))
	prob = np.around(prob)
	acc = np.mean(Y_train.flatten() == prob)
	#print("loss = ", loss)
	print("acc = ", acc)

def train(X_train, Y_train):
	lr = 0.051
	iteration = 2900
	lamda = 0.0
	length = X_train.shape[0]
	num = X_train.shape[1]
	weight = np.array([0.1] * num)
	bias = 0.1
	w_lr = np.array([0.0] * num)
	b_lr = 0.0
	for i in range(iteration):
		z = X_train.dot(weight) + bias
		prob = sigmoid(z)
		w_grad = -1.0 *((X_train.T).dot((Y_train.flatten() - prob)))/length
		b_grad = -1.0 *((Y_train.flatten() - prob).sum())/length
		w_lr = w_lr + w_grad **2
		b_lr = b_lr + b_grad **2
		weight = weight - lr / np.sqrt(w_lr) * w_grad
		bias = bias - lr / np.sqrt(b_lr) * b_grad
		if i % 100 == 0:
			pass
			#evaluate(X_train, Y_train, weight, bias, prob)
			#print(prob)
	return weight, bias

def test(filename, X_test, weight, bias):
	file_out = open(filename, "w")
	file_out.write("id,label\n")
	z = X_test.dot(weight) + bias
	prob = sigmoid(z)
	for i in range(len(prob)):
		if abs(prob[i] - 0.5) < 1e-6:
			prob[i] = 1
	prob = np.around(prob)
	for i in range(len(prob)):
		outstr = str(i+1) + ',' + str(int(prob[i])) + '\n'
		file_out.write(outstr)

def mix(X_data):
	Copy_data = X_data
	#no = np.array([66, 67, 68, 80, 81, 84, 86, 101, 102, 104, 105])
	#no = np.array([66, 67, 68, 80, 81, 84, 86, 101, 102, 104, 105])
	no = np.array([68, 84, 86, 102, 105])
	full = np.array(range(X_data.shape[1]))
	full = np.setdiff1d(full, no)
	take = np.array([0, 1, 3, 4, 5])
	#X_data = np.concatenate((X_data[:, full], X_data[:, take] ** 0.5, X_data[:, take] ** 1.5, X_data[:, take] ** 1.5, X_data[:, [0, 1, 3, 5]] ** 2, X_data[:, take] ** 4), axis = 1)
	X_data = np.concatenate((X_data[:, full], X_data[:, take] ** 0.5, X_data[:, take] ** 1.5, X_data[:, take] ** 1.5, X_data[:, [0, 1, 3, 5]] ** 2, X_data[:, take] ** 4), axis = 1)
	#X_data = np.concatenate((X_data[:, full], X_data[:, full] ** 0.5, X_data[:, two] ** 2), axis = 1)
	return X_data

X_train = readX(sys.argv[1])
Y_train = readY(sys.argv[2])
X_test = readX(sys.argv[3])

X_train = mix(X_train)
X_test = mix(X_test)
#print(X_train.shape)
(X_train, mean, std) = nortrain(X_train)
X_test = nortest(X_test, mean, std)
(weight, bias) = train(X_train, Y_train)
test(sys.argv[4], X_test, weight, bias)