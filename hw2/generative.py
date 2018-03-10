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

def nortrain(X_train):
	mean = np.mean(X_train, axis = 0)
	std = np.std(X_train, axis = 0)
	X_train = (X_train - mean) / std
	return X_train, mean, std

def nortest(X_test, mean, std):
	X_test = (X_test - mean) / std
	return X_test

def sigmoid(z):
	out = 1 / (1.0 + np.exp(-z))
	return out

def evaluate(X_data, mulist, share_cov, inv_cov, numlist):
	# print(mulist[0].shape)
	# print(inv_cov.shape)
	Wt = (mulist[0] - mulist[1]).dot(inv_cov)
	b = (-0.5) * (mulist[0].dot(inv_cov)).dot(mulist[0]) + 0.5 * (mulist[1].dot(inv_cov)).dot(mulist[1]) + np.log(numlist[0] / numlist[1])
	z = Wt.dot(X_data.T) + b
	prob = sigmoid(z)
	return prob

def train(X_train, Y_train):
	share_cov = 0.0
	mulist = []
	numlist = []
	for i in range(2):
		choose = X_train[(Y_train == i).flatten()]
		mue = np.mean(choose, axis = 0)
		mulist.append(mue)
		num = choose.shape[0]
		numlist.append(num)
		#covariance = np.mean([(choose[i] - mue).reshape((-1,1))*(choose[i] - mue).reshape((1,-1)) for i in range(choose.shape[0])], axis = 0)
		full = np.zeros((106, 106))
		for j in range(choose.shape[0]):
			full += (choose[j] - mue).reshape((-1,1))*(choose[j] - mue).reshape((1,-1))
		covariance = full /choose.shape[0]
		share_cov += (choose.shape[0] / X_train.shape[0]) * covariance
	inv_cov = np.linalg.inv(share_cov)
	prob = evaluate(X_train, mulist, share_cov, inv_cov, numlist)
	prob = np.around(1-prob)
	acc = np.mean(Y_train.flatten() == prob)
	# print('acc = ', acc)
	return mulist, share_cov, inv_cov, numlist

def writeout(filename, prob):
	file_out = open(filename, "w")
	file_out.write("id,label\n")
	for i in range(len(prob)):
		outstr = str(i+1) + ',' + str(int(prob[i])) + '\n'
		file_out.write(outstr)

X_train = readX(sys.argv[1])
Y_train = readY(sys.argv[2])
X_test = readX(sys.argv[3])

(X_train, mean, std) = nortrain(X_train)
X_test = nortest(X_test, mean, std)

(mulist, share_cov, inv_cov, numlist) = train(X_train, Y_train)
testprob = evaluate(X_test, mulist, share_cov, inv_cov, numlist)
testprob = np.around(1-testprob)
writeout(sys.argv[4], testprob)