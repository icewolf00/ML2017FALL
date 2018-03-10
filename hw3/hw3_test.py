import numpy as np
import sys
from keras.models import load_model

#python hw3_test_2.py test.csv ans.csv normal.npy models_2\model-0.61940.h5 models_2\model-0.62060.h5 models_2\model-0.62180.h5 models_2\model-0.62480.h5
def readata(filename):
	filein = open(filename, 'r')
	data = []
	for line in filein:
		data.append(line.strip("\r\n").replace(',', ' ').split())
	data = np.array(data[1:])
	#data = data.astype('float32')
	#print(Y_train[0])
	X_test = data[:, 1:].astype('float')
	#print(X_test[0])
	return X_test

X_test = readata(sys.argv[1])
X_test = X_test.reshape((-1, 48, 48, 1))
X_test = X_test / 255
# print(sys.argv[3])
# model = load_model(sys.argv[3])
# model2 = load_model(sys.argv[4])
# model3 = load_model(sys.argv[5])
model4 = load_model(sys.argv[6])
pred = 0.0
# pred += model.predict(X_test)
# pred += model2.predict(X_test)
# pred += model3.predict(X_test)
pred += model4.predict(X_test)
pred = np.argmax(pred, axis = 1)
fileout = open(sys.argv[2], "w")
print("id,label", file = fileout)
for i in range(pred.size):
	print(str(i) + "," + str(pred[i]), file = fileout)
fileout.close()