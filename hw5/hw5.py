import sys
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import load_model
from keras.engine.topology import Layer

def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))

def read_data(filename, user2id, movie2id):
    test_data = pd.read_csv(filename)

    test_data['UserID'] = test_data['UserID'].apply(lambda x: user2id[x])
    test_data['MovieID'] = test_data['MovieID'].apply(lambda x: movie2id[x])

    return test_data[['UserID', 'MovieID']].values

def out_file(filename, pred):
    file_out = open(filename, "w")
    print("TestDataID,Rating", file = file_out)
    for i in range(len(pred)):
        print(str(id[i]) + "," + str(pred[i]), file = file_out)
    file_out.close()

user2id = np.load(sys.argv[3])[()]
movie2id = np.load(sys.argv[4])[()]
X_test = read_data(sys.argv[1], user2id, movie2id)

model = load_model(sys.argv[5], custom_objects={'rmse': rmse})
pred = model.predict([X_test[:, 0], X_test[:, 1]]).squeeze()
# pred = pred.clip(1.0, 5.0)

y = np.load(sys.argv[6])
y_mean = y[0]
y_std = y[1]
pred = pred * y_std + y_mean
pred = pred.clip(1.0, 5.0)

file_out = open(sys.argv[2], "w")
print("TestDataID,Rating", file = file_out)
for i in range(len(pred)):
    print(str(i+1) + "," + str(pred[i]), file = file_out)
file_out.close()