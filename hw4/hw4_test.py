import pandas as pd
import numpy as np
import re
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, GRU, Dropout, TimeDistributed, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing import text
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from collections import Counter
from keras.models import load_model
from gensim.models import Word2Vec
import time
localtime = time.asctime( time.localtime(time.time()))
print(localtime)
MAX_LEN = 40
# dict_train_l = np.load("dict.npy").item()f

file_test = open(sys.argv[1], "r",  encoding = 'utf-8')
test = []
line = file_test.readline()
for line in file_test:
    line = line.strip("\n")
    temp = line.split(",", 1)
    temp = temp[1].split(" ")
    test.append(temp)
test = np.array(test)

encoded = np.empty((len(test), MAX_LEN, 72))
model_gen = Word2Vec.load("dict_gen")
# print(model.wv['happy'])
for i in range(len(test)):
    for j in range(len(test[i])):
        if test[i][j] in model_gen.wv.vocab:
            encoded[i][j] = model_gen.wv[test[i][j]]
        else:
            encoded[i][j] = np.zeros(72)

model = load_model(sys.argv[3])
model2 = load_model(sys.argv[4])
model3 = load_model(sys.argv[5])


result = 0.0
result += model.predict(encoded, batch_size = 512)
# print(result[0])

result += model2.predict(encoded, batch_size = 512)
# print(result)
result += model3.predict(encoded, batch_size = 512)

result = result / 3
result = result.flatten()
result = np.around(result)

# print(result)

file_out = open(sys.argv[2], "w")
file_out.write("id,label\n")
for i in range(len(result)):
	outstr = str(i) + ',' + str(int(result[i])) + '\n'
	file_out.write(outstr)

localtime = time.asctime( time.localtime(time.time()))
print(localtime)