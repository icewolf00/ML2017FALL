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
from gensim.models import Word2Vec
from sklearn.utils import shuffle

MAX_LEN = 40

file_train_l = open(sys.argv[1], "r",  encoding = 'utf-8')
temp = []
labels = []
train_l = []
for line in file_train_l:
    line = line.strip("\n")
    temp = line.split(" +++$+++ ")
    labels.append(temp[0])
    temp = temp[1].split(" ")
    train_l.append(temp)

labels = np.array(labels)
train_l = np.array(train_l)
encoded = np.empty((len(train_l), MAX_LEN, 72))
(train_l, labels) = shuffle(train_l, labels)

model = Word2Vec.load("dict_gen")
# print(model.wv['happy'])
for i in range(len(train_l)):
	for j in range(len(train_l[i])):
		if train_l[i][j] in model.wv.vocab:
			encoded[i][j] = model.wv[train_l[i][j]]
		else:
			encoded[i][j] = np.zeros(72)
# print(encoded.shape)
model_train = Sequential()
model_train.add(GRU(512, input_shape = (40, 72), return_sequences=False))
model_train.add(Dropout(0.2))
# model_train.add(Dense(512))
# model_train.add(Dropout(0.2))
# model_train.add(Dense(256))
# model_train.add(Dropout(0.2))
model_train.add(Dense(32))
model_train.add(Dropout(0.2))
model_train.add(Dense(1))
model_train.add(Activation('sigmoid'))
model_train.compile(loss = 'binary_crossentropy', optimizer = "Adam", metrics = ['accuracy'])

callbacks = []
callbacks.append(ModelCheckpoint('model-{epoch:05d}-{val_acc:.5f}-{val_loss:.5f}.h5', monitor='val_acc', save_best_only=True, mode = 'auto', period=1))

model_train.fit(encoded, labels, batch_size = 256, epochs = 10, validation_split=0.2, callbacks=callbacks)

model.save("model.h5")