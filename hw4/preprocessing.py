import pandas as pd
import numpy as np
import re
import sys
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Flatten, GRU, Dropout, TimeDistributed, Activation
# from keras.layers.embeddings import Embedding
# from keras.preprocessing import sequence
# from keras.preprocessing import text
# from keras.utils import to_categorical
# from keras.callbacks import ModelCheckpoint
# from keras.callbacks import EarlyStopping
# from collections import Counter
from gensim.models import Word2Vec


file_train_l = open("training_label.txt", "r",  encoding = 'utf-8')
file_train_n = open("training_nolabel.txt", "r",  encoding = 'utf-8')

temp = []
labels = []
train_l = []
for line in file_train_l:
    line = line.strip("\n")
    temp = line.split(" +++$+++ ")
    labels.append(temp[0])
    temp = temp[1].split(" ")
    train_l.append(temp)


    
for line in file_train_n:
    line = line.strip("\n")
    temp = line.split(" ")
    train_l.append(temp)

labels = np.array(labels)
train_l = np.array(train_l)
# test = np.array(test)

model = Word2Vec(train_l, size=72, window=4, min_count=20, workers=4, negative = 5, batch_words=10000, iter = 10)
model.save("dict_gen")