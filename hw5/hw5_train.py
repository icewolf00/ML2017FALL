import sys
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import Model
from keras.models import Sequential
from keras.layers import add
from keras.layers import Dot
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Embedding
from keras.regularizers import l2
from keras.initializers import Zeros
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.engine.topology import Layer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
dim = 15

def rmse(y_true, y_pred):
    # y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))*Y_std

def read_data(trainfile, testfile):
    train_file = pd.read_csv(trainfile)
    test_file = pd.read_csv(testfile)

    train_file['test'] = 0
    test_file['test'] = 1

    total = pd.concat([train_file, test_file])

    id2user = total['UserID'].unique()
    id2movie = total['MovieID'].unique()

    user2id = {i: id for id, i in enumerate(id2user)}
    movie2id = {i: id for id, i in enumerate(id2movie)}

    total['UserID'] = total['UserID'].apply(lambda x: user2id[x])
    total['MovieID'] = total['MovieID'].apply(lambda x: movie2id[x])

    train_file = total.loc[total['test'] == 0]

    return train_file[['UserID', 'MovieID']].values, train_file['Rating'].values, total[['UserID', 'MovieID']].values, user2id, movie2id

X_train, Y_train, X, user2id, movie2id = read_data(sys.argv[1], sys.argv[2])
num_users, num_movies = len(user2id), len(movie2id)

np.save('user2id', user2id)
np.save('movie2id', movie2id)

np.random.seed(5)
indices = np.random.permutation(len(X_train))
X_train, Y_train = X_train[indices], Y_train[indices]

Y_mean = np.mean(Y_train, axis = 0)
Y_std = np.std(Y_train, axis = 0)
np.save("normal.npy", [Y_mean, Y_std])
Y_train = (Y_train - Y_mean) / Y_std

u_input = Input(shape=(1,))

U = Embedding(num_users, dim, embeddings_regularizer=l2(0.00001))(u_input)
U = Reshape((dim,))(U)
# U = BatchNormalization()(U)
U = Dropout(0.1)(U)


m_input = Input(shape=(1,))
M = Embedding(num_movies, dim, embeddings_regularizer=l2(0.00001))(m_input)
M = Reshape((dim,))(M)
# M = BatchNormalization()(M)
M = Dropout(0.1)(M)

pred = Dot(axes=-1)([U, M])
U_bias = Reshape((1,))(Embedding(num_movies, 1, embeddings_regularizer=l2(0.00001))(u_input))
M_bias = Reshape((1,))(Embedding(num_users, 1, embeddings_regularizer=l2(0.00001))(m_input))

pred = add([pred, U_bias, M_bias])
# pred = Lambda(lambda x: x + K.constant(3.6, dtype=K.floatx()))(pred)


model = Model(inputs=[u_input, m_input], outputs=[pred])

model.summary()

callbacks = []
callbacks.append(EarlyStopping(monitor='val_rmse', patience=10))
callbacks.append(ModelCheckpoint('model_normal.h5', monitor='val_rmse', save_best_only=True))

model.compile(loss='mse', optimizer='adam', metrics=[rmse])
model.fit([X_train[:, 0], X_train[:, 1]], Y_train, epochs=100, batch_size=1024, validation_split=0.1, callbacks=callbacks) 