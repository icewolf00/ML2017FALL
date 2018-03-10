
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, Activation
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import optimizers


def readata(filename):
	filein = open(filename, 'r')
	data = []
	for line in filein:
		data.append(line.strip("\r\n").replace(',', ' ').split())
	data = np.array(data[1:])
	#data = data.astype('float32')
	Y_train = data[:, 0].astype('int')
	#print(Y_train[0])
	X_train = data[:, 1:].astype('float')
	#print(X_train[0])
	return X_train, Y_train


(X_train, Y_train) = readata(sys.argv[1])
X_train = X_train.reshape((-1, 48, 48, 1))
(X_train, Y_train) = shuffle(X_train, Y_train)

X_train = X_train / 255

X_train, X_valid = X_train[:-5000], X_train[-5000:]
Y_train, Y_valid = Y_train[:-5000], Y_train[-5000:]

X_train = np.concatenate((X_train, X_train[:, :, ::-1]), axis=0)
Y_train = np.concatenate((Y_train, Y_train), axis=0)

Y_train = to_categorical(Y_train, 7)
Y_valid = to_categorical(Y_valid, 7)

datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=[0.9, 1.1],
            shear_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

model = Sequential()

model.add(Convolution2D(64, (5, 5), activation = 'relu', input_shape = (48, 48, 1))) #44*44
model.add(MaxPooling2D((2, 2))) #22*22
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Convolution2D(128, (3, 3), activation = 'relu')) #20*20
model.add(MaxPooling2D((2, 2)))	#10*10
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Convolution2D(256, (3, 3), activation = 'relu')) #8*8
model.add(MaxPooling2D((2, 2)))	#4*4
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Convolution2D(512, (3, 3), activation = 'relu')) #2*2
model.add(MaxPooling2D((2, 2)))	#1*1
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(activation="relu", units=512))
#model.add(Activation('elu'))
model.add(Dropout(0.3))


model.add(Dense(activation="relu", units=512))
#model.add(Activation('elu'))
model.add(Dropout(0.3))


model.add(Dense(activation="relu", units=64))
#model.add(Activation('elu'))
model.add(Dropout(0.3))

model.add(Dense(units = 7))
model.add(Activation('softmax'))

adam = optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

datagen.fit(X_train)

model.summary()

callbacks = []
callbacks.append(ModelCheckpoint('models/model-{epoch:05d}-{val_acc:.5f}-{val_loss:.5f}.h5', monitor='val_acc', save_best_only=True, mode = 'auto', period=1))
#callbacks.append(EarlyStopping(monitor = 'val_acc', min_delta = 1e-6, patience = 10, verbose=0, mode = 'max'))
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=128), steps_per_epoch=len(X_train) / 32, epochs = 512, validation_data=(X_valid, Y_valid), callbacks=callbacks)
# model.save(sys.argv[2])