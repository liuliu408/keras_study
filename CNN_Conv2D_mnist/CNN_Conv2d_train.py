# coding: utf-8
### Learnt from Mofan
### recreated by Mike G on 10th Sep 2018
### Train

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPool2D, Flatten, Dense
from keras.optimizers import Adam

nb_class = 10
nb_epoch = 50
batchsize = 1024

# Prepare your data mnist, MAC /.Keras/datasets linux home .keras/datasets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# setup data shape
X_train = X_train.reshape(-1, 28, 28, 1)  # TFchannel last theano (1,28,28)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train = X_train.astype('float32')
Y_test = Y_test.astype('float32')

X_train = X_train/255.0
X_test = X_test/255.0

# One-hot [0,0,0,0,0,1,0,0,0] =5

Y_train = np_utils.to_categorical(Y_train, nb_class)
Y_test = np_utils.to_categorical(Y_test, nb_class)

# setup model

model = Sequential()

# 1st Conv2D layer
model.add(Convolution2D(
    filters=32,
    kernel_size=[5, 5],
    padding='same',
    input_shape=(28, 28, 1)
))
model.add(Activation('relu'))

model.add(MaxPool2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding="same",
))

# 2nd Conv2D layer
model.add(Convolution2D(
    filters=64,
    kernel_size=(5, 5),
    padding='same',
))

model.add(Activation('relu'))
model.add(MaxPool2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding="same",
))

#1st Fully connected Dense
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

#2nd Fully connected Dense
model.add(Dense(10))
model.add(Activation('softmax'))


#Define Optimizer and setup Parameter
adam = Adam(lr=0.0001)

#compile model
model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

#Run/Fireup network

model.fit(x=X_train,
          y=Y_train,
          epochs=nb_epoch,
          batch_size=batchsize,
          verbose=1,
          validation_data=(X_test,Y_test),
          )

model.save('./liuqtrain.h5')   # 保存训练模型！