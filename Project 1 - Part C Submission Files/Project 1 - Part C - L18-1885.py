import numpy as np
import cv2 as cv
import os
import random
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import TensorBoard



Bise_dir = os.path.dirname(__file__)
cats_Imagesfolder = os.path.join(Bise_dir, "cats")
dogs_Imagesfolder = os.path.join(Bise_dir, "dogs")
print(cats_Imagesfolder)
print(dogs_Imagesfolder)
cats_images = os.listdir(cats_Imagesfolder)
dogs_images = os.listdir(dogs_Imagesfolder)

cats_Array = np.array([cv.imread(os.path.join(os.path.abspath(cats_Imagesfolder), i), cv.IMREAD_GRAYSCALE).reshape(64 ,64 ,1) for i in cats_images ])
dogs_Array = np.array([cv.imread(os.path.join(os.path.abspath(dogs_Imagesfolder), i), cv.IMREAD_GRAYSCALE).reshape(64 ,64 ,1) for i in cats_images ])
cats_Labels = np.array([[1]] * len(cats_Array))
dogs_Labels = np.array([[0]] * len(dogs_Array))
array_d = np.concatenate([cats_Array, dogs_Array])
label_l = np.concatenate([cats_Labels, dogs_Labels])

batch_size = 64
num_classes = 2
epochs = 50

im_width = 64
Im_height = 64
model = Sequential()
model.add(Conv2D(kernel_size=(3,3),filters=3,input_shape=(64, 64,1),activation="relu",padding="valid"))
model.add(Conv2D(kernel_size=(3,3),filters=6,activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(kernel_size=(3,3),filters=3,activation="relu",padding="same"))
model.add(Conv2D(kernel_size=(5,5),filters=6,activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Conv2D(kernel_size=(2,2),strides=(2,2),filters=10))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100,activation="sigmoid"))
model.add(Dense(num_classes,activation="softmax"))
model.summary()
model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
tensorboardCall = TensorBoard(log_dir='./this', histogram_freq=1, write_graph=True, write_images=True)
model.fit(array_d, label_l, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[tensorboardCall], shuffle=True, validation_split=0.20)
