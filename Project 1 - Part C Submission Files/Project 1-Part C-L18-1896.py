import numpy as np

import random
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import keras
import cv2 as cv
import os
from keras import backend as K
from keras.callbacks import TensorBoard



base_dir = os.path.dirname(__file__)
catsF = os.path.join(base_dir, "cats")
dogsF= os.path.join(base_dir, "dogs")


ImageCat = os.listdir(catsF)
ImageDogs = os.listdir(dogs_folder)

ImageCat_array = np.array([cv.imread(os.path.join(os.path.abspath(catsF), i), cv.IMREAD_GRAYSCALE).reshape(64 ,64 ,1) for i in ImageCat ])
ImageDogs_array = np.array([cv.imread(os.path.join(os.path.abspath(dogs_folder), i), cv.IMREAD_GRAYSCALE).reshape(64 ,64 ,1) for i in ImageCat ])

ImageCatlab= np.array([[1]] * len(ImageCat_array))
ImageDogs_labels = np.array([[0]] * len(ImageDogs_array))

cat_ = np.concatenate([ImageCat_array, ImageDogs_array])
ff_ = np.concatenate([ImageCat_labels, ImageDogs_labels])

BatchSize = 100
classes = 2
Epochs = 40

im_width = 64
Im_height = 64
model = Sequential()
model.add(Conv2D(kernel_size=(3,3),filters=3,input_shape=(64, 64, 1),activation="relu",padding="valid"))
model.add(Conv2D(kernel_size=(3,3),filters=2,activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(kernel_size=(3,3),filters=2,activation="relu",padding="same"))
model.add(Conv2D(kernel_size=(3,3),filters=10,activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(kernel_size=(2,2),strides=(2,2),filters=10))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100,activation="sigmoid"))
model.add(Dense(num_classes,activation="softmax"))
model.summary()

model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
Call= TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
model.fit(cat_, dog_, BatchSize=BatchSize, epochs=Epochs, verbose=1, callbacks=[Call], shuffle=True, validation_split=0.20)

