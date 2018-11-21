# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 23:58:48 2018

@author: simra1
"""

import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
####cats-->0
####dogs-->1
train_cats='C:/Users/simra1/Documents/university/MS/Deep Learning/project1/CatsandDogs/cats'

labels=[]
data=[]
im_width=64
im_height=64

train_image_files_cats=[f for f in os.listdir(train_cats)
						if os.path.isfile(os.path.join(train_cats,f))]

for file_name_cats in train_image_files_cats: 
     if file_name_cats!='Thumbs.db':   
        image_file_cats=str(train_cats+'/'+file_name_cats)
        img_cats=cv2.imread(image_file_cats,cv2.IMREAD_GRAYSCALE)
        new_img_cats=cv2.resize(img_cats,(im_width,im_height))
        data.append(new_img_cats)
        labels.append(0)

        
train_dogs='C:/Users/simra1/Documents/university/MS/Deep Learning/project1/CatsandDogs/dogs'
train_image_files_dogs=[f for f in os.listdir(train_dogs)
						if os.path.isfile(os.path.join(train_dogs,f))]
for file_name_dogs in train_image_files_dogs: 
     if file_name_dogs!='Thumbs.db':   
        image_file_dogs=str(train_dogs+'/'+file_name_dogs)
        img_dogs=cv2.imread(image_file_dogs,cv2.IMREAD_GRAYSCALE)
        new_img_dogs=cv2.resize(img_dogs,(im_width,im_height))
        data.append(new_img_dogs)
        labels.append(1)

data=np.array(data)
data=data.reshape((data.shape)[0],(data.shape)[1],(data.shape)[2],1)
labels=np.array(labels)
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(data,labels, test_size=0.2, random_state=1)

from keras import Sequential
from keras.layers import Dense,MaxPooling2D,Conv2D,Flatten,Dropout


model = Sequential()
model.add(Conv2D(kernel_size=(3,3),filters=3,input_shape=(im_width, im_height,1),activation="relu",padding="valid"))
model.add(Conv2D(kernel_size=(3,3),filters=10,activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(kernel_size=(3,3),filters=3,activation="relu",padding="same"))
model.add(Conv2D(kernel_size=(5,5),filters=5,activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Conv2D(kernel_size=(2,2),strides=(2,2),filters=10))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100,activation="sigmoid"))
model.add(Dense(1,activation="sigmoid"))
model.summary()

from keras import optimizers
#sgd=optimizers.SGD(lr=0.001, momentum=0.7, decay=0.0, nesterov=False)
sgd = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error',metrics=["accuracy"], optimizer=sgd)
history1 = model.fit(data,labels, epochs=5, batch_size=300, validation_data=(X_val,Y_val), verbose=1)

print(history.history['val_acc'])
f, ax = plt.subplots()
ax.plot([None] + history.history['loss'], 'o-')
ax.plot([None] + history.history['val_loss'], 'x-')
ax.legend(['Train loss', 'Validation loss'], loc = 0)
ax.set_title('Training/Validation acc per Epoch')
ax.set_xlabel('Epoch')
ax.set_ylabel('loss')
plt.show()





