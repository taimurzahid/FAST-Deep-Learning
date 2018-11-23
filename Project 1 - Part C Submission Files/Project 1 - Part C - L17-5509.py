# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:18:58 2018

CS 5102 - Deep Learning
Part A - Convolutional Neural Network using Keras
with Tensorflow backend

@author: amir.iqbal
"""

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
from keras.callbacks import TensorBoard
from time import time
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

img_width = 64
img_height = 64


train_data_dir = r'D:\amir\Dev\Deep Learning\Project1\CatsandDogs\train'
validation_data_dir = r'D:\amir\Dev\Deep Learning\Project1\CatsandDogs\validation'
nb_train_samples = 18000
nb_validation_samples = 7000
epochs = 100

#batch_sizes = [16, 32, 64, 128, 256, 512]
batch_size = 128  # the best performing for validation accuracy
#weight_initializers = ['glorot_normal', 'he_uniform', 'he_normal']
weight_initializer = 'glorot_uniform'
learning_rates = [0.001]
#learning_rate = 0.001  #default
#optimizers = ['adam', 'adamax', 'nadam']
optimizer = 'adam'
last_saved_model = 'last_model.h5'
# set this to true if you want to load previously saved model and resume training
resume = False
# for quick evaluation of new parameters, we have to run training on small data set to save time
reduce_data_by_factor = 1 # if 1 it means we are training on all data, if 10 then we are training on 1/10th of data



#defaults
#Conv2D kernel_initializer='glorot_uniform', bias_initializer='zeros'
#Dense kernel_initializer='glorot_uniform', bias_initializer='zeros'

for learning_rate in learning_rates:
    K.clear_session()    
    if resume:
        print('loading last saved model')
        model = load_model(last_saved_model)
    else:
        #adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 3)
        model = Sequential()
        model.add(Conv2D(kernel_size=(3,3), filters=16, input_shape=input_shape, activation="relu", padding="same", kernel_initializer=weight_initializer))
        model.add(Conv2D(kernel_size=(3,3), filters=16, activation="relu", padding="same", kernel_initializer=weight_initializer))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Conv2D(kernel_size=(3,3), filters=32, activation="relu", padding="same", kernel_initializer=weight_initializer))
        model.add(Conv2D(kernel_size=(3,3), filters=32, activation="relu", padding="same", kernel_initializer=weight_initializer))
        model.add(Conv2D(kernel_size=(3,3), filters=32, activation="relu", padding="same", kernel_initializer=weight_initializer))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(128, activation="relu", kernel_initializer=weight_initializer))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="sigmoid", kernel_initializer=weight_initializer))
        model.summary()
        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])
    
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')
    
    log_str = "{time}-batch_size-{batch:d}-optimizer-{opt}-learining-rate-{lern}-reduce_by-{red}-resume-{res}".format(time=int(time()), batch=batch_size, opt=optimizer, red=reduce_data_by_factor, lern=learning_rate, res=resume)
    tensorboard = TensorBoard(log_dir="logs/" + log_str)
    checkpoint = ModelCheckpoint('checkpoints/checkpoint-' + log_str + '-{epoch:02d}-{val_acc:.2f}.h5', monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [tensorboard, checkpoint]
    
    start_time = time()
    history = model.fit_generator(train_generator,
                        steps_per_epoch=nb_train_samples // (batch_size*reduce_data_by_factor),
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks_list,
                        validation_data=validation_generator,
                        validation_steps=nb_validation_samples // (batch_size*reduce_data_by_factor))
    elapsed_time = time() - start_time
    print(elapsed_time)
    model.save("models/model-" + log_str + ".h5")
    model.save(last_saved_model)  # this will be loaded next time if resume required
