from keras.layers import LeakyReLU ,Activation
import keras
from keras.models import Sequential,Model

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
num_classes = 7
train_path=r"./trainX"
test_path=r"./testX"
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_data=train_datagen.flow_from_directory(train_path,target_size=(224,224),classes=['1','2','3','4','5','6','7'],batch_size=62)
test_data=test_datagen.flow_from_directory(test_path,target_size=(224,224),classes=['1','2','3','4','5','6','7'],batch_size=62)

generator = Sequential([
    Dense(128, input_shape=(100,)),
    LeakyReLU(alpha=0.01),
    Dense(784),
    Activation('tanh')
], name='generator')
discriminator = Sequential([
    Dense(128, input_shape=(784,)),
    LeakyReLU(alpha=0.01),
    Dense(1),
    Activation('sigmoid')
], name='discriminator')
gan = Sequential([
    generator,
    discriminator
])

# creating the final model
model_final = Model()

model_final.summary()
model_final.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
#tbCallBack = TensorBoard(log_dir='./acha', histogram_freq=1,write_graph=True, write_images=True)
history=model_final.fit_generator(train_data,steps_per_epoch=2,validation_data=test_data,validation_steps=4,epochs=15,verbose=1)


