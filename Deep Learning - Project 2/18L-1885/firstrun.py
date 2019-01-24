
import keras
from keras.models import Sequential
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

train_data=train_datagen.flow_from_directory(train_path,target_size=(64,64),classes=['1','2','3','4','5','6','7'],batch_size=62)
test_data=test_datagen.flow_from_directory(test_path,target_size=(64,64),classes=['1','2','3','4','5','6','7'],batch_size=62)

model = Sequential()
model.add(Conv2D(kernel_size=(3,3),filters=16,input_shape=(64, 64, 3),activation="relu",padding="valid"))
model.add(Conv2D(kernel_size=(3,3),filters=16,activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(kernel_size=(3,3),filters=16,activation="relu",padding="same"))
model.add(Conv2D(kernel_size=(5,5),filters=16,activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Conv2D(kernel_size=(2,2),strides=(2,2),filters=10))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(256,activation="sigmoid"))
model.add(Dense(num_classes,activation="softmax"))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
#tbCallBack = TensorBoard(log_dir='./acha', histogram_freq=1,write_graph=True, write_images=True)
history=model.fit_generator(train_data,steps_per_epoch=20,validation_data=test_data,validation_steps=4,epochs=10,verbose=1)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()