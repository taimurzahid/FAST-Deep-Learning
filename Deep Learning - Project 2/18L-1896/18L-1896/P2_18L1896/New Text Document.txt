
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
train = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)



test = ImageDataGenerator(rescale=1. / 255)

train=train.flow_from_directory(train_path,target_size=(128,128),classes=['1','2','3','4','5','6','7'],batch_size=62)
test=test.flow_from_directory(test_path,target_size=(128,128),classes=['1','2','3','4','5','6','7'],batch_size=62)
model = keras.applications.resnet50.ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.summary()
for layer in model.layers[:]:
    layer.trainable = False
y = model.output
y = Flatten()(x)
y = Dense(512, activation="relu")(y)
y = Dropout(0.5)(y)
predictions = Dense(7, activation="softmax")(y)
model_final = Model(input=model.input, output=predictions)
model_final.summary()
model_final.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
#tbCallBack = TensorBoard(log_dir='./acha', histogram_freq=1,write_graph=True, write_images=True)
history=model_final.fit_generator(train_data,steps_per_epoch=2,validation_data=test_data,validation_steps=4,epochs=10,verbose=1)
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()