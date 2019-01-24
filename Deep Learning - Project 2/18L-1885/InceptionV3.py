import matplotlib
matplotlib.use('TKAgg')
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
num_classes = 7
train_path='/content/gdrive/My Drive/dermnet'
test_path='/content/gdrive/My Drive/dermnet'
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

model = keras.applications.inception_v3.InceptionV3(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

model.summary()

for layer in model.layers[:]:
    layer.trainable = False

# Adding custom Layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation="softmax")(x)

# creating the final model
model_final = Model(input=model.input, output=predictions)

model_final.summary()
model_final.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
#tbCallBack = TensorBoard(log_dir='./acha', histogram_freq=1,write_graph=True, write_images=True)
history=model_final.fit_generator(train_data,steps_per_epoch=20,validation_data=test_data,validation_steps=4,epochs=20,verbose=1)

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