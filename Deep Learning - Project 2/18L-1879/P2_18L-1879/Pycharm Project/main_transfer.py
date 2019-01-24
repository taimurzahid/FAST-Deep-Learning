from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras import applications
from updated_functions import get_data, get_class_weights
import matplotlib.pyplot as plt
import keras


img_width = 224
img_height = 224
num_of_classes = 7

batch_size = 64
epochs = 25



train_dir = r'C:\Users\Work\PycharmProjects\DlProject\datasets\train'
test_dir = r'C:\Users\Work\PycharmProjects\DlProject\datasets\test'

train_data, train_labels = get_data(train_dir, img_height, img_width)

class_weights = get_class_weights(train_labels-1)

model = applications.VGG19(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

model.summary()

for layer in model.layers[:]:
    layer.trainable = False

# Adding custom Layers
x = model.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(num_of_classes, activation="softmax")(x)

# creating the final model
model_final = Model(input=model.input, output=predictions)

model_final.summary()


model_final.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


history = model_final.fit(train_data, train_labels-1,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1, shuffle=True, class_weight=class_weights)




# summarizing accuracy/epoch
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# summarize loss/epoch
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
