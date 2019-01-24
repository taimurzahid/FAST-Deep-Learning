from img_reader import get_data
# from funcs import get_data
# from model_one import ModelOne
# from alexnet import alexnet_model
#from model_one import ModelOne
from model_two import ModelTwo
# from vgga16 import VGG_16
import math
import keras
from keras.metrics import sparse_top_k_categorical_accuracy
from gen_img_reader import img_list, total_files

test_data_folder = r'C:\Users\Work\PycharmProjects\DlProject\datasets_tt\aug_test'
train_data_folder = r'C:\Users\Work\PycharmProjects\DlProject\datasets_tt\aug_train'

epochs = 100
batch_size = 64

#test_data, test_labels = get_data(test_data_folder, 256, 256)
#train_data, train_labels = get_data(train_data_folder, 256, 256)

test_files = total_files(test_data_folder)
train_files = total_files(train_data_folder)

train_generator = img_list(train_data_folder, 256, 256, batch_size)
val_generator = img_list(test_data_folder, 256, 256, batch_size)


steps_per_epoch = math.ceil(train_files/batch_size)
validation_steps = math.ceil(test_files/batch_size)
# labels = labels - 1

# model = ModelOne()
model = ModelTwo()


model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                       validation_data=val_generator,
                       epochs=epochs,
                       verbose=1,
                        validation_steps=validation_steps,)

print("!!! DONE !!!")
