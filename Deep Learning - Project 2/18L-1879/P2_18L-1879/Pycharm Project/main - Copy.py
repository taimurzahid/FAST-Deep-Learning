from img_reader import get_data
# from funcs import get_data
# from model_one import ModelOne
# from alexnet import alexnet_model
#from model_one import ModelOne
from model_two import ModelTwo
# from vgga16 import VGG_16
import keras
from keras.metrics import sparse_top_k_categorical_accuracy

test_data_folder = r'C:\Users\Work\PycharmProjects\DlProject\datasets_tt\aug_test'
train_data_folder = r'C:\Users\Work\PycharmProjects\DlProject\datasets_tt\aug_train'

epochs = 100
batch_size = 64

test_data, test_labels = get_data(test_data_folder, 256, 256)
train_data, train_labels = get_data(train_data_folder, 256, 256)


# labels = labels - 1


def top_3_accuracy(y_true, y_pred):
    return sparse_top_k_categorical_accuracy(y_true, y_pred, k=3)


# model = ModelOne()
model = ModelTwo()
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy', top_3_accuracy])

history = model.fit(train_data, train_labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1, shuffle=True, validation_data=(test_data, test_labels))

print("Working")
