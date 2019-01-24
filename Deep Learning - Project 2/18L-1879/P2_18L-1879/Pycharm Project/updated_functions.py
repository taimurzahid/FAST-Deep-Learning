import os, random
import numpy as np
import cv2 as cv
from sklearn.utils import shuffle
import math
from collections import Counter


def shuffle_data(data, label):
    data = np.array(data, dtype=float)
    label = np.array(label, dtype=int)
    data, label = shuffle(data, label)
    return data, label


def total_files(folder_path=""):
    dirs = os.walk(folder_path)
    files_count = 0
    for s_dir in dirs:
        dir_path = s_dir[0]
        dir_files = s_dir[2]
        dir_name = os.path.basename(dir_path)

        print(f'{dir_name}: {len(dir_files)}')
        files_count = files_count + len(dir_files)
    print(f"Total Filess:= {files_count}")
    return int(files_count)


def read_image(img_path="", img_h=128, img_w=128):
    image = cv.imread(img_path)
    i_height = np.size(image, 0)
    i_width = np.size(image, 1)

    file_data = np.array(cv.imread(img_path))

    if (file_data.any() != None):
        if (i_height != img_h or i_width != img_w):
            file_data = cv.resize(file_data, dsize=(img_h, img_w), interpolation=cv.INTER_LANCZOS4)

        file_data = file_data.reshape((file_data.shape[0]), (file_data.shape[1]), 3)
        return file_data
    else:
        return None


def get_data(folder_path="", img_h=128, img_w=128):
    dirs = os.walk(folder_path)

    master_data = []
    master_labels = []
    image_sizes = []
    none_data = 0

    for s_dir in dirs:
        dir_path = s_dir[0]
        dir_files = s_dir[2]
        dir_name = os.path.basename(dir_path)
        for file in dir_files:
            file_path = os.path.join(dir_path, file)

            image = cv.imread(file_path)
            i_height = np.size(image, 0)
            i_width = np.size(image, 1)

            image_sizes.append([i_height, i_width])

            # file_data = np.array(cv.imread(file_path, cv.IMREAD_GRAYSCALE))
            file_data = np.array(cv.imread(file_path))

            if (file_data.any() != None):

                if (i_height != img_h or i_width != img_w):
                    file_data = cv.resize(file_data, dsize=(img_h, img_w), interpolation=cv.INTER_LANCZOS4)

                # Image Shape: Height x Width x Depth(RGB)
                reshaped_data = file_data.reshape((file_data.shape[0]), (file_data.shape[1]), 3)
                master_data.append(reshaped_data)
                master_labels.append(np.array([dir_name]))
                # master_labels.append(np.array([((int(os.path.basename(dir_path)))-1)]))
            else:
                print(file_path)
                none_data += 1
        print(f'Directory:=> {dir_path}')

    print(f'Total Images: {len(master_data)} || Total Labels: {len(master_labels)} || Total Errors: {none_data}')
    return shuffle_data(master_data, master_labels)


def get_balance_random_test(folder_path="", img_h=300, img_w=300, batch_size=64, num_of_classes=7, test=35):
    dirs = os.walk(folder_path)

    samples = int(batch_size / num_of_classes)

    for num in range(test):
        m_data = []
        m_label = []
        dirs = os.walk(folder_path)

        for s_dir in dirs:
            dir_path = s_dir[0]
            dir_files = s_dir[2]
            dir_name = os.path.basename(dir_path)

            if (len(dir_files) > 0):
                tfiles = random.sample(dir_files, samples)
                for file in tfiles:
                    single_image_path = os.path.join(dir_path, file)
                    single_image_label = dir_name
                    m_data.append(read_image(single_image_path, img_h, img_w))
                    m_label.append([int(dir_name) - 1])
        yield shuffle_data(m_data, m_label)


def get_class_weights(y, smooth_factor=0):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    y = np.ravel(y)
    counter = Counter(y)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority / count) for cls, count in counter.items()}
