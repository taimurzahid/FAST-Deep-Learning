import os
import numpy as np
import cv2 as cv
import random
import math


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


def img_list(folder_path="", img_h=128, img_w=128, batch_size=1000):
    dirs = os.walk(folder_path)

    paths_labels = []

    for s_dir in dirs:
        dir_path = s_dir[0]
        dir_files = s_dir[2]
        for file in dir_files:
            single_image_path = os.path.join(dir_path, file)
            single_image_label = os.path.basename(dir_path)
            paths_labels.append([single_image_path, single_image_label])

    random.shuffle(paths_labels)
    tdata_len = len(paths_labels)
    iterations = math.ceil(tdata_len / batch_size)

    for count in range(iterations):
        m_data = []
        m_label = []

        if ((count + 1) * batch_size < tdata_len):
            to_loop = paths_labels[count * batch_size:(count + 1) * batch_size]
            for img in to_loop:
                m_data.append(read_image(img[0], img_h, img_w))
                m_label.append([int(img[1])-1])
            yield np.array(m_data, dtype=float), np.array(m_label)
        else:
            to_loop = paths_labels[count * batch_size:]
            for img in to_loop:
                m_data.append(read_image(img[0], img_h, img_w))
                m_label.append([int(img[1])-1])
                yield np.array(m_data, dtype=float), np.array(m_label)


def total_files(folder_path=""):
    dirs = os.walk(folder_path)
    files_count =0
    for s_dir in dirs:
        dir_path = s_dir[0]
        dir_files = s_dir[2]

        files_count = files_count + len(dir_files)
    print(f"Files Count:= {files_count}")
    return int(files_count)
