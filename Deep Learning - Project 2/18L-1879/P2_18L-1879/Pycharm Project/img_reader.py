import os
import numpy as np
import cv2 as cv
import random
from sklearn.utils import shuffle


def shuffle_data(data, label):
    data = np.array(data, dtype=float)
    label = np.array(label, dtype=int)
    data, label = shuffle(data, label)
    return data, label


def get_data(folder_path="", img_h=128, img_w=128):
    dirs = os.walk(folder_path)

    master_data = []
    master_labels = []
    image_sizes = []
    none_data = 0

    for s_dir in dirs:
        dir_path = s_dir[0]
        dir_files = s_dir[2]
        for file in dir_files:
            file_path = os.path.join(dir_path, file)

            image = cv.imread(file_path)
            i_height = np.size(image, 0)
            i_width = np.size(image, 1)

            image_sizes.append([i_height, i_width])

            # file_data = np.array(cv.imread(file_path, cv.IMREAD_GRAYSCALE))
            file_data = np.array(cv.imread(file_path))
            # file_data = cv.resize(file_data, dsize=(256, 256), interpolation=cv.INTER_LANCZOS4)
            # cv.imshow("Picture", file_data)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            if (file_data.any() != None):
                if (i_height != img_h or i_width != img_w):
                    file_data = cv.resize(file_data, dsize=(img_h, img_w), interpolation=cv.INTER_LANCZOS4)
                # Image Shape: Height x Width x Depth(RGB)
                reshaped_data = file_data.reshape((file_data.shape[0]), (file_data.shape[1]), 3)
                master_data.append(reshaped_data)
                master_labels.append(np.array([(int(os.path.basename(dir_path)))]))
                # master_labels.append(np.array([((int(os.path.basename(dir_path)))-1)]))
            else:
                print(file_path)
                none_data += 1
        print(dir_path)

    print(f'Total Images: {len(master_data)} || Total Labels: {len(master_labels)} || Total Errors: {none_data}')
    return shuffle_data(master_data, master_labels)
