import os
import numpy as np
import cv2 as cv


def get_data(folder_path=""):
    dirs = os.walk(folder_path)

    master_data = []
    master_labels = []
    none_data = 0

    for s_dir in dirs:
        dir_path = s_dir[0]
        dir_files = s_dir[2]
        for file in dir_files:
            file_path = os.path.join(dir_path, file)
            file_data = np.array(cv.imread(file_path, cv.IMREAD_GRAYSCALE))
            if (file_data.any() != None):
                reshaped_data = file_data.reshape((file_data.shape[0]), (file_data.shape[1]), 1)
                master_data.append(reshaped_data)
                master_labels.append(np.array([(int(os.path.basename(dir_path)))]))
            else:
                print(file_path)
                none_data += 1
        print(dir_path)

    print(f'Total Images: {len(master_data)} || Total Labels: {len(master_labels)} || Total Errors: {none_data}')
    return np.array(master_data), np.array(master_labels)
