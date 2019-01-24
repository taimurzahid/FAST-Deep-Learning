import os, random, shutil
from augmentation import do_augmentation
import time

# master_dir = r'C:\Users\Work\PycharmProjects\DlProject\experiments\dataset'

# This contains all the files
train_dir = r'C:\Users\Work\PycharmProjects\DlProject\traintest\train'
test_dir = r'C:\Users\Work\PycharmProjects\DlProject\traintest\test'

dirs = os.walk(train_dir)
for s_dir in dirs:
    dir_path = s_dir[0]
    dir_files = s_dir[2]
    if (dir_files != 0):
        dir_name = os.path.basename(dir_path)
        test_files = random.sample(dir_files, int(0.2 * len(dir_files)))
        os.mkdir(os.path.join(test_dir, str(dir_name)))
        current_dir = os.path.join(test_dir, str(dir_name))
        for fname in test_files:
            srcpath = os.path.join(dir_path, fname)
            shutil.move(srcpath, current_dir)


aug_test_dir = r'C:\Users\Work\PycharmProjects\DlProject\traintest\aug_test'
aug_train_dir = r'C:\Users\Work\PycharmProjects\DlProject\traintest\aug_train'

a = do_augmentation(test_dir, aug_test_dir, 376, '6')
print(f'Test Data Augmentation: {a}')

time.sleep(10)

b = do_augmentation(train_dir, aug_train_dir, 1504, '6')
print(f'Train Data Augmentation: {b}')
