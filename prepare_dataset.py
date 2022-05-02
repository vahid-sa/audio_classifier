import os
import numpy as np
from os import path as osp

root_dir = osp.abspath('./dataset')
train_file_path = osp.join(root_dir, "train_list.txt")
validation_file_path = osp.join(root_dir, "validation_list.txt")
if osp.isfile(train_file_path):
    os.remove(train_file_path)
if osp.isfile(validation_file_path):
    os.remove(validation_file_path)

validation_list = []
train_list = []
for root, dirs, names in os.walk(root_dir):
    val_numbers = np.random.randint(1, 41, 5)
    for name in names:
        path = osp.join(osp.basename(root), name)
        if int(name.split('-')[0]) in val_numbers:
            validation_list.append(path)
        else:
            train_list.append(path)
train_list.sort()
validation_list.sort()

train = '\n'.join(train_list)
validation = '\n'.join(validation_list)

with open(train_file_path, "w") as f:
    f.write(train)
with open(validation_file_path, "w") as f:
    f.write(validation)
