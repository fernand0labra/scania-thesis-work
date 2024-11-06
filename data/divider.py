import os
import numpy as np

dataset_path =      ''
dataset_folder =    ''  # os.path.join(*os.path.split(dataset_path)[:-1])

set_ratio = [0.6, 0.2, 0.2]                         # [Train, Val, Test]
dataset = open(dataset_path, 'r').readlines()

train_set = open(os.path.join(dataset_folder, 'train.txt'), 'w')
val_set = open(os.path.join(  dataset_folder, 'val.txt'), 'w')
test_set = open(os.path.join( dataset_folder, 'test.txt'), 'w')

n_lines = len(dataset)
dataset = list(map(lambda s: dataset[s], list(np.array(np.random.random(n_lines) * n_lines, dtype=np.uint32))))

for idx, line in enumerate(dataset):
    if   idx < int(n_lines * set_ratio[0]):                     train_set.write(line)
    elif idx < int(n_lines * (set_ratio[0] + set_ratio[1])):    val_set.write(line)
    elif idx < int(n_lines):                                    test_set.write(line)