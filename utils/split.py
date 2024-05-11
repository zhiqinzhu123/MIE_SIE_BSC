"""
slit.py 用于将训练集切分训练/验证集且满足k折交叉验证
17、18 等年份记得随数据集更改
"""

import os
from sklearn.model_selection import StratifiedKFold
import numpy as np
# import shutil

train_root = r'E:\DATASETS\aaa\brats2017\Training'
test_root = r'E:\DATASETS\aaa\brats2017\Validation'

def write(data, fname, root):
    fname = os.path.join(root, fname)
    with open(fname, 'w') as f:
        f.write('\n'.join(data))

def trainfile_txt(root):
    hgg = os.listdir(os.path.join(root, 'HGG'))
    hgg = [os.path.join('HGG', f, f + "_f32.pkl") for f in hgg] # "_pm.pkl/_f32.pkl"
    lgg = os.listdir(os.path.join(root, 'LGG'))
    lgg = [os.path.join('LGG', f, f + "_f32.pkl") for f in lgg]

    X = hgg + lgg

    Y = [1]*len(hgg) + [0]*len(lgg)

    write(X, '17_all.txt', root)

    X, Y = np.array(X), np.array(Y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)

    for k, (train_index, valid_index) in enumerate(skf.split(Y, Y)):
        train_list = list(X[train_index])
        valid_list = list(X[valid_index])

        write(train_list, '17_train_{}.txt'.format(k), root=root)
        write(valid_list, '17_valid_{}.txt'.format(k), root=root)

def testfile_txt(root):

    X = [os.path.join(f, f + "_f32.pkl") for f in os.listdir(root)]

    write(X, '17_test.txt', root)

trainfile_txt(train_root)
testfile_txt(test_root)