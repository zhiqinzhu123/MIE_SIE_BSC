"""
preprocess.py 用于数据预处理，将nii文件打包为pkl文件
xxx_f32.pkl: [flair,t1ce,t1,t2,seg]
xxx_pm.pkl: [flair,t1ce,t1,t2,seg,pm]
"""

import pickle
import os
import numpy as np
import nibabel as nib
from glob import glob
from tqdm import tqdm

modalities = ('flair', 't1ce', 't1', 't2')


# nib 加载
def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data


# 标准化
def normalize(image, mask=None):
    assert len(image.shape) == 3  # shape is [H,W,D]
    assert image[0, 0, 0] == 0  # check the background is zero
    if mask is not None:
        mask = (image > 0)  # The bg is zero

    mean = image[mask].mean()
    std = image[mask].std()
    image = image.astype(dtype=np.float32)
    image[mask] = (image[mask] - mean) / std
    return image


# 保存
def savepkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


# 数据处理
def process(path, pm=False):
    """ Set all Voxels that are outside of the brain mask to 0"""
    name = os.path.basename(path)  # 文件夹名(病例序号)
    label_path = os.path.join((path), name + "_seg.nii.gz")
    label = np.array(nib_load(label_path), dtype='uint8', order='C')
    images = np.stack([
        np.array(nib_load(os.path.join(path, name + '_' + modal + '.nii.gz')), dtype='float32', order='C')
        for modal in modalities], -1)

    mask = images.sum(-1) > 0

    for k in range(4):
        x = images[..., k]  #
        y = x[mask]  #

        lower = np.percentile(y, 0.2)  # 算分位数
        upper = np.percentile(y, 99.8)

        x[mask & (x < lower)] = lower
        x[mask & (x > upper)] = upper

        y = x[mask]

        x -= y.mean()
        x /= y.std()

        images[..., k] = x

    if pm == True:
        pointmap_path = os.path.join((path), name + "_pm.nii.gz")
        pointmap = np.array(nib_load(pointmap_path), dtype='uint8', order='C')
        savepath = os.path.join(path, name + "_pm.pkl")
        savepkl(data=(images, label, pointmap), path=savepath)
        savepath = os.path.join(path, name + "_f32.pkl")
        savepkl(data=(images, label), path=savepath)
    else:
        savepath = os.path.join(path, name + "_f32.pkl")
        savepkl(data=(images, label), path=savepath)


def doit(paths, pm):
    for path in tqdm(paths):
        # name = os.path.basename(path) # 文件夹名(病例序号)
        # print(name)
        process(path, pm)


# 开始处理训练集
root = r"E:\DATASETS\aaa\brats2017\Training"  # 路径
paths = glob(os.path.join(root, "*GG", "Brats17*"))
print(len(paths))
doit(paths, True)

# 开始处理验证集（实验中用于测试）
root = r"E:\DATASETS\aaa\brats2017\Validation"  # 路径
paths = glob(os.path.join(root, "Brats17*"))
print(len(paths))
doit(paths, False)
