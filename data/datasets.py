import os
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from .transforms import NumpyType
from .data_utils import pkload

import numpy as np

class BraTSDataset(Dataset):
    def __init__(self, list_file, root='', pm=False, transforms=''):
        paths, names = [], []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                name = line.split('/')[-1]
                names.append(name)
                path = os.path.join(root, line)
                paths.append(path)

        self.names = names
        self.paths = paths
        self.transforms = eval(transforms or 'Identity()')
        self.pm = pm

    def __getitem__(self, index): 
        path = self.paths[index]
        if self.pm:
            x, y, z = pkload(path)
            # transforms work with nhwtc
            x, y, z = x[None, ...], y[None, ...], z[None, ...]
            # print(x.shape, y.shape)#(1, 240, 240, 155, 4) (1, 240, 240, 155)
            x, y, z = self.transforms([x, y, z])
            x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
            y = np.ascontiguousarray(y)
            z = np.ascontiguousarray(z)
            x, y, z = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)
            return x, y, z
        else:
            x, y = pkload(path)
            # transforms work with nhwtc
            x, y = x[None, ...], y[None, ...]
            # print(x.shape, y.shape)#(1, 240, 240, 155, 4) (1, 240, 240, 155)
            x, y = self.transforms([x, y])
            x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
            y = np.ascontiguousarray(y)
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            return x, y

    def __len__(self):
        return len(self.names)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]
