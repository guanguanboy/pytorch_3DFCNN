import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import h5py

class PaviaDataset(Dataset):
    def __init__(self, file_dir, transform=None):
        # 加载数据
        self.h5f = h5py.File(file_dir, 'r')
        self.file_dir = file_dir
        self.transform = transform
        self.traindata = self.h5f['data']
        self.label = self.h5f['label'].value[:,:,:,4:-4,:]

    def __len__(self):
        return len(self.traindata)

    def __getitem__(self, item):
        image = self.traindata[item]
        y_label = self.label[item]

        if self.transform:
            image = self.transform(image)
            y_label = self.transform(y_label)

        return image, y_label

