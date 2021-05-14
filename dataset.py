import os
import numpy as np
from torchvision.io import read_image
import fnmatch
from torch.utils.data import Dataset
import torch.nn as nn
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.LEN = len( fnmatch.filter(os.listdir(img_dir), '*.jpg') )
        print(self.LEN)

    def __len__(self):
        return self.LEN

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,  str(idx) + '.jpg')
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return (image, '')

class NoisyDataset(nn.Module):
  def __init__(self, rootdir='./', mean = 0, var = 1):
    super(NoisyDataset, self).__init__()
    self.mean = mean
    self.var = var
  
  def forward(self, image):
    if image.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.

    noise = np.random.normal(self.mean, self.var, size = image.shape)
    noisy_image = np.clip(image + noise, low_clip, 1)
    return noisy_image