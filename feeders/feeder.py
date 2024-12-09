import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import os
import tifffile
import albumentations as Alb
from albumentations.pytorch import ToTensorV2
from PIL import Image
import pickle


class SpaceNetFeeder(Dataset):
    ''' https://arxiv.org/abs/2004.06500 '''
    def __init__(self, dataroot, listroot, load_size=900, crop_size=512, hflip=False, vflip=False, rot=False):
        if listroot.endswith('.txt'):
            with open(listroot, 'r') as f:
                eo_paths = f.read().splitlines()
        elif listroot.endswith('.pkl'):
            with open(listroot, 'rb') as f:
                eo_paths = pickle.load(f)
        else:
            print("Unsupported file format.")

        self.eo_paths = [os.path.join(dataroot, eo_path) for eo_path in eo_paths]
        self.sar_paths = [x.replace('PS-RGB', 'SAR-Intensity') for x in self.eo_paths]

        self.load_size = load_size
        self.crop_size = crop_size
        assert (self.load_size >= self.crop_size)

        self.hflip = hflip
        self.vflip = vflip
        self.rot = rot

        if 'train' in listroot:
            self.split = 'train'
        elif 'test' in listroot:
            self.split = 'test'
        else:
            print("Unsupported file format.")

    def im_percent_norm(self, x, p=(1, 99), eps=1 / (2 ** 10)):
        pv = np.percentile(x, p, axis=(0, 1))
        y = x.astype(np.float32)
        pmin = pv[0, ...]
        pmax = pv[1, ...]
        y = np.clip(y, pmin, pmax)
        y = (y - pmin) / np.maximum((pmax - pmin), eps) * 255.0
        return y

    def __getitem__(self, index):
        eo_path = self.eo_paths[index]
        sar_path = self.sar_paths[index]
        EO = np.array(Image.open(eo_path).convert('RGB')).astype(np.float32)
        SAR = tifffile.imread(sar_path).astype(np.float32)
        SAR = self.im_percent_norm(SAR)
        SAR = np.stack((SAR[:, :, 0], (SAR[:, :, 1] + SAR[:, :, 2]) / 2, SAR[:, :, 3]), axis=-1)

        transform = []
        if self.split == 'train':
            transform.append(Alb.RandomCrop(width=self.crop_size, height=self.crop_size))
            if self.hflip:
                transform.append(Alb.HorizontalFlip(p=0.5))
            if self.vflip:
                transform.append(Alb.VerticalFlip(p=0.5))
            if self.rot:
                transform.append(Alb.RandomRotate90(p=0.5))
        else:
            transform.append(Alb.CenterCrop(width=self.crop_size, height=self.crop_size))

        transform.append(Alb.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0))
        transform.append(ToTensorV2())

        transform = Alb.Compose(transform, additional_targets={'image2': 'image'})
        augmented = transform(image=SAR, image2=EO)
        SAR, EO = augmented['image'], augmented['image2']
        return SAR, EO

    def __len__(self):
        return len(self.eo_paths)


class QXSFeeder(Dataset):
    ''' https://arxiv.org/abs/2103.08259 '''
    def __init__(self, dataroot, listroot, load_size=256, crop_size=256, hflip=False, vflip=False, rot=False):
        if listroot.endswith('.txt'):
            with open(listroot, 'r') as f:
                eo_paths = f.read().splitlines()
        elif listroot.endswith('.pkl'):
            with open(listroot, 'rb') as f:
                eo_paths = pickle.load(f)
        else:
            print("Unsupported file format.")

        self.eo_paths = [os.path.join(dataroot, eo_path) for eo_path in eo_paths]
        self.sar_paths = [x.replace('opt_256_oc_0.2', 'sar_256_oc_0.2') for x in self.eo_paths]

        self.load_size = load_size
        self.crop_size = crop_size
        assert (self.load_size >= self.crop_size)

        self.hflip = hflip
        self.vflip = vflip
        self.rot = rot

        if 'train' in listroot:
            self.split = 'train'
        elif 'test' in listroot:
            self.split = 'test'
        else:
            print("Unsupported file format.")

    def __getitem__(self, index):
        eo_path = self.eo_paths[index]
        sar_path = self.sar_paths[index]
        EO = np.array(Image.open(eo_path).convert('RGB'))
        SAR = np.array(Image.open(sar_path).convert('RGB'))

        transform = []
        if self.split == 'train':
            transform.append(Alb.RandomCrop(width=self.crop_size, height=self.crop_size))
            if self.hflip:
                transform.append(Alb.HorizontalFlip(p=0.5))
            if self.vflip:
                transform.append(Alb.VerticalFlip(p=0.5))
            if self.rot:
                transform.append(Alb.RandomRotate90(p=0.5))
        else:
            transform.append(Alb.CenterCrop(width=self.crop_size, height=self.crop_size))

        transform.append(Alb.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0))
        transform.append(ToTensorV2())

        transform = Alb.Compose(transform, additional_targets={'image2': 'image'})
        augmented = transform(image=SAR, image2=EO)
        SAR, EO = augmented['image'], augmented['image2']
        return SAR, EO

    def __len__(self):
        return len(self.eo_paths)


class SAROptFeeder(Dataset):
    ''' https://ieeexplore.ieee.org/document/9779739 '''
    def __init__(self, dataroot, listroot, load_size=600, crop_size=512, hflip=False, vflip=False, rot=False):
        if listroot.endswith('.txt'):
            with open(listroot, 'r') as f:
                eo_paths = f.read().splitlines()
        elif listroot.endswith('.pkl'):
            with open(listroot, 'rb') as f:
                eo_paths = pickle.load(f)
        else:
            print("Unsupported file format.")

        self.eo_paths = [os.path.join(dataroot, eo_path) for eo_path in eo_paths]

        if 'train' in listroot:
            self.split = 'train'
            self.sar_paths = [x.replace('trainB', 'trainA') for x in self.eo_paths]
        elif 'test' in listroot:
            self.split = 'test'
            self.sar_paths = [x.replace('testB', 'testA') for x in self.eo_paths]
        else:
            print("Unsupported file format.")

        self.load_size = load_size
        self.crop_size = crop_size
        assert (self.load_size >= self.crop_size)

        self.hflip = hflip
        self.vflip = vflip
        self.rot = rot

    def __getitem__(self, index):
        eo_path = self.eo_paths[index]
        sar_path = self.sar_paths[index]
        EO = np.array(Image.open(eo_path).convert('RGB'))
        SAR = np.array(Image.open(sar_path).convert('RGB'))

        transform = []
        if self.split == 'train':
            transform.append(Alb.RandomCrop(width=self.crop_size, height=self.crop_size))
            if self.hflip:
                transform.append(Alb.HorizontalFlip(p=0.5))
            if self.vflip:
                transform.append(Alb.VerticalFlip(p=0.5))
            if self.rot:
                transform.append(Alb.RandomRotate90(p=0.5))
        else:
            transform.append(Alb.CenterCrop(width=self.crop_size, height=self.crop_size))

        transform.append(Alb.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0))
        transform.append(ToTensorV2())

        transform = Alb.Compose(transform, additional_targets={'image2': 'image'})
        augmented = transform(image=SAR, image2=EO)
        SAR, EO = augmented['image'], augmented['image2']
        return SAR, EO

    def __len__(self):
        return len(self.eo_paths)