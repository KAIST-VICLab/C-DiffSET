"""Dataset feeders for SAR-to-EO image translation.

Each feeder loads paired (SAR, EO) images, applies optional augmentation, and
returns two tensors normalized to the [-1, 1] range expected by the Stable
Diffusion VAE encoder.

Supported datasets:
    * SEN1-2         (``SENMSFeeder``)   -- https://arxiv.org/abs/1906.07789
    * SpaceNet6      (``SpaceNetFeeder``) -- https://arxiv.org/abs/2004.06500
    * QXS-SAROPT     (``QXSFeeder``)      -- https://arxiv.org/abs/2103.08259
    * SAR2Opt        (``SAROptFeeder``)   -- https://ieeexplore.ieee.org/document/9779739
    * Stellar-Vision (``StellarFeeder``)  -- proprietary
"""

import os
import pickle

import numpy as np
import torch
import tifffile
import albumentations as Alb
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

import PIL.Image
try:
    import pyspng
except ImportError:
    pyspng = None


def _read_path_list(listroot):
    """Read a list of relative EO image paths from a ``.txt`` or ``.pkl`` file."""
    if listroot.endswith('.txt'):
        with open(listroot, 'r') as f:
            return f.read().splitlines()
    if listroot.endswith('.pkl'):
        with open(listroot, 'rb') as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported list file format: {listroot}")


def _split_from_listroot(listroot):
    """Infer the data split ('train' / 'test') from the list-file name."""
    if 'train' in listroot:
        return 'train'
    if 'test' in listroot:
        return 'test'
    raise ValueError(f"Cannot infer split (train/test) from: {listroot}")


def _build_transform(split, crop_size, hflip, vflip, rot):
    """Compose an Albumentations pipeline shared by all paired feeders.

    Training uses random cropping and optional flips/rotations; testing uses a
    deterministic center crop. Both branches normalize to [-1, 1] via
    ``mean=std=0.5`` and convert to a CHW tensor.
    """
    transform = []
    if split == 'train':
        transform.append(Alb.RandomCrop(width=crop_size, height=crop_size))
        if hflip:
            transform.append(Alb.HorizontalFlip(p=0.5))
        if vflip:
            transform.append(Alb.VerticalFlip(p=0.5))
        if rot:
            transform.append(Alb.RandomRotate90(p=0.5))
    else:
        transform.append(Alb.CenterCrop(width=crop_size, height=crop_size))

    transform.append(Alb.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0))
    transform.append(ToTensorV2())
    return Alb.Compose(transform, additional_targets={'image2': 'image'})


class SENMSFeeder(Dataset):
    """SEN1-2 dataset (https://arxiv.org/abs/1906.07789)."""

    def __init__(self, dataroot, listroot, load_size=256, crop_size=256,
                 hflip=False, vflip=False, rot=False):
        eo_paths = _read_path_list(listroot)
        self.eo_paths = [os.path.join(dataroot, p) for p in eo_paths]
        # SAR / EO paths differ only by the ``s1_`` / ``s2_`` prefix.
        self.sar_paths = [p.replace('s2_', 's1_') for p in self.eo_paths]

        self.load_size = load_size
        self.crop_size = crop_size
        assert self.load_size >= self.crop_size

        self.hflip, self.vflip, self.rot = hflip, vflip, rot
        self.split = _split_from_listroot(listroot)

    def __getitem__(self, index):
        EO = np.array(Image.open(self.eo_paths[index]).convert('RGB'))
        SAR = np.array(Image.open(self.sar_paths[index]).convert('RGB'))

        transform = _build_transform(self.split, self.crop_size,
                                     self.hflip, self.vflip, self.rot)
        augmented = transform(image=SAR, image2=EO)
        return augmented['image'], augmented['image2']

    def __len__(self):
        return len(self.eo_paths)


class SpaceNetFeeder(Dataset):
    """SpaceNet6 dataset (https://arxiv.org/abs/2004.06500).

    SAR intensity is stored as multi-channel TIFF; we percentile-normalize each
    channel and map the four polarizations to a 3-channel image using
    ``HH / (HV + VH) / 2 / VV`` (see paper, Table VI).
    """

    def __init__(self, dataroot, listroot, load_size=900, crop_size=512,
                 hflip=False, vflip=False, rot=False):
        eo_paths = _read_path_list(listroot)
        self.eo_paths = [os.path.join(dataroot, p) for p in eo_paths]
        self.sar_paths = [p.replace('PS-RGB', 'SAR-Intensity') for p in self.eo_paths]

        self.load_size = load_size
        self.crop_size = crop_size
        assert self.load_size >= self.crop_size

        self.hflip, self.vflip, self.rot = hflip, vflip, rot
        self.split = _split_from_listroot(listroot)

    def im_percent_norm(self, x, p=(1, 99), eps=1 / (2 ** 10)):
        """Clip each channel to its [p_low, p_high] percentiles and rescale to [0, 255]."""
        pv = np.percentile(x, p, axis=(0, 1))
        y = x.astype(np.float32)
        pmin, pmax = pv[0, ...], pv[1, ...]
        y = np.clip(y, pmin, pmax)
        y = (y - pmin) / np.maximum((pmax - pmin), eps) * 255.0
        return y

    def __getitem__(self, index):
        EO = np.array(Image.open(self.eo_paths[index]).convert('RGB')).astype(np.float32)
        SAR = tifffile.imread(self.sar_paths[index]).astype(np.float32)
        SAR = self.im_percent_norm(SAR)
        # Map (HH, HV, VH, VV) -> (HH, mean(HV, VH), VV).
        SAR = np.stack((SAR[:, :, 0], (SAR[:, :, 1] + SAR[:, :, 2]) / 2, SAR[:, :, 3]), axis=-1)

        transform = _build_transform(self.split, self.crop_size,
                                     self.hflip, self.vflip, self.rot)
        augmented = transform(image=SAR, image2=EO)
        return augmented['image'], augmented['image2']

    def __len__(self):
        return len(self.eo_paths)


class QXSFeeder(Dataset):
    """QXS-SAROPT dataset (https://arxiv.org/abs/2103.08259)."""

    def __init__(self, dataroot, listroot, load_size=256, crop_size=256,
                 hflip=False, vflip=False, rot=False):
        eo_paths = _read_path_list(listroot)
        self.eo_paths = [os.path.join(dataroot, p) for p in eo_paths]
        self.sar_paths = [p.replace('opt_256_oc_0.2', 'sar_256_oc_0.2') for p in self.eo_paths]

        self.load_size = load_size
        self.crop_size = crop_size
        assert self.load_size >= self.crop_size

        self.hflip, self.vflip, self.rot = hflip, vflip, rot
        self.split = _split_from_listroot(listroot)

    def __getitem__(self, index):
        EO = np.array(Image.open(self.eo_paths[index]).convert('RGB'))
        SAR = np.array(Image.open(self.sar_paths[index]).convert('RGB'))

        transform = _build_transform(self.split, self.crop_size,
                                     self.hflip, self.vflip, self.rot)
        augmented = transform(image=SAR, image2=EO)
        return augmented['image'], augmented['image2']

    def __len__(self):
        return len(self.eo_paths)


class SAROptFeeder(Dataset):
    """SAR2Opt dataset (https://ieeexplore.ieee.org/document/9779739)."""

    def __init__(self, dataroot, listroot, load_size=600, crop_size=512,
                 hflip=False, vflip=False, rot=False):
        eo_paths = _read_path_list(listroot)
        self.eo_paths = [os.path.join(dataroot, p) for p in eo_paths]

        self.split = _split_from_listroot(listroot)
        if self.split == 'train':
            self.sar_paths = [p.replace('trainB', 'trainA') for p in self.eo_paths]
        else:
            self.sar_paths = [p.replace('testB', 'testA') for p in self.eo_paths]

        self.load_size = load_size
        self.crop_size = crop_size
        assert self.load_size >= self.crop_size

        self.hflip, self.vflip, self.rot = hflip, vflip, rot

    def __getitem__(self, index):
        EO = np.array(Image.open(self.eo_paths[index]).convert('RGB'))
        SAR = np.array(Image.open(self.sar_paths[index]).convert('RGB'))

        transform = _build_transform(self.split, self.crop_size,
                                     self.hflip, self.vflip, self.rot)
        augmented = transform(image=SAR, image2=EO)
        return augmented['image'], augmented['image2']

    def __len__(self):
        return len(self.eo_paths)


class StellarFeeder(Dataset):
    """Stellar-Vision dataset (proprietary).

    Expects the directory layout::

        data_dir/
            SAR/image/*.png|*.npy
            EO/image/*.png|*.npy

    SAR and EO frames are matched by sorted file name. Images are returned in
    the [-1, 1] range without additional augmentation.
    """

    def __init__(self, data_dir):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.sar_images_dir = os.path.join(data_dir, 'SAR/image')
        self.eo_images_dir = os.path.join(data_dir, 'EO/image')

        self.eo_image_fnames = self._get_fnames(self.eo_images_dir, supported_ext)
        self.sar_image_fnames = self._get_fnames(self.sar_images_dir, supported_ext)

    def _get_fnames(self, directory, supported_ext):
        fnames = {
            os.path.relpath(os.path.join(root, fname), start=directory)
            for root, _dirs, files in os.walk(directory) for fname in files
        }
        return sorted(fname for fname in fnames if self._file_ext(fname) in supported_ext)

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def _load_image(self, directory, fname):
        path = os.path.join(directory, fname)
        ext = self._file_ext(fname)
        if ext == '.npy':
            img = np.load(path)
            img = img.reshape(-1, *img.shape[-2:])
        elif ext == '.png' and pyspng is not None:
            with open(path, 'rb') as f:
                img = pyspng.load(f.read())
            img = img.reshape(*img.shape[:2], -1).transpose(2, 0, 1)
        else:
            img = np.array(PIL.Image.open(path).convert('RGB'))
            img = img.reshape(*img.shape[:2], -1).transpose(2, 0, 1)
        return torch.from_numpy(img)

    def __len__(self):
        assert len(self.eo_image_fnames) == len(self.sar_image_fnames), \
            "Number of EO files and SAR files should be the same"
        return len(self.eo_image_fnames)

    def __getitem__(self, idx):
        eo_image = self._load_image(self.eo_images_dir, self.eo_image_fnames[idx])
        sar_image = self._load_image(self.sar_images_dir, self.sar_image_fnames[idx])
        # Normalize uint8 [0, 255] -> float [-1, 1].
        return sar_image / 255.0 * 2 - 1, eo_image / 255.0 * 2 - 1
