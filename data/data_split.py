"""Generate train/test split lists for each supported dataset.

Each ``*_split`` function writes both a ``.txt`` and a ``.pkl`` list of EO image
paths (relative to ``dataroot``) into ``./data/<Dataset>_split/``, which is the
location referenced by the YAML configs.

Usage::

    python data/data_split.py --dataset spacenet --dataroot /path/to/Space6 --ratio 80
    python data/data_split.py --dataset saropt   --dataroot /path/to/sar2opt
"""

import argparse
import os
import pickle
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in IMG_EXTENSIONS)


def make_dataset_relpath(directory, abs_path, max_dataset_size=float("inf")):
    """List image files under ``directory`` as paths relative to ``abs_path``."""
    assert os.path.isdir(directory), '%s is not a valid directory' % directory
    images = []
    for root, _, fnames in sorted(os.walk(directory)):
        for fname in fnames:
            if is_image_file(fname):
                images.append(os.path.relpath(os.path.join(root, fname), abs_path))
    return images[:min(max_dataset_size, len(images))]


def make_dataset_list_relpath(dir_list, abs_path, max_dataset_size=float("inf")):
    """Same as :func:`make_dataset_relpath` but over a list of directories."""
    images = []
    for directory in sorted(dir_list):
        assert os.path.isdir(directory), '%s is not a valid directory' % directory
        for root, _, fnames in sorted(os.walk(directory)):
            for fname in fnames:
                if is_image_file(fname):
                    images.append(os.path.relpath(os.path.join(root, fname), abs_path))
    return images[:min(max_dataset_size, len(images))]


def _save_split(save_path, train_list, test_list, suffix=''):
    """Write train/test lists as both .txt and .pkl."""
    os.makedirs(save_path, exist_ok=True)
    for name, paths in (('train', sorted(train_list)), ('test', sorted(test_list))):
        base = os.path.join(save_path, f'{name}_eo_list{suffix}')
        with open(base + '.txt', 'w') as f:
            f.write('\n'.join(paths) + '\n')
        with open(base + '.pkl', 'wb') as f:
            pickle.dump(paths, f)


def SENMS_split(dataroot, ratio=80):
    """SEN1-2 (https://arxiv.org/abs/1906.07789)."""
    random.seed(2024)
    train_list, test_list = [], []
    for season in ['ROIs1158_spring', 'ROIs1868_summer', 'ROIs1970_fall', 'ROIs2017_winter']:
        season_dir = os.path.join(dataroot, season)
        paths = [os.path.join(season_dir, x) for x in os.listdir(season_dir) if 's2_' in x]
        test_list += random.sample(paths, int(len(paths) * (1 - ratio / 100)))
        train_list += [x for x in paths if x not in test_list]

    train_eo = make_dataset_list_relpath(sorted(train_list), dataroot)
    test_eo = make_dataset_list_relpath(sorted(test_list), dataroot)
    _save_split('./data/SENMS_split/', train_eo, test_eo, suffix=f'_{ratio:03}')


def SpaceNet_split(dataroot, ratio=80):
    """SpaceNet6 (https://arxiv.org/abs/2004.06500)."""
    random.seed(2024)
    eo_dataroot = os.path.join(dataroot, 'train/AOI_11_Rotterdam/PS-RGB/')
    paths = make_dataset_relpath(eo_dataroot, dataroot)
    test_list = random.sample(paths, int(len(paths) * (1 - ratio / 100)))
    train_list = [x for x in paths if x not in test_list]
    _save_split('./data/SpaceNet_split/', train_list, test_list, suffix=f'_{ratio:03}')


def QXS_split(dataroot, ratio=80):
    """QXS-SAROPT (https://arxiv.org/abs/2103.08259)."""
    random.seed(2024)
    eo_dataroot = os.path.join(dataroot, 'opt_256_oc_0.2')
    paths = make_dataset_relpath(eo_dataroot, dataroot)
    test_list = random.sample(paths, int(len(paths) * (1 - ratio / 100)))
    train_list = [x for x in paths if x not in test_list]
    _save_split('./data/QXS_split/', train_list, test_list, suffix=f'_{ratio:03}')


def SAROpt_split(dataroot):
    """SAR2Opt (https://ieeexplore.ieee.org/document/9779739)."""
    train_list = make_dataset_relpath(os.path.join(dataroot, 'trainB'), dataroot)
    test_list = make_dataset_relpath(os.path.join(dataroot, 'testB'), dataroot)
    _save_split('./data/SAROpt_split/', train_list, test_list)


def main():
    parser = argparse.ArgumentParser(description='Create train/test split lists.')
    parser.add_argument('--dataset', required=True,
                        choices=['senms', 'spacenet', 'qxs', 'saropt'])
    parser.add_argument('--dataroot', required=True, help='root directory of the dataset')
    parser.add_argument('--ratio', type=int, default=80,
                        help='train percentage (ignored for saropt, which has fixed splits)')
    args = parser.parse_args()

    if args.dataset == 'senms':
        SENMS_split(args.dataroot, args.ratio)
    elif args.dataset == 'spacenet':
        SpaceNet_split(args.dataroot, args.ratio)
    elif args.dataset == 'qxs':
        QXS_split(args.dataroot, args.ratio)
    elif args.dataset == 'saropt':
        SAROpt_split(args.dataroot)
    print(f"Split lists for '{args.dataset}' written under ./data/")


if __name__ == '__main__':
    main()
