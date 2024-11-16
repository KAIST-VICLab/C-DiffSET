import os
import random
import pickle


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def make_dataset_relpath(dir, abs_path, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                path = os.path.relpath(path, abs_path)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def make_dataset_list(dir_list, max_dataset_size=float("inf")):
    images = []
    for dir in sorted(dir_list):
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images[:min(max_dataset_size, len(images))]


def make_dataset_list_relpath(dir_list, abs_path, max_dataset_size=float("inf")):
    images = []
    for dir in sorted(dir_list):
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    path = os.path.relpath(path, abs_path)
                    images.append(path)
    return images[:min(max_dataset_size, len(images))]


def SpaceNet_split(dataroot, ratio=80):
    ''' https://arxiv.org/abs/2004.06500 '''
    SAVE_PATH = './SpaceNet_split/'
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    random.seed(2025)

    eo_dataroot = os.path.join(dataroot, 'train/AOI_11_Rotterdam/PS-RGB/')
    list = make_dataset_relpath(eo_dataroot, dataroot)
    test_list = random.sample(list, int(len(list) * (1 - ratio / 100)))
    train_list = [x for x in list if x not in test_list]

    train_list = sorted(train_list)
    test_list = sorted(test_list)

    with open(f'{SAVE_PATH}/train_eo_list_{ratio:03}.txt', 'w') as f:
        for path in train_list:
            f.write(path + '\n')

    with open(f'{SAVE_PATH}/train_eo_list_{ratio:03}.pkl', 'wb') as f:
        pickle.dump(train_list, f)

    with open(f'{SAVE_PATH}/test_eo_list_{ratio:03}.txt', 'w') as f:
        for path in test_list:
            f.write(path + '\n')

    with open(f'{SAVE_PATH}/test_eo_list_{ratio:03}.pkl', 'wb') as f:
        pickle.dump(test_list, f)


def QXS_split(dataroot, ratio=80):
    ''' https://arxiv.org/abs/2103.08259 '''
    SAVE_PATH = './QXS_split/'
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    random.seed(2025)

    eo_dataroot = os.path.join(dataroot, 'opt_256_oc_0.2')
    list = make_dataset_relpath(eo_dataroot, dataroot)
    test_list = random.sample(list, int(len(list) * (1 - ratio / 100)))
    train_list = [x for x in list if x not in test_list]

    train_list = sorted(train_list)
    test_list = sorted(test_list)

    with open(f'{SAVE_PATH}/train_eo_list_{ratio:03}.txt', 'w') as f:
        for path in train_list:
            f.write(path + '\n')

    with open(f'{SAVE_PATH}/train_eo_list_{ratio:03}.pkl', 'wb') as f:
        pickle.dump(train_list, f)

    with open(f'{SAVE_PATH}/test_eo_list_{ratio:03}.txt', 'w') as f:
        for path in test_list:
            f.write(path + '\n')

    with open(f'{SAVE_PATH}/test_eo_list_{ratio:03}.pkl', 'wb') as f:
        pickle.dump(test_list, f)


def SAROpt_split(dataroot):
    ''' https://ieeexplore.ieee.org/document/9779739 '''
    SAVE_PATH = './SAROpt_split/'
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    train_dataroot = os.path.join(dataroot, 'trainB')
    test_dataroot = os.path.join(dataroot, 'testB')

    train_list = make_dataset_relpath(train_dataroot, dataroot)
    test_list = make_dataset_relpath(test_dataroot, dataroot)

    train_list = sorted(train_list)
    test_list = sorted(test_list)

    with open(f'{SAVE_PATH}/train_eo_list.txt', 'w') as f:
        for path in train_list:
            f.write(path + '\n')

    with open(f'{SAVE_PATH}/train_eo_list.pkl', 'wb') as f:
        pickle.dump(train_list, f)

    with open(f'{SAVE_PATH}/test_eo_list.txt', 'w') as f:
        for path in test_list:
            f.write(path + '\n')

    with open(f'{SAVE_PATH}/test_eo_list.pkl', 'wb') as f:
        pickle.dump(test_list, f)


SpaceNet_split('./SpaceNet6', ratio=80)
QXS_split('./QXS_SAROPT', ratio=80)
SAROpt_split('./SAR2Opt')
