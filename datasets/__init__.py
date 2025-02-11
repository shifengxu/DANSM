import os
import torchvision.transforms as T

from datasets.ImageDataset import ImageDataset
from datasets.lsun import LSUN
from datasets.cifar import CIFAR10


def get_train_test_datasets(args, config):
    if config.data.dataset == "LSUN":
        train_folder = "{}_train".format(config.data.category)
        val_folder = "{}_val".format(config.data.category)
        root_dir = os.path.join(args.data_dir, "lsun")
        if config.data.random_flip:
            train_tfm = T.Compose([
                T.Resize(config.data.image_size),
                T.CenterCrop(config.data.image_size),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
            ])
        else:
            train_tfm = T.Compose([
                T.Resize(config.data.image_size),
                T.CenterCrop(config.data.image_size),
                T.ToTensor(),
            ])
        test_tfm = T.Compose([
            T.Resize(config.data.image_size),
            T.CenterCrop(config.data.image_size),
            T.ToTensor(),
        ])
        train_dataset = LSUN(root_dir, classes=[train_folder], transform=train_tfm)
        test_dataset  = LSUN(root_dir, classes=[val_folder], transform=test_tfm)
    elif config.data.dataset == "LSUN2":
        train_folder = "{}_train".format(config.data.category)
        val_folder = "{}_val".format(config.data.category)
        root_dir = os.path.join(args.data_dir, "lsun")
        if config.data.random_flip:
            train_tfm = T.Compose([T.RandomHorizontalFlip(p=0.5), T.ToTensor()])
        else:
            train_tfm = T.Compose([T.ToTensor()])
        train_dataset = ImageDataset(root_dir, classes=[train_folder], transform=train_tfm)
        test_dataset  = ImageDataset(root_dir, classes=[val_folder], transform=T.Compose([T.ToTensor()]))
    elif config.data.dataset == "CIFAR10":
        if config.data.random_flip:
            train_tfm = T.Compose([T.RandomHorizontalFlip(p=0.5), T.ToTensor()])
        else:
            train_tfm = T.Compose([T.ToTensor()])
        test_tfm = T.Compose([T.ToTensor()])
        dir1 = os.path.join(args.data_dir, "datasets", "cifar10")
        dir2 = os.path.join(args.data_dir, "datasets", "cifar10_test")
        train_dataset = CIFAR10(dir1, train=True, download=True, transform=train_tfm)
        test_dataset = CIFAR10(dir2, train=False, download=True, transform=test_tfm)
    elif config.data.dataset == 'FFHQ' or config.data.dataset == 'IMAGENET':
        root_dir = args.data_dir
        tfm1 = T.Compose([T.Resize(config.data.image_size), T.RandomHorizontalFlip(p=0.5), T.ToTensor()])
        tfm2 = T.Compose([T.Resize(config.data.image_size), T.ToTensor()])
        train_tfm = tfm1 if config.data.random_flip else tfm2
        train_dataset = ImageDataset(root_dir, classes=None, transform=train_tfm)
        test_dataset = ImageDataset(root_dir, classes=None, transform=tfm2)
    else:
        raise ValueError(f"Unknown config.data.dataset: {config.data.dataset}")

    return train_dataset, test_dataset

def data_scaler(config, x):
    if config.data.centered:
        # Rescale to [-1, 1]
        return x * 2. - 1.
    else:
        return x


def data_inverse_scaler(config, x):
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return (x + 1.) / 2.
    else:
        return x

def load_samples_noises_from_batches_in_dir(data_dir, device):
    if not os.path.exists(data_dir):
        raise ValueError(f"Dir not exist: {data_dir}")

    import torch
    dir_list = os.listdir(data_dir)
    tmp_list = [d for d in dir_list if "sample" in d]
    if len(tmp_list) == 0: raise ValueError(f"Not found image or latent sub-folder: {data_dir}")
    if len(tmp_list) > 1: raise ValueError(f"Found multiple image or latent sub-folder: {data_dir}")
    image_dir = os.path.join(data_dir, tmp_list[0])
    tmp_list = [d for d in dir_list if "noise" in d]
    if len(tmp_list) == 0: raise ValueError(f"Not found noise sub-folder: {data_dir}")
    if len(tmp_list) > 1: raise ValueError(f"Found multiple noise sub-folder: {data_dir}")
    noise_dir = os.path.join(data_dir, tmp_list[0])
    print(f"load_samples_noises_from_batches_in_dir() data_dir : {data_dir}")
    print(f"load_samples_noises_from_batches_in_dir() image_dir: {image_dir}")
    print(f"load_samples_noises_from_batches_in_dir() noise_dir: {noise_dir}")

    file_list = os.listdir(image_dir)
    file_list.sort()
    print(f"load_samples_noises_from_batches_in_dir() image files cnt: {len(file_list)}")
    print(f"load_samples_noises_from_batches_in_dir() image files[0] : {file_list[0]}")
    print(f"load_samples_noises_from_batches_in_dir() image files[-1]: {file_list[-1]}")
    image_arr = []
    print(f"load_samples_noises_from_batches_in_dir() load image files...")
    for f in file_list:
        f_path = os.path.join(image_dir, f)
        img = torch.load(f_path, map_location=device)
        image_arr.append(img)
    print(f"load_samples_noises_from_batches_in_dir() load image files...Done")
    images = torch.concat(image_arr, dim=0)

    file_list = os.listdir(noise_dir)
    file_list.sort()
    print(f"load_samples_noises_from_batches_in_dir() noise files cnt: {len(file_list)}")
    print(f"load_samples_noises_from_batches_in_dir() noise files[0] : {file_list[0]}")
    print(f"load_samples_noises_from_batches_in_dir() noise files[-1]: {file_list[-1]}")
    noise_arr = []
    print(f"load_samples_noises_from_batches_in_dir() load noise files...")
    for f in file_list:
        f_path = os.path.join(noise_dir, f)
        tmp = torch.load(f_path, map_location=device)
        noise_arr.append(tmp)
    print(f"load_samples_noises_from_batches_in_dir() load noise files...Done")
    noises = torch.concat(noise_arr, dim=0)
    print(f"load_samples_noises_from_batches_in_dir() samples count: {len(images)}")
    print(f"load_samples_noises_from_batches_in_dir() noises count : {len(noises)}")
    return images, noises
