import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils import log_info


class ImageNoiseDataset(Dataset):
    def __init__(self, image_dir, noise_dir, image_transform=None, noise_transform=None):
        self.image_dir = image_dir
        self.noise_dir = noise_dir
        self.image_transform = image_transform
        self.noise_transform = noise_transform
        if not os.path.exists(image_dir):
            raise ValueError(f"Path not exist: {image_dir}")
        if not os.path.isdir(image_dir):
            raise ValueError(f"Path not dir: {image_dir}")
        if not os.path.exists(noise_dir):
            raise ValueError(f"Path not exist: {noise_dir}")
        if not os.path.isdir(noise_dir):
            raise ValueError(f"Path not dir: {noise_dir}")
        image_name_arr = os.listdir(image_dir)
        image_name_arr.sort()
        self.image_name_arr = image_name_arr
        noise_name_arr = os.listdir(noise_dir)
        noise_name_arr.sort()
        self.noise_name_arr = noise_name_arr
        self.image_count = len(self.image_name_arr)
        self.noise_count = len(self.noise_name_arr)
        if self.image_count != self.noise_count:
            raise ValueError(f"image count {self.image_count} not match noise count {self.noise_count}")
        log_info(f"ImageNoiseDataset()")
        log_info(f"  image_dir  : {self.image_dir}")
        log_info(f"  noise_dir  : {self.noise_dir}")
        log_info(f"  image_count: {self.image_count}")
        log_info(f"  noise_count: {self.noise_count}")
        log_info(f"  image[0]   : {self.image_name_arr[0]}")
        log_info(f"  noise[0]   : {self.noise_name_arr[0]}")
        log_info(f"  image[-1]  : {self.image_name_arr[-1]}")
        log_info(f"  noise[-1]  : {self.noise_name_arr[-1]}")

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_name_arr[index])
        image = Image.open(img_path)
        img_rgb = image.convert("RGB")
        if self.image_transform:
            img_rgb = self.image_transform(img_rgb)
        img_rgb = np.array(img_rgb)

        noise_path = os.path.join(self.noise_dir, self.noise_name_arr[index])
        noise = np.load(noise_path)
        if self.noise_transform:
            noise = self.noise_transform(noise)
        return img_rgb, noise

    def __len__(self):
        return self.image_count
