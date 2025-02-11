import os
import numpy as np
from torch.utils.data import Dataset

from utils import log_info


class LatentDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        if not os.path.exists(data_dir):
            raise ValueError(f"Path not exist: {data_dir}")
        if not os.path.isdir(data_dir):
            raise ValueError(f"Path not dir: {data_dir}")
        file_name_arr = os.listdir(data_dir)
        file_name_arr.sort()
        self.file_name_arr = file_name_arr
        self.file_count = len(self.file_name_arr)
        log_info(f"LatentDataset::__init__()...")
        log_info(f"  data_dir  : {self.data_dir}")
        log_info(f"  data_count: {self.file_count}")
        log_info(f"  data[0]   : {self.file_name_arr[0]}")
        log_info(f"  data[-1]  : {self.file_name_arr[-1]}")
        log_info(f"LatentDataset::__init__()...Done")

    def __getitem__(self, index):
        f_path = os.path.join(self.data_dir, self.file_name_arr[index])
        latent = np.load(f_path)
        if self.transform:
            latent = self.transform(latent)
        return latent

    def __len__(self):
        return self.file_count

def get_data_loader_and_dataset_for_latent(data_dir, batch_size, shuffle, num_workers=4, transform=None):
    from torch.utils.data import DataLoader
    ds = LatentDataset(data_dir, transform=transform)
    loader = DataLoader(ds, batch_size, shuffle=shuffle, num_workers=num_workers)
    log_info(f"datasets.LatentDataset::get_data_loader_and_dataset_for_latent()...")
    log_info(f"  data_dir   : {data_dir}")
    log_info(f"  data cnt   : {len(ds)}")
    log_info(f"  batch_cnt  : {len(loader)}")
    log_info(f"  batch_size : {batch_size}")
    log_info(f"  shuffle    : {shuffle}")
    log_info(f"  num_workers: {num_workers}")
    log_info(f"datasets.LatentDataset::get_data_loader_and_dataset_for_latent()...Done")
    return loader, ds
