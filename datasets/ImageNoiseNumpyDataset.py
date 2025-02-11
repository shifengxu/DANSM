import os
import numpy as np
from torch.utils.data import Dataset

from utils import log_info


class ImageNoiseNumpyDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise ValueError(f"Path not exist: {data_dir}")
        if not os.path.isdir(data_dir):
            raise ValueError(f"Path not dir: {data_dir}")
        file_list = os.listdir(data_dir)
        file_list.sort()
        log_info(f"ImageNoiseNumpyDataset()")
        log_info(f"  data_dir   : {data_dir}")
        log_info(f"  file cnt   : {len(file_list)}")
        z0_arr, z1_arr = [], []
        z0_fc, z1_fc = 0, 0 # file count
        for f in file_list:
            f = str(f)
            if f.endswith("z0.npy"):
                z0 = np.load(os.path.join(data_dir, f))
                z0_arr.extend(z0)
                z0_fc += 1
            elif f.endswith("z1.npy"):
                z1 = np.load(os.path.join(data_dir, f))
                z1_arr.extend(z1)
                z1_fc += 1
        # for
        self.z0_fc = z0_fc
        self.z1_fc = z1_fc
        self.z0_arr = z0_arr
        self.z1_arr = z1_arr
        log_info(f"  z0 file cnt: {z0_fc}")
        log_info(f"  z1 file cnt: {z1_fc}")
        log_info(f"  z0 data cnt: {len(z0_arr)}")
        log_info(f"  z1 data cnt: {len(z1_arr)}")

    def __getitem__(self, index):
        # z1 is data; z0 is noise
        return self.z1_arr[index], self.z0_arr[index]

    def __len__(self):
        return len(self.z0_arr)
