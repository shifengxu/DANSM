"""
Rectified Flow sampling
"""
import os.path
import time
import torch
import torch.utils.data as tu_data
import torchvision.transforms as T

from .rectified_flow_base import RectifiedFlowBase
from datasets import get_train_test_datasets, data_scaler
from datasets.ImageNoiseDataset import ImageNoiseDataset
from datasets.ImageNoiseNumpyDataset import ImageNoiseNumpyDataset
from framework_rfm.rectified_flow_sampling import RectifiedFlowSampling
from utils import log_info, calc_fid, get_time_ttl_and_eta

class RectifiedFlowMiscellaneous(RectifiedFlowBase):
    def __init__(self, args):
        super().__init__(args)
        self.model = None
        log_info(f"RectifiedFlowMiscellaneous()")
        log_info(f"  eps        : {self.eps}")
        log_info(f"  device     : {self.device}")

    def get_data_loaders(self, train_shuffle=True, test_shuffle=False):
        args, config = self.args, self.config
        batch_size = args.batch_size
        num_workers = 4
        train_ds, test_ds = get_train_test_datasets(args, config)
        train_loader = tu_data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=train_shuffle,
            num_workers=num_workers,
        )
        log_info(f"train dataset and data loader:")
        log_info(f"  root       : {train_ds.root}")
        log_info(f"  split      : {train_ds.split}") if hasattr(train_ds, 'split') else None
        log_info(f"  classes    : {train_ds.classes}") if hasattr(train_ds, 'classes') else None
        log_info(f"  len        : {len(train_ds)}")
        log_info(f"  batch_cnt  : {len(train_loader)}")
        log_info(f"  batch_size : {batch_size}")
        log_info(f"  shuffle    : {train_shuffle}")
        log_info(f"  num_workers: {num_workers}")

        test_loader = tu_data.DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=test_shuffle,
            num_workers=num_workers,
        )
        log_info(f"test dataset and loader:")
        log_info(f"  root       : {test_ds.root}")
        log_info(f"  split      : {test_ds.split}") if hasattr(test_ds, 'split') else None
        log_info(f"  classes    : {test_ds.classes}") if hasattr(test_ds, 'classes') else None
        log_info(f"  len        : {len(test_ds)}")
        log_info(f"  batch_cnt  : {len(test_loader)}")
        log_info(f"  batch_size : {batch_size}")
        log_info(f"  shuffle    : {test_shuffle}")
        log_info(f"  num_workers: {num_workers}")
        return train_loader, test_loader

    def save_image_one_by_one(self):
        """ Load images from dataset or data-loader, and save them one by one """
        args, config = self.args, self.config
        train_loader, test_loader = self.get_data_loaders(train_shuffle=False)
        import torchvision.utils as tvu
        # config.data.category is like: "church_outdoor"
        root_dir = os.path.join("../ddim/exp/datasets/lsun", f"{config.data.category}_train")
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
            log_info(f"os.makedirs({root_dir})")
        counter = 0
        for x_batch, y_batch in train_loader:
            for x, y in zip(x_batch, y_batch):
                path = os.path.join(root_dir, f"{counter:06d}.png")
                tvu.save_image(x, path)
                counter += 1
                if counter % 100 == 0: log_info(path)
            # for
        # for
        root_dir = os.path.join("../ddim/exp/datasets/lsun", f"{config.data.category}_val")
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
            log_info(f"os.makedirs({root_dir})")
        counter = 0
        for x_batch, y_batch in test_loader:
            for x, y in zip(x_batch, y_batch):
                path = os.path.join(root_dir, f"{counter:06d}.png")
                tvu.save_image(x, path)
                counter += 1
                if counter % 100 == 0: log_info(path)
            # for
        # for
        return True

    def run_delta_between_prediction_and_ground_truth(self):
        """ get the statistic info of the delta between prediction and ground_truth """
        args = self.args
        train_loader, test_loader = self.get_data_loaders()
        self.model = self.load_ckpt(args.resume_ckpt_path, eval_mode=True, only_return_model=True)

        # ts_list = [0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
        ts_list = [99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
        loc_list = [
            (0, 0, 0), (0, 128, 128), (0, 255, 255),
            (1, 0, 0), (1, 128, 128), (1, 255, 255),
            (2, 0, 0), (2, 128, 128), (2, 255, 255),
        ]  # channel-height-width dimension
        b_cnt = len(train_loader)   # batch count
        b_sz = self.args.batch_size or self.config.training.batch_size
        ts_cnt = len(ts_list)
        log_info(f"RectifiedFlowMiscellaneous::run_delta_between_prediction_and_ground_truth()")
        log_info(f"  b_cnt   : {b_cnt}")
        log_info(f"  b_sz    : {b_sz}")
        log_info(f"  ts_cnt  : {ts_cnt}")
        log_info(f"  ts_list : {ts_list}")
        log_info(f"  loc_list: {loc_list}")
        b_ts_total = b_cnt * ts_cnt
        time_start = time.time()
        loc_cnt = len(loc_list)
        with torch.no_grad():
            for ts_idx, ts in enumerate(ts_list):
                train_loader, test_loader = self.get_data_loaders()
                delta_lst_lst = []
                for j in range(loc_cnt): delta_lst_lst.append([])
                for b_idx, (x, y) in enumerate(train_loader):
                    x = x.to(self.device)
                    x = data_scaler(self.config, x)
                    b_sz, ch, h, w = x.shape  # batch_size, channel, height, width
                    t = torch.ones((b_sz,), dtype=torch.float, device=self.device)
                    t = t * ts / 1000
                    t = torch.mul(t, 1.0 - self.eps)
                    t = torch.add(t, self.eps)
                    z0 = torch.randn(b_sz, ch, h, w, device=self.device)
                    t_expand = t.view(-1, 1, 1, 1)
                    perturbed_data = x * t_expand + z0 * (1. - t_expand)
                    target = x - z0
                    predict = self.model(perturbed_data, t * 999)
                    delta = predict - target
                    for i in range(b_sz):
                        for j in range(loc_cnt):
                            c, h, w = loc_list[j]
                            delta_lst_lst[j].append(delta[i][c][h][w].item())
                            # the ".item()" above is important.
                            # Else, the GPU memory will accumulate, and finally out of memory.
                        # for
                    # for
                    elp, eta = get_time_ttl_and_eta(time_start, ts_idx * b_cnt + b_idx, b_ts_total)
                    log_info(f"B{b_idx:03d}/{b_cnt}.ts:{ts:03d}, t:{t[0]:.4f}. elp:{elp}, eta:{eta}")
                # for loader
                for delta_lst, (c, h, w) in zip(delta_lst_lst, loc_list):
                    f_path = f"./output5_lostats/ts{ts:03d}_dim{c}_{h:03d}_{w:03d}.txt"
                    log_info(f"write {len(delta_lst)} lines to {f_path}")
                    with open(f_path, 'w') as fptr:
                        [fptr.write(f"{d:10.7f}\r\n") for d in delta_lst]
                    # with
                # for
                del delta_lst_lst
                del train_loader
                del test_loader
                torch.cuda.empty_cache()
            # for ts
        # with

    def compare_distance(self):
        # compare distance between noise and image
        args = self.args
        data_dir = args.data_dir
        seed = 123
        batch_size = args.batch_size
        if args.config == 'cifar10':
            seed_dir = os.path.join(data_dir, str(seed))
            ds = ImageNoiseNumpyDataset(seed_dir)
        else:
            image_dir = os.path.join(data_dir, f"{seed}_image")
            noise_dir = os.path.join(data_dir, f"{seed}_noise")
            tfm = T.Compose([T.ToTensor()])
            ds = ImageNoiseDataset(image_dir, noise_dir, image_transform=tfm)
        d_loader = tu_data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
        b_cnt = len(d_loader)
        log_info(f"config    : {args.config}")
        log_info(f"data_dir  : {data_dir}")
        log_info(f"batch_size: {batch_size}")
        log_info(f"batch_cnt : {b_cnt}")
        log_info(f"device    : {self.device}")
        match_cnt, total_cnt = 0, 0
        for b_idx, (image_batch, noise_batch) in enumerate(d_loader):
            image_batch, noise_batch = image_batch.to(self.device), noise_batch.to(self.device)
            image_batch = image_batch * 2 - 1
            min_pair_dist = (image_batch - noise_batch).square().mean(dim=(1, 2, 3))
            for i_idx, image in enumerate(image_batch):
                image = image.unsqueeze(0)
                dist_arr = (image - noise_batch).square().mean(dim=(1, 2, 3))
                result = torch.min(dist_arr, dim=0, keepdim=False)
                min_dist, min_idx = result.values, result.indices
                total_cnt += 1
                if min_idx == i_idx:
                    match_cnt += 1
                else:
                    log_info(f"{i_idx:3d}|{min_idx:3d}: {min_pair_dist[i_idx]:.4f} vs {min_dist:.4f}")
            # for
            log_info(f"B{b_idx:3d}/{b_cnt}: match/total: {match_cnt}/{total_cnt} ----------")
        # for

    def sample_all_ts_and_step(self):
        args, config = self.args, self.config
        steps_arr = args.sample_steps_arr
        init_ts_arr = args.sample_init_ts_arr
        basename = os.path.basename(args.sample_ckpt_path)
        stem, ext = os.path.splitext(basename)
        result_file = f"./sample_all_dat_{stem}.txt"
        log_info(f"main_pytorch->sample_all()")
        log_info(f"  init_ts_arr : {init_ts_arr}")
        log_info(f"  steps_arr   : {steps_arr}")
        log_info(f"  result_file : {result_file}")
        for init_ts in init_ts_arr:
            result_file = f"./sample_all_dat_{stem}_initTS{init_ts:.3f}.txt"
            res_arr = []
            for steps in steps_arr:
                runner = RectifiedFlowSampling(args)
                runner.sample(sample_steps=steps, init_ts=init_ts)
                del runner  # delete the GPU memory. And can calculate FID
                torch.cuda.empty_cache()
                log_info(f"sleep 5 seconds to empty the GPU cache. . .")
                time.sleep(5)
                fid = calc_fid(args.gpu_ids[0], True, input1=args.fid_input1, input2=args.sample_output_dir)
                msg = f"FID: {fid:7.3f}. steps:{steps:2d}"
                res_arr.append(msg)
                with open(result_file, 'w') as fptr: [fptr.write(f"{m}\n") for m in res_arr]
                log_info(msg)
                log_info("")
                log_info("")
                log_info("")
            # for
            [log_info(f"{msg}") for msg in res_arr]
        # for

    @staticmethod
    def resize_images():
        import os
        import torchvision.transforms as transforms
        import torchvision.utils as tvu
        from PIL import Image

        dir_src = "../download_dataset/ffhq_512x512_v1"
        dir_dst = "../download_dataset/ffhq_256x256_v1"
        name_list = os.listdir(dir_src)
        name_list.sort()
        f_cnt = len(name_list)
        print(f"dir_src    : {dir_src}")
        print(f"dir_dst    : {dir_dst}")
        print(f"found files: {f_cnt}")

        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), ])
        if not os.path.exists(dir_dst):
            print(f"os.makedirs({dir_dst})")
            os.makedirs(dir_dst)
        for idx, f_name in enumerate(name_list):
            fpath_in = os.path.join(dir_src, f_name)
            nm, ext = os.path.splitext(f_name)
            fpath_out = os.path.join(dir_dst, f"{nm}.png")
            if idx % 1000 == 0 or idx + 1 == f_cnt:
                print(f"{fpath_in} -> {fpath_out}")
            image = Image.open(fpath_in).convert("RGB")
            image = transform(image)
            tvu.save_image(image, fpath_out)
        # for

    # class
