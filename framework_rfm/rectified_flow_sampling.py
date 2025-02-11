"""
Distance-Aware Training: sampling
"""
import os
import time
import torch
import torchvision.utils as tvu
from torch import Tensor

import utils
from .rectified_flow_base import RectifiedFlowBase
from datasets import data_inverse_scaler
from utils import log_info as log_info


class RectifiedFlowSampling(RectifiedFlowBase):
    def __init__(self, args):
        super().__init__(args)
        self.model = None

    def sample(self, sample_steps=10, init_ts=None):
        args, config = self.args, self.config
        if init_ts is None: init_ts = args.sample_init_ts_arr[0]
        log_info(f"RectifiedFlowSampling::sample()")
        log_info(f"  sample_steps: {sample_steps}")
        log_info(f"  init_ts     : {init_ts}")
        self.model = self.load_ckpt(args.sample_ckpt_path, eval_mode=True, only_return_model=True)
        img_cnt = args.sample_count
        b_sz = args.sample_batch_size
        b_cnt = img_cnt // b_sz
        if b_cnt * b_sz < img_cnt:
            b_cnt += 1
        c_data = config.data
        c, h, w = c_data.num_channels, c_data.image_size, c_data.image_size
        log_info(f"  img_cnt: {img_cnt}")
        log_info(f"  b_sz   : {b_sz}")
        log_info(f"  b_cnt  : {b_cnt}")
        log_info(f"  c      : {c}")
        log_info(f"  h      : {h}")
        log_info(f"  w      : {w}")
        log_info(f"  steps  : {sample_steps}")
        time_start = time.time()
        with torch.no_grad():
            for b_idx in range(b_cnt):
                n = img_cnt - b_idx * b_sz if b_idx == b_cnt - 1 else b_sz
                x1 = torch.randn(n, c, h, w, requires_grad=False, device=self.device)
                x0 = self.sample_batch(x1, sample_steps, init_ts=init_ts, b_idx=b_idx)
                self.save_images(x0, b_idx, b_sz, time_start, b_cnt)
            # for
        # with
        return 0

    def sample_batch(self, x1: Tensor, sample_steps, init_ts=0., b_idx=-1):
        """
        sample a batch, starting from x1.
        From losses.py, where z0 is Gaussian noise:
            # standard rectified flow loss
            t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
            t_expand = t.view(-1, 1, 1, 1).repeat(1, batch.shape[1], batch.shape[2], batch.shape[3])
            perturbed_data = t_expand * batch + (1.-t_expand) * z0
            target = batch - z0
            model_fn = mutils.get_model_fn(model, train=train)
            score = model_fn(perturbed_data, t*999)
        """
        b_sz = x1.size(0)
        dt = 1. / sample_steps
        x = x1
        for i in range(sample_steps):
            num_t = i / sample_steps * (1.0 - self.eps) + self.eps
            num_t = init_ts + (1 - init_ts) * num_t
            if b_idx == 0:
                log_info(f"RectifiedFlowSampling::sample_batch() i:{i:2d}, num_t:{num_t:.6f}")
            t = torch.ones(b_sz, requires_grad=False, device=self.device) * num_t
            pred = self.model(x, t * 999)
            x = x + pred * dt
        return x

    def save_images(self, x0, b_idx, b_sz, time_start=None, b_cnt=None, img_dir=None):
        """ save x0 """
        x0 = data_inverse_scaler(self.config, x0)
        img_cnt = len(x0)
        img_dir = img_dir or self.args.sample_output_dir
        if not os.path.exists(img_dir):
            log_info(f"os.makedirs({img_dir})")
            os.makedirs(img_dir)
        img_path = None
        for i in range(img_cnt):
            img_id = b_idx * b_sz + i
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x0[i], img_path)
        if time_start is not None and b_cnt is not None:
            elp, eta = utils.get_time_ttl_and_eta(time_start, b_idx+1, b_cnt)
            log_info(f"saved {img_cnt}. B:{b_idx:3d}/{b_cnt}. {img_path}. elp:{elp}, eta:{eta}")

# class
