import os
import time
import torch

from utils import log_info, get_time_ttl_and_eta, calc_fid
from .diffusion import Diffusion

import torchvision.utils as tvu

class DiffusionSampling(Diffusion):
    def __init__(self, args):
        super().__init__(args)
        self.model = None

    def sample(self, steps=10):
        args, config = self.args, self.config
        model = self.load_ckpt(args.sample_ckpt_path, eval_mode=True)
        self.model = model

        sample_count = args.sample_count
        log_info(f"sample_fid(self, {type(model).__name__})...")
        log_info(f"  args.sample_output_dir : {self.args.sample_output_dir}")
        log_info(f"  num_timesteps          : {self.num_timesteps}")
        log_info(f"  steps                  : {steps}")
        log_info(f"  sample_count           : {sample_count}")
        b_sz = self.args.sample_batch_size or config.sampling.batch_size
        b_cnt = (sample_count - 1) // b_sz + 1  # get the ceiling
        log_info(f"  batch_size             : {b_sz}")
        log_info(f"  batch_cnt              : {b_cnt}")
        log_info(f"  Generating image samples for FID evaluation")
        time_start = time.time()
        with torch.no_grad():
            for b_idx in range(b_cnt):
                n = b_sz if b_idx + 1 < b_cnt else sample_count - b_idx * b_sz
                log_info(f"round: {b_idx}/{b_cnt}. to generate: {n}")
                x_t = torch.randn(  # normal distribution with mean 0 and variance 1
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                x0 = self.sample_batch(x_t, steps, b_idx=b_idx)
                self.save_images(x0, b_idx, b_sz, time_start, b_cnt)
            # for b_idx
        # with

    def sample_batch(self, x_t, steps, b_idx=-1):
        skip = self.num_timesteps // steps
        seq = range(0, self.num_timesteps, skip)
        seq = list(reversed(seq))
        x = self.generalized_steps(x_t, seq, self.model, b_idx=b_idx)
        return x

    def generalized_steps(self, x_T, seq, model, b_idx=-1):
        """
        Original paper: Denoising Diffusion Implicit Models. ICLR. 2021
        :param x_T: x_T in formula; it has initial Gaussian Noise
        :param seq:    timestep t sequence
        :param model:
        :param b_idx:  batch index
        :return:
        """
        msg = f"diffusion::seq=[{seq[0]}~{seq[-1]}], len={len(seq)}"
        b_sz = len(x_T)
        xt = x_T
        seq2 = seq[1:] + [-1]
        with torch.no_grad():
            for i, j in zip(seq, seq2):
                at = self.alphas_cumprod[i+1] # alpha_bar_t
                aq = self.alphas_cumprod[j+1] # alpha_bar_{t-1}
                mt = at / aq
                t = (torch.ones(b_sz, device=self.device) * i)
                et = model(xt, t) # epsilon_t
                if b_idx == 0:
                    log_info(f"Diffusion::generalized_steps() {msg}; ts={i:03d}, ab:{at:.6f}")
                xt_next = (xt - (1 - at).sqrt() * et) / mt.sqrt() + (1 - aq).sqrt() * et
                xt = xt_next
            # for
        # with
        return xt

    def save_images(self, x, b_idx, b_sz, time_start=None, b_cnt=None, img_dir=None):
        x = self.inverse_data_transform(x)
        img_cnt = len(x)
        if img_dir is None: img_dir = self.args.sample_output_dir
        if not os.path.exists(img_dir):
            log_info(f"os.makedirs({img_dir})")
            os.makedirs(img_dir)
        img_path = None
        for i in range(img_cnt):
            img_id = b_idx * b_sz + i
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x[i], img_path)
        if time_start is not None and b_cnt is not None:
            elp, eta = get_time_ttl_and_eta(time_start, b_idx+1, b_cnt)
            log_info(f"B{b_idx:2d}/{b_cnt}: saved {img_cnt} images: {img_path}. elp:{elp}, eta:{eta}")

    def inverse_data_transform(self, X):
        config = self.config
        if config.data.logit_transform:
            X = torch.sigmoid(X)
        elif config.data.rescaled:
            X = (X + 1.0) / 2.0
        return torch.clamp(X, 0.0, 1.0)

    def sample_all(self):
        args, config = self.args, self.config
        steps_arr = args.sample_steps_arr
        basename = os.path.basename(args.sample_ckpt_path)
        stem, ext = os.path.splitext(basename)
        log_info(f"DiffusionSampling::sample_all()")
        log_info(f"  steps_arr   : {steps_arr}")
        result_file = f"./sample_all_dm_{stem}.txt"
        res_arr = []
        for steps in steps_arr:
            runner = DiffusionSampling(args)
            runner.sample(steps=steps)
            del runner  # delete the GPU memory. And can calculate FID
            torch.cuda.empty_cache()
            log_info(f"sleep 2 seconds to empty the GPU cache. . .")
            time.sleep(2)
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

# class
