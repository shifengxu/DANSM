import os
import time
import torch
from torch.backends import cudnn
import torch.utils.data as data

from models.match_scope_manager import MatchScopeManager
from utils import log_info, get_time_ttl_and_eta, calc_fid_isc
from .diffusion_sampling import DiffusionSampling
from .losses import noise_estimation_loss2
from .diffusion import Diffusion
from datasets import get_train_test_datasets
from models.ermongroup.ermongroup_model import ErmongroupModel
from models.ermongroup.ema import EMAHelper


def data_transform(config, X):
    # _rescale_mean = None
    # _rescale_std = None
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X

class DiffusionTraining(Diffusion):
    def __init__(self, args):
        super().__init__(args)
        self.model = None
        self.optimizer = None
        self.ema_helper = None
        self.ema_rate = args.ema_rate or self.config.model.ema_rate
        self.ts_low = 0
        self.ts_high = len(self.betas)
        self.save_ckpt_eval     = args.save_ckpt_eval
        self.sample_output_dir  = args.sample_output_dir
        self.fid_input1         = args.fid_input1
        self.sample_isc_flag    = args.sample_isc_flag
        self.sampler = None
        if self.save_ckpt_eval:
            self.sampler = DiffusionSampling(args)
        self.result_arr = []
        self.ms_size = args.ms_size
        c_data = self.config.data
        c, h, w = c_data.channels, c_data.image_size, c_data.image_size
        self.ms_mgr = MatchScopeManager(self.ms_size, c, h, w, args.device)
        log_info(f"DiffusionTraining()")
        log_info(f"  resume_ckpt_path   : {args.resume_ckpt_path}")
        log_info(f"  save_ckpt_path     : {args.save_ckpt_path}")
        log_info(f"  save_ckpt_interval : {args.save_ckpt_interval}")
        log_info(f"  save_ckpt_eval     : {self.save_ckpt_eval}")
        log_info(f"  sampler            : {type(self.sampler).__name__}")
        log_info(f"  sample_output_dir  : {self.sample_output_dir}")
        log_info(f"  fid_input1         : {self.fid_input1}")
        log_info(f"  sample_isc_flag    : {self.sample_isc_flag}")
        log_info(f"  ts_low     : {self.ts_low}")
        log_info(f"  ts_high    : {self.ts_high}")
        log_info(f"  ms_size    : {self.ms_size}")
        log_info(f"  c          : {c}")
        log_info(f"  h          : {h}")
        log_info(f"  w          : {w}")
        log_info(f"  ms_mgr     : {type(self.ms_mgr).__name__}")

    def get_data_loaders(self):
        args, config = self.args, self.config
        batch_size = args.batch_size or config.training.batch_size
        dataset, test_dataset = get_train_test_datasets(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        log_info(f"train dataset and data loader:")
        log_info(f"  root       : {dataset.root}")
        log_info(f"  split      : {dataset.split}") if hasattr(dataset, 'split') else None
        log_info(f"  len        : {len(dataset)}")
        log_info(f"  batch_cnt  : {len(train_loader)}")
        log_info(f"  batch_size : {batch_size}")
        log_info(f"  shuffle    : True")
        log_info(f"  num_workers: {config.data.num_workers}")

        test_loader = data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        log_info(f"test dataset and loader:")
        log_info(f"  root          : {test_dataset.root}")
        log_info(f"  len           : {len(test_dataset)}")
        log_info(f"  batch_cnt     : {len(test_loader)}")
        log_info(f"  batch_size    : {batch_size}")
        log_info(f"  shuffle       : False")
        log_info(f"  num_workers   : {config.data.num_workers}")
        return train_loader, test_loader

    def init_models(self):
        args, config = self.args, self.config
        model, optimizer, ema_helper, cur_epoch = None, None, None, None
        if self.args.resume_ckpt_path:
            st = self.load_ckpt(args.resume_ckpt_path, eval_mode=False, only_return_model=False)
            model      = st['model']
            optimizer  = st['optimizer']
            ema_helper = st['ema_helper']
            cur_epoch  = st['cur_epoch']
        if model is None:
            model = ErmongroupModel(config)
            log_info(f"model from scratch: {type(model).__name__} ===================")
            log_info(f"  config type: {config.model.type}")
            log_info(f"  ema_rate   : {self.ema_rate}")
            log_info(f"  model.to({self.device})")
            model.to(self.device)
            log_info(f"  torch.nn.DataParallel(device_ids={args.gpu_ids})")
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        if optimizer is None:
            optimizer = self.get_optimizer(model.parameters(), args.lr)
            log_info(f"  optimizer: {type(optimizer).__name__}, lr={args.lr}")
        if ema_helper is None:
            ema_helper = EMAHelper(model, mu=self.ema_rate)
            log_info(f"  ema_helper: EMAHelper(mu={self.ema_rate})")
        if cur_epoch is None:
            cur_epoch = 0
        self.model      = model
        self.optimizer  = optimizer
        self.ema_helper = ema_helper

        cudnn.benchmark = True
        return cur_epoch

    def train(self):
        args, config = self.args, self.config
        train_loader, test_loader = self.get_data_loaders() # data loaders
        start_epoch = self.init_models()
        log_interval = args.log_interval
        save_int = args.save_ckpt_interval
        e_cnt = args.n_epochs
        b_cnt = len(train_loader)               # batch count
        eb_cnt = (e_cnt - start_epoch) * b_cnt  # epoch * batch
        self.model.train()
        log_info(f"DiffusionTraining::train()...")
        log_info(f"log_interval  : {log_interval}")
        log_info(f"start_epoch   : {start_epoch}")
        log_info(f"save_interval : {save_int}")
        log_info(f"e_cnt         : {e_cnt}")
        log_info(f"b_cnt         : {b_cnt}")
        log_info(f"eb_cnt        : {eb_cnt}")
        cd = config.data
        ch, h, w = cd.channels, cd.image_size, cd.image_size,
        log_info(f"ch            : {ch}")
        log_info(f"h             : {h}")
        log_info(f"w             : {w}")
        log_info(f"lr            : {args.lr}")
        log_info(f"ema_rate      : {args.ema_rate}")
        data_start = time.time()
        eb_counter = 0
        for epoch in range(start_epoch+1, e_cnt+1):
            log_info(f"Epoch {epoch}/{e_cnt} ----------")
            loss_ttl, loss_cnt = 0., 0
            d_old_sum, d_new_sum = 0., 0.   # distance sum for old and new
            for i, (x, y) in enumerate(train_loader):
                eb_counter += 1
                x = x.to(self.device)
                x = data_transform(self.config, x)
                loss, decay, dist_old, dist_new = self.train_batch(x)
                loss_ttl += loss
                loss_cnt += 1
                d_old_sum += dist_old
                d_new_sum += dist_new

                if i % log_interval == 0 or i == b_cnt - 1:
                    elp, eta = get_time_ttl_and_eta(data_start, eb_counter, eb_cnt)
                    do_avg, dn_avg = d_old_sum / loss_cnt, d_new_sum / loss_cnt
                    loss_str = f"loss:{loss:8.4f}; ema:{decay:.4f}; dist:{do_avg:.4f}~{dn_avg:.4f}"
                    log_info(f"E{epoch}.B{i:03d}/{b_cnt} {loss_str}. elp:{elp}, eta:{eta}")
            # for loader
            loss_avg = loss_ttl / loss_cnt
            d_old_avg = d_old_sum / loss_cnt
            d_new_avg = d_new_sum / loss_cnt
            log_info(f"E:{epoch}/{e_cnt}: avg_loss:{loss_avg:8.4f}. "
                     f"dist:{d_old_avg:.4f}~{d_new_avg:.4f}. MS total:{self.ms_mgr.state_str()}")
            if 0 < epoch < e_cnt and save_int > 0 and epoch % save_int == 0:
                self.save_model(epoch, epoch_in_file_name=True)
                if self.save_ckpt_eval: self.ema_sample_and_fid(epoch)
        # for epoch
        self.save_model(e_cnt, epoch_in_file_name=False)
        if self.save_ckpt_eval:
            self.ema_sample_and_fid(e_cnt)
            basename = os.path.basename(args.save_ckpt_path)
            stem, ext = os.path.splitext(basename)
            f_path = f"./sample_fid_is_{stem}_all.txt"
            with open(f_path, 'w') as fptr:
                [fptr.write(f"{m}\n") for m in self.result_arr]
            # with

    def train_batch(self, x):
        """
        train model
        :param x: input clean image
        :return:
        """
        epsilon = torch.randn_like(x)
        if self.ms_size > 0:
            dist_old = (x - epsilon).square().mean()
            for i in range(len(x)): # For each image, find the nearest noise in ns
                self.ms_mgr.assign_nearest(x, epsilon, i)
            dist_new = (x - epsilon).square().mean()
        else:
            dist_old, dist_new = 0., 0.

        b_sz = x.size(0)  # batch size
        # t = torch.randint(high=self.num_timesteps, size=(b_sz // 2 + 1,), device=self.device)
        # t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:b_sz]
        t = torch.randint(low=self.ts_low, high=self.ts_high, size=(b_sz,), device=self.device)
        loss, xt = noise_estimation_loss2(self.model, x, t, epsilon, self.alphas_cumprod)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip)
        self.optimizer.step()

        ema_decay = self.ema_helper.update(self.model)
        return loss.item(), ema_decay, dist_old, dist_new

    def ema_sample_and_fid(self, epoch):
        """
        Make samples and calculate the FID.
         """
        log_info(f"DiffusionTraining::ema_sample_and_fid()")
        args, config = self.args, self.config
        self.ema_helper.store(self.model.parameters())
        self.ema_helper.ema_to_module(self.model)
        self.model.eval()
        self.sampler.model = self.model
        img_cnt     = args.sample_count
        b_sz        = args.sample_batch_size
        steps_arr   = args.sample_steps_arr
        b_cnt = img_cnt // b_sz
        if b_cnt * b_sz < img_cnt:
            b_cnt += 1
        c_data = config.data
        c, h, w = c_data.channels, c_data.image_size, c_data.image_size
        s_fid1, s_dir, s_isc = self.fid_input1, self.sample_output_dir, self.sample_isc_flag
        log_info(f"  epoch      : {epoch}")
        log_info(f"  img_cnt    : {img_cnt}")
        log_info(f"  b_sz       : {b_sz}")
        log_info(f"  b_cnt      : {b_cnt}")
        log_info(f"  c          : {c}")
        log_info(f"  h          : {h}")
        log_info(f"  w          : {w}")
        log_info(f"  steps_arr  : {steps_arr}")
        time_start = time.time()
        msg_arr = []
        for steps in steps_arr:
            with torch.no_grad():
                for b_idx in range(b_cnt):
                    n = img_cnt - b_idx * b_sz if b_idx == b_cnt - 1 else b_sz
                    z0 = torch.randn(n, c, h, w, requires_grad=False, device=self.device)
                    x0 = self.sampler.sample_batch(z0, steps, b_idx=b_idx)
                    self.sampler.save_images(x0, b_idx, b_sz, time_start, b_cnt)
                # for
            # with
            torch.cuda.empty_cache()
            log_info(f"sleep 2 seconds to empty the GPU cache. . .")
            time.sleep(2)
            log_info(f"fid_input1       : {s_fid1}")
            log_info(f"sample_output_dir: {s_dir}")
            log_info(f"sample_isc_flag  : {s_isc}")
            fid, is_mean, is_std = calc_fid_isc(args.gpu_ids[0], s_fid1, s_dir, s_isc)
            msg = f"E{epoch:04d}_steps{steps:02d}"
            msg += f"\tFID{fid:7.3f}\tis_mean{is_mean:7.3f}\tis_std{is_std:7.3f}"
            log_info(msg)
            msg_arr.append(msg)
        # for
        self.ema_helper.restore(self.model.parameters())
        self.model.train()
        basename = os.path.basename(args.save_ckpt_path)
        stem, ext = os.path.splitext(basename)
        f_path = f"./sample_fid_is_{stem}_E{epoch:04d}.txt"
        with open(f_path, 'w') as fptr:
            [fptr.write(f"{m}\n") for m in msg_arr]
        # with
        self.result_arr.extend(msg_arr)

    def save_model(self, e_idx, epoch_in_file_name=False):
        real_model = self.model
        if isinstance(real_model, torch.nn.DataParallel):
            real_model = real_model.module
        states = {
            'model'         : real_model.state_dict(),
            'optimizer'     : self.optimizer.state_dict(),
            'ema_helper'    : self.ema_helper.state_dict(),
            'beta_schedule' : self.beta_schedule,
            'cur_epoch'     : e_idx,
        }
        ckpt_path = self.args.save_ckpt_path
        save_ckpt_dir, base_name = os.path.split(ckpt_path)
        if not os.path.exists(save_ckpt_dir):
            log_info(f"os.makedirs({save_ckpt_dir})")
            os.makedirs(save_ckpt_dir)
        if epoch_in_file_name:
            stem, ext = os.path.splitext(base_name)
            ckpt_path = os.path.join(save_ckpt_dir, f"{stem}_E{e_idx:03d}{ext}")
        log_info(f"Save ckpt: {ckpt_path} . . .")
        torch.save(states, ckpt_path)
        log_info(f"Save ckpt: {ckpt_path} . . . Done")

# class
