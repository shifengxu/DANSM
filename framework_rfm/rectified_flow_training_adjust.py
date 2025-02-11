"""
Distance-Aware Training. adjust the data order
"""
import os
import time
import torch
import torch.utils.data as data

from .rectified_flow_sampling import RectifiedFlowSampling
from .rectified_flow_base import RectifiedFlowBase
from datasets import get_train_test_datasets
from datasets import data_scaler
from models.match_scope_manager import MatchScopeManager
from models.ncsn.ncsnpp import NCSNpp
from utils import log_info as log_info, calc_fid_isc
from utils import get_time_ttl_and_eta
from models.ncsn.ema import ExponentialMovingAverage

class RectifiedFlowTrainingAdjust(RectifiedFlowBase):
    def __init__(self, args):
        super().__init__(args)
        self.resume_ckpt_path   = args.resume_ckpt_path
        self.save_ckpt_path     = args.save_ckpt_path
        self.save_ckpt_interval = args.save_ckpt_interval
        self.save_ckpt_eval     = args.save_ckpt_eval
        self.sampler = None
        if self.save_ckpt_eval:
            self.sampler = RectifiedFlowSampling(args)
        self.sample_output_dir  = args.sample_output_dir
        self.fid_input1         = args.fid_input1
        self.sample_isc_flag    = args.sample_isc_flag
        self.data_dir = args.data_dir
        self.seed = args.seed
        self.model = None
        self.ema = None
        self.optimizer = None
        self.ema_rate = args.ema_rate
        self.step = 0
        self.step_new = 0
        self.ms_size = args.ms_size
        c_data = self.config.data
        c, h, w = c_data.num_channels, c_data.image_size, c_data.image_size
        self.ms_mgr = MatchScopeManager(self.ms_size, c, h, w, args.device)
        log_info(f"RectifiedFlowTrainingAdjust()")
        log_info(f"  resume_ckpt_path   : {self.resume_ckpt_path}")
        log_info(f"  save_ckpt_path     : {self.save_ckpt_path}")
        log_info(f"  save_ckpt_interval : {self.save_ckpt_interval}")
        log_info(f"  save_ckpt_eval     : {self.save_ckpt_eval}")
        log_info(f"  sampler            : {type(self.sampler).__name__}")
        log_info(f"  sample_output_dir  : {self.sample_output_dir}")
        log_info(f"  fid_input1         : {self.fid_input1}")
        log_info(f"  sample_isc_flag    : {self.sample_isc_flag}")
        log_info(f"  device     : {self.device}")
        log_info(f"  ema_rate   : {self.ema_rate}")
        log_info(f"  eps        : {self.eps}")
        log_info(f"  step       : {self.step}")
        log_info(f"  step_new   : {self.step_new}")
        log_info(f"  ms_size    : {self.ms_size}")
        log_info(f"  c          : {c}")
        log_info(f"  h          : {h}")
        log_info(f"  w          : {w}")
        log_info(f"  ms_mgr     : {type(self.ms_mgr).__name__}")
        self.start_time = None
        self.batch_counter = 0
        self.batch_total = 0
        self.result_arr = []

    def init_model_ema_optimizer(self):
        """Create the score model."""
        args, config = self.args, self.config
        if self.resume_ckpt_path:
            states = self.load_ckpt(self.resume_ckpt_path, eval_mode=False, only_return_model=False)
            model      = states['model']
            ema        = states['ema']
            optimizer  = states['optimizer']
            step       = states['step']
            ckpt_epoch = states['epoch']
        else:
            model_name = config.model.name
            log_info(f"RectifiedFlowTrainingAdjust::init_model_ema_optimizer()")
            log_info(f"  config.model.name: {model_name}")
            if model_name.lower() == 'ncsnpp':
                model = NCSNpp(config)
            else:
                raise ValueError(f"Unknown model name: {model_name}")
            log_info(f"  model = model.to({self.device})")
            model = model.to(self.device)
            log_info(f"  torch.nn.DataParallel(model, device_ids={self.args.gpu_ids})")
            model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
            ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_rate)
            log_info(f"  ema constructed.")
            log_info(f"  ema.num_updates: {ema.num_updates}")
            log_info(f"  ema.decay      : {ema.decay}")
            optimizer = self.get_optimizer(model.parameters())
            step = 0
            ckpt_epoch = 0

        self.model = model
        self.ema = ema
        self.optimizer = optimizer
        self.step = step
        return ckpt_epoch

    def get_data_loaders(self, train_shuffle=True, test_shuffle=False):
        args, config = self.args, self.config
        batch_size = args.batch_size
        num_workers = 4
        if args.config == 'bedroom2':   # special case
            config.data.dataset = "LSUN2"
        train_ds, test_ds = get_train_test_datasets(args, config)
        train_loader = data.DataLoader(
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

        test_loader = data.DataLoader(
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

    def train(self):
        args, config = self.args, self.config
        train_loader, _ = self.get_data_loaders()
        ckpt_epoch = self.init_model_ema_optimizer() or 0  # change None to 0
        log_interval = args.log_interval
        e_cnt = args.n_epochs       # epoch count
        b_cnt = len(train_loader)   # batch count
        lr = args.lr
        save_int = args.save_ckpt_interval
        self.start_time = time.time()
        self.batch_counter = 0
        self.batch_total = (e_cnt - ckpt_epoch) * b_cnt
        self.model.train()
        log_info(f"RectifiedFlowTrainingAdjust::train()")
        log_info(f"  save_interval : {save_int}")
        log_info(f"  log_interval  : {log_interval}")
        log_info(f"  image_size    : {config.data.image_size}")
        log_info(f"  b_sz          : {args.batch_size}")
        log_info(f"  lr            : {lr}")
        log_info(f"  train_b_cnt   : {b_cnt}")
        log_info(f"  e_cnt         : {e_cnt}")
        log_info(f"  ckpt_epoch    : {ckpt_epoch}")
        log_info(f"  batch_total   : {self.batch_total}")
        for epoch in range(ckpt_epoch+1, e_cnt+1):
            log_info(f"Epoch {epoch}/{e_cnt} ---------- lr={lr:.6f}")
            data_cnt = 0
            loss_sum, loss_cnt = 0., 0
            d_old_sum, d_new_sum = 0., 0.   # distance sum for old and new
            for i, (x, y) in enumerate(train_loader):
                self.batch_counter += 1
                data_cnt += x.size(0)
                x, y = x.to(self.device), y.to(self.device)
                x = data_scaler(config, x)
                loss, decay, dist_old, dist_new = self.train_batch_adjust(x)
                loss_sum += loss
                loss_cnt += 1
                d_old_sum += dist_old
                d_new_sum += dist_new
                if i % log_interval == 0 or i == b_cnt - 1:
                    elp, eta = get_time_ttl_and_eta(self.start_time, self.batch_counter, self.batch_total)
                    do_avg, dn_avg = d_old_sum / loss_cnt, d_new_sum / loss_cnt
                    loss_str = f"loss:{loss:6.4f}; ema:{decay:.4f}; dist:{do_avg:.4f}~{dn_avg:.4f}"
                    log_info(f"E{epoch}.B{i:03d}/{b_cnt} {loss_str}. elp:{elp}, eta:{eta}")
            # for
            loss_avg = loss_sum / loss_cnt
            d_old_avg = d_old_sum / loss_cnt
            d_new_avg = d_new_sum / loss_cnt
            log_info(f"E{epoch}.training_loss_avg: {loss_avg:.6f}. dist:{d_old_avg:.4f}~{d_new_avg:.4f}."
                     f" MS total:{self.ms_mgr.state_str()}")
            if 0 < epoch < e_cnt and save_int > 0 and epoch % save_int == 0:
                self._save_ckpt(epoch)
                if self.save_ckpt_eval: self.ema_sample_and_fid(epoch)
        # for
        self._save_ckpt(e_cnt)
        if self.save_ckpt_eval:
            self.ema_sample_and_fid(e_cnt)
            basename = os.path.basename(args.save_ckpt_path)
            stem, ext = os.path.splitext(basename)
            f_path = f"./sample_fid_isc_{stem}_all.txt"
            with open(f_path, 'w') as fptr:
                [fptr.write(f"{m}\n") for m in self.result_arr]
            # with
        return 0

    def train_batch_adjust(self, x_batch):
        """ train batch: adjust the order or noise inside the batch """
        z0 = torch.randn_like(x_batch, device=self.device)
        if self.ms_size > 0:
            dist_old = (x_batch - z0).square().mean()
            for i in range(len(x_batch)): # For each image, find the nearest noise in ns
                self.ms_mgr.assign_nearest(x_batch, z0, i)
            dist_new = (x_batch - z0).square().mean()
        else:
            dist_old, dist_new = 0., 0.

        self.optimizer.zero_grad()
        loss = self.calc_loss(x_batch, z0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.optim.grad_clip)
        self.optimizer.step()
        decay = self.ema.update(self.model.parameters())
        self.step_new += 1
        self.step += 1
        # here, we must return loss.item(), and not loss.
        # If return loss, it will cause memory leak.
        return loss.item(), decay, dist_old, dist_new

    def _save_ckpt(self, epoch_cnt):
        self.save_ckpt(self.model, self.ema, self.optimizer, epoch_cnt, self.step, self.step_new, True)

    def calc_loss(self, x_batch, z0):
        target = x_batch - z0
        b_sz, c, h, w = x_batch.size()
        t = torch.rand(b_sz, device=self.device)
        t = torch.mul(t, 1.0 - self.eps)
        t = torch.add(t, self.eps)
        t_expand = t.view(-1, 1, 1, 1)
        perturbed_data = t_expand * x_batch + (1. - t_expand) * z0
        predict = self.model(perturbed_data, t * 999)
        loss = (predict - target).square().mean()
        return loss

    def ema_sample_and_fid(self, epoch):
        """
        Make samples and calculate the FID.
         """
        log_info(f"get_ema_fid()")
        args, config = self.args, self.config
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())
        self.model.eval()
        self.sampler.model = self.model
        img_cnt     = args.sample_count
        b_sz        = args.sample_batch_size
        steps_arr   = args.sample_steps_arr
        init_ts_arr = args.sample_init_ts_arr
        b_cnt = img_cnt // b_sz
        if b_cnt * b_sz < img_cnt:
            b_cnt += 1
        c_data = config.data
        c, h, w = c_data.num_channels, c_data.image_size, c_data.image_size
        s_fid1, s_dir, s_isc = self.fid_input1, self.sample_output_dir, self.sample_isc_flag
        log_info(f"  epoch      : {epoch}")
        log_info(f"  img_cnt    : {img_cnt}")
        log_info(f"  b_sz       : {b_sz}")
        log_info(f"  b_cnt      : {b_cnt}")
        log_info(f"  c          : {c}")
        log_info(f"  h          : {h}")
        log_info(f"  w          : {w}")
        log_info(f"  steps_arr  : {steps_arr}")
        log_info(f"  init_ts_arr: {init_ts_arr}")
        time_start = time.time()
        msg_arr = []
        for init_ts in init_ts_arr:
            for steps in steps_arr:
                with torch.no_grad():
                    for b_idx in range(b_cnt):
                        n = img_cnt - b_idx * b_sz if b_idx == b_cnt - 1 else b_sz
                        z0 = torch.randn(n, c, h, w, requires_grad=False, device=self.device)
                        x0 = self.sampler.sample_batch(z0, steps, init_ts=init_ts, b_idx=b_idx)
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
                msg = f"E{epoch:04d}_steps{steps:02d}_initTS{init_ts:.3f}"
                msg += f"\tFID{fid:7.3f}\tis_mean{is_mean:7.3f}\tis_std{is_std:7.3f}"
                log_info(msg)
                msg_arr.append(msg)
            # for
        # for
        self.ema.restore(self.model.parameters())
        self.model.train()
        basename = os.path.basename(args.save_ckpt_path)
        stem, ext = os.path.splitext(basename)
        f_path = f"./sample_fid_is_{stem}_E{epoch:04d}.txt"
        with open(f_path, 'w') as fptr:
            [fptr.write(f"{m}\n") for m in msg_arr]
        # with
        self.result_arr.extend(msg_arr)

# class
