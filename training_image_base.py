"""
Train on clear image space.
"""
import datetime
import os
import time
import torch
import torch.utils.data as data
import torchvision.utils as tvu

from utils import log_info as log_info, calc_fid_isc, get_time_ttl_and_eta
from models.ncsn.ema import ExponentialMovingAverage

class TrainingImageBase:
    def __init__(self, args):
        self.args = args
        self.resume_ckpt_path   = args.resume_ckpt_path
        self.save_ckpt_path     = args.save_ckpt_path
        self.save_ckpt_interval = args.save_ckpt_interval
        self.save_ckpt_eval     = args.save_ckpt_eval
        self.data_dir           = args.data_dir
        self.seed               = args.seed
        self.device             = args.device
        self.ema_rate           = args.ema_rate
        self.ss_size            = args.ms_size     # noise selective-set size
        self.sample_output_dir  = args.sample_output_dir
        self.fid_input1         = args.fid_input1
        self.loss_norm          = 'lpips'
        log_info(f"TrainingImageBase::__init__()...")
        self.sampler = None
        self.model = None
        self.ema = None
        self.optimizer = None
        log_info(f"  resume_ckpt_path   : {self.resume_ckpt_path}")
        log_info(f"  save_ckpt_path     : {self.save_ckpt_path}")
        log_info(f"  save_ckpt_interval : {self.save_ckpt_interval}")
        log_info(f"  save_ckpt_eval     : {self.save_ckpt_eval}")
        log_info(f"  sample_output_dir  : {self.sample_output_dir}")
        log_info(f"  sample_steps_arr   : {args.sample_steps_arr}")
        log_info(f"  fid_input1         : {self.fid_input1}")
        log_info(f"  device             : {self.device}")
        log_info(f"  ema_rate           : {self.ema_rate}")
        log_info(f"  ss_size            : {self.ss_size}")
        log_info(f"  loss_norm          : {self.loss_norm}")
        self.batch_counter = 0
        self.result_arr = []
        self.sample_channel = None
        self.sample_height  = None
        self.sample_width   = None
        self.train_loader   = None
        log_info(f"TrainingImageBase::__init__()...Done")

    # ====================================================================
    # === those not-implemented functions should be defined elsewhere. ===
    # ====================================================================
    # def create_sampler(self):
    #     raise NotImplementedError()
    #
    # def create_model_optimizer(self):
    #     raise NotImplementedError()
    #
    # def create_sample_batch(self, noise_batch, steps, batch_idx=-1):
    #     raise NotImplementedError()
    #
    # def calc_loss(self, x_batch, noise_batch):
    #     raise NotImplementedError()
    #
    # def adjust_noise(self, x_batch, noise_batch, ss_size):
    #     raise NotImplementedError()

    def init_model_ema_optimizer(self):
        log_info(f"TrainingImageBase::init_model_ema_optimizer()...")
        model, optimizer = self.create_model_optimizer()
        log_info(f"  model = model.to({self.device})")
        model = model.to(self.device)
        ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_rate)
        log_info(f"  model    : {type(model).__name__}")
        log_info(f"  optimizer: {type(optimizer).__name__}")
        log_info(f"  ema      : {type(ema).__name__}")
        if self.resume_ckpt_path:
            log_info(f"  load ckpt: {self.resume_ckpt_path}...")
            states = torch.load(self.resume_ckpt_path, map_location=self.device)
            log_info(f"  load ckpt: {self.resume_ckpt_path}...Done")
            log_info(f"  pure_flag    : {states.get('pure_flag', None)}")
            log_info(f"  args.config  : {states.get('args.config', None)}")
            log_info(f"  args.seed    : {states.get('args.seed', None)}")
            log_info(f"  args.model   : {states.get('args.model', None)}")
            log_info(f"  args.todo    : {states.get('args.todo', None)}")
            log_info(f"  args.lr      : {states.get('args.lr', None)}")
            log_info(f"  args.ema_rate: {states.get('args.ema_rate', None)}")
            log_info(f"  args.n_epochs: {states.get('args.n_epochs', None)}")
            log_info(f"  host         : {states.get('host', None)}")
            log_info(f"  cwd          : {states.get('cwd', None)}")
            log_info(f"  pid          : {states.get('pid', None)}")
            log_info(f"  date_time    : {states.get('date_time', None)}")
            log_info(f"  class_name   : {states.get('class_name', None)}")
            log_info(f"  epoch        : {states.get('epoch', None)}")

            log_info(f"  model.load_state_dict(states['model'], strict=True)")
            model.load_state_dict(states['model'], strict=True)

            ema.load_state_dict(states['ema'])
            log_info(f"  ema.load_state_dict(states['ema'])")
            log_info(f"  ema.copy_to(model.parameters())")
            ema.copy_to(model.parameters())
            log_info(f"  ema.num_updates: {ema.num_updates}")
            log_info(f"  ema.decay (old): {ema.decay}")
            ema.decay = self.args.ema_rate
            log_info(f"  ema.decay (new): {ema.decay}")

            log_info(f"  optimizer.load_state_dict(states['optimizer'])")
            optimizer.load_state_dict(states['optimizer'])

        if len(self.args.gpu_ids) > 1:
            log_info(f"model = torch.nn.DataParallel(model, device_ids={self.args.gpu_ids})")
            model = torch.nn.DataParallel(model, device_ids=self.args.gpu_ids)
        self.model = model
        self.ema   = ema
        self.optimizer = optimizer
        log_info(f"TrainingImageBase::init_model_ema_optimizer()...Done")

    def get_data_loader(self, train_shuffle=True):
        args = self.args
        root_dir = args.data_dir
        import torchvision.transforms as T
        if args.config == 'cifar10':
            from datasets.cifar import CIFAR10
            train_tfm = T.Compose([T.RandomHorizontalFlip(p=0.5), T.ToTensor()])
            dir1 = os.path.join(root_dir, "datasets", "cifar10")
            ds = CIFAR10(dir1, train=True, download=True, transform=train_tfm)
        else:
            from datasets.ImageDataset import ImageDataset
            train_tfm = T.Compose([T.RandomHorizontalFlip(p=0.5), T.ToTensor()])
            ds = ImageDataset(root_dir, transform=train_tfm)

        batch_size = args.batch_size
        num_workers = 4
        train_loader = data.DataLoader(ds, batch_size, shuffle=train_shuffle, num_workers=num_workers)
        log_info(f"TrainingImageBase::get_data_loader()...")
        log_info(f"  data_dir   : {root_dir}")
        log_info(f"  len        : {len(ds)}")
        log_info(f"  batch_cnt  : {len(train_loader)}")
        log_info(f"  batch_size : {batch_size}")
        log_info(f"  shuffle    : {train_shuffle}")
        log_info(f"  num_workers: {num_workers}")
        log_info(f"TrainingImageBase::get_data_loader()...Done")
        return train_loader

    def train(self):
        args = self.args
        self.train_loader = train_loader = self.get_data_loader()
        ckpt_epoch = self.init_model_ema_optimizer() or 0  # change None to 0
        log_info(f"TrainingImageBase::train()")
        log_info(f"  save_ckpt_eval: {self.save_ckpt_eval}")
        if self.save_ckpt_eval:
            # init model first, and then init sampler
            log_info(f"  TrainingImageBase::train(): create sampler")
            self.sampler = self.create_sampler()
            self.sampler.model = self.model
        log_info(f"  sampler       : {type(self.sampler).__name__}")

        log_interval = args.log_interval
        e_cnt = args.n_epochs       # epoch count
        b_cnt = len(train_loader)   # batch count
        lr = args.lr
        save_int = args.save_ckpt_interval
        start_time = time.time()
        self.batch_counter = 0  # reset to 0
        batch_total = (e_cnt - ckpt_epoch) * b_cnt
        self.model.train()
        log_info(f"  save_interval : {save_int}")
        log_info(f"  log_interval  : {log_interval}")
        log_info(f"  b_sz          : {args.batch_size}")
        log_info(f"  lr            : {lr}")
        log_info(f"  train_b_cnt   : {b_cnt}")
        log_info(f"  e_cnt         : {e_cnt}")
        log_info(f"  ckpt_epoch    : {ckpt_epoch}")
        log_info(f"  batch_total   : {batch_total}")
        log_info(f"  ss_size       : {self.ss_size}")
        for epoch in range(ckpt_epoch+1, e_cnt+1):
            log_info(f"Epoch {epoch}/{e_cnt} ---------- lr={lr:.6f}")
            loss_sum, loss_cnt = 0., 0
            d_old_sum, d_new_sum = 0., 0.   # distance sum for old and new
            for i, (x, y) in enumerate(train_loader):
                self.batch_counter += 1
                x = x.to(self.device)
                x = x * 2. - 1.
                if self.sample_channel is None:
                    bs, c, h, w = x.size()
                    self.sample_channel, self.sample_height, self.sample_width = c, h, w
                    log_info(f"Sample CHW: {c}, {h}, {w}")
                loss, decay, dist_old, dist_new = self.train_batch_adjust(x)
                loss_sum += loss
                loss_cnt += 1
                d_old_sum += dist_old
                d_new_sum += dist_new
                if i % log_interval == 0 or i == b_cnt - 1:
                    elp, eta = get_time_ttl_and_eta(start_time, self.batch_counter, batch_total)
                    do_avg, dn_avg = d_old_sum / loss_cnt, d_new_sum / loss_cnt
                    loss_str = f"loss:{loss:.5f}; ema:{decay:.5f}; dist:{do_avg:.4f}~{dn_avg:.4f}"
                    log_info(f"E{epoch}.B{i:03d}/{b_cnt} {loss_str}. elp:{elp}, eta:{eta}")
            # for
            loss_avg = loss_sum / loss_cnt
            d_old_avg = d_old_sum / loss_cnt
            d_new_avg = d_new_sum / loss_cnt
            log_info(f"E{epoch}.training_loss_avg: {loss_avg:.6f}. dist:{d_old_avg:.4f}~{d_new_avg:.4f}.")
            if save_int > 0 and epoch % save_int == 0 or epoch == e_cnt:
                self.save_image_ckpt(epoch)
                if self.save_ckpt_eval: self.ema_sample_and_fid(epoch)
        # for
        basename = os.path.basename(args.save_ckpt_path)
        stem, ext = os.path.splitext(basename)
        f_path = f"./sample_fid_{stem}_all.txt"
        with open(f_path, 'w') as fptr:
            [fptr.write(f"{m}\n") for m in self.result_arr]
        # with
        return 0

    def train_batch_adjust(self, x_batch):
        """ train batch: adjust the order or noise inside the batch """
        noise_batch = torch.randn_like(x_batch, device=self.device)
        if self.ss_size > 1:
            dist_old = (x_batch - noise_batch).square().mean()
            self.adjust_noise(x_batch, noise_batch, self.ss_size)
            dist_new = (x_batch - noise_batch).square().mean()
        else:
            dist_old, dist_new = 0., 0.

        self.optimizer.zero_grad()
        loss = self.calc_loss(x_batch, noise_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        decay = self.ema_update()
        # here, we must return loss.item(), and not loss.
        # If return loss, it will cause memory leak.
        return loss.item(), decay, dist_old, dist_new

    def save_image_ckpt(self, epoch, epoch_in_file_name=True):
        real_model = self.model
        if isinstance(real_model, torch.nn.DataParallel):
            real_model = real_model.module
        states = {
            'pure_flag'     : True,    # this pure model, not DataParallel
            'args.config'   : self.args.config,
            'args.seed'     : self.args.seed,
            'args.model'    : self.args.model,
            'args.todo'     : self.args.todo,
            'args.lr'       : self.args.lr,
            'args.ema_rate' : self.args.ema_rate,
            'args.n_epochs' : self.args.n_epochs,
            'host'          : os.uname().nodename,
            'cwd'           : os.getcwd(),
            'pid'           : os.getpid(),
            'date_time'     : datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'class_name'    : self.__class__.__name__,
            'epoch'         : epoch,
            'model'         : real_model.state_dict(),
            'optimizer'     : self.optimizer.state_dict(),
            'ema'           : self.ema.state_dict(),
        }
        ckpt_path = self.args.save_ckpt_path
        save_ckpt_dir, base_name = os.path.split(ckpt_path)
        if not os.path.exists(save_ckpt_dir):
            log_info(f"os.makedirs({save_ckpt_dir})")
            os.makedirs(save_ckpt_dir)
        if epoch_in_file_name:
            stem, ext = os.path.splitext(base_name)
            ckpt_path = os.path.join(save_ckpt_dir, f"{stem}_E{epoch:04d}{ext}")
        log_info(f"Save ckpt: {ckpt_path} . . .")
        torch.save(states, ckpt_path)
        log_info(f"Save ckpt: {ckpt_path} . . . Done")

    def load_image_ckpt(self, epoch, ckpt_path=None):
        if not ckpt_path:
            save_ckpt_dir, base_name = os.path.split(self.args.save_ckpt_path)
            stem, ext = os.path.splitext(base_name)
            ckpt_path = os.path.join(save_ckpt_dir, f"{stem}_E{epoch:04d}{ext}")
        log_info(f"load: {ckpt_path}...")
        states = torch.load(ckpt_path, map_location=self.device)
        log_info(f"load: {ckpt_path}...Done")
        m = self.model
        if isinstance(m, torch.nn.DataParallel):
            m = m.module
        m.load_state_dict(states['model'], strict=True)
        self.ema.load_state_dict(states['ema'])
        self.optimizer.load_state_dict(states['optimizer'])

    def ema_update(self):
        decay = self.ema.update(self.model.parameters())
        return decay

    def ema_sample_and_fid(self, epoch):
        """
        Make samples and calculate the FID.
         """
        log_info(f"ema_sample_and_fid()")
        torch.cuda.empty_cache()
        args = self.args
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
        c, h, w = self.sample_channel, self.sample_height, self.sample_width
        s_fid1, s_dir = self.fid_input1, self.sample_output_dir
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
        for steps in steps_arr:
            with torch.no_grad():
                for b_idx in range(b_cnt):
                    n = img_cnt - b_idx * b_sz if b_idx == b_cnt - 1 else b_sz
                    z0 = torch.randn(n, c, h, w, requires_grad=False, device=self.device)
                    img = self.create_sample_batch(z0, steps, batch_idx=b_idx)
                    img = (img + 1.0) / 2.0
                    img = torch.clamp(img, 0., 1.0)
                    self.save_images(img, b_idx, b_sz, time_start, b_cnt)
                # for
            # with
            torch.cuda.empty_cache()
            log_info(f"sleep 2 seconds to empty the GPU cache. . .")
            time.sleep(2)
            log_info(f"fid_input1       : {s_fid1}")
            log_info(f"sample_output_dir: {s_dir}")
            fid, is_mean, is_std = calc_fid_isc(args.gpu_ids[0], s_fid1, s_dir, isc_flag=False)
            msg = f"E{epoch:04d}_steps{steps:02d}\tFID{fid:7.3f}"
            log_info(msg)
            msg_arr.append(msg)
        # for
        self.ema.restore(self.model.parameters())
        self.model.train()
        basename = os.path.basename(args.save_ckpt_path)
        stem, ext = os.path.splitext(basename)
        f_path = f"./sample_fid_{stem}_E{epoch:04d}.txt"
        with open(f_path, 'w') as fptr:
            [fptr.write(f"{m}\n") for m in msg_arr]
        # with
        self.result_arr.extend(msg_arr)

    def save_images(self, x, b_idx, b_sz, time_start=None, b_cnt=None, img_dir=None):
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

# class
