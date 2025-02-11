import math
import os.path
import numpy as np
import torch
import yaml
import torch.optim as optim
from models.ermongroup.ema import EMAHelper
from models.ermongroup.ermongroup_model import ErmongroupModel
from utils import log_info, dict2namespace

class Diffusion(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        cfg_str = args.config
        self.config_str = cfg_str
        dir_name = os.path.dirname(__file__)
        cfg_dir = os.path.join(dir_name, 'configs')
        if cfg_str == 'bedroom':
            cfg_path = os.path.join(cfg_dir, 'bedroom.yml')
        elif cfg_str == 'cifar10':
            cfg_path = os.path.join(cfg_dir, 'cifar10.yml')
        elif cfg_str == 'latent_bedroom':
            cfg_path = os.path.join(cfg_dir, 'latent_4_32_32.yml')
        else:
            raise ValueError(f"Invalid config for Diffusion: {cfg_str}")
        with open(cfg_path, "r") as f:  # parse config file
            config = yaml.safe_load(f)
        self.config = config = dict2namespace(config)
        log_info(f"runners.Diffusion()")
        log_info(f"  cfg_str       : {cfg_str}")
        log_info(f"  cfg_path      : {cfg_path}")
        log_info(f"  device        : {self.device}")

        self.model_var_type = config.model.var_type
        self.beta_schedule  = config.diffusion.beta_schedule
        self.alphas, self.alphas_cumprod, self.betas = self.get_alphas_and_betas(config)
        self.num_timesteps = self.betas.shape[0]
        log_info(f"  model_var_type: {self.model_var_type}")
        log_info(f"  beta_schedule : {self.beta_schedule}")
        log_info(f"  num_timesteps : {self.num_timesteps}")

        if self.model_var_type == "fixedlarge":
            self.logvar = self.betas.log()
            # torch.cat([posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            ac = self.alphas_cumprod
            alphas_cumprod_prev = torch.cat([torch.ones(1).to(self.device), ac[:-1]], dim=0)
            posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - ac)
            self.logvar = posterior_variance.clamp(min=1e-20).log()

        arr = [torch.ones(1, device=self.device), self.alphas_cumprod]
        self.alphas_cumprod = torch.cat(arr, dim=0)
        log_info(f"  betas         : {len(self.betas)}")
        log_info(f"  alphas        : {len(self.alphas)}")
        log_info(f"  alphas_cumprod: {len(self.alphas_cumprod)}")

    def get_alphas_and_betas(self, config):
        """
        Explanation_1:
            beta:linear     idx: 0          idx: 999
            beta            0.0001          0.02
            alpha           0.9999          0.98
            aacum           0.9999          0.00004
            aacum_sqrt      0.999949999     0.006324555
            noise           0.0001          0.99996
            noise_sqrt      0.01            0.99998
        Notes:
            aacum: is alpha accumulation: a0*a1*a2...
            noise: is just: 1 - accum
        :param config:
        :return:
        """
        device = self.device
        ts_cnt = config.diffusion.num_diffusion_timesteps
        if self.beta_schedule == "cosine":
            # cosine scheduler is from the following paper:
            # ICML. 2021. Alex Nichol. Improved Denoising Diffusion Probabilistic Models
            # In this option, it composes alphas_cumprod firstly, then alphas and betas.
            cos_0 = math.cos(0.008 / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = [] # alpha cumulate array
            for i in range(ts_cnt):
                t = i / ts_cnt
                ac = math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
                ac /= cos_0
                alphas_cumprod.append(ac)
            alphas_cumprod = torch.Tensor(alphas_cumprod).float().to(device)
            divisor = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)
            alphas = torch.div(alphas_cumprod, divisor)
            betas = 1 - alphas
        elif self.beta_schedule.startswith('cos:'):
            expo_str = self.beta_schedule.split(':')[1]  # "cos:2.2"
            expo = float(expo_str)
            alphas_cumprod = []  # alpha cumulate array
            for i in range(ts_cnt):
                t = i / ts_cnt
                ac = math.cos((t + 0.008) / 1.008 * math.pi / 2) ** expo
                alphas_cumprod.append(ac)
            alphas_cumprod = torch.Tensor(alphas_cumprod).float().to(device)
            divisor = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)
            alphas = torch.div(alphas_cumprod, divisor)
            betas = 1 - alphas
        elif self.beta_schedule.startswith("noise_rt_expo:"):
            # noise is: 1 - alpha_accumulated
            expo_str = self.beta_schedule.split(':')[1]  # "noise_rt_expo:2.2"
            expo = float(expo_str)
            n_low, n_high = 0.008, 0.999 # old value
            # n_low, n_high = 0.001, 0.9999  # if "noise_rt_expo:1", got FID 27.792929 on CIFAR-10
            sq_root = np.linspace(n_low, n_high, ts_cnt, dtype=np.float64)
            sq_root = torch.from_numpy(sq_root).float().to(device)
            if expo != 1.0:
                sq_root = torch.pow(sq_root, expo)
            sq = torch.mul(sq_root, sq_root)
            alphas_cumprod = 1 - sq
            divisor = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)
            alphas = torch.div(alphas_cumprod, divisor)
            betas = 1 - alphas
        elif self.beta_schedule.startswith('aacum_rt_expo:'):
            expo_str = self.beta_schedule.split(':')[1]  # "aacum_rt_expo:2.2"
            expo = float(expo_str)
            n_high, n_low = 0.9999, 0.0008 # old value
            # n_high, n_low = 0.9999, 0.001
            # given: 0.9999, 0.001
            #   if "aacum_rt_expo:1",   got FID 22.608681 on CIFAR-10
            #   if "aacum_rt_expo:1.5", got FID 49.226592 on CIFAR-10
            sq_root = np.linspace(n_high, n_low, ts_cnt, dtype=np.float64)
            sq_root = torch.from_numpy(sq_root).float().to(device)
            if expo != 1.0:
                sq_root = torch.pow(sq_root, expo)
            alphas_cumprod = torch.mul(sq_root, sq_root)
            divisor = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)
            alphas = torch.div(alphas_cumprod, divisor)
            betas = 1 - alphas
        else:
            betas = self.get_beta_schedule(ts_cnt)
            betas = torch.from_numpy(betas).float().to(device)
            alphas = 1.0 - betas
            alphas_cumprod = alphas.cumprod(dim=0)
        return alphas, alphas_cumprod, betas

    def get_beta_schedule(self, num_diffusion_timesteps):
        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)

        beta_schedule = self.beta_schedule
        beta_start    = self.config.diffusion.beta_start
        beta_end      = self.config.diffusion.beta_end
        if beta_schedule == "quad":
            betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64)
            betas = betas ** 2
        elif beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "const":
            betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1.0 / np.linspace(
                num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
            )
        elif beta_schedule == "sigmoid":
            betas = np.linspace(-6, 6, num_diffusion_timesteps)
            betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(beta_schedule)
        assert betas.shape == (num_diffusion_timesteps,)
        return betas

    def get_optimizer(self, parameters, lr=None):
        config = self.config
        lr = lr or config.optim.lr
        if config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=lr, weight_decay=config.optim.weight_decay,
                              betas=(config.optim.beta1, 0.999), amsgrad=config.optim.amsgrad,
                              eps=config.optim.eps)
        elif config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=lr, weight_decay=config.optim.weight_decay)
        elif config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=lr, momentum=0.9)
        else:
            raise NotImplementedError(
                'Optimizer {} not understood.'.format(config.optim.optimizer))

    def load_ckpt(self, ckpt_path, eval_mode=True, only_return_model=True):
        def apply_ema():
            if self.config_str == 'latent_bedroom':
                log_info(f"  ema: ExponentialMovingAverage()")
                from models.ncsn.ema import ExponentialMovingAverage
                ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)
                log_info(f"  ema.load_state_dict(states['ema'])")
                ema.load_state_dict(states['ema'])
                log_info(f"  ema.copy_to(model.parameters())")
                ema.copy_to(model.parameters())
            else:
                log_info(f"  ema_helper: EMAHelper()")
                eh = EMAHelper(model)
                k = "ema_helper" if isinstance(states, dict) else -1
                log_info(f"  ema_helper: load from states[{k}]")
                eh.load_state_dict(states[k])
                log_info(f"  ema_helper: apply to model {type(model).__name__}")
                eh.ema_to_module(model)
            # if
        # apply_ema()

        model = ErmongroupModel(self.config)
        log_info(f"load ckpt: {ckpt_path} . . .")
        states = torch.load(ckpt_path, map_location=self.device)
        if 'model' not in states:
            log_info(f"  !!! Not found 'model' in states. Will take it as pure model")
            log_info(f"  model.load_state_dict(states)")
            model.load_state_dict(states)
        else:
            key = 'model' if isinstance(states, dict) else 0
            log_info(f"  load_model_dict(states[{key}])...")
            model.load_state_dict(states[key], strict=True)
            if eval_mode:
                apply_ema()
        # endif
        log_info(f"  model({type(model).__name__}).to({self.device})")
        model = model.to(self.device)
        if len(self.args.gpu_ids) > 1:
            log_info(f"  torch.nn.DataParallel(model, device_ids={self.args.gpu_ids})")
            model = torch.nn.DataParallel(model, device_ids=self.args.gpu_ids)
        if eval_mode:
            model.eval()

        if only_return_model:
            log_info(f"load ckpt: {ckpt_path} . . . Done")
            return model

        if 'model' not in states:
            optimizer  = None
            ema_helper = None
            cur_epoch  = None
        else:
            optimizer = self.get_optimizer(model.parameters(), self.args.lr)
            ema_helper = EMAHelper(model, mu=self.args.ema_rate)
            cur_epoch = states['cur_epoch']
            op_st = states['optimizer']
            op_st["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(op_st)
            log_info(f"  resume optimizer  : eps={op_st['param_groups'][0]['eps']}")
            ema_helper.load_state_dict(states['ema_helper'])
            log_info(f"  resume ema_helper : mu={ema_helper.mu:8.6f}")
            log_info(f"  resume cur_epoch  : {cur_epoch}")
        # endif
        log_info(f"load ckpt: {ckpt_path} . . . Done")
        return {
            'model'     : model,
            'optimizer' : optimizer,
            'ema_helper': ema_helper,
            'cur_epoch' : cur_epoch
        }
