import torch
from .diffusion import Diffusion
from utils import log_info


class DiffusionTrainingBase(Diffusion):
    def __init__(self, args):
        log_info(f"DiffusionTrainingBase::__init__()...")
        Diffusion.__init__(self, args)
        self.ts_high = len(self.betas)
        log_info(f"  ts_high    : {self.ts_high}")
        log_info(f"  unet       : {self.args.unet}")
        log_info(f"DiffusionTrainingBase::__init__()...Done")

    def create_model_optimizer(self):
        unet_type = self.args.unet
        if unet_type is None or unet_type == '' or unet_type == 'ermongroup':
            from models.ermongroup.ermongroup_model import ErmongroupModel
            model = ErmongroupModel(self.config)
        elif unet_type == 'ncsn':
            from framework_rfm.rectified_flow_base import RectifiedFlowBase
            from models.ncsn.ncsnpp import NCSNpp
            base = RectifiedFlowBase(self.args)
            model = NCSNpp(base.config)
        else:
            raise ValueError(f"Invalid unet_type : {unet_type}")
        optimizer = self.get_optimizer(model.parameters(), self.args.lr)
        return model, optimizer

    def create_sampler(self):
        from .diffusion_sampling import DiffusionSampling

        sampler = DiffusionSampling(self.args)
        return sampler

    def create_sample_batch(self, noise_batch, steps, batch_idx=-1):
        x0 = self.sampler.sample_batch(noise_batch, steps, b_idx=batch_idx)
        return x0

    def calc_loss(self, x_batch, noise_batch, ts=None):
        b_sz = x_batch.size(0)  # batch size
        if ts is None:
            t = torch.randint(low=0, high=self.ts_high, size=(b_sz,), device=self.device)
        else:
            t = torch.full((b_sz,), int(ts), device=self.device)
        at = self.alphas_cumprod.index_select(0, t).view(-1, 1, 1, 1)  # alpha_t
        x = x_batch * at.sqrt() + noise_batch * (1.0 - at).sqrt()
        output = self.model(x, t.float())
        return (noise_batch - output).square().mean()

# class
