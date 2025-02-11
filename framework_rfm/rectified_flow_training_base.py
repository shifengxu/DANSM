"""
Distance-Aware Training. adjust the data order.
Train on latent space.
"""
import torch
from .rectified_flow_base import RectifiedFlowBase
from utils import log_info

class RectifiedFlowTrainingBase(RectifiedFlowBase):
    def __init__(self, args):
        log_info(f"RectifiedFlowTrainingBase::__init__()...")
        RectifiedFlowBase.__init__(self, args)
        init_ts_arr = args.sample_init_ts_arr
        log_info(f"  eps                : {self.eps}")
        log_info(f"  args.init_ts_arr   : {init_ts_arr}")
        if len(init_ts_arr) == 0: init_ts_arr = [0.0]
        self.init_ts = init_ts_arr[0]
        log_info(f"  self.init_ts       : {self.init_ts}")
        log_info(f"RectifiedFlowTrainingBase::__init__()...Done")

    def create_model_optimizer(self):
        from models.ncsn.ncsnpp import NCSNpp
        config = self.config
        model_name = config.model.name
        if model_name.lower() == 'ncsnpp':
            model = NCSNpp(config)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        optimizer = self.get_optimizer(model.parameters())
        return model, optimizer

    def create_sampler(self):
        from .rectified_flow_sampling import RectifiedFlowSampling
        sample = RectifiedFlowSampling(self.args)
        return sample

    def create_sample_batch(self, noise_batch, steps, batch_idx=-1):
        ltt = self.sampler.sample_batch(noise_batch, steps, init_ts=self.init_ts, b_idx=batch_idx)
        return ltt

    def calc_loss(self, x_batch, noise_batch, ts=None):
        target = x_batch - noise_batch
        b_sz, c, h, w = x_batch.size()
        if ts is None:
            t = torch.rand(b_sz, device=self.device)
        else:
            t = torch.full((b_sz,), ts, device=self.device)
        t = torch.mul(t, 1.0 - self.eps)
        t = torch.add(t, self.eps)
        t_expand = t.view(-1, 1, 1, 1)
        perturbed_data = t_expand * x_batch + (1. - t_expand) * noise_batch
        predict = self.model(perturbed_data, t * 999)
        loss = (predict - target).square().mean()
        return loss

# class
