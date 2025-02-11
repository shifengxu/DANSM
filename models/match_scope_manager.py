"""
Match scope manager.
match scope is the range in which we adjust the data order.
If match scope size is 1000, then we adjust the order of:
a), 1000 noises;
b), 1000 images;
And try to make their one-to-one distance shorter.
"""
import torch
from utils import log_info as log_info

class MatchScopeManager:
    def __init__(self, ms_size, c, h, w, device):
        self.ms_size = ms_size
        self.ms_method = None
        self.ms_ckpt_path = None
        self.c = c  # channel
        self.h = h  # height
        self.w = w  # width
        self.device = device
        self.matched_count = 0
        self.flush_count = 0
        log_info(f"MatchScopeManager::__init__()")
        log_info(f"  ms_size: {ms_size}")
        log_info(f"  c      : {c}")
        log_info(f"  h      : {h}")
        log_info(f"  w      : {w}")
        log_info(f"  device : {device}")
        if ms_size > 0:
            self.noises = torch.randn(self.ms_size, self.c, self.h, self.w, device=self.device)
        else:
            self.noises = None
        self.nf_vectors = None  # noise feature vectors
        log_info(f"  noise  : {type(self.noises).__name__}")
        self.sampler     = None  # sampler of the generative model
        self.sample_bs   = None  # sample batch size
        self.sample_step = None

    def assign_nearest(self, z1, z0, idx):
        if self.matched_count >= self.ms_size:
            # current noises have been all matched. Need to re-flush.
            self.noises = torch.randn(self.ms_size, self.c, self.h, self.w, device=self.device, requires_grad=False)
            self.matched_count = 0
            self.flush_count += 1
        single_data = z1[idx:idx+1]
        dist_arr = (single_data - self.noises).square().mean(dim=(1, 2, 3))
        result = torch.min(dist_arr, dim=0, keepdim=False)
        min_dist, min_idx = result.values, result.indices
        # have to assign z0 here.
        # Because if:
        #   noise = self.noises[min_idx]
        #   self.noises[min_idx, :] = 10000.
        #   return noise
        # the noise will also be changed to 10000.
        z0[idx, :] = self.noises[min_idx, :]
        self.noises[min_idx, :] = 10000. # make it big enough
        self.matched_count += 1

    def total(self):
        tt = self.ms_size * self.flush_count + self.matched_count
        return tt

    def state_str(self):
        tt = self.ms_size * self.flush_count + self.matched_count
        s = f"MS:{tt}={self.ms_size}*{self.flush_count}+{self.matched_count}"
        return s

# class
