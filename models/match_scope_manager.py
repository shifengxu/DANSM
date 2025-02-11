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
        self.sscd_model  = None  # SSCD model
        self.sampler     = None  # sampler of the generative model
        self.sample_bs   = None  # sample batch size
        self.sample_step = None

    def init_sscd(self, args, c_sscd, sampler):
        self.ms_method    = args.ms_method
        self.ms_ckpt_path = args.ms_ckpt_path
        self.sample_bs    = args.sample_batch_size
        self.sample_step  = args.sample_steps_arr[0]
        self.sampler      = sampler
        log_info(f"MatchScopeManager::init_sscd()")
        log_info(f"  ms_method    : {self.ms_method}")
        log_info(f"  ms_ckpt_path : {self.ms_ckpt_path}")
        log_info(f"  sample_bs    : {self.sample_bs}")
        log_info(f"  sample_step  : {self.sample_step}")
        log_info(f"  sampler      : {type(sampler).__name__}")
        from sscd.models.model import Model as SSCD_Model
        m_backbone, feature_dims, m_pool_param = c_sscd.model_backbone, c_sscd.feature_dims, 3
        log_info(f"  SSCD backbone    : {m_backbone}")
        log_info(f"  SSCD feature_dims: {feature_dims}")
        log_info(f"  SSCD pool_param  : {m_pool_param}")
        sscd_model = SSCD_Model(backbone=m_backbone, dims=feature_dims, pool_param=m_pool_param)
        log_info(f"  load: {self.ms_ckpt_path}")
        state_dict = torch.load(self.ms_ckpt_path, map_location=self.device)
        sscd_model.load_state_dict(state_dict)
        sscd_model = sscd_model.to(args.device)
        log_info(f"  sscd_model = sscd_model.to({args.device})")
        if len(args.gpu_ids) > 1:
            sscd_model = torch.nn.DataParallel(sscd_model, device_ids=args.gpu_ips)
            log_info(f"  sscd_model = torch.nn.DataParallel(sscd_model, device_ids={args.gpu_ips})")
        sscd_model.eval()
        self.sscd_model = sscd_model
        self.noise_to_feature_vector()

    def noise_to_feature_vector(self):
        # step 1: from noise to sample
        # step 2: from sample to sscd vector
        flag = self.flush_count == 0

        def _log(*args):
            if flag: log_info(*args)

        ttl = len(self.noises)
        b_sz = self.sample_bs
        b_cnt = ttl // b_sz
        if b_cnt * b_sz < ttl: b_cnt += 1
        step = self.sample_step
        feature_arr = []
        _log(f"MatchScopeManager::noise_to_feature_vector()...")
        _log(f"  step   : {step}")
        _log(f"  b_size : {b_sz}")
        _log(f"  ttl    : {ttl}")
        noise_clone = self.noises.clone()
        self.sampler.model.eval()
        with torch.no_grad():
            for b_idx in range(b_cnt):
                i, j = b_idx * b_sz, (b_idx + 1) * b_sz
                if j > ttl: j = ttl
                z0 = noise_clone[i:j]   # must use the cloned noise, as sampling process will change it.
                x0 = self.sampler.sample_batch(z0, step)
                feature = self.sscd_model(x0)
                feature_arr.append(feature)
            # for
        # with
        self.sampler.model.train()
        self.nf_vectors = torch.cat(feature_arr, dim=0)
        _log(f"  noises : {self.noises.size()}")
        _log(f"  nf_vec : {self.nf_vectors.size()}")
        _log(f"MatchScopeManager::noise_to_feature_vector()...Done")

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

    def assign_nearest_sscd(self, sfv_batch, noise_batch, idx):
        """
        :param sfv_batch  : sample feature vector, in batch
        :param noise_batch: noise batch. Such noise is used in training
        :param idx        : index on which sample-noise pair to handle
        :return:
        """
        if self.matched_count >= self.ms_size:
            # current noises have been all matched. Need to re-flush.
            self.noises = torch.randn(self.ms_size, self.c, self.h, self.w, device=self.device, requires_grad=False)
            self.matched_count = 0
            self.flush_count += 1
            self.noise_to_feature_vector()
        sfv = sfv_batch[idx:idx+1]
        sim_arr = (sfv * self.nf_vectors).sum(dim=1)
        result = torch.max(sim_arr, dim=0, keepdim=False)
        val, find_idx = result.values, result.indices
        noise_batch[idx, :] = self.noises[find_idx, :]
        # the following concatenation might be time-consuming
        tmp = self.noises
        self.noises = torch.cat([tmp[:find_idx], tmp[find_idx+1:]])
        del tmp
        tmp = self.nf_vectors
        self.nf_vectors = torch.cat([tmp[:find_idx], tmp[find_idx+1:]])
        del tmp
        self.matched_count += 1

    def reorder_noises(self, samples, z0_noises):
        flag = self.matched_count == 0 and self.flush_count == 0

        def _log(*args):
            if flag: log_info(*args)

        if self.ms_size <= 0:
            return 0
        if self.ms_method is None or self.ms_method == '' or self.ms_method.lower() == 'none':
            _log(f"MatchScopeManager::reorder_noises() ms_method='{self.ms_method}'...")
            for i in range(len(samples)):
                self.assign_nearest(samples, z0_noises, i)
            _log(f"MatchScopeManager::reorder_noises() ms_method='{self.ms_method}'...Done")
        elif self.ms_method == 'sscd':
            _log(f"MatchScopeManager::reorder_noises() sscd. sample to feature vector...")
            with torch.no_grad():
                sfv = self.sscd_model(samples) # sample feature vector
            _log(f"MatchScopeManager::reorder_noises() sscd. reorder noises based on fv...")
            for i in range(len(samples)):
                self.assign_nearest_sscd(sfv, z0_noises, i)
            _log(f"MatchScopeManager::reorder_noises() sscd...Done")
        else:
            raise ValueError(f"Invalid ms_method: {self.ms_method}")
        return True

    def total(self):
        tt = self.ms_size * self.flush_count + self.matched_count
        return tt

    def state_str(self):
        tt = self.ms_size * self.flush_count + self.matched_count
        s = f"MS:{tt}={self.ms_size}*{self.flush_count}+{self.matched_count}"
        return s

# class
