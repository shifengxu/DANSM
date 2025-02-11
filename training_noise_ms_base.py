"""
This is a base class, can not work by itself.
It is involved in training process, and the (Gaussian) noise is selected by match-scope.
The noise selection strategy is match-scope, not selective-set.
    selective-set: given one sample, find its noise in a selective-set. The set size could be 10, 20 or others.
    match-scope  : given a batch (e.g. 10) of sample and a batch of noise, re-order the noise and make sure the
                   sample-noise pair has short distance.
"""
from models.match_scope_manager import MatchScopeManager
from utils import log_info as log_info

class TrainingNoiseByMatchScopeBase:
    def __init__(self, args):
        super().__init__(args)
        self.ms_size = args.ms_size
        c, h, w = args.C, args.H // args.f, args.W // args.f
        self.ms_mgr = MatchScopeManager(self.ms_size, c, h, w, args.device)

    def adjust_noise(self, x_batch, noise_batch, ss_size):
        """
        Find proper noise for sample, and the searching scope is called selective set.
        the selective set size is called ss_size.
        Since the noise_batch is already generated, so just need to search other (ss_size-1) noises.
        The normal iteration is to iterate (ss_size-1) times for x_batch.
        But if ss_size is too large ,such as 10000, the normal iteration will be too slow.
        Therefore, in such case, we iterate each sample in x_batch.
        Specifically, for each single sample, we create (ss_size-1) noises to select.
        :param x_batch:
        :param noise_batch:
        :param ss_size:
        :return:
        """
        if self.ms_size <= 1:
            return
        if not hasattr(self, '_adjust_noise_by_ms'):
            log_info(f"TrainingLatentMatchScopeBase::adjust_noise(x, noise, ss_size={ss_size})")
        for i in range(len(x_batch)):  # For each image, find the nearest noise in ns
            self.ms_mgr.assign_nearest(x_batch, noise_batch, i)
        if not hasattr(self, '_adjust_noise_by_ms'):
            setattr(self, '_adjust_noise_by_ms', True)
            log_info(f"  x_batch size: {len(x_batch)}")
            log_info(f"  ms_mgr.state: {self.ms_mgr.state_str()}")
            log_info(f"  ms_mgr.total: {self.ms_mgr.total()}")
        return


# class
