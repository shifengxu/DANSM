from utils import log_info
from training_noise_ms_base import TrainingNoiseByMatchScopeBase
from training_latent_base import TrainingLatentBase
from .diffusion_training_base import DiffusionTrainingBase


class DiffusionTrainingLatentMatchScope(DiffusionTrainingBase,
                                        TrainingNoiseByMatchScopeBase,
                                        TrainingLatentBase):
    def __init__(self, args):
        log_info(f"DiffusionTrainingLatentMatchScope::__init__()...")
        DiffusionTrainingBase.__init__(self, args)
        TrainingNoiseByMatchScopeBase.__init__(self, args)
        TrainingLatentBase.__init__(self, args)
        log_info(f"DiffusionTrainingLatentMatchScope::__init__()...Done")

# class
