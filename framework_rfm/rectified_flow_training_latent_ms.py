"""
Distance-Aware Training. adjust the data order in match-scope.
Train on latent space.
"""
from framework_rfm.rectified_flow_training_base import RectifiedFlowTrainingBase
from training_latent_base import TrainingLatentBase
from training_noise_ms_base import TrainingNoiseByMatchScopeBase
from utils import log_info

class RectifiedFlowTrainingLatentMatchScope(RectifiedFlowTrainingBase,
                                            TrainingNoiseByMatchScopeBase,
                                            TrainingLatentBase):
    def __init__(self, args):
        log_info(f"RectifiedFlowTrainingLatentMatchScope::__init__()...")
        RectifiedFlowTrainingBase.__init__(self, args)
        TrainingNoiseByMatchScopeBase.__init__(self, args)
        TrainingLatentBase.__init__(self, args)
        log_info(f"RectifiedFlowTrainingLatentMatchScope::__init__()...Done")

# class
