import torch.nn as nn


class EMAHelper(object):
    def __init__(self, module, mu=0.999):
        self.mu = mu
        self.num_updates = 0
        self.shadow = {}
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.collected_params = []

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        self.num_updates += 1
        decay = min(self.mu, (1. + self.num_updates) / (10. + self.num_updates))
        d = 1. - decay
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = d * param.data + decay * self.shadow[name].data
        return decay

    def ema_to_module(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def store(self, parameters):
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
