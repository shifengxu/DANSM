import os
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config, count_params
from utils import log_info


class StableDiffusionLatent:
    def __init__(self, args):
        log_info(f"StableDiffusionLatent::__init__()...")
        self.args = args
        abs_path = os.path.abspath(__file__)
        cur_dir = os.path.dirname(abs_path)  # current folder
        cfg_file = os.path.join(cur_dir, "v2-inference.yaml")
        log_info(f"  cfg_file: {cfg_file}")
        self.config = OmegaConf.load(cfg_file)
        self.ld_model = self.init_model()    # latent diffusion model
        log_info(f"StableDiffusionLatent::__init__()...Done")

    def init_model(self):
        """
        Init LatentDiffusion model, but only keep its decoder.
        LatentDiffusion -> DiffusionWrapper -> UNetModel
        """
        args, config = self.args, self.config
        ckpt = args.sd_ckpt_path
        log_info(f"StableDiffusionLatent::init_model()...")
        log_info(f"  ckpt  : {ckpt}")
        # config.model has target: ldm.models.diffusion.ddpm.LatentDiffusion
        log_info(f"  create ldm.models.diffusion.ddpm.LatentDiffusion...")
        ld_model = instantiate_from_config(config.model)
        log_info(f"  create ldm.models.diffusion.ddpm.LatentDiffusion...Done")

        log_info(f"  torch.load({ckpt})...")
        tl_sd = torch.load(ckpt, map_location=args.device, weights_only=False)  # torch loaded state dict
        log_info(f"  torch.load({ckpt})...Done")
        if "global_step" in tl_sd:
            log_info(f"  Global Step: {tl_sd['global_step']}")
        sd = tl_sd["state_dict"]
        log_info(f"  ld_model.load_state_dict()...")
        ld_model.load_state_dict(sd, strict=False)
        log_info(f"  ld_model.load_state_dict()...Done")
        log_info(f"  ld_model.eval()")
        ld_model.eval()
        param_cnt = count_params(ld_model, verbose=False)
        log_info(f"  ld_model size: {param_cnt*1e-6:7.2f} M")
        # now ld_model       is LatentDiffusion class
        # and ld_model.model is DiffusionWrapper class

        # delete unnecessary parts, to save GPU memories
        # delete Unet part. model size (parameters): 1303.60 M -> 437.69 M
        del ld_model.model
        param_cnt = count_params(ld_model, verbose=False)
        log_info(f"  ld_model size: {param_cnt*1e-6:7.2f} M <- after delete ld_model.model")

        # delete EMA part
        if hasattr(ld_model, 'model_ema') and ld_model.model_ema:
            del ld_model.model_ema
            param_cnt = count_params(ld_model, verbose=False)
            log_info(f"  ld_model size: {param_cnt*1e-6:7.2f} M <- after delete ld_model.model_ema")

        # delete cond_stage_model.  model size (parameters): 437.69 M -> 83.65 M
        del ld_model.cond_stage_model
        param_cnt = count_params(ld_model, verbose=False)
        log_info(f"  ld_model size: {param_cnt * 1e-6:7.2f} M <- after delete ld_model.cond_stage_model")

        # ld_model.first_stage_model size (parameters): 83.65 M
        param_cnt = count_params(ld_model.first_stage_model, verbose=False)
        log_info(f"  ld_model.first_stage_model size: {param_cnt*1e-6:7.2f} M")

        log_info(f"  ld_model.to({args.device})")
        ld_model = ld_model.to(args.device)
        log_info(f"StableDiffusionLatent::init_model()...Done")
        return ld_model

    def decode_latent(self, latent_batch):
        img_batch = self.ld_model.decode_first_stage(latent_batch)
        return img_batch

# class
