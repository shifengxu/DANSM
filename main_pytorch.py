"""
op_dat: Optimal Transport - Distance-Aware Training
"""

import sys
import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
prt_dir = os.path.dirname(cur_dir)  # parent dir
if cur_dir not in sys.path:
    sys.path.append(cur_dir)
    print(f"sys.path.append({cur_dir})")
if prt_dir not in sys.path:
    sys.path.append(prt_dir)
    print(f"sys.path.append({prt_dir})")

# This is for "ninja", which is necessary in model construction.
# "ninja" is an exe file, locates in the same folder as "python".
# Sample location: ~/anaconda3/envs/restflow/bin/
exe_dir = os.path.dirname(sys.executable)
env_path = os.environ['PATH']
if exe_dir not in env_path:
    os.environ['PATH'] = f"{exe_dir}:{env_path}"
    print(f"Environment variable PATH has inserted new dir: {exe_dir}")

import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from utils import log_info as log_info

torch.set_printoptions(sci_mode=False)

def parse_args_and_config():
    from utils import str2bool
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--config", type=str, default='cifar10')
    parser.add_argument("--todo", type=str, default='train', help="train|sample")
    parser.add_argument("--model", type=str, default='rfm', help="cm|dm|rfm")
    parser.add_argument("--unet", type=str, default='', help="ermongroup|ncsn")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[7])
    parser.add_argument("--seed", type=int, default=123, help="Random seed. 0 means ignore")
    parser.add_argument("--loss_norm", type=str, default='lpips')
    parser.add_argument("--log_interval", type=int, default=10)

    # data
    parser.add_argument("--data_dir", type=str, default="./download_dataset")
    parser.add_argument("--batch_size", type=int, default=50, help="0 mean to use size from config file")
    parser.add_argument("--ms_size", type=int, default=1000, help="match scope size")
    parser.add_argument("--ms_method", type=str, default="")
    parser.add_argument("--ms_ckpt_path", type=str, default="./ckpt/sscd_TV_RESNET50_E500.pt")
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--C", type=int, default=4)
    parser.add_argument("--f", type=int, default=8)

    # training
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=100, help="0 mean epoch number from config file")
    parser.add_argument('--ema_rate', type=float, default=0.999, help='mu in EMA. 0 means using value from config')
    parser.add_argument("--resume_ckpt_path", type=str, default='')
    parser.add_argument("--save_ckpt_path", type=str, default='./ckpt.pth')
    parser.add_argument("--save_ckpt_interval", type=int, default=50, help="count by epoch")
    parser.add_argument("--save_ckpt_eval", type=str2bool, default=False, help="Calculate FID/IS when save ckpt")

    # sampling
    parser.add_argument("--sample_count", type=int, default=50000)
    parser.add_argument("--sample_batch_size", type=int, default=500)
    parser.add_argument("--sample_ckpt_path", type=str, default='./ckpt.pth')
    parser.add_argument("--sample_output_dir", type=str, default="./generated")
    parser.add_argument("--sample_steps_arr", nargs='*', type=int, default=[1])
    parser.add_argument("--sample_init_ts_arr", nargs='*', type=float, default=[0.])
    parser.add_argument("--sample_isc_flag", type=str2bool, default=False, help="calculate IS for samples")
    parser.add_argument("--fid_input1", type=str, default="cifar10-train")
    parser.add_argument("--sd_ckpt_path", type=str, default='./checkpoints/v2-1_512-ema-pruned.ckpt')

    # parallel generation
    parser.add_argument("--pg_ckpt_rfm", type=str, default="./ckpt_RFM_gnobitab_LSUN_Bedroom.pth")
    parser.add_argument("--pg_ckpt_dm", type=str, default="./ckpt_DM-lsun-bedroom-model-2388000.ckpt")
    parser.add_argument("--pg_ckpt_cm", type=str, default="./ckpt_CT_bedroom256.pt")

    args = parser.parse_args()

    # add device
    gpu_ids = args.gpu_ids
    device = torch.device(f"cuda:{gpu_ids[0]}") if torch.cuda.is_available() and gpu_ids else torch.device("cpu")
    args.device = device
    log_info(f"gpu_ids : {gpu_ids}")
    log_info(f"device  : {device}")

    # set random seed
    seed = args.seed  # if seed is 0. then ignore it.
    log_info(f"args.seed : {seed}")
    if seed:
        log_info(f"  torch.manual_seed({seed})")
        log_info(f"  np.random.seed({seed})")
        torch.manual_seed(seed)
        np.random.seed(seed)
    if seed and torch.cuda.is_available():
        log_info(f"  torch.cuda.manual_seed_all({seed})")
        torch.cuda.manual_seed_all(seed)
    log_info(f"final seed: torch.initial_seed(): {torch.initial_seed()}")

    cudnn.benchmark = True
    return args

def main():
    args = parse_args_and_config()
    log_info(f"pid : {os.getpid()}")
    log_info(f"cwd : {os.getcwd()}")
    log_info(f"host: {os.uname().nodename}")
    log_info(f"args: {args}")

    log_info(f"main_pytorch -> {args.todo} ===================================")
    if args.todo == 'sample':
        from framework_rfm.rectified_flow_sampling import RectifiedFlowSampling
        runner = RectifiedFlowSampling(args)
        runner.sample(sample_steps=args.sample_steps_arr[0])
    elif args.todo == 'train_adjust' or args.todo == 'train':
        if args.model == 'rfm':
            from framework_rfm.rectified_flow_training_adjust import RectifiedFlowTrainingAdjust
            runner = RectifiedFlowTrainingAdjust(args)
            runner.train()
        elif args.model == 'dm':
            from framework_ddim.diffusion_training import DiffusionTraining
            runner = DiffusionTraining(args)
            runner.train()
        else:
            raise ValueError(f"Invalid model {args.model} with todo {args.todo}")
    elif args.todo == 'train_latent_ms':
        if args.model == 'rfm':
            from framework_rfm.rectified_flow_training_latent_ms import RectifiedFlowTrainingLatentMatchScope
            runner = RectifiedFlowTrainingLatentMatchScope(args)
            runner.train()
        elif args.model == 'dm':
            from framework_ddim.diffusion_training_latent_ms import DiffusionTrainingLatentMatchScope
            runner = DiffusionTrainingLatentMatchScope(args)
            runner.train()
        else:
            raise ValueError(f"Invalid model {args.model} with todo {args.todo}")
    elif args.todo == 'train_latent_hungarian':
        if args.model == 'rfm':
            from framework_rfm.rectified_flow_training_latent_ms import RectifiedFlowTrainingLatentMatchScope
            runner = RectifiedFlowTrainingLatentMatchScope(args)
            runner.train_with_hungarian_algo()
        else:
            raise ValueError(f"Invalid model {args.model} with todo {args.todo}")
    elif args.todo == 'sample_all':
        if args.model == 'rfm':
            from framework_rfm.rectified_flow_misc import RectifiedFlowMiscellaneous
            runner = RectifiedFlowMiscellaneous(args)
            runner.sample_all_ts_and_step()
        elif args.model == 'dm':
            from framework_ddim.diffusion_sampling import DiffusionSampling
            runner = DiffusionSampling(args)
            runner.sample_all()
        else:
            raise ValueError(f"Invalid model {args.model} with todo {args.todo}")
    elif args.todo == 'compare_distance':
        from framework_rfm.rectified_flow_misc import RectifiedFlowMiscellaneous
        runner = RectifiedFlowMiscellaneous(args)
        runner.compare_distance()
    elif args.todo == 'ldm':
        from ldm.stable_diffusion_latent import StableDiffusionLatent
        sdl = StableDiffusionLatent(args)
        sdl.init_model()
    elif args.todo == 'resize_images':
        from framework_rfm.rectified_flow_misc import RectifiedFlowMiscellaneous
        RectifiedFlowMiscellaneous.resize_images()
    else:
        raise Exception(f"Invalid todo: {args.todo}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
