import argparse
import subprocess
import re
import torch.nn as nn
import math
import datetime
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def count_parameters(model: nn.Module, log_fn=None):
    def prt(x):
        if log_fn: log_fn(x)

    def convert_size_str(size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])

    prt(f"count_parameters({type(model)}) ------------")
    prt('  requires_grad  name  count  size')
    counter = 0
    for name, param in model.named_parameters():
        s_list = list(param.size())
        prt(f"  {param.requires_grad} {name} {param.numel()} = {s_list}")
        c = param.numel()
        counter += c
    # for
    str_size = convert_size_str(counter)
    prt(f"  total  : {counter} {str_size}")
    return counter, str_size

def log_info(*args):
    dtstr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{dtstr}]", *args)

def get_time_ttl_and_eta(time_start, elapsed_iter, total_iter):
    """
    Get estimated total time and ETA time.
    :param time_start:
    :param elapsed_iter:
    :param total_iter:
    :return: string of elapsed time, string of ETA
    """

    def sec_to_str(sec):
        val = int(sec)  # seconds in int type
        s = val % 60
        val = val // 60  # minutes
        m = val % 60
        val = val // 60  # hours
        h = val % 24
        d = val // 24  # days
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"

    elapsed_time = time.time() - time_start  # seconds elapsed
    elp = sec_to_str(elapsed_time)
    if elapsed_iter == 0:
        eta = 'NA'
    else:
        # seconds
        eta = elapsed_time * (total_iter - elapsed_iter) / elapsed_iter
        eta = sec_to_str(eta)
    return elp, eta

def calc_fid(gpu, fid_subprocess: bool, input1, input2, logger=log_info):
    if fid_subprocess:
        cmd = f"fidelity --gpu {gpu} --fid --input1 {input1} --input2 {input2} --silent"
        logger(f"cmd: {cmd}")
        cmd_arr = cmd.split(' ')
        res = subprocess.run(cmd_arr, stdout=subprocess.PIPE)
        output = str(res.stdout)
        logger(f"out: {output}")  # frechet_inception_distance: 16.5485\n
        m = re.search(r'frechet_inception_distance: (\d+\.\d+)', output)
        fid = float(m.group(1))
    else:
        import torch_fidelity
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=input1,
            input2=input2,
            cuda=True,
            isc=False,
            fid=True,
            kid=False,
            verbose=False,
            samples_find_deep=True,
        )
        fid = metrics_dict['frechet_inception_distance']
    return fid

def calc_fid_isc(gpu, input1, input2, isc_flag=True, logger=log_info):
    if isc_flag:
        cmd = f"fidelity --gpu {gpu} --fid --isc --input1 {input1} --input2 {input2} --silent"
    else:
        cmd = f"fidelity --gpu {gpu} --fid --input1 {input1} --input2 {input2} --silent"
    logger(f"cmd: {cmd}")
    cmd_arr = cmd.split(' ')
    res = subprocess.run(cmd_arr, stdout=subprocess.PIPE)
    output = str(res.stdout)
    # inception_score_mean: 11.24431
    # inception_score_std: 0.09522244
    # frechet_inception_distance: 71.2743
    logger(f"out: {output}")
    m = re.search(r'frechet_inception_distance: (\d+\.\d+)', output)
    fid = float(m.group(1))
    if isc_flag:
        m = re.search(r'inception_score_mean: (\d+\.\d+)', output)
        is_mean = float(m.group(1))
        m = re.search(r'inception_score_std: (\d+\.\d+)', output)
        is_std = float(m.group(1))
    else:
        is_mean, is_std = 0., 0.
    return fid, is_mean, is_std

def calc_line_segment_distance(ls1_p1, ls1_p2, ls2_p1, ls2_p2, dansm_style=False):
    """
    Line segment distance, which is the minimum distance between 2 points based on timestep t.
    :param ls1_p1: line segment 1, point 1. z_1 in DANSM
    :param ls1_p2: line segment 1, point 2. x_1 in DANSM
    :param ls2_p1: line segment 2, point 1. z_2 in DANSM
    :param ls2_p2: line segment 2, point 2. x_2 in DANSM
    :param dansm_style: True or False. Distance-aware noise-sample matching, which t=0 is sample and t=1 is noise
    :return:
    """
    import torch

    if dansm_style:
        vec_a = torch.sub(ls1_p2, ls2_p2)   # vector V in DANSM
        vec_b = torch.sub(ls1_p1, ls2_p1)   # vector U in DANSM
    else:
        vec_a = torch.sub(ls1_p1, ls2_p1)
        vec_b = torch.sub(ls1_p2, ls2_p2)
    bs_a, bs_b = len(vec_a), len(vec_b)
    vec_a = vec_a.view(bs_a, -1)
    vec_b = vec_b.view(bs_b, -1)
    a_m_b = vec_a - vec_b
    numerator = (vec_a * a_m_b).sum(dim=1)
    denominator = (a_m_b * a_m_b).sum(dim=1)
    denominator += 1e-8 # to avoid zero
    t_star = numerator / denominator
    flag1 = t_star < 0
    result = numerator # reuse this variable
    result[flag1] = vec_a.norm(dim=1)[flag1]
    t_star[flag1] = 0
    flag2 = t_star > 1
    result[flag2] = vec_b.norm(dim=1)[flag2]
    t_star[flag2] = 1
    flag3 = ~flag1 & ~flag2
    weight_a, weight_b = (1. - t_star).unsqueeze(1), t_star.unsqueeze(1)
    result[flag3] = (weight_a * vec_a + weight_b * vec_b).norm(dim=1)[flag3]
    return result, t_star

def calc_avg_min_inter_path_distance(noises, images, dansm_style=False, keep_dim=False, log_fn=log_info):
    """
    Calculate inter-path distance for all paths.
    dansm_style means distance-aware noise-sample matching, where timestep t=0 is sample and t=1 is noise.
    """
    import torch
    noise_cnt = len(noises)    # path count
    image_cnt = len(images)
    func_args_str = f"noise_cnt={noise_cnt}, image_cnt={image_cnt}, dansm_style={dansm_style}, keep_dim={keep_dim}"
    if log_fn is None: log_fn = lambda s: None
    log_fn(f"calc_avg_min_inter_path_distance({func_args_str})...")
    assert noise_cnt == image_cnt
    if noise_cnt <= 1:  # handle special case
        return 0., 0.
    p_dist_sum, t_star_sum = None, None
    p_dist_min, t_star_min = None, None     # for minimal inter-path distance
    t_start = time.time()
    for k in range(1, noise_cnt):    # shift
        noi2 = torch.roll(noises, shifts=k, dims=0)
        img2 = torch.roll(images, shifts=k, dims=0)
        p_dist, t_star = calc_line_segment_distance(noises, images, noi2, img2, dansm_style=dansm_style)
        if p_dist_sum is None:
            p_dist_sum, t_star_sum = p_dist.detach().clone().float(), t_star.detach().clone().float()
            p_dist_min, t_star_min = p_dist.detach().clone().float(), t_star.detach().clone().float()
        else:
            p_dist_sum += p_dist.float()
            t_star_sum += t_star.float()
            flag = torch.lt(p_dist, p_dist_min)
            p_dist_min[flag] = p_dist[flag].float()
            t_star_min[flag] = t_star[flag].float()
        if k < 10 or k < 100 and k % 10 == 0 or k < 1000 and k % 100 == 0 or\
                k < 10000 and k % 1000 == 0 or k % 10000 == 0:
            elp, eta = get_time_ttl_and_eta(t_start, k, noise_cnt-1)
            log_fn(f"  shift:{k:05d}, elp:{elp}, eta:{eta}")
    # for
    p_dist_avg = p_dist_sum / (noise_cnt - 1)
    t_star_avg = t_star_sum / (noise_cnt - 1)
    if not keep_dim:
        p_dist_avg = p_dist_avg.mean()
        t_star_avg = t_star_avg.mean()
        p_dist_min = p_dist_min.mean()
        t_star_min = t_star_min.mean()
    log_fn(f"calc_avg_min_inter_path_distance({func_args_str})...Done")
    return p_dist_avg, t_star_avg, p_dist_min, t_star_min

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
