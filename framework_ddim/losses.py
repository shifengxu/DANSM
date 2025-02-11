import torch

def noise_estimation_loss2(model,
                           x0: torch.Tensor,
                           t: torch.LongTensor,
                           e: torch.Tensor,
                           a: torch.Tensor,
                           keepdim=False):
    """
    Assume batch size is 250; and image size is 32*32.
    :param model:
    :param x0:  shape [250, 3, 32, 32]  x0
    :param t:   shape [250]             timestep
    :param e:   shape [250, 3, 32, 32]  epsilon
    :param a:   shape [1000].           alpha: (1-b).cumprod(dim=0)
    :param keepdim:
    :return:
    """
    at = a.index_select(0, t).view(-1, 1, 1, 1)  # alpha_t
    x = x0 * at.sqrt() + e * (1.0 - at).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3)), x
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0), x
