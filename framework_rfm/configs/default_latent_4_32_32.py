import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 128
    training.n_iters = 1300001
    training.snapshot_freq = 100000
    training.log_freq = 50
    training.eval_freq = 100
    # store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 10000
    # produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.continuous = False
    training.reduce_mean = True
    training.sde = 'rectified_flow'

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.16
    sampling.sigma_variance = 0.0  # NOTE: sigma variance for turning ODE to SDE
    sampling.init_noise_scale = 1.0
    sampling.use_ode_sampler = 'rk45'
    sampling.ode_tol = 1e-5
    sampling.sample_N = 1000
    sampling.method = 'rectified_flow'
    sampling.init_type = 'gaussian'

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 9
    evaluate.end_ckpt = 26
    evaluate.batch_size = 1024
    evaluate.enable_sampling = False
    evaluate.num_samples = 50000
    evaluate.enable_loss = False
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = 'test'

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'latent_4_32_32'
    data.category = 'unknown'
    data.image_size = 32
    data.random_flip = True
    data.centered = True
    data.uniform_dequantization = False
    data.num_channels = 4

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 50
    model.num_scales = 1000
    model.beta_min = 0.1
    model.beta_max = 20.
    model.dropout = 0.15
    model.embedding_type = 'positional'
    model.name = 'ncsnpp'
    model.scale_by_sigma = False
    model.ema_rate = 0.999999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = False
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'none'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.
    model.fourier_scale = 16
    model.conv_size = 3

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0.
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config
