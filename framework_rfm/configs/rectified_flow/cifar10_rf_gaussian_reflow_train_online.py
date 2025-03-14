# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Use the generated data for training new rectified flow with reflow"""
import ml_collections

from framework_rfm.configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'rectified_flow'
  training.continuous = False
  training.snapshot_freq = 20000
  training.reduce_mean = True

  # sampling
  sampling = config.sampling
  sampling.method = 'rectified_flow'
  sampling.init_type = 'gaussian' 
  sampling.init_noise_scale = 1.0
  sampling.use_ode_sampler = 'rk45'

  # reflow
  config.reflow = reflow = ml_collections.ConfigDict()
  reflow.reflow_type = 'train_online_reflow' # NOTE: generate_data_from_z0, train_reflow, train_online_reflow
  reflow.reflow_t_schedule = 'uniform' # NOTE; t0, t1, uniform, or an integer k > 1
  reflow.reflow_loss = 'l2' # NOTE: l2, lpips, lpips+l2
  reflow.last_flow_ckpt = 'ckpt_path' # NOTE: the rectified flow model to fine-tune
  reflow.data_root = 'data_path' # NOTE: the folder to load the generated data

  # data
  data = config.data
  data.centered = True

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
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
  model.embedding_type = 'positional'
  model.fourier_scale = 16
  model.conv_size = 3

  return config
