# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ConvSSM/blob/main/LICENSE
#
# Written by Jimmy Smith
# ------------------------------------------------------------------------------

import os
import numpy as np
from typing import Optional, Any
import optax
import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import flaim

from src.models.convS5.conv_ops import VmapBasicConv
from src.models.convS5.diagonal_ssm import init_ConvS5SSM
from src.models.base import ResNetEncoder, ResNetDecoder
from src.models.convS5.layers import StackedLayers


def reshape_data(frames):
    # Make(seq_len, dev_bsz, H, W, in_dim)
    frames = frames.transpose(1, 0, 2, 3, 4)
    return frames


class PF_HCONVS5_NOVQ(nn.Module):
    config: Any
    training: bool
    parallel: bool
    dtype: Optional[Any] = jnp.float32

    @property
    def metrics(self):
        metrics = ['loss', 'mse_loss', 'l1_loss']
        return metrics

    def setup(self):
        config = self.config

        # Sequence Model
        self.ssm = init_ConvS5SSM(config.ssm['ssm_size'],
                                  config.ssm['blocks'],
                                  config.ssm['clip_eigs'],
                                  config.d_model,
                                  config.ssm['B_kernel_size'],
                                  config.ssm['C_kernel_size'],
                                  config.ssm['D_kernel_size'],
                                  config.ssm['dt_min'],
                                  config.ssm['dt_max'],
                                  config.ssm['C_D_config'])
        self.sequence_model = StackedLayers(**self.config.seq_model,
                                            ssm=self.ssm,
                                            training=self.training,
                                            d_model=self.config.d_model,
                                            parallel=self.parallel)

        initial_states = []
        bsz_device, _ = divmod(config.batch_size, jax.device_count())
        for i in range(config.seq_model['n_layers']):
            state = jnp.zeros((bsz_device,
                config.latent_height,
                config.latent_width,
                config.ssm['ssm_size']//2))
            initial_states.append((state, state))  # Add states for E/I
        self.initial_states = initial_states
        # self.action_embeds = nn.Embed(config.action_dim + 1, config.action_embed_dim, dtype=self.dtype)
        self.action_conv = VmapBasicConv(k_size=1,
                                         out_channels=config.d_model)
        self.readout = nn.Dense(2)
        self.preproc = nn.Conv(config.ssm["ssm_size"], kernel_size=[1, 1])

        # Encoder
        # self.encoder = ResNetEncoder(**config.encoder, dtype=self.dtype)
        # model, weights = flaim.get_model(
        #     # model_name='convnext_small',
        #     model_name='resnet50',
        #     pretrained='in1k_256',
        #     n_classes=-1,
        #     # jit=True,
        # )
        # self.encoder = model
        # self.weights = weights
        kernel = np.load(os.path.join("weights", "gabors_for_contours_7.npy"), allow_pickle=True, encoding="latin1").item()["s1"][0]
        ks = kernel.shape
        kernel = kernel.reshape(ks[3], ks[2], ks[0], ks[1])
        kernel = kernel[1:]  # Reduce to 24 channels
        kernel = np.ascontiguousarray(kernel)
        self.weights = kernel

        # # Decoder
        # out_dim = self.config.channels
        # self.decoder = ResNetDecoder(**config.decoder, image_size=0,
        #                              out_dim=out_dim, dtype=self.dtype)

    def sample_timestep(self, encoding, initial_states, action):
        inp = self.encode(encoding)

        action = self.action_embeds(action)
        action = jnp.tile(action[:, :, None, None], (1, 1, *inp.shape[2:4], 1))
        inp = jnp.concatenate([inp[:, :-1], action[:, 1:]], axis=-1)

        # inp is BTHWC, convS5 model needs TBHWC
        inp = reshape_data(inp)
        inp = self.action_conv(inp)
        last_states, deter = self.sequence_model(inp, initial_states)
        deter = reshape_data(deter)  # Now BTHWC

        recon_logits, recon = self.reconstruct(deter)
        return recon, recon_logits, recon, last_states

    def encode(self, encodings):
        # out = jax.vmap(self.encoder, 1, 1)(encodings)
        # encodings = encodings.repeat(3, -1)
        # _, out = self.encoder.apply(self.weights, encodings, training=False)  # , mutable=['intermediates'])
        # _, out = self.encoder.apply(self.weights, encodings, training=False, mutable=['intermediates'])
        # out = jax.lax.stop_gradient(out["intermediates"]["stage_1"][0])
        # out = self.preproc(out)
        out = lax.conv_general_dilated(encodings, self.weights, (1, 1), padding="SAME", dimension_numbers=["NHWC", "OIHW", "NHWC"])
        # out = jax.lax.stop_gradient(out)

        # Replicate over timesteps
        out = out[None].repeat(self.config.seq_len, 0)

        return out

    # def gabor_encode(self, encodings):

    def condition(self, encodings, actions, initial_states=None):
        if initial_states is None:
            initial_states = self.initial_states

        inp = self.encode(encodings)

        # # inp is BTHWC, convS5 model needs TBHWC
        # inp = reshape_data(inp)
        inp = self.action_conv(inp)
        # inp = inp[0].max((1, 2))
        last_states, deter = self.sequence_model(inp, initial_states)
        # deter = reshape_data(deter)  # swap back to BTHWC
        # out = deter[-1].mean((1, 2))
        out = deter[-1].max((1, 2))
        encodings = self.readout(out)

        return None, encodings, None, None, None

    def reconstruct(self, deter):
        recon_logits = jax.vmap(self.decoder, 1, 1)(deter)
        recon = nn.tanh(recon_logits)
        return recon, recon

    def __call__(self, video, actions, deterministic=False):
        # video: BTHWC, actions: BT
        print(video.shape)
        encodings = video
        _, encodings, _, _, _ = self.condition(encodings, actions)

        loss = optax.softmax_cross_entropy(encodings, jax.nn.one_hot(actions, 2)).mean()
        mse_loss = loss
        l1_loss = loss

        out = dict(loss=loss, mse_loss=mse_loss, l1_loss=l1_loss)
        return out

