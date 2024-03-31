# ------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2022 BAIR OPEN RESEARCH COMMONS REPOSITORY
# To view a copy of this license, visit
# https://github.com/wilson1yan/teco/tree/master
# ------------------------------------------------------------------------------
#
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
import os.path as osp
import socket
import numpy as np
import time
import argparse
import yaml
import pickle
import wandb
import glob
from functools import partial

import jax
from jax import random, lax
import jax.numpy as jnp
from flax.training import checkpoints
from flax import jax_utils
from flax.serialization import to_bytes, from_bytes
import orbax.checkpoint as ocp

from src.data import Data
from src.train_utils import init_model_state, \
        get_first_device, ProgressMeter, seed_all
from src.utils import flatten, add_border, save_video_grid, add_border_mnist
from src.models import get_model
from src.models.sampling import sample_convSSM_noVQ, sample_transformer_noVQ, sample_convSSM, \
    sample_transformer
from src import runtime_metrics


def main():
    rng = random.PRNGKey(config.seed)
    rng, init_rng = random.split(rng)
    seed_all(config.seed)

    options = ocp.CheckpointManagerOptions(max_to_keep=3)
    mngr = ocp.CheckpointManager(config.output_dir, options=options)
    files = mngr.all_steps()

    if len(files):
        print('Found previous checkpoints: {}'.format(files))
        config.ckpt = mngr.latest_step()
    else:
        config.output_dir = ocp.test_utils.erase_and_create_empty(config.output_dir)
        config.ckpt = None

    if is_master_process:
        root_dir = os.environ['DATA_DIR']
        os.makedirs(osp.join(root_dir, 'wandb'), exist_ok=True)
        run = wandb.init(project='teco_old', config=config,
                   dir=root_dir, id=config.run_id, resume='allow')
        wandb.run.name = config.run_id
        # wandb.run.save()

    data = Data(config)
    train_loader = data.create_iterator(train=True)
    test_loader = data.create_iterator(train=False)
    p_get_eval_metrics = jax.pmap(partial(get_eval_metrics,
                                          open_loop_ctx=config.open_loop_ctx_1),
                                  axis_name='batch')
    steps_per_eval = int(config.eval_size/config.batch_size)
    batch = next(train_loader)
    batch = get_first_device(batch)
    model = get_model(config)
    print(config.model)
    print(model)

        
    model_par_eval = model(parallel=True, training=False)
    model_seq_eval = model(parallel=False, training=False)
    model = model(parallel=True, training=True)
    p_encode = None
    p_decode = None

    p_train_step = jax.pmap(train_step, axis_name='batch')
    p_test_step = jax.pmap(test_step, axis_name='batch')
    state, schedule_fn = init_model_state(init_rng, model, batch, config)
    if config.ckpt is not None:
        state = mngr.restore(config.ckpt, args=ocp.args.StandardRestore(state))
        print('Restored from checkpoint {}'.format(config.ckpt))
    iteration = int(state.step)
    state = jax_utils.replicate(state)

    rngs = random.split(rng, jax.local_device_count())

    best_test = 1000

    while iteration <= config.total_steps:
        test_loss = test(iteration, model, p_test_step, state, test_loader, schedule_fn, rngs)
        iteration, state, rngs = train(iteration, model, p_train_step, state, train_loader,
                                       schedule_fn, rngs)

        if iteration % config.save_interval == 0:
            test_loss = test(iteration, model_par_eval, p_test_step, state, test_loader, schedule_fn, rngs)
            # test_loss = test(iteration, model, p_test_step, state, test_loader, schedule_fn, rngs)
            if test_loss < best_test:
                best_test = min(test_loss, best_test)
                if is_master_process:
                    state_ = jax_utils.unreplicate(state)
                    success = mngr.save(state_.step, args=ocp.args.StandardSave(state_))
                    if success:
                        print('Saved checkpoint {} to {}'.format(state_.step, config.output_dir))
                    del state_  # Needed to prevent a memory leak bug
        iteration += 1
    mngr.wait_until_finished()


def train_step(batch, state, rng):
    new_rng, *rngs = random.split(rng, len(config.rng_keys) + 1)
    rngs = {k: r for k, r in zip(config.rng_keys, rngs)}

    def loss_fn(params):
        variables = {'params': params, **state.model_state}
        out = state.apply_fn(
            variables,
            video=batch['video'],
            actions=batch['actions'],
            deterministic=False,
            rngs=rngs
        )
        loss = out['loss']
        return loss, out

    aux, grads = jax.value_and_grad(
        loss_fn, has_aux=True)(state.params)
    out = aux[1]
    grads = jax.lax.pmean(grads, axis_name='batch')
    new_state = state.apply_gradients(
        grads=grads,
    )

    return new_state, out, new_rng


def test_step(batch, state, rng):
    # new_rng, *rngs = random.split(rng, len(config.rng_keys) + 1)
    # rngs = {k: r for k, r in zip(config.rng_keys, rngs)}

    def loss_fn(params):
        variables = {'params': params, **state.model_state}
        out = state.apply_fn(
            variables,
            video=batch['video'],
            actions=batch['actions'],
            # rngs=rngs
        )
        loss = out['loss']
        return loss, out

    loss, out = loss_fn(state.params)
    return loss, out


def train(iteration, model, p_train_step, state, train_loader, schedule_fn, rngs):
    progress = ProgressMeter(
        config.total_steps,
        ['time', 'data'] + model.metrics
    )

    end = time.time()
    while True:
        batch = next(train_loader)
        batch_size = batch['video'].shape[1]
        progress.update(data=time.time() - end)

        state, return_dict, rngs = p_train_step(batch=batch, state=state, rng=rngs)

        metrics = {k: return_dict[k].mean() for k in model.metrics}
        metrics = {k: v.astype(jnp.float32) for k, v in metrics.items()}
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})

        if is_master_process and iteration % config.log_interval == 0:
            wandb.log({'train/lr': schedule_fn(iteration)}, step=iteration)
            wandb.log({**{f'train/{metric}': val
                        for metric, val in metrics.items()}
                    }, step=iteration)

        progress.update(time=time.time() - end)
        end = time.time()

        if iteration % config.log_interval == 0:
            progress.display(iteration)

        if iteration % config.save_interval == 0 or \
        iteration % config.viz_interval == 0 or \
        iteration >= config.total_steps:
            return iteration, state, rngs
        
        iteration += 1


def test(iteration, model, p_test_step, state, test_loader, schedule_fn, rngs):
    progress = ProgressMeter(
        config.test_steps,
        ['time', 'data'] + model.metrics,
        prefix="Test:"
    )

    end = time.time()
    losses = []
    while True:
        batch = next(test_loader)
        batch_size = batch['video'].shape[1]
        progress.update(data=time.time() - end)

        loss, return_dict = p_test_step(batch=batch, state=state, rng=rngs)

        metrics = {k: return_dict[k].mean() for k in model.metrics}
        metrics = {k: v.astype(jnp.float32) for k, v in metrics.items()}
        progress.update(n=batch_size, **{k: v for k, v in metrics.items()})

        if is_master_process and iteration % config.log_interval == 0:
            # wandb.log({'test/lr': schedule_fn(iteration)}, step=iteration)
            wandb.log({**{f'test/{metric}': val
                        for metric, val in metrics.items()}
                    }, step=iteration)

        progress.update(time=time.time() - end)
        end = time.time()

        if iteration % config.log_interval == 0:
            progress.display(iteration)

        losses.append(loss)

        if iteration >= config.test_steps:
            return jnp.mean(jnp.stack(losses))
        iteration += 1


def visualize(model, iteration, state, test_loader, action_conditioned, open_loop_ctx, log_num, p_observe, p_imagine, p_encode, p_decode):
    batch = next(test_loader)

    if config.model in ["teco_convS5", "convS5", "teco_S5", "S5"]:
        predictions, real = sample_convSSM.sample(model, state, batch['video'], batch['actions'],
                                                  action_conditioned, open_loop_ctx, p_observe,
                                                  p_imagine,
                                                  p_encode, p_decode)

    elif config.model in ["teco_transformer", "transformer", "performer"]:
        predictions, real = sample_transformer.sample(model, state, batch['video'], batch['actions'],
                                                      action_conditioned, open_loop_ctx, p_observe, p_imagine,
                                                      p_encode, p_decode)

    predictions, real = jax.device_get(predictions), jax.device_get(real)
    predictions, real = predictions * 0.5 + 0.5, real * 0.5 + 0.5
    predictions = flatten(predictions, 0, 2)
    add_border(predictions[:, :open_loop_ctx], (0., 1., 0.))
    add_border(predictions[:, open_loop_ctx:], (1., 0., 0.))

    original = flatten(real, 0, 2)
    video = np.stack((predictions, original), axis=1)  # (NB)2THWC
    video = flatten(video, 0, 2)  # (NB2)THWC
    video = save_video_grid(video)
    video = np.transpose(video, (0, 3, 1, 2))
    if is_master_process:
        wandb.log({'viz/sample_{}'.format(log_num): wandb.Video(video, fps=20, format='gif')}, step=iteration)


def visualize_noVQ(model, iteration, state, test_loader, action_conditioned, open_loop_ctx, log_num, p_observe, p_imagine, p_encode, p_decode, eval_seq_len):
    batch = next(test_loader)

    if config.model in ["convS5_noVQ", 'convLSTM_noVQ', "pf_convS5_noVQ"]:
        predictions, real = sample_convSSM_noVQ.sample(model, state, batch['video'], batch['actions'], action_conditioned, open_loop_ctx, p_observe, p_imagine, p_encode, p_decode, eval_seq_len)
    else:
        predictions, real = sample_transformer_noVQ.sample(model, state, batch['video'], batch['actions'], action_conditioned, open_loop_ctx, p_observe, p_imagine, p_encode, p_decode, eval_seq_len)

    predictions, real = jax.device_get(predictions), jax.device_get(real)
    predictions, real = predictions * 0.5 + 0.5, real * 0.5 + 0.5
    predictions = flatten(predictions, 0, 2)
    add_border_mnist(predictions[:, :open_loop_ctx], (0., 1., 0.))
    add_border_mnist(predictions[:, open_loop_ctx:], (1., 0., 0.))

    original = flatten(real, 0, 2)
    print(predictions.shape, original.shape)
    video = np.stack((predictions, original), axis=1)  # (NB)2THWC
    video = flatten(video, 0, 2)  # (NB2)THWC
    video = save_video_grid(video)
    video = np.transpose(video, (0, 3, 1, 2))
    if video.shape[1] == 1:
        video = np.repeat(video, 3, axis=1)
    if is_master_process:
        wandb.log({'viz/sample_{}'.format(log_num): wandb.Video(video, fps=20, format='gif')}, step=iteration)


def get_eval_metrics(predictions, real, open_loop_ctx):
    predictions = predictions[:, open_loop_ctx:] * 0.5 + 0.5
    real = real[:, open_loop_ctx:] * 0.5 + 0.5

    ssim_val = runtime_metrics.get_ssim(predictions, real).mean()
    loss_lpips = runtime_metrics.get_lpips(predictions, real, net='alexnet').mean()
    psnr_val = runtime_metrics.get_psnr(predictions, real).mean()
    return lax.pmean(ssim_val, axis_name='batch'), \
           lax.pmean(loss_lpips, axis_name='batch'), lax.pmean(psnr_val, axis_name='batch')


def validate(p_get_eval_metrics, model, iteration, state, test_loader, steps_per_eval, action_conditioned,
             open_loop_ctx, p_observe, p_imagine, p_encode, p_decode):
    ssim_vals, losses_lpips, psnr_vals = [], [], []

    for batch_idx in range(steps_per_eval):
        batch = next(test_loader)

        if config.model in ["teco_convS5", "convS5", "teco_S5", "S5"]:
            predictions, real = sample_convSSM.sample(model, state, batch['video'], batch['actions'],
                                                      action_conditioned, open_loop_ctx, p_observe, p_imagine,
                                                      p_encode, p_decode)

        elif config.model in ["teco_transformer", "transformer", "performer"]:
            predictions, real = sample_transformer.sample(model, state, batch['video'], batch['actions'],
                                                          action_conditioned, open_loop_ctx, p_observe, p_imagine,
                                                          p_encode, p_decode)

        ssim_val, loss_lpips, psnr_val = p_get_eval_metrics(predictions, real)

        ssim_vals.append(ssim_val[0])
        losses_lpips.append(loss_lpips[0])
        psnr_vals.append(psnr_val[0])

    ssim_val = np.mean(np.array(ssim_vals))
    loss_lpips = np.mean(np.array(losses_lpips))
    psnr_val = np.mean(np.array(psnr_vals))

    if is_master_process:
        metrics = dict(lpips=loss_lpips,
                       ssim=ssim_val,
                       psnr=psnr_val)

        wandb.log({**{f'eval/{metric}': val
                      for metric, val in metrics.items()}
                   }, step=iteration)


def validate_noVQ(p_get_eval_metrics, model, iteration, state, test_loader, steps_per_eval, action_conditioned,
                  open_loop_ctx, p_observe, p_imagine, p_encode, p_decode, eval_seq_len):
    ssim_vals, losses_lpips, psnr_vals = [], [], []

    for batch_idx in range(steps_per_eval):
        batch = next(test_loader)

        if config.model in ["convS5_noVQ", 'convLSTM_noVQ', "pf_convS5_noVQ"]:
            predictions, real = sample_convSSM_noVQ.sample(model, state, batch['video'], batch['actions'],
                                                           action_conditioned, open_loop_ctx, p_observe,
                                                           p_imagine, p_encode, p_decode, eval_seq_len)
        else:
            predictions, real = sample_transformer_noVQ.sample(model, state, batch['video'], batch['actions'],
                                                               action_conditioned, open_loop_ctx, p_observe, p_imagine,
                                                               p_encode, p_decode, eval_seq_len)

        ssim_val, loss_lpips, psnr_val = p_get_eval_metrics(predictions, real)

        ssim_vals.append(ssim_val[0])
        losses_lpips.append(loss_lpips[0])
        psnr_vals.append(psnr_val[0])

    ssim_val = np.mean(np.array(ssim_vals))
    loss_lpips = np.mean(np.array(losses_lpips))
    psnr_val = np.mean(np.array(psnr_vals))

    if is_master_process:
        metrics = dict(lpips=loss_lpips,
                       ssim=ssim_val,
                       psnr=psnr_val)

        wandb.log({**{f'eval/{metric}': val
                      for metric, val in metrics.items()}
                   }, step=iteration)


def validate_cwvae_noVQ(p_get_eval_metrics, iteration, state, test_loader, steps_per_eval,
                  p_sample, eval_seq_len, open_loop_ctx):
    ssim_vals, losses_lpips, psnr_vals = [], [], []

    for batch_idx in range(steps_per_eval):
        batch = next(test_loader)

        predictions, real = sample_cwvae_noVQ.sample(state, batch['video'],
                                                     p_sample, eval_seq_len, open_loop_ctx)

        ssim_val, loss_lpips, psnr_val = p_get_eval_metrics(predictions, real)

        ssim_vals.append(ssim_val[0])
        losses_lpips.append(loss_lpips[0])
        psnr_vals.append(psnr_val[0])

    ssim_val = np.mean(np.array(ssim_vals))
    loss_lpips = np.mean(np.array(losses_lpips))
    psnr_val = np.mean(np.array(psnr_vals))

    if is_master_process:
        metrics = dict(lpips=loss_lpips,
                       ssim=ssim_val,
                       psnr=psnr_val)

        wandb.log({**{f'eval/{metric}': val
                      for metric, val in metrics.items()}
                   }, step=iteration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='/raid/moving_mnist_long_2')
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    if "gs://" in args.output_dir:
        args.run_id = "{}_{}".format(args.output_dir.split("/")[-1], np.random.randint(0, 100000))
    else:
        args.run_id = args.output_dir

    if not osp.isabs(args.output_dir):
        if 'DATA_DIR' not in os.environ:
            os.environ['DATA_DIR'] = 'logs'
            print('DATA_DIR environment variable not set, default to logs/')
        root_folder = os.environ['DATA_DIR']
        if "gs://" not in args.output_dir:
            args.output_dir = osp.join(root_folder, args.output_dir)

    config = yaml.safe_load(open(args.config, 'r'))
    config['data_path'] = args.data_dir
    if os.environ.get('DEBUG') == '1':
        config['viz_interval'] = 10
        config['save_interval'] = 10
        config['log_interval'] = 1
        args.output_dir = osp.join(osp.dirname(args.output_dir), f'DEBUG_{osp.basename(args.output_dir)}')
        args.run_id = f'DEBUG_{args.run_id}'

    print(f"Logging to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    args_d = vars(args)
    args_d.update(config)
    pickle.dump(args, open(osp.join(args.output_dir, 'args'), 'wb'))
    config = args

    if config.multinode:
        print("Configuring multinode")
        process_id = int(os.environ.get('NGC_ARRAY_INDEX'))
        if process_id == 0:
            master_address = socket.gethostbyname(socket.gethostname())
        else:
            master_address = socket.gethostbyname(os.environ.get('NGC_MASTER_ADDR'))
        coordinator_address = master_address + ":29500"
        num_processes = int(os.environ.get('NGC_ARRAY_SIZE'))
        process_id = int(os.environ.get('NGC_ARRAY_INDEX'))
        print("initializing process {}".format(process_id))
        print('coordinator_address', coordinator_address)
        print('num_processes', num_processes)
        print('process_id', process_id)
        jax.distributed.initialize(coordinator_address=coordinator_address,
                                   num_processes=num_processes,
                                   process_id=process_id)

    print(f'JAX process: {jax.process_index()} / {jax.process_count()}')
    print(f'JAX total devices: {jax.device_count()}')

    print(f'JAX local devices: {jax.local_device_count()}')

    is_master_process = jax.process_index() == 0

    main()
