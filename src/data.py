# ------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2022 BAIR OPEN RESEARCH COMMONS REPOSITORY
# To view a copy of this license, visit
# https://github.com/wilson1yan/teco/tree/master
# ------------------------------------------------------------------------------

import os
import glob
import socket
import os.path as osp
import numpy as np
from flax import jax_utils
import jax
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")
import tensorflow_datasets as tfds
import tensorflow_io as tfio
from tensorflow.python.lib.io import file_io
import io
# from skimage.io import imread
import cv2


def is_tfds_folder(path):
    path = osp.join(path, '1.0.0')
    if path.startswith('gs://'):
        return tf.io.gfile.exists(path)
    else:
        return osp.exists(path)


def load_npz(config, split, num_ds_shards, ds_shard_id):
    folder = osp.join(config.data_path, split, '*', '*.npz')
    if folder.startswith('gs://'):
        fns = tf.io.gfile.glob(folder)
    else:
        fns = list(glob.glob(folder))
    fns = np.array_split(fns, num_ds_shards)[ds_shard_id].tolist()

    def read(path):
        path = path.decode('utf-8')
        if path.startswith('gs://'):
            path = io.BytesIO(file_io.FileIO(path, 'rb').read())
        data = np.load(path)
        video, actions = data['video'].astype(np.float32), data['actions'].astype(np.int32)
        video = 2 * (video / 255.) - 1 
        return video, actions

    dataset = tf.data.Dataset.from_tensor_slices(fns)
    dataset = dataset.map(
        lambda item: tf.numpy_function(
            read,
            [item],
            [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(
        lambda video, actions: dict(video=video, actions=actions),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    return dataset


def load_mnist_npz(config, split, num_ds_shards, ds_shard_id):
    folder = osp.join(config.data_path, split, '*.npz')

    if folder.startswith('gs://'):
        fns = tf.io.gfile.glob(folder)
    else:
        fns = list(glob.glob(folder))
    fns = np.array_split(fns, num_ds_shards)[ds_shard_id].tolist()
    print("fns shape", len(fns))
    def read(path):
        path = path.decode('utf-8')
        if path.startswith('gs://'):
            path = io.BytesIO(file_io.FileIO(path, 'rb').read())
        data = np.load(path)
        video, actions = data['data'].astype(np.float32), None
        video = 2 * (video / 255.) - 1
        return video

    dataset = tf.data.Dataset.from_tensor_slices(fns)
    dataset = dataset.map(
        lambda item: tf.numpy_function(
            read,
            [item],
            [tf.float32]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(
        lambda video: dict(video=video, actions=None),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    return dataset


def load_pf(config, split, num_ds_shards, ds_shard_id, pnode=False):

    # meta = np.load("/media/data_cifs/pathfinder_small/curv_contour_length_14/metadata/combined.npy")
    # root = "/media/data_cifs/pathfinder_small/curv_contour_length_14"
    # root = "/mnt/disks/pathfinder"
    if pnode:
        root = "/media/data_cifs/curvy_2snakes/curv_contour_length_14_full/"
    else:
        root = "/mnt/disks/pathfinder"

    if split == "train":
        # fns = np.array_split(meta, num_ds_shards)[ds_shard_id].tolist()
        # fns = np.asarray(glob.glob(os.path.join("/media/data_cifs/curvy_2snakes/curv_contour_length_14_full/train/*.PNG"))).reshape(-1, 1)
        fns = np.asarray(glob.glob(os.path.join(root, "train", "*.PNG")))
    elif split == "val" or split == "validation" or split == "test":
        # Use a hardcoded set of val images for now. Quick hack.
        # fns = np.asarray(glob.glob(os.path.join("/media/data_cifs/curvy_2snakes/curv_contour_length_14_full/val/*.PNG"))).reshape(-1, 1)
        fns = np.asarray(glob.glob(os.path.join(root, "val", "*.PNG")))
    else:
        raise NotImplementedError("Split {} not recognized.".format(split))
    print("fns shape", len(fns))
    def read(path):
        # path = path[0].decode('UTF-8')
        path = path.decode('UTF-8')
        label = int(path.split("_")[-1].split(".")[0])
        fl = path
        # if len(path) == 1:
        #     path = path[0].decode('UTF-8')
        #     label = int(path.split("_")[-1].split(".")[0])
        #     fl = path
        # else:
        #     label = int(path[3])
        #     p0 = path[0].decode('UTF-8')
        #     p1 = path[1].decode('UTF-8')
        #     fl = os.path.join(root, p0, p1)
        #     if fl.startswith('gs://'):
        #         fl = io.BytesIO(file_io.FileIO(fl, 'rb').read())
        # image = imread(fl).astype(np.float32)[..., None]
        image = cv2.imread(fl, 0).astype(np.float32)[..., None]
        image = jax.image.resize(image, shape=config.resize, method="bilinear")
        image = (image - image.min()) / (image.max() - image.min())  # Renormalize to [0, 1]
        # image = 2 * (image / 255.) - 1
        image = 2 * image - 1  # Normalize to [-1, 1]
        video, actions = image, label
        return video, actions

    dataset = tf.data.Dataset.from_tensor_slices(fns)
    dataset = dataset.map(
        lambda item: tf.numpy_function(
            read,
            [item],
            [tf.float32, tf.int64]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(
        lambda video, actions: dict(video=video, actions=actions),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    return dataset


def load_video(config, split, num_ds_shards, ds_shard_id):
    folder = osp.join(config.data_path, split, '*', '*.mp4')
    if folder.startswith('gs://'):
        fns = tf.io.gfile.glob(folder)
    else:
        fns = list(glob.glob(folder))
    fns = np.array_split(fns, num_ds_shards)[ds_shard_id].tolist()

    # TODO resizing video
    def read(path):
        path = path.decode('utf-8')

        video = tfio.experimental.ffmpeg.decode_video(tf.io.read_file(path)).numpy()
        start_idx = np.random.randint(0, video.shape[0] - config.seq_len + 1)
        video = video[start_idx:start_idx + config.seq_len]
        video = 2 * (video / np.array(255., dtype=np.float32)) - 1
        
        np_path = path[:-3] + 'npz'
        if tf.io.gfile.exists(np_path):
            if path.startswith('gs://'):
                np_path = io.BytesIO(file_io.FileIO(np_path, 'rb').read())
            np_data = np.load(np_path)
            actions = np_data['actions'].astype(np.int32)
            actions = actions[start_idx:start_idx + config.seq_len]
        else:
            actions = np.zeros((video.shape[0],), dtype=np.int32)
        
        return video, actions

    dataset = tf.data.Dataset.from_tensor_slices(fns)
    dataset = dataset.map(
        lambda item: tf.numpy_function(
            read,
            [item],
            [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(
        lambda video, actions: dict(video=video, actions=actions),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    return dataset
                

class Data:
    def __init__(self, config, xmap=False):
        self.config = config
        self.xmap = xmap
        if "serre" in socket.gethostname():
            self.pnode = True
        else:
            self.pnode = False
        print('Dataset:', config.data_path)

    @property
    def train_itr_per_epoch(self):
        return self.train_size // self.config.batch_size

    @property
    def test_itr_per_epoch(self):
        return self.test_size // self.config.batch_size

    def create_iterator(self, train, repeat=True, prefetch=True):
        if self.xmap:
            num_data = jax.device_count() // self.config.num_shards
            num_data_local = max(1, jax.local_device_count() // self.config.num_shards)
            if num_data >= jax.process_count():
                num_ds_shards = jax.process_count()
                ds_shard_id = jax.process_index()
            else:
                num_ds_shards = num_data
                n_proc_per_shard = jax.process_count() // num_data
                ds_shard_id = jax.process_index() // n_proc_per_shard
        else:
            num_data_local = jax.local_device_count()
            num_ds_shards = jax.process_count()
            ds_shard_id = jax.process_index()

        batch_size = self.config.batch_size // num_ds_shards
        split_name = 'train' if train else 'test'
        if not is_tfds_folder(self.config.data_path):
            if 'dmlab' in self.config.data_path:
                dataset = load_npz(self.config, split_name, num_ds_shards, ds_shard_id)
            elif 'mnist' in self.config.data_path:
                print('loading mnist')
                dataset = load_mnist_npz(self.config, split_name, num_ds_shards, ds_shard_id)
            elif 'pathfinder' in self.config.data_path:
                dataset = load_pf(self.config, split_name, num_ds_shards, ds_shard_id, pnode=self.pnode)
            else:
                dataset = load_video(self.config, split_name, num_ds_shards, ds_shard_id)
        else:
            seq_len = self.config.seq_len

            def process(features):
                video = tf.cast(features['video'], tf.int32)
                T = tf.shape(video)[0]
                start_idx = tf.random.uniform((), 0, T - seq_len + 1, dtype=tf.int32)
                video = tf.identity(video[start_idx:start_idx + seq_len])
                actions = tf.cast(features['actions'], tf.int32)
                actions = tf.identity(actions[start_idx:start_idx + seq_len])
                return dict(video=video, actions=actions)

            split = tfds.split_for_jax_process(split_name, process_index=ds_shard_id,
                                               process_count=num_ds_shards)
            dataset = tfds.load(osp.basename(self.config.data_path), split=split,
                                data_dir=osp.dirname(self.config.data_path))

            # caching only for pre-encoded since raw video will probably
            # run OOM on RAM
            if self.config.cache:
                dataset = dataset.cache()

            options = tf.data.Options()
            options.threading.private_threadpool_size = 48
            options.threading.max_intra_op_parallelism = 1
            dataset = dataset.with_options(options)
            dataset = dataset.map(process)

        if repeat:
            dataset = dataset.repeat()
        if train:
            dataset = dataset.shuffle(batch_size * 2, seed=self.config.seed)

        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(batch_size)

        def prepare_tf_data(xs):
            def _prepare(x):
                x = x._numpy()
                x = x.reshape((num_data_local, -1) + x.shape[1:])
                return x
            xs = jax.tree_util.tree_map(_prepare, xs)
            return xs

        iterator = map(prepare_tf_data, dataset)

        if prefetch:
            iterator = jax_utils.prefetch_to_device(iterator, 2)

        return iterator
