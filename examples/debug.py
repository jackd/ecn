# import os
# os.environ['NUMBA_DISABLE_JIT'] = '1'

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ecn.ops import grid as grid_ops
from ecn.ops import neighbors as neigh_ops
from ecn.ops import ragged as ragged_ops
from ecn.ops import spike as spike_ops
from ecn.problems import sources


def run_example(features, labels, grid_shape, num_classes):
    decay_time = 2000
    spatial_buffer = 32
    reset_potential = -2.0
    threshold = 1.0
    decay_time_expansion_rate = np.sqrt(2)
    num_levels = 5

    times = features["time"]
    coords = features["coords"]
    # polarity = features['polarity']

    max_coords = tf.reduce_max(coords, axis=0)
    offset = (tf.constant(grid_shape, dtype=coords.dtype) - max_coords) // 2
    coords = coords + offset

    coords = grid_ops.ravel_multi_index(coords, grid_shape)

    spike_kwargs = dict(reset_potential=reset_potential, threshold=threshold)

    grid_kwargs = dict(kernel_shape=(3, 3), padding=(1, 1), strides=(2, 2))
    num_partitions = np.prod(grid_kwargs["kernel_shape"])
    in_shape = grid_shape

    for _ in range(num_levels):
        (
            grid_partitions,
            grid_indices,
            grid_splits,
            out_shape,
        ) = grid_ops.sparse_neighborhood(in_shape, **grid_kwargs)

        grid_indices_T, grid_splits_T, partitions_T = ragged_ops.transpose_csr(
            grid_indices, grid_splits, grid_partitions
        )

        out_times, out_coords = spike_ops.spike_threshold(
            times, coords, grid_indices_T, grid_splits_T, decay_time, **spike_kwargs
        )

        (event_partitions, event_indices, event_splits) = neigh_ops.compute_neighbors(
            times,
            coords,
            out_times=out_times,
            out_coords=out_coords,
            grid_partitions=grid_partitions,
            grid_indices=grid_indices,
            grid_splits=grid_splits,
            spatial_buffer_size=spatial_buffer,
            event_duration=4 * decay_time,
        )
        rowids = tf.ragged.row_splits_to_segment_ids(event_splits, out_type=tf.int64)
        ij = tf.stack((rowids, event_indices), axis=-1)
        partitions = tf.cast(event_partitions, tf.int32)
        ijs = tf.dynamic_partition(ij, partitions, num_partitions=num_partitions)

        times = out_times
        coords = out_coords
        in_shape = out_shape
        decay_time = decay_time * decay_time_expansion_rate


# source = sources.ncaltech101_source()
source = sources.asl_dvs_source()
meta = source.meta
ds = source.get_dataset("train")
total = source.epoch_length("train")

for features, labels in tqdm(ds, total=total):
    run_example(features, labels, **meta)
