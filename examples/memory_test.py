from typing import Sequence
from tqdm import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds

from kblocks.keras import layers
from ecn.problems import sources
from ecn import components as comp
from ecn.problems.utils import multi_graph_trainable
from ecn.problems.builders import inception_multi_graph
from ecn.ops import spike as spike_ops
from ecn.ops import neighbors as neigh_ops
from ecn.ops import grid as grid_ops


def simple_multi_graph(features,
                       labels,
                       weights=None,
                       num_classes=10,
                       grid_shape=(34, 34),
                       decay_time=10000,
                       spatial_buffer=32,
                       reset_potential=-2.0,
                       threshold=1.0,
                       filters0: int = 32,
                       kt0: int = 4,
                       hidden_units: Sequence[int] = (128,),
                       dropout_rate: float = 0.4):
    times = features['time']
    coords = features['coords']
    polarity = features['polarity']
    filters = filters0

    spike_kwargs = dict(reset_potential=reset_potential, threshold=threshold)

    grid = comp.Grid(grid_shape)
    link = grid.link((3, 3), (1, 1), (0, 0))

    in_stream = comp.SpatialStream(grid, times, coords, min_mean_size=5000)
    # in_stream = comp.SpatialStream(grid, times, coords, min_mean_size=None)

    out_stream = comp.spike_threshold(in_stream,
                                      link,
                                      decay_time=decay_time,
                                      min_mean_size=1024,
                                      **spike_kwargs)

    features = in_stream.prepare_model_inputs(polarity)
    features = tf.keras.layers.Lambda(lambda x: tf.identity(x.values))(features)

    convolver = comp.spatio_temporal_convolver(
        link,
        in_stream,
        out_stream,
        decay_time=decay_time,
        spatial_buffer_size=spatial_buffer)
    features = convolver.convolve(features,
                                  filters=filters,
                                  temporal_kernel_size=kt0,
                                  activation='relu')
    features = layers.BatchNormalization()(features)
    in_stream = out_stream
    del out_stream
    del convolver
    decay_time *= 2

    for min_mean_size in (512, 128):
        # in place
        link = in_stream.grid.self_link((3, 3))
        ip_convolver = comp.spatio_temporal_convolver(
            link,
            in_stream,
            in_stream,
            decay_time=decay_time,
            spatial_buffer_size=spatial_buffer)
        features = ip_convolver.convolve(features,
                                         filters=filters,
                                         temporal_kernel_size=kt0,
                                         activation='relu')
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(dropout_rate)(features)

        link = in_stream.grid.link((5, 5), (2, 2), (2, 2))
        out_stream = comp.spike_threshold(in_stream,
                                          link,
                                          decay_time=decay_time,
                                          min_mean_size=min_mean_size,
                                          **spike_kwargs)

        ds_convolver = comp.spatio_temporal_convolver(
            link,
            in_stream,
            out_stream,
            decay_time=decay_time,
            spatial_buffer_size=spatial_buffer)

        features = ds_convolver.convolve(features,
                                         filters=filters,
                                         temporal_kernel_size=kt0,
                                         activation='relu')
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(dropout_rate)(features)

        in_stream = out_stream
        del out_stream
        decay_time *= 2
        filters *= 2

    global_stream = comp.global_spike_threshold(in_stream,
                                                decay_time=decay_time,
                                                min_mean_size=32,
                                                **spike_kwargs)
    flat_convolver = comp.flatten_convolver(in_stream, global_stream,
                                            decay_time)
    features = flat_convolver.convolve(features,
                                       filters=filters,
                                       temporal_kernel_size=kt0,
                                       activation='relu')
    features = layers.BatchNormalization()(features)
    decay_time *= 2
    filters *= 2
    temporal_convolver = comp.temporal_convolver(global_stream, global_stream,
                                                 decay_time)
    features = temporal_convolver.convolve(features,
                                           filters=filters,
                                           temporal_kernel_size=kt0 * 2,
                                           activation='relu')
    features = layers.BatchNormalization()(features)
    features = layers.Dropout(dropout_rate)(features)
    for h in hidden_units:
        features = layers.Dense(h, activation='relu')(features)
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(dropout_rate)(features)
    logits = layers.Dense(num_classes, activation=None, name='stream')(features)
    final_logits = tf.keras.layers.Lambda(
        lambda args: tf.gather(*args),
        name='final')([logits, global_stream.model_row_ends])

    outputs = (final_logits, logits)

    labels = global_stream.prepare_labels(labels)
    weights = global_stream.prepare_weights(weights)
    return outputs, labels, weights


# source = sources.nmnist_source(
#     read_config=tfds.ReadConfig(
#         interleave_block_length=1, interleave_parallel_reads=1))
source = sources.nmnist_source2()
split = 'train'

# build_fn = simple_multi_graph
build_fn = inception_multi_graph
batch_size = 32
trainable = multi_graph_trainable(build_fn,
                                  source,
                                  batch_size,
                                  use_cache=True,
                                  cache_dir='/tmp/ecn_tests/memory_test',
                                  clear_cache=True)
source = trainable.source
ds = source.get_dataset(split)
total = source.examples_per_epoch(split)

# def map_fn(events, labels):
#     shape = tf.constant((34, 34), dtype=tf.int64)
#     (grid_partitions, grid_indices, grid_splits,
#      out_shape) = grid_ops.sparse_neighborhood(shape, (3, 3), (1, 1), (1, 1))
#     coords = grid_ops.ravel_multi_index(events['coords'], shape)
#     times = events['time']
#     out_times, out_coords = spike_ops.spike_threshold(times,
#                                                       coords,
#                                                       grid_indices,
#                                                       grid_splits,
#                                                       decay_time=10000)
#     partitions, indices, splits = neigh_ops.compute_neighbors(
#         times, coords, out_times, out_coords, grid_partitions, grid_indices,
#         grid_splits, 1000, 10000)
#     return partitions, indices, splits

# ds = source.get_dataset(split).map(map_fn)
# total = source.examples_per_epoch(split)

for args in tqdm(ds, total=total):
    pass

# with tf.Graph().as_default():
#     ds = trainable.source.get_dataset(split)
#     out = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
#     with tf.compat.v1.Session() as sess:
#         for i in tqdm(range(trainable.source.examples_per_epoch(split))):
#             sess.run(out)
