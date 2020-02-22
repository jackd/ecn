from typing import Sequence
import numpy as np
import tensorflow as tf
import gin
from kblocks.keras import layers
from ecn import components as comp
from ecn import multi_graph as mg


@gin.configurable(module='ecn.builders')
def simple1d_graph(features,
                   labels,
                   weights=None,
                   num_classes=11,
                   grid_shape=(64,),
                   decay_time=50000,
                   filters0=32,
                   spatial_buffer=32,
                   reset_potential=-1.0,
                   threshold=0.5,
                   kt0=8,
                   dropout_rate=0.5,
                   hidden_units=(256,)):
    channels = features['channel']
    with mg.post_batch_context():
        times = tf.cast(1e6 * features['time'], tf.int64)
        channels = tf.cast(channels, tf.int64)
    spike_kwargs = dict(reset_potential=reset_potential, threshold=threshold)
    filters = filters0
    grid = comp.Grid(grid_shape)

    in_stream = comp.SpatialStream(grid, times, channels)
    features = None
    for _ in range(3):
        # in-place
        link = in_stream.grid.link((9,), (1, 1), (4, 4))
        out_stream = comp.spike_threshold(in_stream,
                                          link=link,
                                          decay_time=decay_time,
                                          min_mean_size=None,
                                          **spike_kwargs)

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
        decay_time *= 2
        features *= 2
        in_stream = out_stream
        link = in_stream.grid.link((9,), (2, 2), (4, 4))
        out_stream = comp.spike_threshold(in_stream,
                                          link=link,
                                          decay_time=decay_time,
                                          min_mean_size=None,
                                          **spike_kwargs)

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
        features = layers.Dropout(dropout_rate)(features)
        in_stream = out_stream
        decay_time *= 2
        features *= 2

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


@gin.configurable(module='ecn.builders')
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


@gin.configurable(module='ecn.builders')
def inception_multi_graph(features,
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
    # link = grid.link((5, 5), (1, 1), (1, 1))

    in_stream = comp.SpatialStream(grid, times, coords, min_mean_size=5000)
    # in_stream = comp.SpatialStream(grid, times, coords, min_mean_size=None)

    out_stream = comp.spike_threshold(in_stream,
                                      link,
                                      decay_time=decay_time,
                                      min_mean_size=2048,
                                      reset_potential=reset_potential,
                                      threshold=threshold)

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
    features = layers.Dropout(dropout_rate)(features)
    in_stream = out_stream
    del out_stream
    del convolver
    decay_time *= 2

    t_kernel = np.zeros((5, 5), dtype=np.bool)
    t_kernel[2] = True
    t_kernel[:, 2] = True

    for min_mean_size in (512, 128):
        # in place
        link = in_stream.grid.partial_self_link(t_kernel)
        t_convolver = comp.spatio_temporal_convolver(
            link,
            in_stream,
            in_stream,
            decay_time=decay_time,
            spatial_buffer_size=spatial_buffer)
        p_convolver = comp.pointwise_convolver(
            in_stream,
            in_stream,
            spatial_buffer_size=spatial_buffer,
            decay_time=decay_time * 4)

        # (5x1 + 1x5)xt
        ft = t_convolver.convolve(features,
                                  filters=filters,
                                  temporal_kernel_size=kt0)
        # 1x1x4t
        fp = p_convolver.convolve(features,
                                  filters=filters,
                                  temporal_kernel_size=4 * kt0)
        # 1x1x1
        fc = layers.Dense(units=filters * 4, activation='relu')(features)
        fc = layers.Dense(units=filters)(fc)
        branched = tf.nn.relu(ft + fp + fc)

        branched = layers.BatchNormalization()(branched)
        # branched = layers.Dropout(dropout_rate)(branched)
        features = features + branched

        link = in_stream.grid.link((3, 3), (2, 2), (1, 1))
        out_stream = comp.spike_threshold(in_stream,
                                          link,
                                          decay_time=decay_time,
                                          min_mean_size=min_mean_size,
                                          **spike_kwargs)

        filters *= 2
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

    global_stream = comp.global_spike_threshold(in_stream,
                                                decay_time=decay_time,
                                                min_mean_size=64,
                                                **spike_kwargs)
    flat_convolver = comp.flatten_convolver(in_stream, global_stream,
                                            decay_time)
    features = flat_convolver.convolve(features,
                                       filters=filters,
                                       temporal_kernel_size=kt0,
                                       activation='relu')
    features = layers.BatchNormalization()(features)
    # features = layers.Dropout(dropout_rate)(features)
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
    return (outputs, labels, weights)


@gin.configurable(module='ecn.builders')
def inception128_multi_graph(features,
                             labels,
                             weights=None,
                             num_classes=10,
                             grid_shape=(128, 128),
                             decay_time=10000,
                             spatial_buffer=32,
                             reset_potential=-2.0,
                             threshold=2.0,
                             filters0: int = 16,
                             kt0: int = 4,
                             hidden_units: Sequence[int] = (128,),
                             dropout_rate: float = 0.4,
                             decay_time_expansion_rate: float = 1.5):
    times = features['time']
    coords = features['coords']
    polarity = features['polarity']
    filters = filters0

    spike_kwargs = dict(reset_potential=reset_potential, threshold=threshold)

    grid = comp.Grid(grid_shape)
    link = grid.link((5, 5), (2, 2), (2, 2))

    in_stream = comp.SpatialStream(grid, times, coords, min_mean_size=220000)
    # in_stream = comp.SpatialStream(grid, times, coords, min_mean_size=None)

    out_stream = comp.spike_threshold(in_stream,
                                      link,
                                      decay_time=decay_time,
                                      min_mean_size=24000,
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
    features = layers.Dropout(dropout_rate)(features)
    in_stream = out_stream
    del out_stream
    del convolver
    decay_time = int(decay_time * decay_time_expansion_rate)

    t_kernel = np.zeros((5, 5), dtype=np.bool)
    t_kernel[2] = True
    t_kernel[:, 2] = True

    for min_mean_size in (5000, 1024, 256):
        # in place
        link = in_stream.grid.partial_self_link(t_kernel)
        t_convolver = comp.spatio_temporal_convolver(
            link,
            in_stream,
            in_stream,
            decay_time=decay_time,
            spatial_buffer_size=spatial_buffer)
        p_convolver = comp.pointwise_convolver(
            in_stream,
            in_stream,
            spatial_buffer_size=spatial_buffer,
            decay_time=decay_time * 4)

        # (5x1 + 1x5)xt
        ft = t_convolver.convolve(features,
                                  filters=filters,
                                  temporal_kernel_size=kt0)
        # 1x1x4t
        fp = p_convolver.convolve(features,
                                  filters=filters,
                                  temporal_kernel_size=4 * kt0)
        # 1x1x1
        fc = layers.Dense(units=filters * 4, activation='relu')(features)
        fc = layers.Dense(units=filters)(fc)
        branched = tf.nn.relu(ft + fp + fc)

        branched = layers.BatchNormalization()(branched)
        features = features + branched

        link = in_stream.grid.link((3, 3), (2, 2), (1, 1))
        out_stream = comp.spike_threshold(in_stream,
                                          link,
                                          decay_time=decay_time,
                                          min_mean_size=min_mean_size,
                                          **spike_kwargs)

        filters *= 2
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
        decay_time = int(decay_time * decay_time_expansion_rate)

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
    features = layers.Dropout(dropout_rate)(features)
    filters *= 2
    temporal_convolver = comp.temporal_convolver(global_stream, global_stream,
                                                 decay_time * 4)
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
    return (outputs, labels, weights)
