from typing import Sequence
import numpy as np
import tensorflow as tf
import gin
from kblocks.keras import layers
from ecn import components as comp
from ecn import multi_graph as mg

Lambda = tf.keras.layers.Lambda

class Printer(tf.keras.layers.Layer):
    def __init__(self, fn, **kwargs):
        self._fn = fn
        super().__init__(**kwargs)

    def call(self, inputs):
        with tf.control_dependencies([tf.print(self._fn(inputs))]):
            return tf.nest.map_structure(tf.identity, inputs)



def _complex(args):
    return tf.complex(*args)


def dropout(x, dropout_rate: float):
    if dropout_rate == 0:
        return x
    layer = layers.Dropout(dropout_rate)

    if x.dtype.is_complex:
        real = layer(Lambda(tf.math.real)(x))
        imag = layer(Lambda(tf.math.imag)(x))
        return Lambda(_complex)([real, imag])
    else:
        return layer(x)


def flatten_complex(x, axis=-1):
    return tf.concat((tf.math.real(x), tf.math.imag(x)), axis=axis)


def apply_dense(layer, x):
    if x.dtype.is_complex:
        assert (layer.activation is None or
                layer.activation.__name__ == 'linear')
        return Lambda(_complex)(
            [layer(Lambda(tf.math.real)(x)),
             layer(Lambda(tf.math.imag)(x))])
    else:
        return layer(x)


def _with_numerics_check(x, message='not numeric'):
    if x.dtype.is_complex:
        tf.debugging.check_numerics(tf.math.real(x), f'{message} (real)')
        tf.debugging.check_numerics(tf.math.imag(x), f'{message} (imag)')
    else:
        tf.debugging.check_numerics(x, message)
    return tf.identity(x)


def with_numerics_check(x, message):
    return Lambda(_with_numerics_check, arguments=dict(message=message))(x)


@gin.configurable(module='ecn.builders')
def ncars_inception_graph(features,
                          labels,
                          weights=None,
                          activation='relu',
                          reset_potential=-1.0,
                          threshold=0.6,
                          filters0=32,
                          decay_time=10000,
                          kt0=4,
                          spatial_buffer=32,
                          dropout_rate=0.5,
                          hidden_units=(128,)):
    times = features['time']
    coords = features['coords']
    polarity = features['polarity']
    filters = filters0

    activation = tf.keras.activations.get(activation)

    spike_kwargs = dict(reset_potential=reset_potential, threshold=threshold)

    with mg.pre_cache_context():
        grid = comp.Grid(tf.reduce_max(coords, axis=0) + 1)
    # link = grid.link((2, 2), (1, 1), (0, 0))
    # link = grid.link((3, 3), (1, 1), (1, 1))
    link = grid.link((5, 5), (1, 1), (0, 0))

    in_stream = comp.SpatialStream(grid, times, coords, min_mean_size=None)
    # in_stream = comp.SpatialStream(grid, times, coords, min_mean_size=None)

    out_stream = comp.spike_threshold(in_stream,
                                      link,
                                      decay_time=decay_time,
                                      min_mean_size=None,
                                      reset_potential=reset_potential,
                                      threshold=threshold)

    features = in_stream.prepare_model_inputs(polarity)
    features = Lambda(lambda x: tf.identity(x.values))(features)

    convolver = comp.spatio_temporal_convolver(
        link,
        in_stream,
        out_stream,
        decay_time=decay_time,
        spatial_buffer_size=spatial_buffer)
    features = convolver.convolve(features,
                                  filters=filters,
                                  temporal_kernel_size=kt0,
                                  activation=activation)
    features = layers.BatchNormalization()(features)
    features = dropout(features, dropout_rate)
    in_stream = out_stream
    del out_stream
    del convolver
    decay_time *= 2

    t_kernel = np.zeros((5, 5), dtype=np.bool)
    t_kernel[2] = True
    t_kernel[:, 2] = True

    # for min_mean_size in (512, 128):
    for _ in range(2):
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
        fc = apply_dense(layers.Dense(units=filters * 4), features)
        fc = activation(fc)
        fc = apply_dense(layers.Dense(units=filters), fc)

        branched = ft + fp + fc
        branched = activation(branched)
        branched = layers.BatchNormalization()(branched)
        # branched = layers.Dropout(dropout_rate)(branched)
        features = features + branched

        # down sample conv
        link = in_stream.grid.link((3, 3), (2, 2), (1, 1))
        out_stream = comp.spike_threshold(in_stream,
                                          link,
                                          decay_time=decay_time,
                                          min_mean_size=None,
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
                                         activation=activation)
        features = layers.BatchNormalization()(features)
        features = dropout(features, dropout_rate)

        in_stream = out_stream
        del out_stream
        decay_time *= 2

    global_stream = comp.global_spike_threshold(in_stream,
                                                decay_time=decay_time,
                                                min_mean_size=None,
                                                **spike_kwargs)
    flat_convolver = comp.temporal_convolver(in_stream, global_stream,
                                             decay_time)
    features = flat_convolver.convolve(features,
                                       filters=filters,
                                       temporal_kernel_size=kt0,
                                       activation=activation)
    features = layers.BatchNormalization()(features)
    # features = dropout(features, dropout_rate)
    decay_time *= 2
    filters *= 2
    temporal_convolver = comp.temporal_convolver(global_stream, global_stream,
                                                 decay_time)
    features = temporal_convolver.convolve(features,
                                           filters=filters,
                                           temporal_kernel_size=kt0 * 2,
                                           activation=activation)
    features = layers.BatchNormalization()(features)
    features = dropout(features, dropout_rate)
    features = Lambda(flatten_complex)(features)

    for h in hidden_units:
        features = layers.Dense(h, activation=activation)(features)
        features = layers.BatchNormalization()(features)
        features = dropout(features, dropout_rate)
    logits = layers.Dense(1, activation=None, name='stream')(features)
    # logits = Lambda(tf.squeeze, arguments=dict(axis=-1), name='stream')(logits)
    final_logits = Lambda(lambda args: tf.gather(*args),
                          name='final')([logits, global_stream.model_row_ends])

    outputs = (final_logits, logits)

    labels = global_stream.prepare_labels(labels)
    weights = global_stream.prepare_weights(weights)
    return (outputs, labels, weights)


@gin.configurable(module='ecn.builders')
def simple1d_half_graph(
        features,
        labels,
        weights=None,
        num_classes=11,
        grid_shape=(64,),
        decay_time=5000,
        filters0=16,
        spatial_buffer=32,
        reset_potential=-2.0,
        threshold=1.0,
        kt0=4,
        dropout_rate=0.5,
        hidden_units=(128,),
        kernel_size=5,
        initial_size=None,
        activation='relu',
        use_batch_norm=True,
):
    channels = features['channel']
    with mg.post_batch_context():
        times = features['time']
        channels = tf.cast(channels, tf.int64)
        # num_events = tf.cast(tf.size(times), tf.float32)
        # start = tf.cast(num_events * 0.05, tf.int64)
        # end = tf.cast(num_events * 0.95, tf.int64)
        # times = times[start:end]
        # times = times - times[0]
        valid = times < 0.96 * times[-1]
        times = times[valid]
        channels = channels[valid]
        # channels = channels[start:end][valid]
        # channels = channels[start:end]
        times = tf.cast(1e6 * times, tf.int64)

    def batch_norm(x):
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        return x

    size = initial_size

    spike_kwargs = dict(reset_potential=reset_potential, threshold=threshold)
    filters = filters0
    grid = comp.Grid(grid_shape)

    in_stream = comp.SpatialStream(grid, times, channels, min_mean_size=size)
    size = None if size is None else size // 4
    features = None
    assert (kernel_size % 2) == 1
    padding = (kernel_size - 1) // 2
    for _ in range(3):
        link = in_stream.grid.link((kernel_size,), (2,), (padding,))
        out_stream = comp.spike_threshold(in_stream,
                                          link=link,
                                          decay_time=decay_time,
                                          min_mean_size=size,
                                          **spike_kwargs)
        size = None if size is None else size // 4
        convolver = comp.spatio_temporal_convolver(
            link,
            in_stream,
            out_stream,
            decay_time=decay_time,
            spatial_buffer_size=spatial_buffer)

        features = convolver.convolve(features,
                                      filters=filters,
                                      temporal_kernel_size=kt0,
                                      activation=activation)
        features = batch_norm(features)
        features = dropout(features, dropout_rate)
        in_stream = out_stream
        filters *= 2
        decay_time *= 2

    global_stream = comp.global_spike_threshold(in_stream,
                                                decay_time=decay_time,
                                                min_mean_size=size,
                                                **spike_kwargs)
    flat_convolver = comp.flatten_convolver(in_stream, global_stream,
                                            decay_time)
    features = flat_convolver.convolve(features,
                                       filters=filters,
                                       temporal_kernel_size=kt0,
                                       activation=activation)
    features = batch_norm(features)
    decay_time *= 2
    filters *= 2
    temporal_convolver = comp.temporal_convolver(global_stream, global_stream,
                                                 decay_time)
    features = temporal_convolver.convolve(features,
                                           filters=filters,
                                           temporal_kernel_size=kt0 * 2,
                                           activation=activation)
    features = batch_norm(features)
    features = dropout(features, dropout_rate)
    features = Lambda(flatten_complex)(features)
    for h in hidden_units:
        features = layers.Dense(h, activation=activation)(features)
        features = batch_norm(features)
        features = dropout(features, dropout_rate)
    logits = layers.Dense(num_classes, activation=None, name='stream')(features)
    final_logits = Lambda(lambda args: tf.gather(*args),
                          name='final')([logits, global_stream.model_row_ends])

    outputs = (final_logits, logits)

    labels = global_stream.prepare_labels(labels)
    weights = global_stream.prepare_weights(weights)
    return outputs, labels, weights


@gin.configurable(module='ecn.builders')
def simple1d_graph(features,
                   labels,
                   weights=None,
                   num_classes=11,
                   grid_shape=(64,),
                   decay_time=10000,
                   filters0=32,
                   spatial_buffer=32,
                   reset_potential=-1.0,
                   threshold=0.5,
                   kt0=8,
                   dropout_rate=0.5,
                   hidden_units=(256,),
                   kernel_size=9,
                   initial_size=None,
                   activation='relu',
                   use_batch_norm=True):
    channels = features['channel']
    with mg.post_batch_context():
        times = features['time']
        channels = tf.cast(channels, tf.int64)
        num_events = tf.cast(tf.size(times), tf.float32)
        start = tf.cast(num_events * 0.05, tf.int64)
        end = tf.cast(num_events * 0.95, tf.int64)
        times = times[start:end]
        times = times - times[0]
        valid = times < 0.98 * times[-1]
        times = times[valid]
        channels = channels[start:end][valid]
        # channels = channels[start:end]
        times = tf.cast(1e6 * times, tf.int64)

    def batch_norm(x):
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        return x

    def dropout(x):
        if dropout_rate == 0:
            return x
        elif activation == 'selu':
            return layers.AlphaDropout(dropout_rate)(x)
        else:
            return layers.Dropout(dropout_rate)(x)

    size = initial_size

    spike_kwargs = dict(reset_potential=reset_potential, threshold=threshold)
    filters = filters0
    grid = comp.Grid(grid_shape)

    in_stream = comp.SpatialStream(grid, times, channels, min_mean_size=size)
    size = None if size is None else size // 2
    features = None
    assert (kernel_size % 2) == 1
    padding = (kernel_size - 1) // 2
    for _ in range(3):
        # in-place
        link = in_stream.grid.link((kernel_size,), (1,), (padding,))
        out_stream = comp.spike_threshold(in_stream,
                                          link=link,
                                          decay_time=decay_time,
                                          min_mean_size=size,
                                          **spike_kwargs)
        size = None if size is None else size // 2

        convolver = comp.spatio_temporal_convolver(
            link,
            in_stream,
            out_stream,
            decay_time=decay_time,
            spatial_buffer_size=spatial_buffer)

        features = convolver.convolve(features,
                                      filters=filters,
                                      temporal_kernel_size=kt0,
                                      activation=activation)
        features = batch_norm(features)
        in_stream = out_stream
        filters *= 2
        in_stream = out_stream
        link = in_stream.grid.link((kernel_size,), (2,), (padding,))
        out_stream = comp.spike_threshold(in_stream,
                                          link=link,
                                          decay_time=decay_time,
                                          min_mean_size=size,
                                          **spike_kwargs)
        size = None if size is None else size // 2
        convolver = comp.spatio_temporal_convolver(
            link,
            in_stream,
            out_stream,
            decay_time=decay_time,
            spatial_buffer_size=spatial_buffer)

        features = convolver.convolve(features,
                                      filters=filters,
                                      temporal_kernel_size=kt0,
                                      activation=activation)
        features = batch_norm(features)
        # features = layers.Dropout(dropout_rate)(features)
        in_stream = out_stream
        decay_time *= 2

    global_stream = comp.global_spike_threshold(in_stream,
                                                decay_time=decay_time,
                                                min_mean_size=size,
                                                **spike_kwargs)
    flat_convolver = comp.flatten_convolver(in_stream, global_stream,
                                            decay_time)
    features = flat_convolver.convolve(features,
                                       filters=filters,
                                       temporal_kernel_size=kt0,
                                       activation=activation)
    features = batch_norm(features)
    decay_time *= 2
    filters *= 2
    temporal_convolver = comp.temporal_convolver(global_stream, global_stream,
                                                 decay_time)
    features = temporal_convolver.convolve(features,
                                           filters=filters,
                                           temporal_kernel_size=kt0 * 2,
                                           activation=activation)
    features = batch_norm(features)
    features = dropout(features)
    for h in hidden_units:
        features = layers.Dense(h, activation=activation)(features)
        features = batch_norm(features)
        features = dropout(features)
    logits = layers.Dense(num_classes, activation=None, name='stream')(features)
    final_logits = Lambda(lambda args: tf.gather(*args),
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
                       dropout_rate: float = 0.4,
                       static_sizes=True):
    times = features['time']
    coords = features['coords']
    polarity = features['polarity']
    filters = filters0

    spike_kwargs = dict(reset_potential=reset_potential, threshold=threshold)

    grid = comp.Grid(grid_shape)
    link = grid.link((3, 3), (1, 1), (0, 0))

    in_stream = comp.SpatialStream(grid,
                                   times,
                                   coords,
                                   min_mean_size=5000 if static_sizes else None)
    # in_stream = comp.SpatialStream(grid, times, coords, min_mean_size=None)

    out_stream = comp.spike_threshold(
        in_stream,
        link,
        decay_time=decay_time,
        min_mean_size=1024 if static_sizes else None,
        **spike_kwargs)

    features = in_stream.prepare_model_inputs(polarity)
    features = Lambda(lambda x: tf.identity(x.values))(features)

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
        out_stream = comp.spike_threshold(
            in_stream,
            link,
            decay_time=decay_time,
            min_mean_size=min_mean_size if static_sizes else None,
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

    global_stream = comp.global_spike_threshold(
        in_stream,
        decay_time=decay_time,
        min_mean_size=32 if static_sizes else None,
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
    final_logits = Lambda(lambda args: tf.gather(*args),
                          name='final')([logits, global_stream.model_row_ends])

    outputs = (final_logits, logits)

    labels = global_stream.prepare_labels(labels)
    weights = global_stream.prepare_weights(weights)
    return outputs, labels, weights


@gin.configurable(module='ecn.builders')
def inception_multi_graph(
        features,
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
        dropout_rate: float = 0.4,
        static_sizes=True,
        activation='relu',
):
    times = features['time']
    coords = features['coords']
    polarity = features['polarity']
    filters = filters0

    activation = tf.keras.activations.get(activation)

    spike_kwargs = dict(reset_potential=reset_potential, threshold=threshold)

    grid = comp.Grid(grid_shape)
    link = grid.link((3, 3), (1, 1), (1, 1))
    # link = grid.link((5, 5), (1, 1), (1, 1))

    in_stream = comp.SpatialStream(grid,
                                   times,
                                   coords,
                                   min_mean_size=5000 if static_sizes else None)
    # in_stream = comp.SpatialStream(grid, times, coords, min_mean_size=None)

    out_stream = comp.spike_threshold(
        in_stream,
        link,
        decay_time=decay_time,
        min_mean_size=2048 if static_sizes else None,
        reset_potential=reset_potential,
        threshold=threshold)

    features = in_stream.prepare_model_inputs(polarity)
    features = Lambda(lambda x: tf.identity(x.values))(features)

    convolver = comp.spatio_temporal_convolver(
        link,
        in_stream,
        out_stream,
        decay_time=decay_time,
        spatial_buffer_size=spatial_buffer)
    features = convolver.convolve(features,
                                  filters=filters,
                                  temporal_kernel_size=kt0,
                                  activation=activation)
    features = layers.BatchNormalization()(features)
    features = dropout(features, dropout_rate)
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
        fc = apply_dense(layers.Dense(units=filters * 4), features)
        fc = activation(fc)
        fc = apply_dense(layers.Dense(units=filters), fc)

        branched = ft + fp + fc
        branched = activation(branched)
        branched = layers.BatchNormalization()(branched)
        # branched = layers.Dropout(dropout_rate)(branched)
        features = features + branched

        # down sample conv
        link = in_stream.grid.link((3, 3), (2, 2), (1, 1))
        out_stream = comp.spike_threshold(
            in_stream,
            link,
            decay_time=decay_time,
            min_mean_size=min_mean_size if static_sizes else None,
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
                                         activation=activation)
        features = layers.BatchNormalization()(features)
        features = dropout(features, dropout_rate)

        in_stream = out_stream
        del out_stream
        decay_time *= 2

    global_stream = comp.global_spike_threshold(
        in_stream,
        decay_time=decay_time,
        min_mean_size=64 if static_sizes else None,
        **spike_kwargs)
    flat_convolver = comp.flatten_convolver(in_stream, global_stream,
                                            decay_time)
    features = flat_convolver.convolve(features,
                                       filters=filters,
                                       temporal_kernel_size=kt0,
                                       activation=activation)
    features = layers.BatchNormalization()(features)
    # features = dropout(features, dropout_rate)
    decay_time *= 2
    filters *= 2
    temporal_convolver = comp.temporal_convolver(global_stream, global_stream,
                                                 decay_time)
    features = temporal_convolver.convolve(features,
                                           filters=filters,
                                           temporal_kernel_size=kt0 * 2,
                                           activation=activation)
    features = layers.BatchNormalization()(features)
    features = dropout(features, dropout_rate)
    features = Lambda(flatten_complex)(features)

    for h in hidden_units:
        features = layers.Dense(h, activation=activation)(features)
        features = layers.BatchNormalization()(features)
        features = dropout(features, dropout_rate)
    logits = layers.Dense(num_classes, activation=None, name='stream')(features)
    final_logits = Lambda(lambda args: tf.gather(*args),
                          name='final')([logits, global_stream.model_row_ends])

    outputs = (final_logits, logits)

    labels = global_stream.prepare_labels(labels)
    weights = global_stream.prepare_weights(weights)
    return (outputs, labels, weights)


@gin.configurable(module='ecn.builders')
def inception128_multi_graph(
        features,
        labels,
        weights=None,
        num_classes=10,
        grid_shape=(128, 128),
        decay_time=10000,
        spatial_buffer=32,
        reset_potential=-3.0,
        threshold=2.0,
        filters0: int = 16,
        kt0: int = 4,
        hidden_units: Sequence[int] = (128,),
        dropout_rate: float = 0.4,
        decay_time_expansion_rate: float = 2.0,
        # start_mean_size=262144,  # 2 ** 18
        # hidden_mean_sizes=(32768, 8192, 1024, 256),  # generally conservative
        # final_mean_size=32,
        bucket_sizes=False,
        activation='relu',
):
    times = features['time']
    coords = features['coords']
    polarity = features['polarity']
    filters = filters0
    activation = tf.keras.activations.get(activation)

    spike_kwargs = dict(reset_potential=reset_potential, threshold=threshold)

    grid = comp.Grid(grid_shape)
    # link = grid.link((5, 5), (2, 2), (2, 2))
    link = grid.link((3, 3), (2, 2), (1, 1))

    in_stream = comp.SpatialStream(
        grid,
        times,
        coords,
        #    min_mean_size=start_mean_size
        bucket_sizes=bucket_sizes)
    # in_stream = comp.SpatialStream(grid, times, coords, min_mean_size=None)

    out_stream = comp.spike_threshold(
        in_stream,
        link,
        decay_time=decay_time,
        #   min_mean_size=hidden_mean_sizes[0],
        bucket_sizes=bucket_sizes,
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

    # for min_mean_size in hidden_mean_sizes[1:]:
    for _ in range(3):
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
        branched = activation(ft + fp + fc)

        branched = layers.BatchNormalization()(branched)
        features = features + branched

        link = in_stream.grid.link((3, 3), (2, 2), (1, 1))
        out_stream = comp.spike_threshold(
            in_stream,
            link,
            decay_time=decay_time,
            #   min_mean_size=min_mean_size,
            bucket_sizes=bucket_sizes,
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

    global_stream = comp.global_spike_threshold(
        in_stream,
        decay_time=decay_time,
        # min_mean_size=final_mean_size,
        bucket_sizes=bucket_sizes,
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


@gin.configurable(module='ecn.builders')
def inception_multi_graph_v2(
        features,
        labels,
        weights=None,
        num_classes=101,
        grid_shape=(234, 174),
        decay_time=2000,
        spatial_buffer=32,
        reset_potential=-2.0,
        threshold=1.0,
        filters0: int = 8,
        kt0: int = 4,
        hidden_units: Sequence[int] = (256,),
        dropout_rate: float = 0.5,
        decay_time_expansion_rate: float = np.sqrt(2),
        num_levels: int=5,
        activation='relu',
        bucket_sizes: bool=False,
        recenter: bool=True
):
    times = features['time']
    coords = features['coords']
    polarity = features['polarity']
    if recenter:
        with mg.pre_cache_context():
            max_coords = tf.reduce_max(coords, axis=0)
            offset = (tf.constant(grid_shape, dtype=coords.dtype) - max_coords) // 2
            coords = coords + offset
    filters = filters0
    activation = tf.keras.activations.get(activation)

    spike_kwargs = dict(reset_potential=reset_potential, threshold=threshold)

    grid = comp.Grid(grid_shape)
    # link = grid.link((5, 5), (2, 2), (2, 2))
    link = grid.link((3, 3), (2, 2), (1, 1))

    in_stream = comp.SpatialStream(grid,
                                   times,
                                   coords,
                                   bucket_sizes=bucket_sizes)

    out_stream = comp.spike_threshold(in_stream,
                                      link,
                                      decay_time=decay_time,
                                      bucket_sizes=bucket_sizes,
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

    for _ in range(num_levels):
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
        branched = activation(ft + fp + fc)

        branched = layers.BatchNormalization()(branched)
        features = features + branched

        link = in_stream.grid.link((3, 3), (2, 2), (1, 1))
        out_stream = comp.spike_threshold(in_stream,
                                          link,
                                          decay_time=decay_time,
                                          bucket_sizes=bucket_sizes,
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
                                                bucket_sizes=bucket_sizes,
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


@gin.configurable(module='ecn.builders')
def inception_vox_pooling(
        features,
        labels,
        weights=None,
        num_classes=10,
        grid_shape=(128, 128),
        decay_time=2000,
        spatial_buffer=32,
        reset_potential=-3.0,
        threshold=1.5,
        filters0: int = 8,
        kt0: int = 4,
        hidden_units: Sequence[int] = (256,),
        dropout_rate: float = 0.5,
        decay_time_expansion_rate: float = 2.0,
        num_levels: int=5,
        activation='relu',
        bucket_sizes: bool=False,
        recenter: bool=True,
        vox_reduction: str = 'mean',
        vox_start = 2,
        initial_pooling = None,
        max_events = None
):
    if vox_reduction == 'max':
        reduction = tf.math.unsorted_segment_max
    else:
        assert(vox_reduction == 'mean')
        reduction = tf.math.unsorted_segment_mean
    times = features['time']
    coords = features['coords']
    polarity = features['polarity']
    with mg.pre_cache_context():
        if max_events is not None:
            times = times[:max_events]
            coords = coords[:max_events]
            polarity = polarity[:max_events]
        if initial_pooling is not None:
            if grid_shape is not None:
                grid_shape = tuple(g // initial_pooling for g in grid_shape)
            coords = coords // initial_pooling
        if recenter:
            max_coords = tf.reduce_max(coords, axis=0)
            offset = (tf.constant(grid_shape, dtype=coords.dtype) - max_coords) // 2
            coords = coords + offset
        times = times - times[0]
        t_start = None

    filters = filters0
    activation = tf.keras.activations.get(activation)

    spike_kwargs = dict(reset_potential=reset_potential, threshold=threshold)

    grid = comp.Grid(grid_shape)
    # link = grid.link((5, 5), (2, 2), (2, 2))
    link = grid.link((3, 3), (2, 2), (1, 1))

    in_stream: comp.SpatialStream = comp.SpatialStream(grid,
                                   times,
                                   coords,
                                   bucket_sizes=bucket_sizes)
    with mg.pre_batch_context():
        t_end = in_stream.cached_times[-1] + 1
    t_end = mg.batch(t_end)

    out_stream = comp.spike_threshold(in_stream,
                                      link,
                                      decay_time=decay_time,
                                      bucket_sizes=bucket_sizes,
                                      **spike_kwargs)

    features = in_stream.prepare_model_inputs(polarity)

    batch_size, features = tf.keras.layers.Lambda(
        lambda x: (x.nrows(), tf.identity(x.values)))(features)
    num_frames = 2**(num_levels - 1)

    convolver = comp.spatio_temporal_convolver(
        link,
        in_stream,
        out_stream,
        decay_time=decay_time,
        spatial_buffer_size=spatial_buffer)
    features = convolver.convolve(features,
                                  filters=filters,
                                  temporal_kernel_size=kt0,
                                  activation=activation)
    features = layers.BatchNormalization()(features)
    features = layers.Dropout(dropout_rate)(features)

    in_stream = out_stream
    del out_stream
    del convolver
    decay_time = int(decay_time * decay_time_expansion_rate)

    t_kernel = np.zeros((5, 5), dtype=np.bool)
    t_kernel[2] = True
    t_kernel[:, 2] = True


    def do_in_place(in_stream: comp.SpatialStream, features, filters):
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
        fc = layers.Dense(units=filters * 4, activation=activation)(features)
        fc = layers.Dense(units=filters)(fc)
        branched = activation(ft + fp + fc)

        branched = layers.BatchNormalization()(branched)
        features = features + branched
        return features

    def merge_voxel_features(
            in_stream: comp.SpatialStream, features, voxel_features,
            num_frames):
        out_voxel_features = in_stream.voxelize(
            reduction, features, t_start, t_end, num_frames, batch_size)
        out_voxel_features = layers.BatchNormalization()(out_voxel_features)
        if voxel_features is None:
            return out_voxel_features
        else:
            voxel_features = layers.Conv3D(
                features.shape[-1], 2, 2, activation=activation,
                padding='same')(voxel_features)
            voxel_features = layers.BatchNormalization()(voxel_features)
            voxel_features = voxel_features + out_voxel_features
        return voxel_features


    voxel_features = None

    for i in range(num_levels-1):
        # in place
        features = do_in_place(in_stream, features, filters)
        if i >= vox_start:
            voxel_features = merge_voxel_features(
                in_stream, features, voxel_features, num_frames)
        num_frames //= 2
        filters *= 2

        link = in_stream.grid.link((3, 3), (2, 2), (1, 1))
        out_stream = comp.spike_threshold(in_stream,
                                          link,
                                          decay_time=decay_time,
                                          bucket_sizes=bucket_sizes,
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
                                         activation=activation)
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(dropout_rate)(features)
        in_stream = out_stream
        del out_stream
        decay_time = int(decay_time * decay_time_expansion_rate)

    features = do_in_place(in_stream, features, filters)
    voxel_features = merge_voxel_features(
        in_stream, features, voxel_features, num_frames)
    assert (num_frames == 1)
    assert (voxel_features.shape[1] == 1)
    image_features = Lambda(tf.squeeze, arguments=dict(axis=1))(voxel_features)
    image_features = layers.Dense(2*filters)(image_features)
    features = tf.keras.layers.GlobalMaxPooling2D()(image_features)
    features = layers.BatchNormalization()(features)
    features = layers.Dropout(dropout_rate)(features)

    for h in hidden_units:
        features = layers.Dense(h, activation=activation)(features)
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(dropout_rate)(features)
    logits = layers.Dense(num_classes, activation=None, name='logits')(features)

    labels = mg.batch(mg.cache(labels))
    if weights is None:
        return logits, labels
    else:
        weights = mg.batch(mg.cache(weights))
        return logits, labels, weights


@gin.configurable(module='ecn.builders')
def inception_pooling(
        features,
        labels,
        weights=None,
        num_classes=101,
        grid_shape=(234, 174),
        decay_time=2000,
        spatial_buffer=32,
        reset_potential=-2.0,
        threshold=1.5,
        filters0: int = 16,
        kt0: int = 4,
        pooled_units: int = 256,
        pooled_kt: int = 4,
        hidden_units: Sequence[int] = (256,),
        dropout_rate: float = 0.5,
        decay_time_expansion_rate: float = 2.,
        num_levels: int=3,
        activation='relu',
        bucket_sizes: bool=False,
        recenter: bool=True,
):
    times = features['time']
    coords = features['coords']
    polarity = features['polarity']
    with mg.pre_cache_context():
        if recenter:
            max_coords = tf.reduce_max(coords, axis=0)
            offset = (tf.constant(
                grid_shape, dtype=coords.dtype) - max_coords) // 2
            coords = coords + offset
        times = times - times[0]
        t_end = times[-1]
    filters = filters0
    activation = tf.keras.activations.get(activation)

    spike_kwargs = dict(reset_potential=reset_potential, threshold=threshold)

    grid = comp.Grid(grid_shape)
    # link = grid.link((5, 5), (2, 2), (2, 2))
    link = grid.link((3, 3), (2, 2), (1, 1))

    in_stream: comp.SpatialStream = comp.SpatialStream(
        grid,
        times,
        coords,
        bucket_sizes=bucket_sizes)
    t_end = tf.gather(in_stream.model_times, in_stream.model_row_ends)

    out_stream = comp.spike_threshold(in_stream,
                                      link,
                                      decay_time=decay_time,
                                      bucket_sizes=bucket_sizes,
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
                                  activation=activation)
    features = layers.BatchNormalization()(features)
    features = layers.Dropout(dropout_rate)(features)
    in_stream = out_stream
    del out_stream
    del convolver
    decay_time = int(decay_time * decay_time_expansion_rate)

    t_kernel = np.zeros((5, 5), dtype=np.bool)
    t_kernel[2] = True
    t_kernel[:, 2] = True

    pooled = []

    def do_in_place(in_stream: comp.SpatialStream, features, filters):
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
        fc = layers.Dense(units=filters * 4, activation=activation)(features)
        fc = layers.Dense(units=filters)(fc)
        branched = activation(ft + fp + fc)

        branched = layers.BatchNormalization()(branched)
        features = features + branched
        pooled.append(in_stream.pool_features(
            t_end, features, pooled_units, pooled_kt,
            activation=activation))
        return features


    for _ in range(num_levels):
        # in place
        features = do_in_place(in_stream, features, filters)

        link = in_stream.grid.link((3, 3), (2, 2), (1, 1))
        out_stream = comp.spike_threshold(in_stream,
                                          link,
                                          decay_time=decay_time,
                                          bucket_sizes=bucket_sizes,
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
                                         activation=activation)
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(dropout_rate)(features)

        in_stream = out_stream
        del out_stream
        decay_time = int(decay_time * decay_time_expansion_rate)

    features = do_in_place(in_stream, features, filters)
    features = Lambda(tf.concat, arguments=dict(axis=-1))(pooled)
    features = layers.BatchNormalization()(features)
    features = layers.Dropout(dropout_rate)(features)
    for h in hidden_units:
        features = layers.Dense(h, activation=activation)(features)
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(dropout_rate)(features)
    logits = layers.Dense(num_classes, activation=None, name='stream')(features)
    labels = mg.batch(mg.cache(labels))
    if weights is None:
        return logits, labels
    else:
        return logits, labels, mg.batch(mg.cache(weights))
