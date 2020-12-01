from typing import Callable, Sequence, Tuple, Union

import gin
import numpy as np
import tensorflow as tf

import meta_model.pipeline as pl
from ecn import components as comp
from kblocks.keras import layers

Lambda = tf.keras.layers.Lambda


@gin.configurable(module="ecn.builders")
def inception_vox_pooling(
    features,
    labels,
    weights=None,
    num_classes: int = 10,
    grid_shape: Tuple[int, int] = (128, 128),
    decay_time: int = 2000,
    spatial_buffer: int = 32,
    reset_potential: float = -3.0,
    threshold: float = 1.5,
    filters0: int = 8,
    kt0: int = 4,
    hidden_units: Sequence[int] = (256,),
    dropout_rate: float = 0.5,
    decay_time_expansion_rate: float = 2.0,
    num_levels: int = 5,
    activation: Union[str, Callable] = "relu",
    recenter: bool = True,
    vox_reduction: str = "mean",
    vox_start: int = 2,
    initial_pooling=None,
    max_events=None,
):
    if vox_reduction == "max":
        reduction = tf.math.unsorted_segment_max
    else:
        assert vox_reduction == "mean"
        reduction = tf.math.unsorted_segment_mean
    times = features["time"]
    coords = features["coords"]
    polarity = features["polarity"]
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

    lif_kwargs = dict(reset_potential=reset_potential, threshold=threshold)

    grid = comp.Grid(grid_shape)
    link = grid.link((3, 3), (2, 2), (1, 1))

    in_stream: comp.SpatialStream = comp.SpatialStream(grid, times, coords)
    t_end = in_stream.cached_times[-1] + 1
    t_end = pl.batch(t_end)

    out_stream = comp.spatial_leaky_integrate_and_fire(
        in_stream, link, decay_time=decay_time, **lif_kwargs,
    )

    features = pl.model_input(pl.batch(pl.cache(polarity)))

    batch_size, features = tf.keras.layers.Lambda(
        lambda x: (x.nrows(), tf.identity(x.values))
    )(features)
    num_frames = 2 ** (num_levels - 1)

    convolver = comp.spatio_temporal_convolver(
        link,
        in_stream,
        out_stream,
        decay_time=decay_time,
        spatial_buffer_size=spatial_buffer,
    )
    features = convolver.convolve(
        features, filters=filters, temporal_kernel_size=kt0, activation=activation
    )
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
            spatial_buffer_size=spatial_buffer,
        )
        p_convolver = comp.pointwise_convolver(
            in_stream,
            in_stream,
            spatial_buffer_size=spatial_buffer,
            decay_time=decay_time * 4,
        )

        # (5x1 + 1x5)xt
        ft = t_convolver.convolve(features, filters=filters, temporal_kernel_size=kt0)
        # 1x1x4t
        fp = p_convolver.convolve(
            features, filters=filters, temporal_kernel_size=4 * kt0
        )
        # 1x1x1
        fc = layers.Dense(units=filters * 4, activation=activation)(features)
        fc = layers.Dense(units=filters)(fc)
        branched = activation(ft + fp + fc)

        branched = layers.BatchNormalization()(branched)
        features = features + branched
        return features

    def merge_voxel_features(
        in_stream: comp.SpatialStream, features, voxel_features, num_frames
    ):
        out_voxel_features = in_stream.voxelize(
            reduction, features, t_start, t_end, num_frames, batch_size
        )
        out_voxel_features = layers.BatchNormalization()(out_voxel_features)
        if voxel_features is None:
            return out_voxel_features

        # residual connection
        voxel_features = layers.Conv3D(
            features.shape[-1], 2, 2, activation=activation, padding="same"
        )(voxel_features)
        voxel_features = layers.BatchNormalization()(voxel_features)
        voxel_features = voxel_features + out_voxel_features
        return voxel_features

    voxel_features = None

    for i in range(num_levels - 1):
        # in place
        features = do_in_place(in_stream, features, filters)
        if i >= vox_start:
            voxel_features = merge_voxel_features(
                in_stream, features, voxel_features, num_frames
            )
        num_frames //= 2
        filters *= 2

        link = in_stream.grid.link((3, 3), (2, 2), (1, 1))
        out_stream = comp.spatial_leaky_integrate_and_fire(
            in_stream, link, decay_time=decay_time, **lif_kwargs,
        )

        ds_convolver = comp.spatio_temporal_convolver(
            link,
            in_stream,
            out_stream,
            decay_time=decay_time,
            spatial_buffer_size=spatial_buffer,
        )

        features = ds_convolver.convolve(
            features, filters=filters, temporal_kernel_size=kt0, activation=activation
        )
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(dropout_rate)(features)
        in_stream = out_stream
        del out_stream
        decay_time = int(decay_time * decay_time_expansion_rate)

    features = do_in_place(in_stream, features, filters)
    voxel_features = merge_voxel_features(
        in_stream, features, voxel_features, num_frames
    )
    assert num_frames == 1
    assert voxel_features.shape[1] == 1
    image_features = Lambda(tf.squeeze, arguments=dict(axis=1))(voxel_features)
    image_features = layers.Dense(2 * filters)(image_features)
    features = tf.keras.layers.GlobalMaxPooling2D()(image_features)
    features = layers.BatchNormalization()(features)
    features = layers.Dropout(dropout_rate)(features)

    for h in hidden_units:
        features = layers.Dense(h, activation=activation)(features)
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(dropout_rate)(features)
    logits = layers.Dense(num_classes, activation=None, name="logits")(features)

    labels = pl.batch(pl.cache(labels))
    if weights is None:
        return logits, labels
    weights = pl.batch(pl.cache(weights))
    return logits, labels, weights
