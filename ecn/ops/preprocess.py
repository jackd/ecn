from typing import NamedTuple
import tensorflow as tf
import ecn.np_utils.preprocess as _pp

IntTensor = tf.Tensor


class SpatialConvArgs(NamedTuple):
    out_times: IntTensor
    out_coords: IntTensor
    neigh: tf.RaggedTensor


class GlobalConvArgs(NamedTuple):
    out_times: IntTensor
    neigh: tf.RaggedTensor


def preprocess_network(times: IntTensor,
                       coords: IntTensor,
                       stride: int,
                       decay_time: int,
                       event_duration: int,
                       spatial_buffer_size: int,
                       num_layers: int,
                       threshold: float = 2.,
                       reset_potential: float = -1):

    def f(times, coords):
        return tf.nest.flatten(
            _pp.preprocess_network(
                times.numpy(),
                coords.numpy(),
                stride=stride,
                decay_time=decay_time,
                event_duration=event_duration,
                spatial_buffer_size=spatial_buffer_size,
                num_layers=num_layers,
                threshold=threshold,
                reset_potential=reset_potential,
            ))

    ndims = coords.shape[1]
    dtypes = (((tf.int64,) * 4,) * num_layers, (tf.int64,) * 3)
    out = tf.py_function(f, (times, coords), tf.nest.flatten(dtypes))
    layer_args, global_args = tf.nest.pack_sequence_as(dtypes, out)
    out_layer_args = []
    for out_times, out_coords, indices, splits in layer_args:
        out_coords.set_shape((None, ndims))
        for t in (out_times, indices, splits):
            t.set_shape((None,))
        out_layer_args.append(
            SpatialConvArgs(out_times, out_coords,
                            tf.RaggedTensor.from_row_splits(indices, splits)))

    for t in global_args:
        t.set_shape((None,))
    times, indices, splits = global_args
    return tuple(out_layer_args), GlobalConvArgs(
        times, tf.RaggedTensor.from_row_splits(indices, splits))
