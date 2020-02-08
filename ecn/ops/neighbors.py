from typing import Tuple
import tensorflow as tf
from ecn.np_utils import neighbors as _np_neigh
IntTensor = tf.Tensor


def compute_global_neighbors(in_times: IntTensor, out_times: IntTensor,
                             event_duration: int,
                             max_neighbors: int) -> Tuple[IntTensor, IntTensor]:
    padded_size = out_times.size

    def fn(in_times, out_times):
        return _np_neigh.compute_global_neighbors(
            in_times.numpy(),
            out_times.numpy(),
            event_duration=event_duration,
            max_neighbors=max_neighbors,
            out_size=padded_size,
        )[:2]

    indices, splits = tf.py_function(fn, [in_times, out_times], [tf.int64] * 2)
    indices.set_shape((max_neighbors,))
    if padded_size is not None:
        splits.set_shape((padded_size + 1,))
    return indices, splits


def compute_neighbors(in_times: IntTensor, in_coords: IntTensor,
                      out_times: IntTensor, out_coords: IntTensor,
                      event_duration: int, spatial_buffer_size: int,
                      max_neighbors: int) -> Tuple[IntTensor, IntTensor]:

    def fn(in_times, in_coords, out_times, out_coords):
        return _np_neigh.compute_neighbors(
            in_times.numpy(),
            in_coords.numpy(),
            out_times.numpy(),
            out_coords.numpy(),
            event_duration=event_duration,
            spatial_buffer_size=spatial_buffer_size,
            max_neighbors=max_neighbors,
        )[:2]

    indices, splits = tf.py_function(
        fn, [in_times, in_coords, out_times, out_coords], [tf.int64] * 2)
    indices.set_shape((max_neighbors,))
    n_out = out_times.size
    if n_out is not None:
        splits.set_shape((n_out + 1,))
    return indices, splits
