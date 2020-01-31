from typing import Tuple
import tensorflow as tf
from ecn.np_utils import neighbors as _np_neigh
IntTensor = tf.Tensor


def compute_global_neighbors(in_times: IntTensor, in_size: IntTensor,
                             out_times: IntTensor, out_size: IntTensor,
                             event_duration: int,
                             max_neighbors: int) -> Tuple[IntTensor, IntTensor]:
    padded_size = out_times.size

    def fn(in_times, in_size, out_times, out_size):
        in_size = in_size.numpy()
        out_size = out_size.numpy()
        return _np_neigh.compute_global_neighbors(
            in_times.numpy()[:in_size],
            out_times.numpy()[:out_size],
            event_duration=event_duration,
            max_neighbors=max_neighbors,
            out_size=padded_size,
        )[:2]

    indices, splits = tf.py_function(fn,
                                     [in_times, in_size, out_times, out_size],
                                     [tf.int64] * 2)
    indices.set_shape((max_neighbors,))
    if padded_size is not None:
        splits.set_shape((padded_size + 1,))
    return indices, splits


def compute_neighbors(in_times: IntTensor, in_coords: IntTensor,
                      in_size: IntTensor, out_times: IntTensor,
                      out_coords: IntTensor, out_size: IntTensor, stride: int,
                      event_duration: int, spatial_buffer_size: int,
                      max_neighbors: int) -> Tuple[IntTensor, IntTensor]:

    def fn(in_times, in_coords, in_size, out_times, out_coords, out_size):
        in_size = in_size.numpy()
        out_size = out_size.numpy()
        return _np_neigh.compute_neighbors(
            in_times.numpy()[:in_size],
            in_coords.numpy()[:in_size],
            out_times.numpy()[:out_size],
            out_coords.numpy()[:out_size],
            stride=stride,
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
