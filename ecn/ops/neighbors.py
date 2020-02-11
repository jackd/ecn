from typing import Tuple
import tensorflow as tf
from ecn.np_utils import neighbors as _np_neigh
IntTensor = tf.Tensor


def compute_global_neighbors(in_times: IntTensor,
                             out_times: IntTensor,
                             event_duration: int,
                             max_neighbors: int = -1
                            ) -> Tuple[IntTensor, IntTensor]:

    def fn(in_times, out_times):
        return _np_neigh.compute_global_neighbors(
            in_times.numpy(),
            out_times.numpy(),
            event_duration=event_duration,
            max_neighbors=max_neighbors,
        )

    indices, splits = tf.py_function(fn, [in_times, out_times], [tf.int64] * 2)
    indices.set_shape((None,))
    n_out = out_times.shape[0]
    splits.set_shape(((None if n_out is None else (n_out + 1)),))

    return indices, splits


def compute_neighbors(
        in_times: IntTensor,
        in_coords: IntTensor,
        out_times: IntTensor,
        out_coords: IntTensor,
        event_duration: int,
        spatial_buffer_size: int,
        max_neighbors: int = -1,
) -> Tuple[IntTensor, IntTensor]:

    def fn(in_times, in_coords, out_times, out_coords):
        return _np_neigh.compute_neighbors(
            in_times.numpy(),
            in_coords.numpy(),
            out_times.numpy(),
            out_coords.numpy(),
            event_duration=event_duration,
            spatial_buffer_size=spatial_buffer_size,
            max_neighbors=max_neighbors)

    indices, splits = tf.py_function(
        fn, [in_times, in_coords, out_times, out_coords], [tf.int64] * 2)
    indices.set_shape((None,))
    n_out = out_times.shape[0]
    splits.set_shape(((None if n_out is None else (n_out + 1)),))
    return indices, splits
