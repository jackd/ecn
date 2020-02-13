from typing import Tuple
import functools
import tensorflow as tf
from ecn.np_utils import neighbors as _np_neigh
IntTensor = tf.Tensor
BoolTensor = tf.Tensor


def compute_global_neighbors(in_times: IntTensor,
                             out_times: IntTensor,
                             event_duration: int,
                             max_neighbors: int = -1
                            ) -> Tuple[IntTensor, IntTensor]:

    indices, splits = tf.numpy_function(
        functools.partial(_np_neigh.compute_global_neighbors,
                          event_duration=event_duration,
                          max_neighbors=max_neighbors), [in_times, out_times],
        [tf.int64] * 2)
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

    indices, splits = tf.numpy_function(
        functools.partial(_np_neigh.compute_neighbors,
                          event_duration=event_duration,
                          spatial_buffer_size=spatial_buffer_size,
                          max_neighbors=max_neighbors),
        [in_times, in_coords, out_times, out_coords], [tf.int64] * 2)
    indices.set_shape((None,))
    n_out = out_times.shape[0]
    splits.set_shape(((None if n_out is None else (n_out + 1)),))
    return indices, splits


# def reindex_index(mask: BoolTensor) -> IntTensor:
#     return tf.math.cumsum(tf.cast(mask, tf.int64)) - 1

# def reindex(original_indices: IntTensor, reindex_index: IntTensor) -> IntTensor:
#     return tf.gather(reindex_index, original_indices)

# def mask_ragged_rows(indices: IntTensor, row_splits: IntTensor,
#                      mask: BoolTensor) -> Tuple[IntTensor, IntTensor]:
#     rt = tf.RaggedTensor.from_row_splits(indices, row_splits)
#     rt = tf.ragged.boolean_mask(rt, mask)
#     return rt.values, rt.row_splits
