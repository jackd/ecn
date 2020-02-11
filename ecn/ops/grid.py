from typing import Union
import tensorflow as tf


def ravel_multi_index(indices, dims: Union[int, tf.Tensor], axis=-1):
    if axis < 0:
        axis += indices.shape.ndims
    if axis < 0:
        raise ValueError(
            'axis must be positive, but after adding ndim still got {}'.format(
                axis))

    if isinstance(dims, int):
        dims_ = tf.constant(dims, dtype=tf.int64)
    else:
        dims_ = dims
    assert (isinstance(dims_, tf.Tensor))
    if dims_.shape.ndims == 0:
        ndim = tf.shape(indices, out_type=tf.int64)[axis]
        # scalar
        offset = dims_**tf.range(ndim - 1, -1, -1, dtype=dims_.dtype)
    else:
        offset = tf.math.cumprod(dims_, exclusive=True, reverse=True)

    num_expands = indices.shape.ndims - axis - 1
    if num_expands > 0:
        offset = tf.reshape(offset, (-1,) + (1,) * num_expands)
    return tf.reduce_sum(indices * offset, axis=axis)


def partition_strided_neighbors(in_coords: tf.Tensor, out_coords: tf.Tensor,
                                in_indices: tf.Tensor, out_indices: tf.Tensor,
                                stride: int):
    in_coords = tf.gather(in_coords // stride, in_indices)
    out_coords = tf.gather(out_coords, out_indices)

    diff = out_coords - in_coords
    return ravel_multi_index(diff, stride)
