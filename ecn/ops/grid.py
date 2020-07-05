from typing import Tuple

import tensorflow as tf

from kblocks.ops.ragged import lengths_to_splits, mask_to_lengths

BoolTensor = tf.Tensor
IntTensor = tf.Tensor
FloatTensor = tf.Tensor

# def partition_strided_neighbors(in_coords: tf.Tensor, out_coords: tf.Tensor,
#                                 in_indices: tf.Tensor, out_indices: tf.Tensor,
#                                 stride: int):
#     in_coords = tf.gather(in_coords // stride, in_indices)
#     out_coords = tf.gather(out_coords, out_indices)

#     diff = out_coords - in_coords
#     return ravel_multi_index(diff, stride)


def ravel_multi_index(indices, dims: tf.Tensor, axis=-1):
    if axis < 0:
        axis += indices.shape.ndims
    if axis < 0:
        raise ValueError(
            "axis must be positive, but after adding ndim still got {}".format(axis)
        )

    dtype = indices.dtype

    dims_ = tf.convert_to_tensor(dims, dtype_hint=dtype)
    if dims_.shape.ndims == 0:
        ndim = tf.shape(indices, out_type=dtype)[axis]
        # scalar
        offset = dims_ ** tf.range(ndim - 1, -1, -1, dtype=dtype)
    else:
        offset = tf.math.cumprod(dims_, exclusive=True, reverse=True)

    num_expands = indices.shape.ndims - axis - 1
    if num_expands > 0:
        offset = tf.reshape(offset, (-1,) + (1,) * num_expands)
    return tf.reduce_sum(indices * offset, axis=axis)


def unravel_index_transpose(indices, dims):
    return tf.transpose(tf.unravel_index(indices, dims), (1, 0))


def base_grid_coords(shape: IntTensor):
    return unravel_index_transpose(tf.range(tf.math.reduce_prod(shape)), shape)


def output_shape(in_shape, kernel_shape, strides, padding):
    """Can be used with numpy arrays or tensors."""
    return (in_shape + 2 * padding - kernel_shape) // strides + 1


def grid_coords(
    in_shape: IntTensor, kernel_shape: IntTensor, strides: IntTensor, padding: IntTensor
) -> Tuple[IntTensor, IntTensor]:
    in_shape = tf.convert_to_tensor(in_shape, dtype_hint=tf.int64)
    kernel_shape = tf.convert_to_tensor(kernel_shape, dtype_hint=tf.int64)
    strides = tf.convert_to_tensor(strides, dtype_hint=tf.int64)
    padding = tf.convert_to_tensor(padding, dtype_hint=tf.int64)

    out_shape = output_shape(in_shape, kernel_shape, strides, padding)
    coords_nd = base_grid_coords(out_shape)
    coords_nd = (coords_nd * strides) - padding
    return coords_nd, out_shape


def shift_grid_coords(out_coords, kernel_shape, strides, padding):
    kernel_shape = tf.convert_to_tensor(kernel_shape, dtype_hint=tf.int64)
    strides = tf.convert_to_tensor(strides, dtype_hint=tf.int64)
    padding = tf.convert_to_tensor(padding, dtype_hint=tf.int64)
    return out_coords * strides + (kernel_shape - 1) // 2 - padding


def _valid_partitions(coords, offset, in_shape):
    out_size = tf.shape(coords)[0]
    coords = tf.expand_dims(coords, axis=-2) + offset
    valid = tf.logical_and(
        tf.reduce_all(coords >= 0, axis=-1), tf.reduce_all(coords < in_shape, axis=-1)
    )
    num_partitions = tf.shape(offset, out_type=in_shape.dtype)[0]
    partitions = tf.tile(
        tf.expand_dims(tf.range(num_partitions), axis=0), (out_size, 1)
    )

    splits = lengths_to_splits(mask_to_lengths(valid, dtype=offset.dtype))
    partitions = tf.boolean_mask(partitions, valid)
    coords = tf.boolean_mask(coords, valid)
    coords = ravel_multi_index(coords, in_shape)
    return partitions, coords, splits


def sparse_neighborhood(
    in_shape: IntTensor, kernel_shape: IntTensor, strides: IntTensor, padding: IntTensor
) -> Tuple[IntTensor, IntTensor, IntTensor, IntTensor]:
    in_shape = tf.convert_to_tensor(in_shape, dtype_hint=tf.int64)
    dtype = in_shape.dtype
    kernel_shape = tf.convert_to_tensor(kernel_shape, dtype=dtype)
    strides = tf.convert_to_tensor(strides, dtype=dtype)
    padding = tf.convert_to_tensor(padding, dtype=dtype)

    coords, out_shape = grid_coords(in_shape, kernel_shape, strides, padding)
    partitions, coords, splits = _valid_partitions(
        coords, base_grid_coords(kernel_shape), in_shape
    )
    return partitions, coords, splits, out_shape


def sparse_neighborhood_in_place(in_shape, kernel_shape):
    in_shape = tf.convert_to_tensor(in_shape, dtype_hint=tf.int64)
    kernel_shape = tf.convert_to_tensor(kernel_shape, dtype_hint=in_shape.dtype)
    coords = base_grid_coords(in_shape) - ((kernel_shape - 1) // 2)
    partitions, coords, splits = _valid_partitions(
        coords, base_grid_coords(kernel_shape), in_shape
    )
    return partitions, coords, splits


def sparse_neighborhood_from_mask(
    in_shape: IntTensor, kernel_mask: BoolTensor, strides: IntTensor, padding: IntTensor
):
    in_shape = tf.convert_to_tensor(in_shape, dtype_hint=tf.int64)
    kernel_mask = tf.convert_to_tensor(kernel_mask, dtype_hint=tf.bool)
    strides = tf.convert_to_tensor(strides, dtype_hint=tf.int64)
    padding = tf.convert_to_tensor(padding, dtype_hint=tf.int64)

    kernel_shape = tf.shape(kernel_mask, out_type=tf.int64)
    coords, out_shape = grid_coords(in_shape, kernel_shape, strides, padding)
    partitions, coords, splits = _valid_partitions(
        coords, tf.cast(tf.where(kernel_mask), coords.dtype), in_shape
    )
    return partitions, coords, splits, out_shape


def sparse_neighborhood_from_mask_in_place(
    in_shape: IntTensor, kernel_mask: BoolTensor
):
    in_shape = tf.convert_to_tensor(in_shape, dtype_hint=tf.int64)
    kernel_mask = tf.convert_to_tensor(kernel_mask, dtype=tf.bool)
    kernel_shape = tf.shape(kernel_mask, out_type=in_shape.dtype)
    coords = base_grid_coords(in_shape) - ((kernel_shape - 1) // 2)
    partitions, coords, splits = _valid_partitions(
        coords, tf.cast(tf.where(kernel_mask), coords.dtype), in_shape
    )
    return partitions, coords, splits
