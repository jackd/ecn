# import functools
# from typing import Tuple

# import tensorflow as tf

from ecn.layers.utils import as_layer
from ecn.ops import grid as _grid_ops

ravel_multi_index = as_layer(_grid_ops.ravel_multi_index)
unravel_index_transpose = as_layer(_grid_ops.unravel_index_transpose)
base_grid_coords = as_layer(_grid_ops.base_grid_coords)
output_shape = as_layer(_grid_ops.output_shape)
grid_coords = as_layer(_grid_ops.grid_coords)
shift_grid_coords = as_layer(_grid_ops.shift_grid_coords)
sparse_neighborhood = as_layer(_grid_ops.sparse_neighborhood)
sparse_neighborhood_in_place = as_layer(_grid_ops.sparse_neighborhood_in_place)
sparse_neighborhood_from_mask = as_layer(_grid_ops.sparse_neighborhood_from_mask)
sparse_neighborhood_from_mask_in_place = as_layer(
    _grid_ops.sparse_neighborhood_from_mask_in_place
)


# @functools.wraps(_grid_ops.ravel_multi_index)
# def ravel_multi_index(indices, dims: tf.Tensor, axis=-1):
#     return wrap(_grid_ops.ravel_multi_index, indices, dims, axis=axis)


# @functools.wraps(_grid_ops.unravel_index_transpose)
# def unravel_index_transpose(indices, dims):
#     return wrap(_grid_ops.unravel_index_transpose, indices, dims)


# @functools.wraps(_grid_ops.base_grid_coords)
# def base_grid_coords(shape: IntTensor):
#     return wrap(_grid_ops.base_grid_coords, shape)


# @functools.wraps(_grid_ops.output_shape)
# def output_shape(in_shape, kernel_shape, strides, padding):
#     return wrap(_grid_ops.output_shape, in_shape, kernel_shape, strides, padding)


# @functools.wraps(_grid_ops.grid_coords)
# def grid_coords(
#     in_shape: IntTensor, kernel_shape: IntTensor, strides: IntTensor, padding: IntTensor
# ) -> Tuple[IntTensor, IntTensor]:
#     return wrap(_grid_ops.grid_coords, in_shape, kernel_shape, strides, padding)


# @functools.wraps(_grid_ops.shift_grid_coords)
# def shift_grid_coords(out_coords, kernel_shape, strides, padding):
#     return wrap(_grid_ops.shift_grid_coords, out_coords, kernel_shape, strides, padding)


# @functools.wraps(_grid_ops.sparse_neighborhood)
# def sparse_neighborhood(
#     in_shape: IntTensor, kernel_shape: IntTensor, strides: IntTensor, padding: IntTensor
# ) -> Tuple[IntTensor, IntTensor, IntTensor, IntTensor]:
#     return wrap(_grid_ops.sparse_neighborhood, in_shape, kernel_shape, strides, padding)


# @functools.wraps(_grid_ops.sparse_neighborhood_in_place)
# def sparse_neighborhood_in_place(in_shape, kernel_shape):
#     return wrap(_grid_ops.sparse_neighborhood_in_place, in_shape, kernel_shape)


# @functools.wraps(_grid_ops.sparse_neighborhood_from_mask)
# def sparse_neighborhood_from_mask(
#     in_shape: IntTensor, kernel_mask: BoolTensor, strides: IntTensor, padding: IntTensor
# ):
#     return wrap(
#         _grid_ops.sparse_neighborhood_from_mask, in_shape, kernel_mask, strides, padding
#     )


# @functools.wraps(_grid_ops.sparse_neighborhood_from_mask_in_place)
# def sparse_neighborhood_from_mask_in_place(
#     in_shape: IntTensor, kernel_mask: BoolTensor
# ):
#     return wrap(_grid_ops.sparse_neighborhood_from_mask_in_place, in_shape, kernel_mask)
