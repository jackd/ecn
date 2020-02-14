from typing import Iterable, Tuple, Union
import numpy as np
import numba as nb
from . import utils
from . import ragged

IntArray = np.ndarray

# @nb.njit()
# def strided_grid_neighbors_1d(coord: int, grid_start: int, grid_stop: int,
#                               stride: int, size: int) -> Iterable[int]:
#     """
#     Get neighboring indices on a 1D grid.

#     E.g. the neighbors of coordinate 5 on a 1D convolution with stride 2,
#     kernel_size 3 corresponds to
#     coord == 5.
#     grid_start == -1.
#     grid_stop == 2.
#     stride == 2.

#     Returns:
#         iterable of `int`s corresponding to the input coordinates in the
#         neighborhood of coord
#     """
#     c0 = coord * stride
#     for i in range(grid_start, grid_stop):
#         c = c0 + i
#         if c >= 0 and c < size:
#             yield c

# @nb.njit()
# def strided_grid_neighbors(coords, grid_starts, grid_stops, strides,
#                            shape) -> Iterable[Tuple[int, ...]]:
#     """
#     Get neighboring indices on a grid.

#     E.g. the neighborhood of a 2D kernel with stride 2, kernel_size 3 would have
#         grid_starts == (-1, -1),
#         grid_stops == (2, 2),
#         strides == (2, 2,),
#         shape = (H, W).

#     For coords == (5, 3) and H > 11, W > 7, this would return
#         (( 9, 5), ( 9, 6), ( 9, 7),
#          (10, 5), (10, 6), (10, 7),
#          (11, 5), (11, 6), (11, 7))

#     Args:
#         coords: [nd] integer coordinates.
#         grid_starts: [nd] integer offset for start of neighbors.
#         grid_stop: [nd] integer offset for one past end of last neighbor.
#         strides: [nd] integer stride.
#         shape: [nd] shape of base grid.

#     Returns:
#         Iterable [nd] integers of neighbors.
#     """
#     nd = len(coords)
#     if nd == 1:
#         for c in strided_grid_neighbors_1d(coords[0], grid_starts[0],
#                                            grid_stops[0], strides[0], shape[0]):
#             yield c,
#     else:
#         for c0 in strided_grid_neighbors_1d(coords[0], grid_starts[0],
#                                             grid_stops[0], strides[0],
#                                             shape[0]):
#             for rest in strided_grid_neighbors(coords[1:], grid_starts[1:],
#                                                grid_stops[1:], strides[1:],
#                                                shape[1:]):
#                 yield (c0, *rest)


@nb.njit(inline='always')
def ravel_multi_index(indices, dims):
    """
    See `np.ravel_multi_index`.

    Note this version accepts negative indices without raising.
    """
    ndim = len(dims)
    assert (len(indices) == ndim)
    acc = indices[-1].copy()
    stride = dims[-1]
    for i in range(ndim - 2, -1, -1):
        acc += indices[i] * stride
        stride *= dims[i]
    return acc


# @nb.njit(inline='always')
def unravel_index(indices: IntArray, shape, dtype=None) -> IntArray:
    ndim = len(shape)
    out = np.empty((ndim, indices.size),
                   dtype=indices.dtype if dtype is None else dtype)
    indices = indices.copy()
    for n in range(ndim - 1, 0, -1):
        s = shape[n]
        out[n] = indices % s
        indices //= shape[n]
    out[0] = indices

    return out


@nb.njit(inline='always')
def unravel_index_transpose(indices: IntArray, shape, dtype=None) -> IntArray:
    ndim = len(shape)
    out = np.empty((indices.size, ndim),
                   dtype=indices.dtype if dtype is None else dtype)
    indices = indices.copy()
    for n in range(ndim - 1, 0, -1):
        s = shape[n]
        out[..., n] = indices % s
        indices //= s
    out[..., 0] = indices
    return out


@nb.njit(inline='always')
def ravel_multi_index_transpose(indices: IntArray, dims):
    """
    Equivalent to `ravel_multi_index(indices.T, dims)`.

    Args:
        indices: [n, ndim] ints
        dims: tuple of ints, shape of the nd array.

    Returns:
        [n] int array of 1D indices.
    """
    ndim = len(dims)
    # assert (indices.shape[-1] == ndim)
    ndim = indices.shape[1]
    acc = indices[..., -1].copy()
    stride = dims[-1].item()
    for i in range(ndim - 2, -1, -1):
        acc += indices[..., i] * stride
        stride *= dims[i]
    return acc


# @nb.njit(inline='always')
# def _neighbor_offset_1d(kernel_size: int):
#     offset = -((kernel_size - 1) // 2)
#     return range(offset, offset + kernel_size)

# @nb.njit(inline='always')
# def _neighbor_offsets_1d_prealloc(kernel_size: int, out: IntArray):
#     offset = (kernel_size - 1) // 2
#     for k in range(kernel_size):
#         out[k] = k - offset
#     return kernel_size

# @nb.njit()
# def _neighbor_offsets(kernel_shape, out: IntArray) -> int:
#     if len(kernel_shape) == 1:
#         return _neighbor_offsets_1d_prealloc(kernel_shape[0], out[:, 0])
#     else:
#         # size, *rest = kernel_shape
#         size = kernel_shape[0]
#         rest = kernel_shape[1:]
#         start_index = 0
#         for i in _neighbor_offset_1d(size):
#             step_size = _neighbor_offsets(rest, out[start_index:, 1:])
#             stop_index = start_index + step_size
#             out[start_index:stop_index, 0] = i
#             start_index = stop_index
#         return start_index


@nb.njit(inline='always')
def neighbor_offsets(kernel_shape: IntArray):
    """
    Get neighborhood offsets associated with rectangle centered at the origin.

    e.g.
    ```python
    neighbor_offsets((3, 3)) == [
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 0],
        [0, 1],
        [1, -1],
        [1, 0],
        [1, 1],
    ]
    ```
    """
    # p = 1
    # for k in kernel_shape:
    #     p *= k

    # out = np.empty((p, len(kernel_shape)), dtype=dtype)
    # _neighbor_offsets(kernel_shape, out)
    # return out
    values = unravel_index_transpose(np.arange(utils.prod(kernel_shape)),
                                     kernel_shape)
    return values - ((kernel_shape - 1) // 2)


@nb.njit()
def grid_grid_coords(in_shape: IntArray, kernel_shape: IntArray,
                     strides: IntArray,
                     padding: IntArray) -> Tuple[IntArray, IntArray]:
    out_shape = (in_shape + padding) // strides
    out_size = utils.prod(out_shape)
    coords = np.arange(out_size)
    coords_nd = unravel_index_transpose(coords, out_shape)
    coords_nd *= strides
    coords_nd += ((kernel_shape - 1) // 2) - padding
    return coords_nd, out_shape


@nb.njit()
def sparse_neighborhood(in_shape: IntArray, kernel_shape: IntArray,
                        strides: IntArray, padding: IntArray
                       ) -> Tuple[IntArray, IntArray, IntArray, IntArray]:
    ndim = len(in_shape)
    coords_nd, out_shape = grid_grid_coords(in_shape, kernel_shape, strides,
                                            padding)
    offset = neighbor_offsets(kernel_shape)
    coords_expanded = np.expand_dims(coords_nd, axis=-2) + offset
    flat_coords = np.reshape(coords_expanded, (-1, ndim))
    num_partitions = offset.shape[0]
    num_coords = flat_coords.shape[0]
    partitions = np.arange(num_coords)
    partitions %= num_partitions
    valid = np.ones((num_coords,), dtype=np.bool_)
    for i in range(ndim):
        ci = flat_coords[:, i]
        valid[ci < 0] = False
        valid[ci >= in_shape[i]] = False

    valid_coords = flat_coords[valid]
    valid_partitions = partitions[valid]
    indices = ravel_multi_index_transpose(valid_coords, in_shape)
    lengths = np.count_nonzero(np.reshape(valid, (-1, offset.shape[0])),
                               axis=-1)
    splits = ragged.lengths_to_splits(lengths)
    return valid_partitions, indices, splits, out_shape
