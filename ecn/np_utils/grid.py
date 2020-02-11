from typing import Iterable, Tuple
import numpy as np
import numba as nb

IntArray = np.ndarray


@nb.njit()
def strided_grid_neighbors_1d(coord: int, grid_start: int, grid_stop: int,
                              stride: int, size: int) -> Iterable[int]:
    """
    Get neighboring indices on a 1D grid.

    E.g. the neighbors of coordinate 5 on a 1D convolution with stride 2,
    kernel_size 3 corresponds to
    coord == 5.
    grid_start == -1.
    grid_stop == 2.
    stride == 2.

    Returns:
        iterable of `int`s corresponding to the input coordinates in the
        neighborhood of coord
    """
    c0 = coord * stride
    for i in range(grid_start, grid_stop):
        c = c0 + i
        if c >= 0 and c < size:
            yield c


@nb.njit()
def strided_grid_neighbors(coords, grid_starts, grid_stops, strides,
                           shape) -> Iterable[Tuple[int, ...]]:
    """
    Get neighboring indices on a grid.

    E.g. the neighborhood of a 2D kernel with stride 2, kernel_size 3 would have
        grid_starts == (-1, -1),
        grid_stops == (2, 2),
        strides == (2, 2,),
        shape = (H, W).

    For coords == (5, 3) and H > 11, W > 7, this would return
        (( 9, 5), ( 9, 6), ( 9, 7),
         (10, 5), (10, 6), (10, 7),
         (11, 5), (11, 6), (11, 7))

    Args:
        coords: [nd] integer coordinates.
        grid_starts: [nd] integer offset for start of neighbors.
        grid_stop: [nd] integer offset for one past end of last neighbor.
        strides: [nd] integer stride.
        shape: [nd] shape of base grid.

    Returns:
        Iterable [nd] integers of neighbors.
    """
    nd = len(coords)
    if nd == 1:
        for c in strided_grid_neighbors_1d(coords[0], grid_starts[0],
                                           grid_stops[0], strides[0], shape[0]):
            yield c,
    else:
        for c0 in strided_grid_neighbors_1d(coords[0], grid_starts[0],
                                            grid_stops[0], strides[0],
                                            shape[0]):
            for rest in strided_grid_neighbors(coords[1:], grid_starts[1:],
                                               grid_stops[1:], strides[1:],
                                               shape[1:]):
                yield (c0, *rest)


@nb.njit(inline='always')
def identity_neighbors(coords):
    return (coords,)


@nb.njit(inline='always')
def ravel_multi_index(indices, dims):
    """See `np.ravel_multi_index`."""
    nd = len(dims)
    acc = indices[-1].copy()
    stride = dims[-1]
    for i in range(nd - 2, -1, -1):
        acc += indices[i] * stride
        stride *= dims[i]
    return acc
