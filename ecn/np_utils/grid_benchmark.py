import numba as nb
import numpy as np
import ecn.np_utils.grid as grid
from ecn.benchmark_utils import benchmark, run_benchmarks

in_shape = np.array((300, 300), dtype=np.int64)
kernel_shape = np.array((3, 3), dtype=np.int64)
strides = np.array((2, 2), dtype=np.int64)
top_left = np.array((1, 1), dtype=np.int64)


@benchmark()
@nb.njit()
def create_sparse_neighborhood():
    return grid.sparse_neighborhood(in_shape, kernel_shape, strides, top_left)


run_benchmarks(5, 100)
