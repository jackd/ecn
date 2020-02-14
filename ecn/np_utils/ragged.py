from typing import Tuple
from .utils import PARALLEL
import numpy as np
import numba as nb

IntArray = np.ndarray
BoolArray = np.ndarray


@nb.njit(inline='always')
def lengths_to_ids(row_lengths: IntArray) -> IntArray:
    return np.repeat(np.arange(row_lengths.size), row_lengths)


@nb.njit(inline='always')
def ids_to_lengths(ids: IntArray) -> IntArray:
    """ids are assumed to be sorted."""
    return np.bincount(ids)


@nb.njit(inline='always')
def splits_to_lengths(row_splits: IntArray) -> IntArray:
    return row_splits[1:] - row_splits[:-1]


@nb.njit(inline='always')
def splits_to_ids(row_splits) -> IntArray:
    return lengths_to_ids(splits_to_lengths(row_splits))


@nb.njit(inline='always')
def ids_to_splits(ids) -> IntArray:
    return lengths_to_splits(ids_to_lengths(ids))


@nb.njit(inline='always')
def lengths_to_splits(row_lengths: IntArray) -> IntArray:
    out = np.empty((row_lengths.size + 1,), dtype=row_lengths.dtype)
    # np.cumsum(row_lengths, out=out[1:])  # not supported yet
    out[1:] = np.cumsum(row_lengths)
    out[0] = 0
    return out


@nb.njit(inline='always', parallel=PARALLEL)
def row_sum(values: IntArray, splits: IntArray, out=None) -> IntArray:
    nrows = splits.size - 1
    if out is None:
        out = np.zeros((nrows,), dtype=values.dtype)
    for i in nb.prange(nrows):  # pylint: disable=not-an-iterable
        out[i] = values[splits[i]:splits[i + 1]].sum()
    return out


@nb.njit(inline='always')
def row_sorted(row_indices: IntArray,
               col_indices: IntArray) -> Tuple[IntArray, IntArray]:
    order = np.argsort(row_indices)
    return row_indices[order], col_indices[order]


@nb.njit(inline='always')
def row(values, splits, row_index) -> IntArray:
    return values[splits[row_index]:splits[row_index + 1]]


@nb.njit(inline='always')
def rows(values, splits):
    s0 = splits[0]
    for s in splits[1:]:
        yield values[s0:s]
        s0 = s


@nb.njit(inline='always', parallel=PARALLEL)
def col_sort(values, splits) -> None:
    for i in nb.prange(splits.size - 1):  # pylint: disable=not-an-iterable
        row(values, splits, i).sort()


@nb.njit(inline='always')
def transpose_csr(indices: IntArray,
                  splits: IntArray) -> Tuple[IntArray, IntArray]:
    col_indices = splits_to_ids(splits)  # initial row indices
    row_indices, col_indices = row_sorted(indices, col_indices)
    return col_indices, ids_to_splits(row_indices)


@nb.njit()
def mask_rows(indices: IntArray, row_splits: IntArray,
              mask: BoolArray) -> Tuple[IntArray, IntArray]:
    row_lengths = row_splits[1:] - row_splits[:-1]
    row_lengths = row_lengths[:mask.size][mask]
    total = np.sum(row_lengths)
    out_indices = np.zeros((total,), dtype=indices.dtype)
    out_row_splits = np.empty((np.count_nonzero(mask) + 1,),
                              dtype=row_splits.dtype)
    jj = out_row_splits[0] = row_splits[0]
    j = 0

    for i in range(mask.size):
        if mask[i]:
            for ii in range(row_splits[i], row_splits[i + 1]):
                out_indices[jj] = indices[ii]
                jj += 1
            j += 1
            out_row_splits[j] = jj
    return out_indices, out_row_splits


# class RaggedArray(object):

#     def __init__(self, values, row_splits):
#         self.values = values
#         self.row_splits = row_splits

#     def row(self, i: int):
#         return self.values[self.row_splits[i]:self.row_splits[i + 1]]

#     def value_rowids(self):
#         row_lengths = self.row_lengths()
#         return np.repeat(np.arange(row_lengths.size), row_lengths)

#     def row_lengths(self):
#         return self.row_splits[1:] - self.row_splits[:-1]
