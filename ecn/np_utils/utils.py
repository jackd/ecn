from typing import Optional, Tuple
import os
import numpy as np
import numba as nb

PARALLEL = os.environ.get('NUMBA_PARALLEL', '1') != '0'
IntArray = np.ndarray


@nb.njit(inline='always')
def double_length(x):
    return np.concatenate((x, np.empty_like(x)), axis=0)


@nb.njit(inline='always')
def max_on_axis(x: np.ndarray, axis=0, out: Optional[np.ndarray] = None):
    if axis != 0 or x.ndim != 2:
        raise NotImplementedError('TODO')
    nd = x.shape[1]
    if out is None:
        out = np.empty((nd,), dtype=x.dtype)

    # errors using nb.prange...
    for i in range(nd):  # pylint: disable=not-an-iterable
        out[i] = np.max(x[:, i])
    return out


@nb.njit(inline='always')
def min_on_axis(x: np.ndarray, axis=0, out: Optional[np.ndarray] = None):
    if axis != 0 or x.ndim != 2:
        raise NotImplementedError('TODO')
    nd = x.shape[1]
    if out is None:
        out = np.empty((nd,), dtype=x.dtype)

    # errors using nb.prange...
    for i in range(nd):  # pylint: disable=not-an-iterable
        out[i] = np.min(x[:, i])
    return out


@nb.njit(inline='always')
def min_on_leading_axis(x: np.ndarray):
    out = np.empty((x.shape[1],), dtype=x.dtype)
    # errors using nb.prange...
    nd = x.shape[1]
    for i in range(nd):  # pylint: disable=not-an-iterable
        out[i] = np.min(x[:, i])
    return out


@nb.njit()
def iter_product(first, *rest):
    if len(rest) == 0:
        for f in first:
            yield f,
    else:
        for f in first:
            for others in iter_product(*rest):  # pylint: disable=no-value-for-parameter
                yield (f,) + others


@nb.njit()
def iter_product_array(first: np.ndarray, *rest: np.ndarray) -> np.ndarray:
    n = len(first)
    for r in rest:
        n *= len(r)
    nd = len(rest) + 1
    out = np.empty((n, nd), dtype=first.dtype)
    i = 0
    for indices in iter_product(first, *rest):
        out[i] = indices
        i += 1
    return out


@nb.njit()
def merge(times0: IntArray, coords0: IntArray, times1: IntArray,
          coords1: IntArray) -> Tuple[IntArray, IntArray]:
    total = times0.size + times1.size
    i0 = 0
    i1 = 0
    t0 = times0[0]
    t1 = times1[0]
    out_times = np.empty((total,), dtype=times0.dtype)
    out_coords = np.empty((total, coords0.shape[1]), dtype=coords0.dtype)
    for i in range(total):
        if t1 < t0:
            out_times[i] = t1
            out_coords[i] = coords1[i1]
            i1 += 1
            t1 = times1[i1]
        else:
            out_times[i] = t0
            out_coords[i] = coords0[i0]
            i0 += 1
            t0 = times0[i0]
    return out_times, out_coords


@nb.njit(inline='always')
def prod(x):
    p = 1
    for xi in x:
        p *= xi
    return p


# @nb.njit()
# def iter_ranges(limits):
#     diffs = limits[:, 1] - limits[:, 0]
#     n = np.prod(diffs)
#     out = np.empty((n, diffs.size), dtype=limits.dtype)
