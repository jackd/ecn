from typing import Callable, Tuple

import numpy as np
import numba as nb

import ecn.np_utils.buffer as bu
import ecn.np_utils.grid as grid
import ecn.np_utils.utils as utils

FloatArray = np.ndarray
IntArray = np.ndarray
BoolArray = np.ndarray


@nb.njit()
def compute_global_neighbors(in_times: IntArray,
                             out_times: IntArray,
                             event_duration: int,
                             max_neighbors: int = -1):

    out_size = out_times.size
    if max_neighbors == -1:
        max_neighbors = out_size * 8
    indices = np.empty((max_neighbors,), dtype=np.int64)
    splits = np.empty((out_size + 1,), dtype=np.int64)
    splits[0] = 0

    in_events = in_times.size
    if in_events == 0:
        splits[1:] = 0
        return indices, splits
    out_events = out_times.size
    i = 0
    it = in_times[i]

    ii = 0
    for o in range(out_events):
        ot = out_times[o]
        start_time = ot - event_duration

        # skip events that have already expired
        while it < start_time:
            i += 1
            if i == in_events:
                splits[o:] = splits[ii]
                return indices, splits
            it = in_times[i]

        # add events
        j = i
        while j < in_events and in_times[j] <= ot:
            j += 1

        for k in range(i, j):
            indices[ii] = k
            ii += 1
            if ii == max_neighbors:
                indices = utils.double_length(indices)
                max_neighbors *= 2

        splits[o + 1] = ii

    return indices[:splits[-1]], splits


@nb.njit()
def compute_neighbors(
        in_times: IntArray,
        in_coords: IntArray,
        out_times: IntArray,
        out_coords: IntArray,
        event_duration: int,
        spatial_buffer_size: int,
        max_neighbors: int = -1,
        spatial_neighbors_fn: Callable = grid.identity_neighbors,
) -> Tuple[IntArray, IntArray]:
    """
    Compute neighboring indices.

    Args:

    Returns:
        index_values: [max_neighbors]
        index_splits: [out_events]
    """
    num_out_events = out_times.size
    num_in_events = in_times.size

    index_splits = np.empty((num_out_events + 1,), dtype=np.int64)
    index_splits[0] = 0

    if num_out_events == 0 or num_in_events == 0:
        index_splits[1:] = 0
        return np.empty((0,), dtype=np.int64), index_splits

    if max_neighbors == -1:
        max_neighbors = spatial_buffer_size * num_out_events
    if in_coords.ndim != 2 or in_coords.shape[1] != 2:
        raise ValueError('in_coords.shape must be (n, 2)')
    X = np.max(in_coords[:, 0]) + 1
    Y = np.max(in_coords[:, 1]) + 1

    buffer_start_stops = np.zeros((X, Y, 2), dtype=np.int64)
    buffer_values = np.empty((X, Y, spatial_buffer_size), dtype=np.int64)

    index_values = np.empty((max_neighbors), dtype=np.int64)

    i = 0
    j = 0
    it = in_times[i]
    mod = buffer_values.shape[-1]

    ii = 0
    max_neighbors = index_values.size

    for o in range(num_out_events):
        ot = out_times[o]
        start_time = ot - event_duration

        # skip events that have already expired
        while it < start_time:
            i += 1
            it = in_times[i]

        # add events from input stream to spatial buffer
        j = i
        jt = it
        while jt <= ot:
            # push right
            x, y = in_coords[j]
            coords = (x, y)
            bu.push_right(j, buffer_values[coords], buffer_start_stops[coords],
                          mod)
            j += 1
            if j == num_in_events:
                break
            jt = in_times[j]

        for coords_ in spatial_neighbors_fn(out_coords[o]):
            # below is related to output events
            x, y = coords_
            coords = (x, y)

            buff_vals = buffer_values[coords]
            start_stop = buffer_start_stops[coords]

            # trim spatial buffer of expired events
            for b in bu.indices(start_stop, mod):
                bv = buff_vals[b]
                bt = in_times[bv]
                if bt >= start_time:
                    break
                start_stop[0] += 1
            start_stop[0] %= mod

            # add valid events
            for b in bu.indices(start_stop, mod):
                index_values[ii] = buff_vals[b]
                ii += 1
                if ii == max_neighbors:
                    index_values = utils.double_length(index_values)
                    max_neighbors *= 2

        index_splits[o + 1] = ii
    return index_values[:index_splits[-1]], index_splits


@nb.njit()
def present_mask(indices: IntArray, max_index: int = -1) -> BoolArray:
    if max_index == -1:
        max_index = np.max(indices)
    mask = np.zeros((max_index,), dtype=np.bool)
    for i in indices:
        mask[i] = True
    return mask


@nb.njit(inline='always')
def reindex_index(mask: BoolArray) -> IntArray:
    return np.cumsum(mask, dtype=np.int64) - 1


@nb.njit(inline='always')
def reindex(original_indices: IntArray, reindex_index: IntArray) -> IntArray:
    return reindex_index[original_indices]


@nb.njit()
def mask_ragged_indices_row_splits(indices: IntArray, row_splits: IntArray,
                                   mask: IntArray) -> Tuple[IntArray, IntArray]:
    nrows = row_splits.size - 1
    out_indices = np.empty_like(indices)
    out_row_splits = np.empty_like(row_splits)
    out_row_splits[0] = row_splits[0]
    j = 0
    for row in range(nrows):
        for i in range(indices[row], indices[row + 1]):
            ii = indices[i]
            if mask[ii]:
                out_indices[j] = ii
                j += 1
        out_row_splits[row + 1] = j
    return out_indices[:out_row_splits[-1]], out_row_splits
