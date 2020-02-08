from typing import Tuple, Optional

import numpy as np
import numba as nb

import ecn.np_utils.buffer as bu

FloatArray = np.ndarray
IntArray = np.ndarray


@nb.njit()
def compute_global_neighbors_prealloc(in_times: IntArray, out_times: IntArray,
                                      indices: IntArray, splits: IntArray,
                                      event_duration: int):
    in_events = in_times.size
    if in_events == 0:
        return 0
    out_events = out_times.size
    i = 0
    it = in_times[i]
    max_neighbors = indices.size

    ii = 0
    for o in range(out_events):
        ot = out_times[o]
        start_time = ot - event_duration

        # skip events that have already expired
        while it < start_time:
            i += 1
            if i == in_events:
                splits[o:] = splits[ii]
                return i
            it = in_times[i]

        # add events
        j = i
        while j < in_events and in_times[j] <= ot:
            j += 1

        for k in range(i, j):
            indices[ii] = k
            ii += 1
            if ii == max_neighbors:
                splits[o + 1:] = k
                return i

        splits[o + 1] = ii
    return i


@nb.njit()
def compute_global_neighbors(in_times: IntArray,
                             out_times: IntArray,
                             event_duration: int,
                             max_neighbors: int,
                             out_size: Optional[int] = None):
    if out_size is None:
        out_size = out_times.size
    indices = np.empty((max_neighbors,), dtype=np.int64)
    splits = np.empty((out_size + 1,), dtype=np.int64)
    splits[0] = 0
    i = compute_global_neighbors_prealloc(
        in_times=in_times,
        out_times=out_times,
        indices=indices,
        splits=splits,
        event_duration=event_duration,
    )
    return indices, splits, i


@nb.njit()
def compute_neighbors_prealloc(in_times: IntArray, in_coords: IntArray,
                               out_times: IntArray, out_coords: IntArray,
                               buffer_start_stops: IntArray,
                               buffer_values: IntArray, index_values: IntArray,
                               index_splits: IntArray,
                               event_duration: int) -> Tuple[int, int]:
    out_events = out_times.size
    in_events = in_times.size
    if out_events == 0 or in_events == 0:
        return 0, 0
    i = 0
    j = 0
    it = in_times[i]
    mod = buffer_values.shape[-1]
    index_splits[0] = 0

    ii = 0
    max_neighbors = index_values.size

    for o in range(out_events):
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
            bu.push_right(j, buffer_values[y, x], buffer_start_stops[y, x], mod)
            j += 1
            if j == in_events:
                break
            jt = in_times[j]

        # below is related to output events
        x, y = out_coords[o]
        buff_vals = buffer_values[y, x]
        start_stop = buffer_start_stops[y, x]

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
                index_splits[o + 1:] = ii
                return i, j

        index_splits[o + 1] = ii

    return i, j


@nb.njit()
def compute_neighbors(in_times: IntArray,
                      in_coords: IntArray,
                      out_times: IntArray,
                      out_coords: IntArray,
                      event_duration: int,
                      spatial_buffer_size: int,
                      max_neighbors: Optional[int] = None,
                      num_out_events: Optional[int] = None
                     ) -> Tuple[IntArray, IntArray]:
    """
    Compute neighboring indices.

    Args:

    Returns:
        index_values: [stride, stride, max_neighbors]
        index_splits: [stride, stride, out_events]
        in_events: number of input events consumed.

    """
    H = np.max(in_coords[:, 1]) + 1
    W = np.max(in_coords[:, 0]) + 1
    if num_out_events is None:
        num_out_events = out_times.size
    if max_neighbors is None:
        max_neighbors = spatial_buffer_size * num_out_events

    buffer_start_stops = np.zeros((H, W, 2), dtype=np.int64)
    buffer_values = np.empty((H, W, spatial_buffer_size), dtype=np.int64)

    index_values = np.empty((max_neighbors), dtype=np.int64)
    index_splits = np.empty((num_out_events + 1,), dtype=np.int64)

    compute_neighbors_prealloc(
        in_times=in_times,
        in_coords=in_coords,
        out_times=out_times,
        out_coords=out_coords,
        buffer_start_stops=buffer_start_stops,
        buffer_values=buffer_values,
        index_values=index_values,
        index_splits=index_splits,
        event_duration=event_duration,
    )

    return index_values, index_splits
