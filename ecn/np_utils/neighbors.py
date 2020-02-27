from typing import Tuple, Optional, Union

import numpy as np
import numba as nb

from . import buffer
from . import utils

FloatArray = np.ndarray
IntArray = np.ndarray
BoolArray = np.ndarray


@nb.njit()
def compute_pooled_neighbors(in_times: IntArray,
                             out_times: IntArray,
                             event_duration: Optional[int] = None,
                             max_neighbors: int = -1):
    dtype = in_times.dtype
    out_size = out_times.size
    if max_neighbors == -1:
        max_neighbors = out_size * 8
    splits = np.empty((out_size + 1,), dtype=dtype)
    splits[0] = 0

    in_size = in_times.size
    if in_size == 0:
        splits[1:] = 0
        return np.empty((0,), dtype=dtype), splits
    indices = np.empty((max_neighbors,), dtype=dtype)
    out_events = out_times.size
    i = 0
    it = in_times[i]

    ii = 0
    for o in range(out_events):
        ot = out_times[o]
        if event_duration is not None:
            start_time = ot - event_duration

            # skip events that have already expired
            while it < start_time:
                i += 1
                if i == in_size:
                    # early exit
                    splits[o + 1:] = ii
                    return indices[:ii], splits
                it = in_times[i]

        # add events
        j = i
        while j < in_size and in_times[j] <= ot:
            j += 1

        for k in range(i, j):
            indices[ii] = k
            ii += 1
            if ii == max_neighbors:
                indices = utils.double_length(indices)
                max_neighbors *= 2

        splits[o + 1] = ii

    return indices[:ii], splits


@nb.njit()
def compute_full_neighbors(in_times: IntArray,
                           in_coords: IntArray,
                           out_times: IntArray,
                           event_duration: Optional[int] = None,
                           max_neighbors: int = -1
                          ) -> Tuple[IntArray, IntArray, IntArray]:
    """
    Same as compute_pooled_neighbors, except we record the coordinates as well.
    """
    dtype = in_times.dtype
    out_size = out_times.size
    in_size = in_times.size
    if in_size == 0 or out_size == 0:
        splits = np.zeros((out_size + 1,), dtype=dtype)
        return np.empty((0,), dtype=dtype), np.empty((0,), dtype=dtype), splits

    if max_neighbors == -1:
        max_neighbors = out_size * 8

    splits = np.empty((out_size + 1,), dtype=dtype)
    splits[0] = 0
    indices = np.empty((max_neighbors,), dtype=dtype)
    partitions = np.empty((max_neighbors,), dtype=dtype)
    out_events = out_times.size
    i = 0
    it = in_times[i]

    ii = 0
    for o in range(out_events):
        ot = out_times[o]
        if event_duration is not None:
            start_time = ot - event_duration

            # skip events that have already expired
            while it < start_time:
                i += 1
                if i == in_size:
                    # early exit
                    splits[o + 1:] = ii
                    return partitions[:ii], indices[:ii], splits
                it = in_times[i]

        # add events
        j = i
        while j < in_size and in_times[j] <= ot:
            j += 1

        for k in range(i, j):
            indices[ii] = k
            partitions[ii] = in_coords[k]
            ii += 1
            if ii == max_neighbors:
                indices = utils.double_length(indices)
                partitions = utils.double_length(partitions)
                max_neighbors *= 2

        splits[o + 1] = ii
    return partitions[:ii], indices[:ii], splits


@nb.njit()
def compute_pointwise_neighbors(
        in_times: IntArray,
        in_coords: IntArray,
        out_times: IntArray,
        out_coords: IntArray,
        spatial_buffer_size: int,
        event_duration: Optional[int] = None,
) -> Tuple[IntArray, IntArray]:
    """Pointwise optimization of compute_neighbors."""
    assert (in_times.ndim == 1)
    assert (in_coords.ndim == 1)
    assert (out_times.ndim == 1)
    assert (out_coords.ndim == 1)
    assert (in_times.size == in_coords.size)
    assert (out_times.size == out_coords.size)

    dtype = in_times.dtype

    # grid_size = grid_splits.size - 1
    num_out_events = out_times.size
    num_in_events = in_times.size

    if num_out_events == 0 or num_in_events == 0:
        index_splits = np.zeros((num_out_events + 1,), dtype=dtype)
        return (
            np.empty((0,), dtype=dtype),
            index_splits,
        )

    index_splits = np.empty((num_out_events + 1,), dtype=dtype)
    index_splits[0] = 0

    max_neighbors = spatial_buffer_size * num_out_events
    grid_size = max(np.max(in_coords), np.max(out_coords)) + 1

    buffer_start_stops = np.zeros((grid_size, 2), dtype=np.int64)
    buffer_values = np.empty((grid_size, spatial_buffer_size), dtype=np.int64)
    indices = np.empty((max_neighbors,), dtype=dtype)

    i = 0
    it = in_times[0]
    j = 0
    jt = it

    ii = 0

    for o in range(num_out_events):
        ot = out_times[o]

        if event_duration is not None:
            start_time = ot - event_duration

            # skip events that have already expired
            while it < start_time:
                i += 1
                it = in_times[i]
        else:
            start_time = None

        j = max(i, j)
        jt = max(it, jt)
        if j < num_in_events:
            while jt <= ot:
                # push right
                coord = in_coords[j]
                buffer.push_right(j, buffer_values[coord],
                                  buffer_start_stops[coord],
                                  spatial_buffer_size)
                j += 1
                if j == num_in_events:
                    break
                jt = in_times[j]

        coord = out_coords[o]
        buff_vals = buffer_values[coord]
        start_stop = buffer_start_stops[coord]

        if event_duration is not None:
            # trim spatial buffer of expired events
            for b in buffer.indices(start_stop, spatial_buffer_size):
                bv = buff_vals[b]
                bt = in_times[bv]
                if bt >= start_time:
                    break
                start_stop[0] += 1
            start_stop[0] %= spatial_buffer_size

        # add valid events
        for b in buffer.indices(start_stop, spatial_buffer_size):
            indices[ii] = buff_vals[b]
            ii += 1
            if ii == max_neighbors:
                indices = utils.double_length(indices)
                max_neighbors *= 2

        index_splits[o + 1] = ii

    return indices[:ii], index_splits


@nb.njit()
def compute_neighbors(
        in_times: IntArray,
        in_coords: IntArray,
        out_times: IntArray,
        out_coords: IntArray,
        grid_partitions: IntArray,
        grid_indices: IntArray,
        grid_splits: IntArray,
        spatial_buffer_size: int,
        event_duration: Optional[int] = None,
) -> Tuple[IntArray, IntArray, IntArray]:
    """
    Compute neighboring indices for flattened events.

    Args:
        in_times: [n_in] int input event times.
        in_coords: [n_in] input event 1D coordinates.
        out_times: [n_out] output event times.
        out_coords: [n_out] output event 1D coordinates.
        grid_partitions: ragged values denoting the partition of the
            corresponding neighborhood value.
        grid_indices: ragged values for components of neighborhoods.
            `grid_indices[grid_splits[i]: grid_splits[i+1]] == p, q, r, ...`
            indicates the output grid `i` has input neighbors `p, q, r, ...`.
        grid_splits: row splits associated with grid_indices/partitions.
        event_duration: duration of each event. If None, the full spatial buffer
            is used.
        spatial_buffer_size: maximum number of events buffered at each spatial
            location.

    Returns:
        event_partition: [num_neighbors] int array of partition values.
        event_indices: [num_neighbors] array of index values into input events.
        event_splits: [out_events + 1] array of row_splits for index_values.
            splits[-1] == num_neighbors
    """
    assert (in_times.ndim == 1)
    assert (in_coords.ndim == 1)
    assert (out_times.ndim == 1)
    assert (out_coords.ndim == 1)
    assert (grid_indices.ndim == 1)
    assert (grid_partitions.ndim == 1)
    assert (grid_splits.ndim == 1)
    assert (in_times.size == in_coords.size)
    assert (out_times.size == out_coords.size)

    num_out_events = out_times.size
    num_in_events = in_times.size

    if num_out_events == 0 or num_in_events == 0:
        index_splits = np.zeros((num_out_events + 1,), dtype=grid_indices.dtype)
        return (
            np.empty((0,), dtype=grid_partitions.dtype),
            np.empty((0,), dtype=grid_indices.dtype),
            index_splits,
        )

    index_splits = np.empty((num_out_events + 1,), dtype=grid_indices.dtype)
    index_splits[0] = 0

    max_neighbors = spatial_buffer_size * num_out_events
    grid_size = max(np.max(in_coords), np.max(grid_indices)) + 1

    buffer_start_stops = np.zeros((grid_size, 2), dtype=np.int64)
    buffer_values = np.empty((grid_size, spatial_buffer_size), dtype=np.int64)
    partitions = np.empty((max_neighbors,), dtype=grid_partitions.dtype)
    indices = np.empty((max_neighbors,), dtype=grid_indices.dtype)

    i = 0
    it = in_times[0]
    j = 0
    jt = it

    ii = 0

    for o in range(num_out_events):
        ot = out_times[o]

        if event_duration is not None:
            start_time = ot - event_duration

            # skip events that have already expired
            while it < start_time:
                i += 1
                it = in_times[i]
        else:
            start_time = None

        j = max(i, j)
        jt = max(it, jt)
        if j < num_in_events:
            while jt <= ot:
                # push right
                coord = in_coords[j]
                buffer.push_right(j, buffer_values[coord],
                                  buffer_start_stops[coord],
                                  spatial_buffer_size)
                j += 1
                if j == num_in_events:
                    break
                jt = in_times[j]

        base_coord = out_coords[o]
        for ri in range(grid_splits[base_coord], grid_splits[base_coord + 1]):
            coord = grid_indices[ri]
            p = grid_partitions[ri]

            buff_vals = buffer_values[coord]
            start_stop = buffer_start_stops[coord]

            if event_duration is not None:
                # trim spatial buffer of expired events
                for b in buffer.indices(start_stop, spatial_buffer_size):
                    bv = buff_vals[b]
                    bt = in_times[bv]
                    if bt >= start_time:
                        break
                    start_stop[0] += 1
                start_stop[0] %= spatial_buffer_size

            # add valid events
            for b in buffer.indices(start_stop, spatial_buffer_size):
                indices[ii] = buff_vals[b]
                partitions[ii] = p
                ii += 1
                if ii == max_neighbors:
                    indices = utils.double_length(indices)
                    partitions = utils.double_length(partitions)
                    max_neighbors *= 2

        index_splits[o + 1] = ii

    return partitions[:ii], indices[:ii], index_splits


@nb.njit(inline='always')
def reindex_index(mask: BoolArray) -> IntArray:
    return np.cumsum(mask) - 1


@nb.njit(inline='always')
def reindex(original_indices: IntArray, reindex_index: IntArray) -> IntArray:
    return reindex_index[original_indices]
