from typing import Tuple, Optional, Union

import numpy as np
import numba as nb

from . import buffer
from . import grid
from . import utils

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


# @nb.njit(inline='always')
# def neighborhood_args_as_1d(in_coords, out_coords, neighbor_offsets):
#     """
#     Convert coordinates to the equivalent coords for 1D.

#     This effectively 'pads' the grid such that
#     ```python
#     np.min(out_coords, axis=0) + np.min(neighbor_offsets, axis=0) == 0
#     ```
#     and
#     ```python
#     (np.max(out_coords, axis=0) + np.max(neighbor_offsets, axis=0) <
#         in_coords.shape)
#     ```

#     It then ravels indices to 1D equivalent.

#     Causes side effects on in_coords, out_coords.

#     Args:
#         in_coords: [n_in, ndim] input coordinates.
#         out_coords: [n_out, ndim] output coordinates.
#         neighbor_offsets: [n_neigh, ndim] neighbor offsets. Can be negative

#     Returns:
#         flat_in_coords: [n_in]
#         flat_out_coords: [n_out]
#         neighbor_offsets: [n_neigh]
#         grid_size: int, 1D size of the resulting array.
#     """
#     # assert (neighbor_offsets.ndim == 2)
#     ndim = neighbor_offsets.shape[1]
#     # assert (in_coords.ndim == 2)
#     # assert (out_coords.ndim == 2)
#     # if out_coords.shape[1] != ndim:
#     #     raise ValueError('Inconsistent out_coords dimension sizes')
#     # if in_coords.shape[1] != ndim:
#     #     raise ValueError('Inconsistent in_coords dimension sizes')

#     grid_size = 1
#     grid_shape = np.empty((ndim,), dtype=neighbor_offsets.dtype)
#     for i in range(ndim):
#         offset = np.min(neighbor_offsets[:, i])
#         ic = in_coords[:, i]

#         oc = out_coords[:, i]
#         min_in = np.min(ic)
#         min_out = np.min(oc) + offset
#         min_min = min(min_in, min_out)
#         ic -= min_min
#         oc -= min_min

#         s = max(np.max(ic), np.max(oc) + np.max(neighbor_offsets[:, i])) + 1
#         grid_shape[i] = s
#         grid_size *= s
#     return (grid.ravel_multi_index_transpose(in_coords, grid_shape),
#             grid.ravel_multi_index_transpose(out_coords, grid_shape),
#             grid.ravel_multi_index_transpose(neighbor_offsets,
#                                              grid_shape), grid_size)

# @nb.njit()
# def compute_neighbors(
#         in_times: IntArray,
#         in_coords: IntArray,
#         out_times: IntArray,
#         out_coords: IntArray,
#         neighbor_offsets: IntArray,
#         event_duration: Optional[int],
#         spatial_buffer_size: int,
# ) -> Tuple[IntArray, IntArray, IntArray]:
#     """Mutates coordinates in place."""
#     assert (in_times.ndim == 1)
#     assert (out_times.ndim == 1)
#     assert (in_coords.shape[0] == in_times.size)
#     assert (out_coords.shape[0] == out_times.size)
#     (in_coords_, out_coords_, neighbor_offsets_,
#      grid_size) = neighborhood_args_as_1d(in_coords, out_coords,
#                                           neighbor_offsets)

#     return compute_neighbors_1d_base(in_times=in_times,
#                                      in_coords=in_coords_,
#                                      out_times=out_times,
#                                      out_coords=out_coords_,
#                                      neighbor_offsets=neighbor_offsets_,
#                                      event_duration=event_duration,
#                                      spatial_buffer_size=spatial_buffer_size,
#                                      grid_size=grid_size)


@nb.njit()
def compute_neighbors_1d(
        in_times: IntArray,
        in_coords: IntArray,
        out_times: IntArray,
        out_coords: IntArray,
        neigh_indices: IntArray,
        neigh_splits: IntArray,
        neigh_partitions: IntArray,
        event_duration: Optional[int],
        spatial_buffer_size: int,
) -> Tuple[IntArray, IntArray, IntArray]:
    """
    Compute neighboring indices for 1D events.

    This is the base method for all neighborhood computations. See
    `compute_neighbors` for generalized dimension implementations.

    To implement a custom `stride`, use
        compute_neighbors_1d_base(out_coords=out_coords * stride, **kwargs)

    To implement padding, use:
        compute_neighbors_1d_base(
            out_coords=stride*out_coords + left_padding,
            # consider updating grid_size if not using the default
            **kwargs)

    To implement higher dimensions, use ravel_multi_index . E.g.
        to implement 2x2 neighborhood on a pixel grid H x W, use
        compute_neighbors_1d_base(
            in_times=in_times,
            in_coords=grid.ravel_multi_index_transpose(in_coords, (H, W)),
            out_times=out_times,
            out_coords=grid.ravel_multi_index_transpose(out_coords, (H, W)),
            neighbor_offsets=grid.ravel_multi_index_transpose(
                [[0, 0], [0, 1], [1, 0], [1, 1]], (H, W))
            **kwargs)

    Args:
        in_times: input event times.
        in_coords: input event 1D coordinates.
        out_times: output event times.
        out_coords: output event 1D coordinates.
        event_duration: duration of each event.
        spatial_buffer_size: maximum number of events buffered at each spatial
            location.
        neighbor_offset: integer offsets for each neighbor. Must not contain
            any duplicates.
        grid_size: spatial size. If None, uses a safe value (but must calculate
            it based on maximum values in both sets of coordinates).

    Returns:
        partition: [num_neighbors] int array of partition values.
        indices: [num_neighbors] array of index values into input events.
        splits: [out_events + 1] array of row_splits for index_values.
            splits[-1] == num_neighbors
    """
    assert (in_times.ndim == 1)
    assert (in_coords.ndim == 1)
    assert (out_times.ndim == 1)
    assert (out_coords.ndim == 1)
    assert (neigh_indices.ndim == 1)
    assert (neigh_splits.ndim == 1)
    assert (in_times.size == in_coords.size)
    assert (out_times.size == out_coords.size)

    grid_size = neigh_splits.size - 1
    num_out_events = out_times.size
    num_in_events = in_times.size

    index_splits = np.empty((num_out_events + 1,), dtype=np.int64)
    index_splits[0] = 0

    if num_out_events == 0 or num_in_events == 0:
        index_splits[1:] = 0
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            index_splits,
        )

    max_neighbors = spatial_buffer_size * num_out_events

    buffer_start_stops = np.zeros((grid_size, 2), dtype=np.int64)
    buffer_values = np.empty((grid_size, spatial_buffer_size), dtype=np.int64)
    partitions = np.empty((max_neighbors,), dtype=np.int64)
    indices = np.empty((max_neighbors,), dtype=np.int64)

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
        while jt <= ot:
            # push right
            coord = in_coords[j]
            buffer.push_right(j, buffer_values[coord],
                              buffer_start_stops[coord], spatial_buffer_size)
            j += 1
            if j == num_in_events:
                break
            jt = in_times[j]

        base_coord = out_coords[o]
        for ri in range(neigh_splits[base_coord], neigh_splits[base_coord + 1]):
            coord = neigh_indices[ri]
            p = neigh_partitions[ri]

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
