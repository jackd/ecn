from typing import Tuple, Optional

import numpy as np
import numba as nb

FloatArray = np.ndarray
IntArray = np.ndarray


@nb.njit()
def compute_global_neighbors_prealloc(in_times: IntArray, out_times: IntArray,
                                      indices: IntArray, splits: IntArray,
                                      event_duration: int):
    in_events = in_times.size
    out_events = out_times.size
    i = -1
    it = -event_duration - 1

    start = 0
    for o in range(out_events):
        ot = out_times[o]
        start_time = ot - event_duration

        # skip events that have already expired
        while it < start_time:
            i += 1
            it = in_times[i]
        j = 0
        while it <= ot:
            indices[start + j] = it
            j = j + 1
            ij_sum = i + j
            if ij_sum == in_events:
                break
            it = in_times[ij_sum]
        start = splits[o + 1] = i + j
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
                               buffer_starts: IntArray, buffer_stops: IntArray,
                               buffer_values: IntArray, index_values: IntArray,
                               index_splits: IntArray, stride: int,
                               event_duration: int) -> int:
    out_events = out_times.size
    in_events = in_times.size
    if out_events == 0:
        return 0
    i = -1
    it = -event_duration - 1
    spatial_buffer_size = buffer_values.shape[-1]
    index_splits[:, :, 0] = 0
    max_neighbors = index_values.shape[-1]

    for o in range(out_events):
        ot = out_times[o]
        start_time = ot - event_duration

        # skip events that have already expired
        while it < start_time:
            i += 1
            it = in_times[i]

        # add events to spatial buffer
        while it <= ot:
            # push right
            x, y = in_coords[i]
            buffer_values[y, x, buffer_stops[y, x] % spatial_buffer_size] = i
            buffer_stops[y, x] += 1
            if buffer_starts[y, x] == buffer_stops[y, x]:
                buffer_starts[y, x] += 1
                print('Warning: buffer overrun')
            i += 1
            if i == in_events:
                break
            it = in_times[i]

        ox, oy = out_coords[o] * stride

        # search spatial buffer
        for dy in range(stride):
            y = oy + dy
            for dx in range(stride):
                x = ox + dx
                buff = buffer_values[y, x]

                # trim left
                stop_index = buffer_stops[y, x]
                for k in range(buffer_starts[y, x], stop_index):
                    it = in_times[buff[k % spatial_buffer_size]]
                    if it > start_time:
                        buffer_starts[y, x] = k
                        break

                # process left
                start_index = buffer_starts[y, x]
                num_neigh = stop_index - start_index
                splits = index_splits[dy, dx]
                value_start = splits[o]
                num_neigh = min(max_neighbors - value_start, num_neigh)
                splits[o + 1] = value_start + num_neigh
                offset_values = index_values[dy, dx]
                for j in range(num_neigh):
                    k = (start_index + j) % spatial_buffer_size
                    offset_values[value_start + j] = buff[k]

    return i


@nb.njit()
def compute_neighbors(in_times: IntArray,
                      in_coords: IntArray,
                      out_times: IntArray,
                      out_coords: IntArray,
                      stride: int,
                      event_duration: int,
                      spatial_buffer_size: int,
                      max_neighbors: int,
                      num_out_events: Optional[int] = None
                     ) -> Tuple[IntArray, IntArray, int]:
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
    H = H + H % stride
    W = W + W % stride
    if num_out_events is None:
        num_out_events = out_times.size

    buffer_starts = np.zeros((H, W), dtype=np.int64)
    buffer_stops = np.zeros((H, W), dtype=np.int64)

    buffer_values = np.empty((H, W, spatial_buffer_size), dtype=np.int64)
    index_values = np.empty((stride, stride, max_neighbors), dtype=np.int64)
    index_splits = np.empty((stride, stride, num_out_events + 1),
                            dtype=np.int64)

    i = compute_neighbors_prealloc(
        in_times=in_times,
        in_coords=in_coords,
        out_times=out_times,
        out_coords=out_coords,
        buffer_starts=buffer_starts,
        buffer_stops=buffer_stops,
        buffer_values=buffer_values,
        index_values=index_values,
        index_splits=index_splits,
        stride=stride,
        event_duration=event_duration,
    )

    return index_values, index_splits, i
