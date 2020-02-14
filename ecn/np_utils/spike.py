from typing import Callable, Tuple, Iterable

import numpy as np
import numba as nb
from . import utils
from . import grid
from . import ragged

FloatArray = np.ndarray
IntArray = np.ndarray


@nb.njit()
def global_spike_threshold(times: IntArray,
                           decay_time: int,
                           threshold: float = 2.,
                           reset_potential: float = -1.,
                           max_out_events: int = -1) -> IntArray:
    """

    Returns:
        out_times: [out_events] int array
    """
    if max_out_events == -1:
        max_out_events = times.size
    out_times = np.empty((max_out_events,), dtype=np.int64)

    out_events = 0
    potential = 0.
    t0 = 0
    for t1 in times:
        potential *= np.exp((t0 - t1) / decay_time)
        potential += 1
        if potential > threshold:
            potential = reset_potential
            # fire event
            out_times[out_events] = t1
            out_events += 1
            if out_events == max_out_events:
                out_times = utils.double_length(out_times)
                max_out_events *= 2
        t0 = t1
    return out_times[:out_events]


# @nb.njit(inline='always')
# def identity(x):
#     return ((x, 1.0),)

# @nb.njit()
# def spike_threshold(times: IntArray,
#                     coords: IntArray,
#                     stride: int,
#                     kernel_size: int,
#                     decay_time: int,
#                     threshold: float = 2.,
#                     padding: str = 'valid',
#                     reset_potential: float = 2.):
#     shape = utils.max_on_axis(coords, axis=0) + 1
#     fn = get_neighbors_fn(shape, kernel_size, stride, padding)

# @nb.njit()
# def spike_threshold_in_place(times: IntArray,
#                              coords: IntArray,
#                              decay_time: int,
#                              threshold: float = 2.,
#                              reset_potential: float = -2.):
#     """Optimized nd version for when strides and kernel sizes are all 1."""
#     dims = utils.max_on_axis(coords, axis=0) + 1
#     grid_size = utils.prod(dims)
#     coords = grid.ravel_multi_index_transpose(coords, dims)
#     times, coords = spike_threshold_1d_in_place(times, coords, decay_time,
#                                                 threshold, reset_potential,
#                                                 grid_size)
#     coords = grid.unravel_index_transpose(coords, dims, coords.dtype)
#     return times, coords

# @nb.njit()
# def spike_threshold_1d_in_place(times: IntArray,
#                                 coords: IntArray,
#                                 decay_time: int,
#                                 threshold: float = 2.,
#                                 reset_potential: float = -2.,
#                                 grid_size=-1):
#     """Optimized 1D version for when stride and kernel_size are both 1."""
#     max_out_events = times.size
#     in_events = times.size
#     if in_events == 0:
#         return np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.int64)
#     if grid_size == -1:
#         grid_size = np.max(coords) + 1
#     potentials = np.zeros((grid_size,), dtype=np.float32)
#     potential_times = np.zeros((grid_size,), dtype=np.int64)
#     out_times = np.empty((max_out_events,), dtype=np.int64)
#     out_coords = np.empty((max_out_events,), dtype=np.int64)

#     out_events = 0
#     for i in range(in_events):
#         t = times[i]
#         coord = coords[i]
#         out_t = potential_times[coord]
#         p = potentials[coord] * np.exp((out_t - t) / decay_time) + 1
#         if p > threshold:
#             p = reset_potential
#             # fire event
#             out_times[out_events] = t
#             out_coords[out_events] = coord
#             out_events += 1
#             if out_events == max_out_events:
#                 # update output pixel
#                 out_times = utils.double_length(out_times)
#                 out_coords = utils.double_length(out_coords)
#                 max_out_events *= 2

#         # update output pixel
#         potentials[coord] = p
#         potential_times[coord] = t

#     return out_times[:out_events], out_coords[:out_events]


@nb.njit()
def spike_threshold_1d(
        times: IntArray,
        coords: IntArray,
        neigh_indices: IntArray,
        neigh_splits: IntArray,
        decay_time: int,
        threshold: float = 2.,
        reset_potential: float = -2.,
        out_size: int = -1,
) -> Tuple[IntArray, IntArray]:
    """
    Get event outputs.

    Args:
        times: [in_events] ints of event times.
        coords: [in_events] ints coordinates of events.
        neigh_indices: sparse indices of neighbors of coords into out_coords.
        neigh_splits: row_splits corresponding to neigh_indices.
        decay_time: rate at which potential decays in units of time.
        gride_shape: ints. If not provided, uses reversed coords bounds.
        threshold: upper threshold at which to fire subsequent events.
        reset_potential: value potential is set to after event fires.
        out_size: output size. If -1, we use `np.max(neigh_indices) + 1`.

    Returns:
        out_times: [out_events] IntArray
        out_coords: [out_events] IntArray
    """
    if out_size == -1:
        out_size = np.max(neigh_indices) + 1
    max_out_events = times.size
    in_events = times.size
    # if in_events == 0:
    #     return np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.int64)

    potentials = np.zeros((out_size,), dtype=np.float32)
    potential_times = np.zeros((out_size,), dtype=np.int64)
    out_times = np.empty((max_out_events,), dtype=np.int64)
    out_coords = np.empty((max_out_events,), dtype=np.int64)

    out_events = 0
    for i in range(in_events):
        t = times[i]
        c = coords[i]
        row = neigh_indices[neigh_splits[c]:neigh_splits[c + 1]]
        weight = 1 / row.size
        for coord in row:
            out_t = potential_times[coord]
            p = potentials[coord] * np.exp((out_t - t) / decay_time) + weight
            if p > threshold:
                p = reset_potential
                # fire event
                out_times[out_events] = t
                out_coords[out_events] = coord
                out_events += 1
                if out_events == max_out_events:
                    # update output pixel
                    out_times = utils.double_length(out_times)
                    out_coords = utils.double_length(out_coords)
                    max_out_events *= 2

            # update output pixel
            potentials[coord] = p
            potential_times[coord] = t

    return out_times[:out_events], out_coords[:out_events]
