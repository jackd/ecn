from typing import Tuple

import numba as nb
import numpy as np

from . import utils

FloatArray = np.ndarray
IntArray = np.ndarray


@nb.njit()
def global_spike_threshold(
    times: IntArray,
    decay_time: int,
    threshold: float = 2.0,
    reset_potential: float = -2.0,
    max_out_events: int = -1,
) -> IntArray:
    """

    Returns:
        out_times: [out_events] int array
    """
    if max_out_events == -1:
        max_out_events = times.size
    out_times = np.empty((max_out_events,), dtype=times.dtype)

    out_events = 0
    potential = 0.0
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


@nb.njit()
def spike_threshold(
    times: IntArray,
    coords: IntArray,
    grid_indices: IntArray,
    grid_splits: IntArray,
    decay_time: int,
    threshold: float = 2.0,
    reset_potential: float = -2.0,
    out_size: int = -1,
) -> Tuple[IntArray, IntArray]:
    """
    Get event outputs.

    Args:
        times: [in_events] ints of event times.
        coords: [in_events] ints coordinates of events.
        grid_indices, grid_splits: ragged components of output coordinates of
            neighbors. Output neighbors in `in_coord` are given by
            `grid_indices[grid_splits[in_coord] : grid_splits[in_coord + 1]]`
        decay_time: rate at which potential decays in units of time.
        threshold: upper threshold at which to fire subsequent events.
        reset_potential: value potential is set to after event fires.
        out_size: output size. If -1, we use `np.max(neigh_indices) + 1`.

    Returns:
        out_times: [out_events] IntArray
        out_coords: [out_events] IntArray
    """
    max_out_events = times.size
    in_events = times.size
    if in_events == 0:
        return np.empty((0,), dtype=times.dtype), np.empty((0,), dtype=coords.dtype)
    if out_size == -1:
        out_size = np.max(grid_indices) + 1

    potentials = np.zeros((out_size,), dtype=np.float32)
    potential_times = np.zeros((out_size,), dtype=times.dtype)
    out_times = np.empty((max_out_events,), dtype=times.dtype)
    out_coords = np.empty((max_out_events,), dtype=coords.dtype)

    out_events = 0
    for i in range(in_events):
        t = times[i]
        c = coords[i]
        lower = grid_splits[c]
        upper = grid_splits[c + 1]
        row = grid_indices[lower:upper]
        size = row.size
        if size == 0:
            continue
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
