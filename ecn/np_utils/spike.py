from typing import Callable, Optional, Tuple

import numpy as np
import numba as nb

import ecn.np_utils.grid as grid
import ecn.np_utils.utils as utils

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


@nb.njit()
def spike_threshold(
        times: IntArray,
        coords: IntArray,
        decay_time: int,
        threshold: float = 2.,
        reset_potential: float = -1.,
        max_out_events: int = -1,
        spatial_neighbors_fn: Optional[Callable] = grid.identity_neighbors,
) -> Tuple[IntArray, IntArray]:
    """
    Get event outputs.

    Args:
        times: [in_events] ints of event times.
        coords: [in_events, 2] ints coordinates of events.
        decay_time: rate at which potential decays in units of time.
        max_out_events: int, maximum number of output events
        gride_shape: ints. If not provided, uses reversed coords bounds.
        threshold: upper threshold at which to fire subsequent events.
        reset_potential: value potential is set to after event fires.
        stride: stride of convolution.

    Returns:
        out_times: [out_events] IntArray
        out_coords: [out_events] IntArray
    """
    if max_out_events == -1:
        max_out_events = times.size
    in_events = times.size
    if in_events == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.int64)

    ndim = 2
    if (coords.ndim != 2 or coords.shape[1] != 2):
        raise ValueError('coords must have shape (n, 2)')
    X = np.max(coords[:, 0]) + 1
    Y = np.max(coords[:, 1]) + 1

    potentials = np.zeros((X, Y), dtype=np.float32)
    potential_times = np.zeros((X, Y), dtype=np.int64)
    out_times = np.empty((max_out_events,), dtype=np.int64)
    out_coords = np.empty((max_out_events, ndim), dtype=np.int64)

    out_events = 0
    for i in range(in_events):
        t = times[i]
        for cn_ in spatial_neighbors_fn(coords[i]):
            x, y = cn_
            cn = (x, y)
            out_t = potential_times[cn]
            p = potentials[cn] * np.exp((out_t - t) / decay_time) + 1
            if p > threshold:
                p = reset_potential
                # fire event
                out_times[out_events] = t
                out_coords[out_events] = cn
                out_events += 1
                if out_events == max_out_events:
                    # update output pixel
                    out_times = utils.double_length(out_times)
                    out_coords = utils.double_length(out_coords)
                    max_out_events *= 2

            # update output pixel
            potentials[cn] = p
            potential_times[cn] = t

    return out_times[:out_events], out_coords[:out_events]
