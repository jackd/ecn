from typing import Optional, Union, Tuple

import numpy as np
import numba as nb

FloatArray = np.ndarray
IntArray = np.ndarray


@nb.njit()
def global_spike_threshold_prealloc(times: IntArray,
                                    out_times: IntArray,
                                    decay_time: int,
                                    threshold: float = 2.,
                                    reset_potential: float = -1.) -> int:
    """
    Returns:
        out_events: int, number of output events
    """
    out_events = 0
    max_out_events = out_times.size
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
                break
        t0 = t1
    return out_events


@nb.njit()
def global_spike_threshold(
        times: IntArray,
        decay_time: int,
        threshold: float = 2.,
        reset_potential: float = -1.,
        max_out_events: Optional[int] = None,
):
    """

    Returns:
        out_times: [max_out_events] int array
        out_events: int
    """
    if max_out_events is None:
        max_out_events = times.size
    out_times = np.empty((max_out_events,), dtype=np.int64)
    out_events = global_spike_threshold_prealloc(
        times=times,
        out_times=out_times,
        decay_time=decay_time,
        threshold=threshold,
        reset_potential=reset_potential,
    )
    return out_times, out_events


@nb.njit()
def spike_threshold_prealloc(
        times: IntArray,
        coords: IntArray,
        # prealloc
        potentials: FloatArray,
        potential_times: IntArray,
        # output prealloc
        out_times: IntArray,
        out_coords: IntArray,
        # config
        decay_time: int,
        threshold: float = 2.,
        reset_potential: float = -1.) -> int:
    """
    Get event outputs.

    Args:
        times: [in_events] ints of event times.
        coords: [in_events, 2] ints of pixel coordinates of each event.
        potentials:
        potential_times:
        decay_time: rate at which potential decays.
        threshold: upper threshold at which to fire subsequent events.
        reset_potential: value potential is set to after event fires.

    Returns:
        out_events: int, number of output events
    """
    in_events = times.size
    out_events = 0
    max_out_events = out_times.size
    for i in range(in_events):
        x, y = c = coords[i]
        t = times[i]
        out_t = potential_times[y, x]
        p = potentials[y, x] * np.exp((out_t - t) / decay_time) + 1
        if p > threshold:
            p = reset_potential
            # fire event
            out_times[out_events] = t
            out_coords[out_events] = c
            out_events += 1
            if out_events == max_out_events:
                # update output pixel
                potentials[y, x] = p
                potential_times[y, x] = t
                break

        # update output pixel
        potentials[y, x] = p
        potential_times[y, x] = t
    return out_events


@nb.njit()
def spike_threshold(
        times: IntArray,
        coords: IntArray,
        decay_time: int,
        threshold: float = 2.,
        reset_potential: float = -1.,
        shape: Optional[Union[IntArray, Tuple[int, int]]] = None,
        max_out_events: Optional[int] = None,
) -> Tuple[IntArray, IntArray, int]:
    """
    Get event outputs.

    Args:
        times: [in_events] ints of event times.
        coords: [in_events, 2] ints of pixel coordinates of each event.
        decay_time: rate at which potential decays.
        max_out_events: int, maximum number of output events
        shape: (H, W), ints. If not provided, uses reversed coords bounds.
        threshold: upper threshold at which to fire subsequent events.
        reset_potential: value potential is set to after event fires.
        stride: stride of convolution.

    Returns:
        out_times: [max_out_events] IntArray
        out_coords: [max_out_events] IntArray
        out_events: int, number of output events
    """
    if shape is None:
        # max_x, max_y = np.max(coords, axis=0)
        max_x = np.max(coords[:, 0]) + 1
        max_y = np.max(coords[:, 1]) + 1
    else:
        max_y, max_x = shape
    if max_out_events is None:
        max_out_events = times.size

    potentials = np.zeros((max_y, max_x), dtype=np.float32)
    potential_times = np.zeros((max_y, max_x), dtype=np.int64)
    out_times = np.empty((max_out_events,), dtype=np.int64)
    out_coords = np.empty((max_out_events, 2), dtype=np.int64)

    out_events = spike_threshold_prealloc(
        times=times,
        coords=coords,
        potentials=potentials,
        potential_times=potential_times,
        out_times=out_times,
        out_coords=out_coords,
        decay_time=decay_time,
        threshold=threshold,
        reset_potential=reset_potential,
    )

    return out_times, out_coords, out_events
