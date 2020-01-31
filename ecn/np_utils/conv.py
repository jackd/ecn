from typing import Tuple, Optional

import numpy as np
import numba as nb

BoolArray = np.ndarray
FloatArray = np.ndarray
IntArray = np.ndarray


@nb.njit()
def unlearned_polarity_event_conv_prealloc(
        polarity: BoolArray, in_times: IntArray, in_coords: IntArray,
        out_times: IntArray, out_coords: IntArray, decayed_polarity: FloatArray,
        decayed_time: IntArray, event_polarities: FloatArray, decay_time: int,
        stride: int) -> int:
    num_in_events = in_times.size
    if num_in_events == 0:
        return 0
    num_out_events = out_times.size
    i = 0
    it = in_times[0]
    pol = int(polarity[i])

    it = in_times[0]

    for o in range(num_out_events):
        ot = out_times[o]
        while it <= ot:
            x, y = in_coords[i]
            decayed_polarity[y, x, pol] = decayed_polarity[y, x, pol] * np.exp(
                (decayed_time[y, x, pol] - it) / decay_time) + 1
            decayed_time[y, x, pol] = it
            i += 1
            it = in_times[i]
            pol = int(polarity[i])

        ox, oy = out_coords[o]

        for dy in range(stride):
            y = oy * stride + dy
            for dx in range(stride):
                x = ox * stride + dx
                for pol in (0, 1):
                    decayed_polarity[
                        y, x, pol] = decayed_polarity[y, x, pol] * np.exp(
                            (decayed_time[y, x, pol] - ot) / decay_time)
                    decayed_time[y, x, pol] = ot
                event_polarities[o, dy, dx] = decayed_polarity[y, x]
    return i


@nb.njit()
def unlearned_polarity_event_conv(
        polarity: BoolArray,
        in_times: IntArray,
        in_coords: IntArray,
        out_times: IntArray,
        out_coords: IntArray,
        decay_time: int,
        stride: int,
        shape: Optional[Tuple[int, int]] = None,
        out_length: Optional[int] = None,
) -> Tuple[FloatArray, int]:
    if shape is None:
        W = np.max(in_coords[:, 0]) + 1
        H = np.max(in_coords[:, 1]) + 1
    else:
        H, W = shape
    decayed_polarity = np.zeros((H, W, 2), dtype=np.float32)
    decayed_time = np.zeros((H, W, 2), dtype=np.int64)
    if out_length is None:
        out_length = out_times.size
    event_polarities = np.empty((out_length, stride, stride, 2),
                                dtype=np.float32)
    i = unlearned_polarity_event_conv_prealloc(polarity, in_times, in_coords,
                                               out_times, out_coords,
                                               decayed_polarity, decayed_time,
                                               event_polarities, decay_time,
                                               stride)
    return event_polarities, i
