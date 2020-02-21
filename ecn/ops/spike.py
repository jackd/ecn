from typing import Tuple, Optional, Callable
import functools
import tensorflow as tf
from ecn.np_utils import spike as _np_spike
IntTensor = tf.Tensor
FloatTensor = tf.Tensor


def global_spike_threshold(times: IntTensor,
                           decay_time: int,
                           threshold: float = 1.,
                           reset_potential: float = -1.,
                           max_out_events: int = -1) -> IntTensor:
    out_times = tf.numpy_function(
        functools.partial(_np_spike.global_spike_threshold,
                          max_out_events=max_out_events,
                          decay_time=decay_time,
                          threshold=threshold,
                          reset_potential=reset_potential), (times,),
        times.dtype)
    out_times.set_shape((None,))
    return out_times


def spike_threshold(
        times: IntTensor,
        coords: IntTensor,
        grid_indices: IntTensor,
        grid_splits: IntTensor,
        decay_time: int,
        threshold: float = 1.,
        reset_potential: float = -1.,
        out_size: int = -1,
) -> Tuple[IntTensor, IntTensor]:

    out_times, out_coords = tf.numpy_function(
        functools.partial(
            _np_spike.spike_threshold,
            decay_time=decay_time,
            threshold=threshold,
            reset_potential=reset_potential,
            out_size=out_size,
        ), (times, coords, grid_indices, grid_splits),
        (times.dtype, coords.dtype))
    out_times.set_shape((None,))
    out_coords.set_shape((None,))
    return out_times, out_coords
