from typing import Tuple
import tensorflow as tf
from ecn.np_utils import spike as _np_spike
IntTensor = tf.Tensor
FloatTensor = tf.Tensor


def global_spike_threshold(times: IntTensor,
                           decay_time: int,
                           threshold: float = 2.,
                           reset_potential: float = -1.,
                           max_out_events: int = -1) -> IntTensor:

    def fn(times):
        return _np_spike.global_spike_threshold(times.numpy(),
                                                max_out_events=max_out_events,
                                                decay_time=decay_time,
                                                threshold=threshold,
                                                reset_potential=reset_potential)

    out_times = tf.py_function(fn, (times,), tf.int64)
    out_times.set_shape((None,))
    return out_times


def spike_threshold(
        times: IntTensor,
        coords: IntTensor,
        decay_time: int,
        threshold: float = -1.,
        reset_potential: float = -1.,
        max_out_events: int = -1,
) -> Tuple[IntTensor, IntTensor]:

    def fn(times, coords):
        return _np_spike.spike_threshold(times.numpy(),
                                         coords.numpy(),
                                         decay_time=decay_time,
                                         threshold=threshold,
                                         reset_potential=reset_potential,
                                         max_out_events=max_out_events)

    out_times, out_coords = tf.py_function(fn, (times, coords),
                                           (tf.int64, tf.int64))
    out_times.set_shape((None,))
    out_coords.set_shape((None, 2))
    return out_times, out_coords
