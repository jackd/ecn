from typing import Tuple
import tensorflow as tf
from ecn.np_utils import spike as _np_spike
IntTensor = tf.Tensor
FloatTensor = tf.Tensor


def global_spike_threshold(times: IntTensor,
                           in_size: IntTensor,
                           max_out_events: int,
                           decay_time: int,
                           threshold: float = 2.,
                           reset_potential: float = -1.
                          ) -> Tuple[IntTensor, IntTensor]:

    def fn(times, in_size):
        return _np_spike.global_spike_threshold(
            times.numpy()[:in_size.numpy()],
            max_out_events=max_out_events,
            decay_time=decay_time,
            threshold=threshold,
            reset_potential=reset_potential,
        )

    out_times, out_events = tf.py_function(fn, [times, in_size],
                                           [tf.int64, tf.int64])
    out_times.set_shape((max_out_events,))
    out_events.set_shape(())
    return out_times, out_events


def spike_threshold(times: IntTensor,
                    coords: IntTensor,
                    in_size: IntTensor,
                    decay_time: int,
                    max_out_events: int,
                    threshold: float = -1.,
                    reset_potential: float = -1.
                   ) -> Tuple[IntTensor, IntTensor, IntTensor]:

    def fn(times, coords, in_size):
        in_size = in_size.numpy()
        return _np_spike.spike_threshold(times.numpy()[:in_size],
                                         coords.numpy()[:in_size],
                                         decay_time=decay_time,
                                         max_out_events=max_out_events,
                                         threshold=threshold,
                                         reset_potential=reset_potential)

    out_times, out_coords, out_events = tf.py_function(
        fn, [times, coords], [tf.int64, tf.int64, tf.int64])
    out_times.set_shape((max_out_events,))
    out_coords.set_shape((max_out_events, 2))
    out_events.set_shape(())
    return out_times, out_coords, out_events
