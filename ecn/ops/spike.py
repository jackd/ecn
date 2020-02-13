from typing import Tuple, Optional, Callable
import functools
import tensorflow as tf
from ecn.np_utils import spike as _np_spike
IntTensor = tf.Tensor
FloatTensor = tf.Tensor


def global_spike_threshold(times: IntTensor,
                           decay_time: int,
                           threshold: float = 2.,
                           reset_potential: float = -1.,
                           max_out_events: int = -1) -> IntTensor:
    out_times = tf.numpy_function(
        functools.partial(_np_spike.global_spike_threshold,
                          max_out_events=max_out_events,
                          decay_time=decay_time,
                          threshold=threshold,
                          reset_potential=reset_potential), (times,), tf.int64)
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

    out_times, out_coords = tf.numpy_function(
        functools.partial(
            _np_spike.spike_threshold,
            decay_time=decay_time,
            threshold=threshold,
            reset_potential=reset_potential,
            max_out_events=max_out_events,
        ), (times, coords), (tf.int64, tf.int64))
    out_times.set_shape((None,))
    out_coords.set_shape((None, 2))
    return out_times, out_coords


# @tf.function
# def global_spike_threshold_tf(times: IntTensor,
#                               decay_time: int,
#                               threshold: float = 2.,
#                               reset_potential: float = -1.,
#                               max_out_events: int = -1) -> IntTensor:
#     """
#     Stupidly slow, sigh...

#     Returns:
#         out_times: [out_events] int tensor

#     """
#     if max_out_events == -1:
#         max_out_events = times.shape[-1]
#     out_times = tf.TensorArray(
#         dtype=tf.int64,
#         size=max_out_events,
#         dynamic_size=False,
#         element_shape=(),
#     )
#     reset_potential = tf.convert_to_tensor(reset_potential, tf.float32)
#     decay_time = tf.convert_to_tensor(decay_time, tf.float32)
#     threshold = tf.convert_to_tensor(threshold, tf.float32)

#     def fold_fn(acc, t):
#         out_times, potential, last_t, out_events = acc
#         potential = potential * tf.exp(
#             tf.cast(last_t - t, tf.float32) / decay_time) + 1
#         if potential > threshold:
#             potential = reset_potential
#             out_times = out_times.write(out_events, t)
#             out_events += 1
#         return out_times, potential, t, out_events

#     potential = tf.zeros((), dtype=tf.float32)
#     last_t = tf.zeros((), dtype=times.dtype)
#     out_events = tf.zeros((), dtype=tf.int32)
#     return tf.foldl(fold_fn, times,
#                     (out_times, potential, last_t, out_events))[0].stack()

# out_events = 0
# potential = 0.
# t0 = tf.zeros((), dtype=times.dtype)
# for t1 in times:
#     potential = potential * tf.exp(
#         tf.cast(t0 - t1, tf.float32) / decay_time) + 1.
#     if potential > threshold:
#         potential = reset_potential
#         # fire event
#         out_times = out_times.write(out_events, t1)
#         out_events += 1
#     t0 = t1
# return out_times.stack()

# def spike_threshold_tf(
#         times: IntTensor,
#         coords: IntTensor,
#         decay_time: int,
#         threshold: float = 2.,
#         reset_potential: float = -1.,
#         max_out_events: int = -1,
#         spatial_neighbors_fn: Callable = lambda x: (x,),
# ) -> Tuple[IntTensor, IntTensor]:
#     """
#     Get event outputs.

#     Args:
#         times: [in_events] ints of event times.
#         coords: [in_events, 2] ints coordinates of events.
#         decay_time: rate at which potential decays in units of time.
#         max_out_events: int, maximum number of output events
#         gride_shape: ints. If not provided, uses reversed coords bounds.
#         threshold: upper threshold at which to fire subsequent events.
#         reset_potential: value potential is set to after event fires.
#         stride: stride of convolution.

#     Returns:
#         out_times: [out_events] IntArray
#         out_coords: [out_events] IntArray
#     """
#     if max_out_events == -1:
#         max_out_events = times.shape[0]
#     in_events = times.shape[0]
#     if in_events == 0:
#         return tf.zeros((0,), dtype=tf.int64), tf.zeros((0, 2), dtype=tf.int64)

#     ndims = 2
#     if (coords.shape.ndims != 2 or coords.shape[1] != ndims):
#         raise ValueError('coords must have shape (n, 2)')

#     shape = tf.reduce_max(coords, axis=0) + 1

#     potentials = tf.Variable(tf.zeros(shape, dtype=tf.float32))
#     potential_times = tf.Variable(tf.zeros(shape, dtype=tf.int64))
#     out_times = tf.TensorArray(
#         dtype=tf.int64,
#         size=max_out_events,
#         dynamic_size=True,
#         element_shape=(),
#     )
#     out_coords = tf.TensorArray(dtype=tf.int64,
#                                 size=max_out_events,
#                                 dynamic_size=True,
#                                 element_shape=(ndims,))
#     out_times = tf.zeros((max_out_events,), dtype=tf.int64)
#     out_coords = tf.zeros((max_out_events, ndims), dtype=tf.int64)

#     out_events = 0
#     for i in range(in_events):
#         t = times[i]
#         for cn in spatial_neighbors_fn(coords[i]):
#             last_t = tf.gather(potential_times, cn)
#             p = tf.gather(potentials, cn) * tf.exp(
#                 (last_t - t) / decay_time) + 1
#             if p > threshold:
#                 p = reset_potential
#                 # fire event
#                 out_times.write(out_events, t)
#                 out_coords.write(out_events, cn)
#                 out_events += 1
#             potentials.scatter_nd_assign(tf.expand_dims(cn, axis=0),
#                                          tf.expand_dims(p, axis=0))

#             # update output pixel
#             potentials[cn].assign(p)
#             potential_times[cn].assign(t)

#     return out_times.stack(), out_coords.stack()
