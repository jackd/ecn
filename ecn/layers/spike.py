from typing import Tuple
import tensorflow as tf
from ecn.ops import spike as _spike_ops

Lambda = tf.keras.layers.Lambda
IntTensor = tf.Tensor


def global_spike_threshold(times: IntTensor,
                           decay_time: int,
                           threshold: float = 2.,
                           reset_potential: float = -1.,
                           max_out_events: int = -1) -> IntTensor:
    return Lambda(_spike_ops.global_spike_threshold,
                  arguments=dict(
                      decay_time=decay_time,
                      threshold=threshold,
                      reset_potential=reset_potential,
                      max_out_events=max_out_events,
                  ))(times)


def _spike_threshold(args, **kwargs):
    return _spike_ops.spike_threshold(*args, **kwargs)


def spike_threshold(times: IntTensor,
                    coords: IntTensor,
                    grid_indices: IntTensor,
                    grid_splits: IntTensor,
                    decay_time: int,
                    threshold: float = -1.,
                    reset_potential: float = -1.,
                    out_size: int = -1) -> Tuple[IntTensor, IntTensor]:
    return Lambda(_spike_threshold,
                  arguments=dict(
                      decay_time=decay_time,
                      threshold=threshold,
                      reset_potential=reset_potential,
                      out_size=out_size,
                  ))([times, coords, grid_indices, grid_splits])
