from typing import Tuple
import tensorflow as tf
from ecn.ops import neighbors as _neigh_ops

IntTensor = tf.Tensor
Lambda = tf.keras.layers.Lambda


def _compute_global_neighbors(args, **kwargs):
    return _neigh_ops.compute_global_neighbors(*args, **kwargs)


def compute_global_neighbors(in_times: IntTensor,
                             out_times: IntTensor,
                             event_duration: int,
                             max_neighbors: int = -1
                            ) -> Tuple[IntTensor, IntTensor]:
    return Lambda(_compute_global_neighbors,
                  arguments=dict(
                      event_duration=event_duration,
                      max_neighbors=max_neighbors,
                  ))([in_times, out_times])


def _compute_neighbors(args, **kwargs):
    return _neigh_ops.compute_neighbors(*args, **kwargs)


def compute_neighbors(in_times: IntTensor,
                      in_coords: IntTensor,
                      out_times: IntTensor,
                      out_coords: IntTensor,
                      event_duration: int,
                      spatial_buffer_size: int,
                      max_neighbors: int = -1) -> Tuple[IntTensor, IntTensor]:
    return Lambda(_compute_neighbors,
                  arguments=dict(
                      event_duration=event_duration,
                      spatial_buffer_size=spatial_buffer_size,
                      max_neighbors=max_neighbors,
                  ))([in_times, in_coords, out_times, out_coords])
