raise NotImplementedError("deprecated")
# # import os
# # os.environ['NUMBA_DISABLE_JIT'] = '1'
# from typing import Tuple
# import numpy as np
# import tensorflow as tf
# import ecn.ops.spike as spike_ops
# import ecn.ops.neighbors as neigh_ops

# BoolTensor = tf.Tensor
# IntTensor = tf.Tensor

# # class SpatialConvArgs(NamedTuple):
# #     out_times: IntTensor
# #     out_coords: IntTensor
# #     indices: IntTensor
# #     splits: IntTensor

# # class GlobalConvArgs(NamedTuple):
# #     out_times: IntTensor
# #     indices: IntTensor
# #     splits: IntTensor

# def preprocess_spatial_conv(times: IntTensor,
#                             coords: IntTensor,
#                             stride: int,
#                             decay_time: int,
#                             event_duration: int,
#                             spatial_buffer_size: int,
#                             threshold: float = 2.,
#                             reset_potential: float = -1.):
#     pooled_coords = coords // stride
#     out_times, out_coords = spike_ops.spike_threshold(
#         times=times,
#         coords=pooled_coords,
#         threshold=threshold,
#         reset_potential=reset_potential,
#         decay_time=decay_time)

#     indices, splits = neigh_ops.compute_neighbors(
#         in_times=times,
#         in_coords=pooled_coords,
#         out_times=out_times,
#         out_coords=out_coords,
#         event_duration=event_duration,
#         spatial_buffer_size=spatial_buffer_size)
#     return out_times, out_coords, indices, splits

# def preprocess_global_conv(times: IntTensor,
#                            decay_time: int,
#                            event_duration: int,
#                            threshold: float = 2.,
#                            reset_potential: float = -1.
#                           ) -> Tuple[IntTensor, IntTensor, IntTensor]:
#     out_times = spike_ops.global_spike_threshold(
#         times, decay_time, threshold=threshold, reset_potential=reset_potential)
#     indices, splits = neigh_ops.compute_pooled_neighbors(
#         in_times=times, out_times=out_times, event_duration=event_duration)
#     return out_times, indices, splits

# def preprocess_network(
#         times: IntTensor,
#         coords: IntTensor,
#         stride: int,
#         decay_time: int,
#         event_duration: int,
#         spatial_buffer_size: int,
#         num_layers: int,
#         threshold: float = 2.,
#         reset_potential: float = -1.,
# ):  # -> Tuple[Tuple[SpatialConvArgs, ...], GlobalConvArgs]:
#     layer_args = []
#     for _ in range(num_layers):
#         out_times, out_coords, indices, splits = args = preprocess_spatial_conv(
#             times=times,
#             coords=coords,
#             stride=stride,
#             decay_time=decay_time,
#             event_duration=event_duration,
#             spatial_buffer_size=spatial_buffer_size,
#             threshold=threshold,
#             reset_potential=reset_potential)
#         del indices, splits

#         layer_args.append(args)
#         times = out_times
#         coords = out_coords
#         event_duration *= 2
#         decay_time *= 2

#     global_args = preprocess_global_conv(
#         times=times,
#         decay_time=decay_time,
#         event_duration=event_duration,
#         threshold=threshold,
#         reset_potential=reset_potential,
#     )
#     return layer_args, global_args

# def trim_args(times: IntTensor, coords: IntTensor, polarity: IntTensor,
#               layer_args, global_args):
#     base_times = times
#     base_coords = coords
#     times, indices, splits = global_args
#     unique_indices, ri = tf.unique(indices)
#     indices = tf.gather(ri, indices)
#     global_args = (times, indices, splits)
#     for i in range(len(layer_args) - 1, -1, -1):
#         times, coords, indices, splits = layer_args[i]
#         rt = tf.RaggedTensor.from_row_splits(indices, splits)
#         rt = tf.gather(rt, unique_indices)
#         indices = rt.values
#         splits = rt.row_splits
#         times = tf.gather(times, unique_indices)
#         coords = tf.gather(coords, unique_indices)

#         unique_indices, ri = tf.unique(indices)
#         layer_args[i] = (times, coords, indices, splits)

#     times, coords, polarity = (tf.gather(x, unique_indices)
#                                for x in (base_times, base_coords, polarity))
#     return times, coords, polarity, layer_args, global_args

# def preprocess_network_trimmed(
#         times: IntTensor,
#         coords: IntTensor,
#         polarity: BoolTensor,
#         stride: int,
#         decay_time: int,
#         event_duration: int,
#         spatial_buffer_size: int,
#         num_layers: int,
#         threshold: float = 2.,
#         reset_potential: float = -1.,
# ):
#     layer_args, global_args = preprocess_network(
#         times=times,
#         coords=coords,
#         stride=stride,
#         decay_time=decay_time,
#         event_duration=event_duration,
#         spatial_buffer_size=spatial_buffer_size,
#         num_layers=num_layers,
#         threshold=threshold,
#         reset_potential=reset_potential,
#     )
#     return trim_args(times, coords, polarity, layer_args, global_args)

# if __name__ == '__main__':
#     from events_tfds.events.nmnist import NMNIST
#     ds = NMNIST().as_dataset(split='train', as_supervised=True)
#     for example, label in ds.take(5):
#         coords = example['coords']
#         times = example['time']
#         polarity = example['polarity']
#         preprocess_network_trimmed(times,
#                                    coords,
#                                    polarity=polarity,
#                                    stride=2,
#                                    decay_time=10000,
#                                    event_duration=10000 * 8,
#                                    spatial_buffer_size=32,
#                                    num_layers=3,
#                                    reset_potential=-2.)
#         print('worked!')
