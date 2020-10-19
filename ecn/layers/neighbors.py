import tensorflow as tf

from ecn.layers.utils import as_layer
from ecn.ops import neighbors as _neigh_ops

IntTensor = tf.Tensor
Lambda = tf.keras.layers.Lambda


compute_pooled_neighbors = as_layer(_neigh_ops.compute_pooled_neighbors)
compute_full_neighbors = as_layer(_neigh_ops.compute_full_neighbors)
compute_pointwise_neighbors = as_layer(_neigh_ops.compute_pointwise_neighbors)
compute_neighbors = as_layer(_neigh_ops.compute_neighbors)
# def _compute_pooled_neighbors(args, **kwargs):
#     return _neigh_ops.compute_pooled_neighbors(*args, **kwargs)


# def compute_pooled_neighbors(
#     in_times: IntTensor,
#     out_times: IntTensor,
#     event_duration: int,
#     max_neighbors: int = -1,
# ) -> Tuple[IntTensor, IntTensor]:
#     return Lambda(
#         _compute_pooled_neighbors,
#         arguments=dict(event_duration=event_duration, max_neighbors=max_neighbors,),
#     )([in_times, out_times])


# def _compute_full_neighbors(args, **kwargs):
#     return _neigh_ops.compute_full_neighbors(*args, **kwargs)


# def compute_full_neighbors(
#     in_times: IntTensor,
#     in_coords: IntTensor,
#     out_times: IntTensor,
#     event_duration: int = -1,
#     max_neighbors: int = -1,
# ) -> Tuple[IntTensor, IntTensor, IntTensor]:
#     return Lambda(
#         _compute_full_neighbors,
#         arguments=dict(event_duration=event_duration, max_neighbors=max_neighbors),
#     )([in_times, in_coords, out_times])


# def _compute_pointwise_neighbors(args, **kwargs):
#     return _neigh_ops.compute_pointwise_neighbors(*args, **kwargs)


# def compute_pointwise_neighbors(
#     in_times: IntTensor,
#     in_coords: IntTensor,
#     out_times: IntTensor,
#     out_coords: IntTensor,
#     spatial_buffer_size: int,
#     event_duration: Optional[int] = None,
# ) -> Tuple[IntTensor, IntTensor]:
#     # 1x1 conv
#     return Lambda(
#         _compute_pointwise_neighbors,
#         arguments=dict(
#             spatial_buffer_size=spatial_buffer_size, event_duration=event_duration
#         ),
#     )([in_times, in_coords, out_times, out_coords])


# #
# # @tf.function
# def _compute_neighbors(args, **kwargs):
#     for arg in args:
#         print(arg, tf.keras.backend.is_keras_tensor(arg))
#     for k in sorted(kwargs):
#         print(k, kwargs[k])
#     # assert not tf.executing_eagerly()
#     out = _neigh_ops.compute_neighbors(*args, **kwargs)
#     print("done")
#     return out


# def compute_neighbors(
#     in_times: IntTensor,
#     in_coords: IntTensor,
#     out_times: IntTensor,
#     out_coords: IntTensor,
#     grid_partitions: IntTensor,
#     grid_indices: IntTensor,
#     grid_splits: IntTensor,
#     spatial_buffer_size: int,
#     event_duration: Optional[int] = None,
# ) -> Tuple[IntTensor, IntTensor, IntTensor]:

#     args = (
#         in_times,
#         in_coords,
#         out_times,
#         out_coords,
#         grid_partitions,
#         grid_indices,
#         grid_splits,
#     )
#     for arg in args:
#         print(tf.keras.backend.is_keras_tensor(arg), is_symbolic_tensor(arg))
#     exit()
#     layer = Lambda(
#         _compute_neighbors,
#         arguments=dict(
#             event_duration=event_duration, spatial_buffer_size=spatial_buffer_size,
#         ),
#     )
#     out = layer(
#         (
#             in_times,
#             in_coords,
#             out_times,
#             out_coords,
#             grid_partitions,
#             grid_indices,
#             grid_splits,
#         )
#     )
#     print("got outputs!")
#     return out
