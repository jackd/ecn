from ecn.layers.utils import as_layer
from ecn.ops import lif as _lif_ops

leaky_integrate_and_fire = as_layer(_lif_ops.leaky_integrate_and_fire)
spatial_leaky_integrate_and_fire = as_layer(_lif_ops.spatial_leaky_integrate_and_fire)

# Lambda = tf.keras.layers.Lambda
# IntTensor = tf.Tensor


# def leaky_integrate_and_fire(
#     times: IntTensor,
#     decay_time: int,
#     threshold: float = 2.0,
#     reset_potential: float = -1.0,
#     max_out_events: int = -1,
# ) -> IntTensor:
#     return Lambda(
#         _lif_ops.leaky_integrate_and_fire,
#         arguments=dict(
#             decay_time=decay_time,
#             threshold=threshold,
#             reset_potential=reset_potential,
#             max_out_events=max_out_events,
#         ),
#     )(times)


# def _spatial_leaky_integrate_and_fire(args, **kwargs):
#     return _lif_ops.spatial_leaky_integrate_and_fire(*args, **kwargs)


# def spatial_leaky_integrate_and_fire(
#     times: IntTensor,
#     coords: IntTensor,
#     grid_indices: IntTensor,
#     grid_splits: IntTensor,
#     decay_time: int,
#     threshold: float = -1.0,
#     reset_potential: float = -1.0,
#     out_size: int = -1,
# ) -> Tuple[IntTensor, IntTensor]:
#     assert isinstance(out_size, int)
#     assert isinstance(decay_time, int)
#     assert isinstance(threshold, (int, float))
#     assert isinstance(reset_potential, (int, float))
#     return Lambda(
#         _spatial_leaky_integrate_and_fire,
#         arguments=dict(
#             decay_time=decay_time,
#             threshold=threshold,
#             reset_potential=reset_potential,
#             out_size=out_size,
#         ),
#     )([times, coords, grid_indices, grid_splits])
