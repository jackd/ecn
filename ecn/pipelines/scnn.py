raise NotImplementedError('deprecated')
# import functools
# import gin
# import tensorflow as tf
# from kblocks.framework.problems.pipelines import CustomPipeline
# from kblocks.ops import ragged as ragged_ops
# from kblocks.ops import sparse as sparse_ops
# from kblocks.ops import repeat as tf_repeat
# # import ecn.ops.spike as spike_ops
# # import ecn.ops.neighbors as neigh_ops
# import ecn.ops.grid as grid_ops
# import ecn.ops.preprocess as pp

# def ijdt_to_sparse(ijdt, dense_shape=(-1, -1)):
#     ij, dt = tf.split(ijdt, [2, 1], axis=-1)
#     dt = tf.squeeze(dt, axis=-1)
#     return tf.SparseTensor(ij, dt, dense_shape)

# def maybe_pad(values, diff):
#     return tf.cond(diff > 0, lambda: tf.pad(values, [[0, diff]]),
#                    lambda: values)

# def pre_map(events,
#             labels,
#             weights=None,
#             stride=2,
#             num_layers=3,
#             decay_time=10000,
#             event_duration=None,
#             threshold=2.,
#             reset_potential=-2.,
#             spatial_buffer_size=32):
#     times = events.pop('time')
#     times = times - tf.reduce_min(times)
#     coords = events.pop('coords')
#     coords = coords - tf.reduce_min(coords, axis=0)
#     polarity = events.pop('polarity')
#     out = {}
#     del events
#     ndims = coords.shape.ndims
#     sk = stride**ndims

#     if event_duration is None:
#         event_duration = decay_time * 4  # ~ 2% potential remaining by this point

#     (times, coords, polarity, spatial_args,
#      global_args) = pp.preprocess_network_trimmed(
#          times,
#          coords,
#          polarity,
#          stride=stride,
#          decay_time=decay_time,
#          event_duration=event_duration,
#          spatial_buffer_size=spatial_buffer_size,
#          num_layers=num_layers,
#          threshold=threshold,
#          reset_potential=reset_potential,
#      )

#     out['polarity'] = ragged_ops.pre_batch_ragged(polarity)

#     spatial_neighs = []
#     all_sizes = [tf.size(times, out_type=tf.int64)]

#     for layer in range(num_layers):
#         out_times, out_coords, neigh = spatial_args[layer]
#         all_sizes.append(tf.size(out_times, out_type=tf.int64))
#         i = neigh.value_rowids()
#         j = neigh.values
#         partitions = grid_ops.partition_strided_neighbors(
#             coords, out_coords, j, i, stride)
#         dt = tf.gather(out_times, i) - tf.gather(times, j)
#         dt = tf.cast(dt, tf.float32) / decay_time
#         indices = tf.stack((partitions, i, j), axis=-1)
#         neighs = sparse_ops.unstack(tf.SparseTensor(indices, dt, (sk, -1, -1)),
#                                     axis=0,
#                                     num_partitions=sk)
#         spatial_neighs.append(tuple(neighs))

#         times = out_times
#         coords = out_coords
#         decay_time *= 2

#     out['spatial_neighs'] = tuple(spatial_neighs)

#     # global
#     global_times, neigh = global_args
#     all_sizes.append(tf.size(global_times, out_type=tf.int64))
#     i = neigh.value_rowids()
#     j = neigh.values
#     dt = tf.gather(times, j) - tf.gather(global_times, i)
#     dt = tf.cast(dt, tf.float32) / decay_time
#     out['global_neigh'] = tf.SparseTensor(tf.stack((i, j), axis=-1), dt,
#                                           (-1, -1))

#     out['lengths'] = tf.stack(all_sizes)
#     return out, labels

# def post_batch_map(inputs, labels, weights=None, final_only=False):

#     lengths = inputs.pop('lengths')
#     polarity = inputs.pop('polarity')
#     out = {}
#     row_ends = tf.math.cumsum(lengths, axis=0)
#     row_splits = tf.pad(row_ends, [[1, 0], [0, 0]])
#     row_starts, totals = tf.split(row_splits, [-1, 1], axis=0)
#     totals = tf.squeeze(totals, axis=0)
#     row_starts = tf.unstack(row_starts, axis=-1)
#     totals = tf.unstack(totals)

#     batch_size = polarity.nrows()
#     static_batch_size = tf.get_static_value(batch_size)
#     if static_batch_size is not None:
#         batch_size = static_batch_size
#     padded_size = 4096 * batch_size
#     # fix plarity
#     pv = polarity.flat_values
#     rs = row_starts[0]
#     pv = maybe_pad(pv, padded_size - totals[0])
#     pv = tf.one_hot(tf.cast(pv, tf.int64),
#                     depth=2,
#                     dtype=tf.float32,
#                     name='one_hot_polarity')

#     out['features'] = tf.RaggedTensor.from_row_starts(pv, rs)
#     # fix totals
#     totals[0] = tf.maximum(totals[0], padded_size)
#     for i in range(len(totals) - 1):
#         padded_size = padded_size // 4
#         totals[i + 1] = tf.maximum(totals[i + 1], padded_size)

#     all_neighs = inputs['spatial_neighs']
#     all_out_neighs = []

#     for n, neighs in enumerate(all_neighs):
#         dense_shape = (batch_size, totals[n + 1], totals[n])
#         out_neighs = []
#         for neigh in neighs:

#             b, i, j = tf.unstack(neigh.indices, axis=-1)
#             i = sparse_ops.apply_offset(b, i, row_starts[n + 1])
#             j = sparse_ops.apply_offset(b, j, row_starts[n])
#             out_neighs.append(
#                 tf.SparseTensor(
#                     tf.stack((b, i, j), axis=-1),
#                     neigh.values,
#                     dense_shape,
#                 ))
#         all_out_neighs.append(tuple(out_neighs))

#     # if sk is not None:
#     #     for neigh in out_neighs:
#     #         shape = neigh.shape.as_list()
#     #         shape[1] = sk
#     #         neigh.set_shape(shape)

#     out['spatial_neighs'] = tuple(all_out_neighs)

#     # global
#     neigh = inputs['global_neigh']
#     b, i, j = tf.unstack(neigh.indices, axis=-1)
#     i = sparse_ops.apply_offset(b, i, row_starts[-1])
#     j = sparse_ops.apply_offset(b, j, row_starts[-2])
#     out['global_neigh'] = tf.SparseTensor(tf.stack(
#         (b, i, j), axis=-1), neigh.values, (batch_size, totals[-1], totals[-2]))
#     out['final_indices'] = row_ends[:, -1] - 1

#     out_labels = {}
#     out_labels['final'] = labels

#     out_weights = {}
#     if weights is not None:
#         out_weights['final'] = weights

#     repeats = lengths[:, -1]
#     if weights is None:
#         weights = 1. / tf.cast(batch_size, tf.float32) / tf.cast(
#             repeats, dtype=tf.float32)
#     else:
#         weights = weights / tf.cast(batch_size * repeats, tf.float32)
#     weights = tf_repeat(weights, repeats, axis=0)
#     labels = tf_repeat(labels, repeats, axis=0)
#     diff = padded_size - tf.size(labels, out_type=totals[-1].dtype)
#     out_labels['stream'] = maybe_pad(labels, diff)
#     out_weights['stream'] = maybe_pad(weights, diff)

#     return out, out_labels, out_weights

# @gin.configurable(module='ecn.pipelines')
# def scnn_pipeline(
#         batch_size: int,
#         stride=2,
#         num_layers=3,
#         use_cache=True,
#         final_only=False,
#         #   ndims=2,
#         **kwargs):

#     pre_map_fn = functools.partial(pre_map,
#                                    stride=stride,
#                                    num_layers=num_layers)

#     if use_cache:
#         pre_batch_map = None
#         pre_cache_map = pre_map_fn
#     else:
#         pre_batch_map = pre_map_fn
#         pre_cache_map = None

#     post_batch_map_fn = functools.partial(post_batch_map, final_only=final_only)
#     return CustomPipeline(batch_size, pre_cache_map, pre_batch_map,
#                           post_batch_map_fn, **kwargs)

# # def scnn_pipeline_backup(batch_size: int, num_layers=3, **kwargs):
# #     stride = 2
# #     ndims = 2

# #     def pre_map(events, labels, weights=None):
# #         times = events.pop('time')
# #         coords = events.pop('coords')
# #         coords = coords - tf.reduce_min(coords, axis=0)
# #         polarity = events['polarity']
# #         out = dict(
# #             coords=ragged_ops.pre_batch_ragged(coords),
# #             polarity=ragged_ops.pre_batch_ragged(polarity),
# #         )
# #         del events
# #         assert (coords.shape[1] == ndims)

# #         decay_time = 10000
# #         # event_duration = 10000
# #         # event_duration = np.iinfo(np.int64).max - 2  # effectively inf
# #         event_duration = decay_time * 8
# #         spatial_buffer_size = 32
# #         threshold = 2.
# #         reset_potential = -1.

# #         spatial_neighs = []
# #         all_sizes = [tf.size(times, out_type=tf.int64)]

# #         for _ in range(num_layers):
# #             out_times, out_coords = spike_ops.spike_threshold(
# #                 times=times,
# #                 coords=coords // stride,
# #                 decay_time=decay_time,
# #                 threshold=threshold,
# #                 reset_potential=reset_potential)
# #             all_sizes.append(tf.size(out_times, out_type=tf.int64))
# #             indices, splits = neigh_ops.compute_neighbors(
# #                 in_times=times,
# #                 in_coords=coords // stride,
# #                 out_times=out_times,
# #                 out_coords=out_coords,
# #                 event_duration=event_duration,
# #                 spatial_buffer_size=spatial_buffer_size)
# #             neigh = tf.RaggedTensor.from_row_splits(indices, splits)
# #             i = neigh.value_rowids()
# #             j = indices
# #             partitions = grid_ops.partition_strided_neighbors(
# #                 coords, out_coords, j, i, stride)
# #             dt = tf.gather(out_times, i) - tf.gather(times, j)
# #             dt = tf.cast(dt, tf.float32) / decay_time
# #             indices = tf.stack((partitions, i, j), axis=-1)

# #             spatial_neighs.append(
# #                 tf.SparseTensor(indices, dt, (stride**ndims, -1, -1)))

# #             times = out_times
# #             coords = out_coords
# #             decay_time *= 2
# #             event_duration *= 2

# #         out['spatial_neighs'] = tuple(spatial_neighs)

# #         # global
# #         global_times = spike_ops.global_spike_threshold(
# #             times,
# #             decay_time,
# #             threshold=threshold,
# #             reset_potential=reset_potential)
# #         all_sizes.append(tf.size(global_times, out_type=tf.int64))

# #         global_indices, global_splits = neigh_ops.compute_pooled_neighbors(
# #             in_times=times,
# #             out_times=global_times,
# #             event_duration=event_duration)
# #         neigh = tf.RaggedTensor.from_row_splits(global_indices, global_splits)
# #         i = neigh.value_rowids()
# #         j = global_indices
# #         dt = tf.gather(times, j) - tf.gather(global_times, i)
# #         dt = tf.cast(dt, tf.float32) / decay_time
# #         out['global_neigh'] = tf.SparseTensor(tf.stack((i, j), axis=-1), dt,
# #                                               (-1, -1))

# #         out['lengths'] = tf.stack(all_sizes)
# #         return out, labels

# #     def post_batch_map(inputs, labels, weights=None):

# #         lengths = inputs.pop('lengths')
# #         polarity = inputs.pop('polarity')
# #         out = {}

# #         row_splits = tf.pad(tf.math.cumsum(lengths, axis=0), [[1, 0], [0, 0]])
# #         row_starts, totals = tf.split(row_splits, [-1, 1], axis=0)
# #         totals = tf.squeeze(totals, axis=0)
# #         row_starts = tf.unstack(row_starts, axis=-1)
# #         totals = tf.unstack(totals)

# #         # batch_size = polarity.nrows()
# #         padded_size = 4096 * batch_size
# #         # fix plarity
# #         pv = polarity.flat_values
# #         rs = row_starts[0]
# #         pv = maybe_pad(pv, padded_size - totals[0])
# #         pv = tf.one_hot(tf.cast(pv, tf.int64),
# #                         depth=2,
# #                         dtype=tf.float32,
# #                         name='one_hot_polarity')

# #         out['features'] = tf.RaggedTensor.from_row_starts(pv, rs)
# #         out_neighs = []
# #         # fix totals
# #         totals[0] = tf.maximum(totals[0], padded_size)
# #         for i in range(len(totals) - 1):
# #             padded_size = padded_size // 4
# #             totals[i + 1] = tf.maximum(totals[i + 1], padded_size)

# #         neighs = inputs['spatial_neighs']
# #         for n, neigh in enumerate(neighs):
# #             dense_shape = (batch_size, stride**ndims, totals[n], totals[n + 1])
# #             b, p, i, j = tf.unstack(neigh.indices, axis=-1)
# #             i = sparse_ops.apply_offset(b, i, row_starts[n + 1])
# #             j = sparse_ops.apply_offset(b, j, row_starts[n])
# #             out_neighs.append(
# #                 tf.SparseTensor(
# #                     tf.stack((b, p, i, j), axis=-1),
# #                     neigh.values,
# #                     dense_shape,
# #                 ))

# #         out['spatial_neighs'] = tuple(out_neighs)

# #         # global
# #         neigh = inputs['global_neigh']
# #         b, i, j = tf.unstack(neigh.indices, axis=-1)
# #         i = sparse_ops.apply_offset(b, i, row_starts[-1])
# #         j = sparse_ops.apply_offset(b, j, row_starts[-2])
# #         out['global_neigh'] = tf.SparseTensor(tf.stack(
# #             (b, i, j),
# #             axis=-1), neigh.values, (batch_size, totals[-1], totals[-2]))

# #         labels = tf.gather(labels, b)
# #         if weights is None:
# #             weights = 1.
# #         weights = tf.gather(weights / tf.cast(lengths[:, -1], tf.float32), b)
# #         diff = totals[-1] - tf.size(b, out_type=totals[-1].dtype)
# #         b = maybe_pad(b, diff)
# #         weights = maybe_pad(weights, diff)
# #         return out, labels, weights

# #     pre_batch_map = None
# #     pre_cache_map = pre_map
# #     # pre_batch_map = pre_map
# #     # pre_cache_map = None
# #     return CustomPipeline(batch_size, pre_cache_map, pre_batch_map,
# #                           post_batch_map, **kwargs)

# if __name__ == '__main__':
#     from ecn.problems.nmnist import nmnist
#     batch_size = 2
#     pipeline = scnn_pipeline(batch_size)
#     problem = nmnist(pipeline)
#     pre_map_fn = pipeline._pre_cache_map
#     if pre_map_fn is None:
#         pre_map_fn = pipeline._pre_batch_map

#     # test
#     base_dataset = problem.get_base_dataset('train')
#     for example in base_dataset.take(1):
#         pre_map_fn(*example)
#     print('pre_map_fn good!')

#     dataset = base_dataset.map(pre_map_fn).batch(batch_size,
#                                                  drop_remainder=True)
#     for example in dataset.take(1):
#         pipeline._post_batch_map(*example)
#     print('post_batch_map good!')

#     dataset = dataset.map(pipeline._post_batch_map)
#     for example in dataset.take(1):
#         pass
#     print('manual pipeline good!')

#     dataset = pipeline(base_dataset.take(500))
#     for example in dataset.take(1):
#         pass
#     print('full pipeline good!')
