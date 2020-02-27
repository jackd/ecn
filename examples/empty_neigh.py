import tensorflow as tf

from ecn.ops import grid as grid_ops
from ecn.ops import neighbors as neigh_ops

in_shape = (16, 16)
kernel_shape = (3, 3)
strides = (2, 2)
padding = (1, 1)

spatial_buffer_size = 32

with tf.Graph().as_default() as g:
    (grid_partitions, grid_indices, grid_splits,
     out_shape) = grid_ops.sparse_neighborhood(in_shape, kernel_shape, strides,
                                               padding)

    in_times = tf.constant([], dtype=tf.int64)
    in_coords = tf.constant([], dtype=tf.int64)
    out_times = tf.constant([], dtype=tf.int64)
    out_coords = tf.constant([], dtype=tf.int64)

    # event_partitions, event_indices, event_splits = neigh_ops.compute_neighbors(
    #     in_times,
    #     in_coords,
    #     out_times,
    #     out_coords,
    #     grid_partitions,
    #     grid_indices,
    #     grid_splits,
    #     spatial_buffer_size,
    #     event_duration=100)

    (event_partitions, event_indices,
     event_splits) = neigh_ops.compute_full_neighbors(in_times,
                                                      in_coords,
                                                      out_times,
                                                      event_duration=100)

    # event_indices, event_splits = neigh_ops.compute_pointwise_neighbors(
    #     in_times, in_coords, out_times, out_coords, event_duration=100)

    # (event_indices, event_splits) = neigh_ops.compute_pooled_neighbors(in_times,
    #                                                   out_times,
    #                                                   event_duration=100)

    rowids = tf.ragged.row_splits_to_segment_ids(event_splits,
                                                 out_type=tf.int64)
    print(rowids)
    print(event_indices)

    ij = tf.stack((rowids, event_indices), axis=-1)
    ijs = tf.dynamic_partition(ij,
                               tf.cast(event_partitions, tf.int32),
                               num_partitions=9)


@tf.function
def f():
    return tf.graph_util.import_graph_def(g.as_graph_def(),
                                          return_elements=[ij.name])


print(f()[0].numpy())
print('success')
