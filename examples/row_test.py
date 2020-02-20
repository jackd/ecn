import tensorflow as tf
from ecn.ops import grid as grid_ops
from ecn.ops import ragged as ragged_ops

grid_shape = (9, 9)
partitions, indices, splits, out_shape = grid_ops.sparse_neighborhood(
    grid_shape, (5, 5), (2, 2), (2, 2))
lengths = splits[1:] - splits[:-1]
print(indices.numpy())
print(lengths.numpy())

indices_T, splits_T, partitions_T = ragged_ops.transpose_csr(
    indices, splits, partitions)
lengths_T = splits_T[1:] - splits_T[:-1]

print('---')
print(indices_T.numpy())
print(lengths_T.numpy())

print(out_shape)
for sp in (splits, splits_T):
    lengths = sp[1:] - sp[:-1]
    print(tf.reduce_min(lengths).numpy())
