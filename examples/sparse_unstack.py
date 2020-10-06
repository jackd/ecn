import tensorflow as tf

import kblocks.ops.sparse as sparse_ops

values = tf.range(5) ** 2
indices = tf.constant([[0, 2], [0, 5], [1, 1], [1, 2], [1, 4],], dtype=tf.int64,)
dense_shape = tf.constant((2, 6), dtype=tf.int64)
st = tf.SparseTensor(indices, values, dense_shape)

unstacked = sparse_ops.unstack(st, axis=0)
print(unstacked[0])
print(unstacked[1])
