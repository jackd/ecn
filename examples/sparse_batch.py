import tensorflow as tf

n = 100
m = 3
maxval = 10
ij = tf.random.uniform((n, m, 2), minval=0, maxval=maxval, dtype=tf.int64)

dataset = tf.data.Dataset.from_tensor_slices(ij)


def map_fn(ij):
    return tf.SparseTensor(ij, tf.ones((tf.shape(ij)[0],), dtype=tf.bool), (-1, -1))


dataset = dataset.map(map_fn).batch(3)
for ds in dataset:
    print(ds.dense_shape)
