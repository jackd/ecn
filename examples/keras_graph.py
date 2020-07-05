import tensorflow as tf

with tf.Graph().as_default() as g:
    x = tf.keras.Input(shape=())
    y = tf.keras.Input(shape=())
    z = x + y
    model = tf.keras.Model((x, y), z)

out = z + 3

print(model([[2.0], [3.0]]))
