import tensorflow as tf


def to_nearest_power(x, base=2):
    x = tf.convert_to_tensor(x, dtype_hint=tf.int64)
    base = tf.convert_to_tensor(base, dtype_hint=x.dtype)
    assert (x.dtype.is_integer)
    return base**tf.cast(
        tf.math.ceil(
            tf.math.log(tf.cast(x, tf.float32)) /
            tf.math.log(tf.cast(base, tf.float32))), x.dtype)


print(to_nearest_power(3, 2))
print(to_nearest_power(7, 2))
print(to_nearest_power(8, 2))
print(to_nearest_power(9, 2))
