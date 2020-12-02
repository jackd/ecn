msg = "tensorflow>=2.3 required, but {} found"
try:
    import tensorflow as tf

    if tf.version.VERSION < "2.3":
        raise ImportError(msg.format(tf.version.VERSION))
except ImportError as e:
    raise ImportError(msg.format("no installation")) from e

# clean up namespace
del msg
