msg = "tensorflow>=2.0 required, but {} found"
try:
    import tensorflow as tf

    v = tf.version.VERSION
    major, minor, patch = v.split(".")
    if int(major) < 2:
        raise ImportError(msg.format(v))
except ImportError:
    raise ImportError(msg.format("no tf installation found"))

# clean up namespace
del msg
