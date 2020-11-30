import tensorflow as tf
from absl import app, flags, logging

from ecn.ops import conv, conv_v2

flags.DEFINE_boolean("jit", default=False, help="use XLA jit compilation")
flags.DEFINE_boolean("sort", default=False, help="use sorted indices")
flags.DEFINE_boolean(
    "backward", default=False, help="benchmark forward and backward pass"
)
flags.DEFINE_boolean("csr", default=False, help="use csr implementation")
flags.DEFINE_integer("burn_iters", default=10, help="number of burn in iterations")
flags.DEFINE_integer("ni", default=100000, help="number of input events")
flags.DEFINE_integer(
    "no", default=-1, help="number of output events. Defaults to ni // 4"
)
flags.DEFINE_integer(
    "ne", default=-1, help="number of edges, -1 will result in using 10*nv"
)
flags.DEFINE_integer("sk", default=9, help="spatial kernel size")
flags.DEFINE_integer("tk", default=4, help="temporal kernel size")
flags.DEFINE_integer("fi", default=32, help="number of input features")
flags.DEFINE_integer("fo", default=64, help="number of output features")
flags.DEFINE_integer(
    "min_iters", default=20, help="minimum number of iterations to benchmark"
)

FLAGS = flags.FLAGS


def summarize(result, print_fn=print):
    """
    Args:
        result: output of a tf.test.Benchmark.run_op_benchmark call.
        print_fn: print-like function.
    """
    print_fn("Wall time (ms): {}".format(result["wall_time"] * 1000))
    gpu_mem = result["extras"].get("allocator_maximum_num_bytes_GPU_0_bfc", 0)
    print_fn("Memory (Mb):  {}".format(gpu_mem / 1024 ** 2))


def get_kwargs():
    if FLAGS.csr and not FLAGS.sort:
        raise ValueError("sort must be True if csr is True")
    n_in = FLAGS.ni
    n_out = FLAGS.no
    if n_out == -1:
        n_out = n_in // 4
    E = FLAGS.ne
    if E == -1:
        E = 10 * n_out
    f_in = FLAGS.fi
    f_out = FLAGS.fo

    tk = FLAGS.tk
    sk = FLAGS.sk

    features = tf.random.normal((n_in, f_in))
    dt_values = tf.random.uniform((E,), dtype=tf.float32)
    i = tf.random.uniform((E,), maxval=n_out, dtype=tf.int64)
    s = tf.sort(tf.random.uniform((E,), maxval=sk, dtype=tf.int64))
    j = tf.random.uniform((E,), maxval=n_in, dtype=tf.int64)
    indices = tf.stack((i, s, j), axis=-1)
    dt = tf.SparseTensor(indices, dt_values, (n_out, sk, n_in))

    kernel = tf.random.normal((tk, sk, f_in, f_out))
    decay = tf.random.uniform((tk, sk))

    if FLAGS.sort:
        dt = tf.sparse.reorder(dt)

    v2_kwargs = dict(features=features, dt=dt, kernel=kernel, decay=decay)

    dt.indices.assert_has_rank(2)
    assert dt.indices.shape[1] == 3
    i, s, j = tf.unstack(dt.indices, axis=-1)
    s = tf.cast(s, tf.int32)
    ij = tf.stack((i, j), axis=-1)
    ijs = tf.dynamic_partition(ij, s, sk)
    vals = tf.dynamic_partition(dt.values, s, sk)
    dt = tuple(tf.SparseTensor(ij, val, (n_out, n_in)) for ij, val in zip(ijs, vals))
    kernel = tf.transpose(kernel, (1, 0, 2, 3))
    decay = tf.transpose(decay, (1, 0))
    v1_kwargs = dict(features=features, dt=dt, kernel=kernel, decay=decay)
    for kwargs in v1_kwargs, v2_kwargs:
        kwargs["use_csr"] = FLAGS.csr
    return v1_kwargs, v2_kwargs


def v1_train_op(features, dt, kernel, decay, use_csr: bool):
    features = tf.constant(features.numpy())
    dt = [
        tf.SparseTensor(d.indices.numpy(), d.values.numpy(), d.dense_shape.numpy())
        for d in dt
    ]
    with tf.GradientTape() as tape:
        kernel = tf.Variable(kernel.numpy())
        decay = tf.Variable(decay.numpy())
        tape.watch(kernel)
        tape.watch(decay)
        forward = conv.spatio_temporal_event_conv(
            features, dt, kernel, decay, use_csr=use_csr
        )
    if FLAGS.backward:
        return tape.gradient(forward, (kernel, decay))
    return forward


def v2_train_op(features, dt, kernel, decay, use_csr: bool):
    features = tf.constant(features.numpy())
    dt = tf.SparseTensor(dt.indices.numpy(), dt.values.numpy(), dt.dense_shape.numpy())
    with tf.GradientTape() as tape:
        kernel = tf.Variable(kernel.numpy())
        decay = tf.Variable(decay.numpy())
        tape.watch(kernel)
        tape.watch(decay)
        out = conv_v2.spatio_temporal_event_conv(
            features, dt, kernel, decay, use_csr=use_csr
        )
    if FLAGS.backward:
        return tape.gradient(out, (kernel, decay))
    return out


def main(_):
    tf.config.optimizer.set_jit(FLAGS.jit)
    bm = tf.test.Benchmark()
    bm_kwargs = dict(burn_iters=FLAGS.burn_iters, min_iters=FLAGS.min_iters)
    v1_kwargs, v2_kwargs = get_kwargs()
    with tf.Graph().as_default():
        v1_op = v1_train_op(**v1_kwargs)
        v2_op = v2_train_op(**v2_kwargs)
        with tf.compat.v1.Session() as sess:
            logging.info("Initializing variables...")

            sess.run(tf.compat.v1.global_variables_initializer())

            logging.info("Starting v1 benchmarking...")
            v1_result = bm.run_op_benchmark(sess, v1_op, **bm_kwargs)
            logging.info("Starting v2 benchmarking...")
            v2_result = bm.run_op_benchmark(sess, v2_op, **bm_kwargs)
    print("v1")
    summarize(v1_result)
    print("v2")
    summarize(v2_result)


if __name__ == "__main__":
    app.run(main)
