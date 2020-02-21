import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    x = tf.keras.backend.placeholder(shape=(), dtype=tf.float32)
    y = tf.keras.backend.placeholder(shape=(), dtype=tf.float32)
    training = tf.keras.backend.placeholder(shape=(), dtype=tf.bool)
    z = tf.cond(training, lambda: x * 10, lambda: y * 2)

graph_def = graph.as_graph_def(add_shapes=True)


def fn(*args):
    x_value, y_value, training_value = args
    return tf.graph_util.import_graph_def(graph_def,
                                          input_map={
                                              x.op.name: x_value,
                                              y.op.name: y_value,
                                              training.op.name: training_value,
                                          },
                                          return_elements=[z.name])


class ImportedCondTest(tf.test.TestCase):

    def test_in_graph_mode(self):
        with tf.Graph().as_default():
            out = tf.graph_util.import_graph_def(graph_def,
                                                 input_map={
                                                     x.op.name: 0.,
                                                     y.op.name: 1.0,
                                                     training.op.name: True,
                                                 },
                                                 return_elements=[z.name])
            with tf.compat.v1.Session() as sess:
                sess.run(out)

    def test_fn(self):
        fn(0., 1., True)

    def test_tf_function_fn(self):
        tf.function(fn)(0., 1., True)

    def test_map(self):
        dataset = tf.data.Dataset.from_tensor_slices(([0.], [1.], [True]))
        mapped = dataset.map(fn)
        for _ in mapped:
            pass


if __name__ == '__main__':
    tf.test.main()
