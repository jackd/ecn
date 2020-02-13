import numpy as np
import tensorflow as tf
from ecn import meta
# from ecn.meta import core
# from ecn.meta import debug
# from ecn.meta import multi_graph as mg
# from ecn.meta import multi_model as mm


def pre_batch_map(xy, label):
    x, y = xy
    return x * y, label


def post_batch_map(z, label):
    f = z**2
    # lr = core.learning_phase()
    # print('----------')
    # print(lr)
    # print('----------')
    # fo = tf.cond(lr, lambda: tf.identity(f), lambda: f)
    fo = f
    return (fo,), label


def model_fn(z):
    return tf.keras.layers.Dense(2, kernel_initializer='ones')(z)


def build_fn(xy, label):
    args = pre_batch_map(xy, label)
    args = tf.nest.map_structure(meta.batch, args)
    features, labels = post_batch_map(*args)
    model_inp = tf.nest.map_structure(meta.model_input, features)
    model_out = model_fn(*model_inp)
    return model_out, labels


class CoreMetaTest(tf.test.TestCase):

    def test_subgraph(self):
        graph = tf.Graph()

        def f(x, y):
            y0, y1 = y
            y = y0 + y1
            return {'z0': x**2 + y, 'z1': x + y**2}

        with graph.as_default():
            x = tf.constant(2.)
            y0 = tf.constant(3.)
            y1 = tf.constant(4.)
            z = f(x, (y0, y1))
            tf.keras.layers.Dense(3)(tf.random.uniform((10, 5),
                                                       dtype=tf.float32))

        fn = meta.subgraph(graph.as_graph_def(), (x, (y0, y1)), z)
        x = tf.constant(5.)
        y0 = tf.constant(6.)
        y1 = tf.constant(7.)
        y = (y0, y1)
        actual = fn(x, y)
        expected = f(x, y)

        actual, expected = self.evaluate((actual, expected))
        np.testing.assert_allclose(actual['z0'], expected['z0'])
        np.testing.assert_allclose(actual['z1'], expected['z1'])

    def test_debug_builder(self):
        batch_size = 3
        x = tf.random.uniform(shape=(5,), dtype=tf.float32)
        y = tf.random.uniform(shape=(5,), dtype=tf.float32)
        x = tf.constant(x)
        y = tf.constant(y)

        label = tf.zeros((), dtype=tf.int64)
        actual_out, actual_labels = meta.debug_builder(build_fn,
                                                       ((x, y), label),
                                                       batch_size=batch_size)

        expected_out = tf.tile(tf.expand_dims(x * y, 0), (batch_size, 1))
        expected_labels = tf.zeros((batch_size,), dtype=tf.int64)
        expected_out, expected_labels = post_batch_map(expected_out, label)
        expected_out = model_fn(*expected_out)

        (actual_out, actual_labels, expected_out,
         expected_labels) = self.evaluate(
             (actual_out, actual_labels, expected_out, expected_labels))
        np.testing.assert_allclose(actual_out, expected_out)
        np.testing.assert_allclose(actual_labels, expected_labels)

    def test_build_multi_graph(self):
        batch_size = 3
        x = tf.random.uniform(shape=(100, 5), dtype=tf.float32)
        y = tf.random.uniform(shape=(100, 5), dtype=tf.float32)
        labels = tf.random.uniform(shape=(100,), maxval=10, dtype=tf.int64)
        dataset = tf.data.Dataset.from_tensor_slices(((x, y), labels))

        # actual
        built = meta.build_multi_graph(build_fn, dataset.element_spec)
        processed = dataset.map(built.pre_batch_map)
        processed = processed.batch(batch_size)
        # print(processed.element_spec)
        processed = processed.map(built.post_batch_map)
        # print(processed.element_spec)
        # exit()
        actual_z = None
        actual_labels = None
        for actual_z, actual_labels in processed.take(1):
            pass
        actual_out = built.trained_model()(actual_z)

        # expected
        processed = dataset.map(pre_batch_map).batch(batch_size).map(
            post_batch_map)
        expected_z = None
        expected_labels = None
        for expected_z, expected_labels in processed.take(1):
            break
        expected_out = model_fn(*expected_z)

        # compare
        actual_out, actual_label, expected_out, expected_label = self.evaluate(
            (actual_out, actual_labels, expected_out, expected_labels))
        np.testing.assert_allclose(actual_out, expected_out)
        np.testing.assert_allclose(actual_label, expected_label)

    def test_build_multi_model(self):
        batch_size = 3
        x = tf.random.uniform(shape=(100, 5), dtype=tf.float32)
        y = tf.random.uniform(shape=(100, 5), dtype=tf.float32)
        labels = tf.random.uniform(shape=(100,), maxval=10, dtype=tf.int64)
        dataset = tf.data.Dataset.from_tensor_slices(((x, y), labels))

        # actual
        built = meta.build_multi_model(build_fn, dataset.element_spec)
        processed = dataset.map(built.pre_batch_map)
        processed = processed.batch(batch_size)
        # print(processed.element_spec)
        processed = processed.map(built.post_batch_map)
        # print(processed.element_spec)
        # exit()
        actual_z = None
        actual_labels = None
        for actual_z, actual_labels in processed.take(1):
            pass
        actual_out = built.trained_model(actual_z)

        # expected
        processed = dataset.map(pre_batch_map).batch(batch_size).map(
            post_batch_map)
        expected_z = None
        expected_labels = None
        for expected_z, expected_labels in processed.take(1):
            break
        expected_out = model_fn(*expected_z)

        # compare
        actual_out, actual_label, expected_out, expected_label = self.evaluate(
            (actual_out, actual_labels, expected_out, expected_labels))
        np.testing.assert_allclose(actual_out, expected_out)
        np.testing.assert_allclose(actual_label, expected_label)


if __name__ == '__main__':
    tf.test.main()

    # t = CoreMetaTest()
    # t.test_subgraph()
    # t.test_debug_meta_graph()
    # t.test_build_multi_graph()
    # t.test_build_multi_model()
