import tensorflow as tf
import numpy as np
import ecn.ops.grid as grid


class GridOpsTest(tf.test.TestCase):

    def test_ravel_multi_index_simple(self):
        dims = (5, 6, 7)
        size = 100
        indices = tuple(np.random.randint(0, d, size=size) for d in dims)
        expected = np.ravel_multi_index(indices, dims)

        indices = tf.constant(np.stack(indices, axis=-1), dtype=tf.int64)
        dims = tf.constant(dims, dtype=tf.int64)
        actual = self.evaluate(grid.ravel_multi_index(indices, dims))

        np.testing.assert_equal(actual, expected)

    def test_ravel_multi_index_int_dims(self):
        dims = 5
        size = 100
        indices = np.random.randint(0, dims, size=(3, size))
        expected = np.ravel_multi_index(tuple(indices), (dims, dims, dims))
        indices = tf.constant(indices.T, dtype=tf.int64)
        actual = self.evaluate(grid.ravel_multi_index(indices, dims))
        np.testing.assert_equal(actual, expected)

    def test_ravel_multi_index_axis(self):
        dims = 5
        size = 100
        indices = np.random.randint(0, dims, size=(3, size))
        expected = np.ravel_multi_index(tuple(indices), (dims, dims, dims))
        indices = tf.constant(indices, dtype=tf.int64)
        actual = self.evaluate(grid.ravel_multi_index(indices, dims, axis=0))
        np.testing.assert_equal(actual, expected)


if __name__ == '__main__':
    tf.test.main()
    # GridOpsTest().test_ravel_multi_index_axis()
