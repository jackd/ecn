import numpy as np
import tensorflow as tf
from ecn.ops import conv


def random_dt(n_in, n_out, E):
    sparse_values = tf.random.uniform(shape=(E,), dtype=tf.float32)
    i = tf.random.uniform(shape=(E,), maxval=n_out, dtype=tf.int64)
    j = tf.random.uniform(shape=(E,), maxval=n_in, dtype=tf.int64)
    indices = tf.stack((i, j), axis=-1)
    dt = tf.SparseTensor(indices, sparse_values, (n_out, n_in))
    return dt


class ConvTest(tf.test.TestCase):

    def test_complex_sparse_matmul(self):
        n_in = 15
        n_out = 13
        E = 11
        features = 7
        i = tf.random.uniform((E,), maxval=n_out, dtype=tf.int64)
        j = tf.random.uniform((E,), maxval=n_in, dtype=tf.int64)
        values = tf.complex(tf.random.uniform((E,), dtype=tf.float32),
                            tf.random.uniform((E,), dtype=tf.float32))
        sparse = tf.SparseTensor(tf.stack((i, j), axis=-1), values,
                                 [n_out, n_in])
        dense = tf.complex(
            tf.random.uniform((n_in, features), dtype=tf.float32),
            tf.random.uniform((n_in, features), dtype=tf.float32),
        )
        actual = conv.sparse_dense_matmul(sparse, dense)
        expected = tf.sparse.sparse_dense_matmul(sparse, dense)
        actual, expected = self.evaluate((actual, expected))
        np.testing.assert_allclose(actual, expected)

    def test_featureless_temporal_conv(self):
        n_in = 100
        n_out = 10
        f_out = 5
        tk = 3
        E = 200

        dt = random_dt(n_in, n_out, E)

        kernel = tf.random.normal((tk, 1, f_out))
        decay = tf.random.uniform((tk,))

        expected = conv.temporal_event_conv(
            tf.ones((n_in, 1), dtype=tf.float32), dt, kernel, decay)
        kernel = tf.squeeze(kernel, axis=1)
        actual = conv.featureless_temporal_event_conv(dt, kernel, decay)

        actual, expected = self.evaluate((actual, expected))
        np.testing.assert_allclose(actual, expected, atol=1e-3)

    def test_featureless_spatio_temporal_conv(self):
        n_in = 100
        n_out = 10
        f_out = 5
        tk = 3
        E = 200
        sk = 7

        dts = [random_dt(n_in, n_out, E) for _ in range(sk)]

        kernel = tf.random.normal((sk, tk, 1, f_out))
        decay = tf.random.uniform((
            sk,
            tk,
        ))

        expected = conv.spatio_temporal_event_conv(
            tf.ones((n_in, 1), dtype=tf.float32), dts, kernel, decay)
        kernel = tf.squeeze(kernel, axis=2)
        kernel = tf.reshape(kernel, (sk, tk, f_out))
        actual = conv.featureless_spatio_temporal_event_conv(dts, kernel, decay)

        actual, expected = self.evaluate((actual, expected))
        np.testing.assert_allclose(actual, expected, atol=1e-3)

    def test_binary_temporal_conv(self):
        n_in = 100
        n_out = 10
        f_out = 5
        tk = 3
        E = 200

        bool_features = tf.random.uniform(shape=(n_in,)) > 0.5
        dt = random_dt(n_in, n_out, E)

        kernel = tf.random.normal((tk, 2, f_out))
        decay = tf.random.uniform((tk,))

        expected = conv.temporal_event_conv(
            tf.one_hot(tf.cast(bool_features, tf.uint8), 2, dtype=tf.float32),
            dt, kernel, decay)
        kernel = tf.transpose(kernel, (1, 0, 2))
        kernel = tf.reshape(kernel, (2 * tk, f_out))
        actual = conv.binary_temporal_event_conv(bool_features, dt, kernel,
                                                 decay)

        actual, expected = self.evaluate((actual, expected))
        np.testing.assert_allclose(actual, expected, atol=1e-3)

    def test_binary_spatio_temporal_conv(self):
        n_in = 100
        n_out = 10
        f_out = 5
        tk = 3
        E = 200
        sk = 7

        bool_features = tf.random.uniform(shape=(n_in,)) > 0.5
        dts = [random_dt(n_in, n_out, E) for _ in range(sk)]

        kernel = tf.random.normal((sk, tk, 2, f_out))
        decay = tf.random.uniform((
            sk,
            tk,
        ))

        expected = conv.spatio_temporal_event_conv(
            tf.one_hot(tf.cast(bool_features, tf.uint8), 2, dtype=tf.float32),
            dts, kernel, decay)
        kernel = tf.transpose(kernel, (0, 2, 1, 3))
        kernel = tf.reshape(kernel, (sk, 2 * tk, f_out))
        actual = conv.binary_spatio_temporal_event_conv(bool_features, dts,
                                                        kernel, decay)

        actual, expected = self.evaluate((actual, expected))
        np.testing.assert_allclose(actual, expected, atol=1e-3)


if __name__ == '__main__':
    tf.test.main()
    # ConvTest().test_binary_temporal_conv()
