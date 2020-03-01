import numpy as np
import tensorflow as tf
from ecn.ops import conv_v2
from ecn.ops import conv


class ConvOpsV2Test(tf.test.TestCase):

    def test_spatio_temporal_event_conv(self):
        n_in = 7
        n_out = 17
        f_in = 11
        f_out = 13
        sk = 3
        tk = 5
        E = 51

        features = tf.random.normal((n_in, f_in))
        dt_values = tf.random.uniform((E,),)
        i = tf.random.uniform((E,), maxval=n_out, dtype=tf.int64)
        s = tf.sort(tf.random.uniform((E,), maxval=sk, dtype=tf.int64))
        j = tf.random.uniform((E,), maxval=n_in, dtype=tf.int64)
        indices = tf.stack((i, s, j), axis=-1)
        dt = tf.SparseTensor(indices, dt_values, (n_out, sk, n_in))
        kernel = tf.random.normal((tk, sk, f_in, f_out))
        decay = tf.random.uniform((tk, sk))

        v2 = conv_v2.spatio_temporal_event_conv(features, dt, kernel, decay)

        kernel = tf.transpose(kernel, (1, 0, 2, 3))
        decay = tf.transpose(decay, (1, 0))
        ij = tf.stack((i, j), axis=-1)
        s = tf.cast(s, dtype=tf.int32)
        ijs = tf.dynamic_partition(ij, s, sk)
        dt_values = tf.dynamic_partition(dt_values, s, sk)
        dts = [
            tf.SparseTensor(ij, vals, (n_out, n_in))
            for ij, vals in zip(ijs, dt_values)
        ]

        v1 = conv.spatio_temporal_event_conv(features, dts, kernel, decay)
        v1, v2 = self.evaluate((v1, v2))
        np.testing.assert_allclose(v1, v2, atol=1e-3)


if __name__ == '__main__':
    # tf.test.main()

    ConvOpsV2Test().test_spatio_temporal_event_conv()
