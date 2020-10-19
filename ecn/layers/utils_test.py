import numpy as np
import tensorflow as tf

from ecn.layers import utils


@utils.as_layer
def f(x, y):
    return x + y


class AsLayerTest(tf.test.TestCase):
    def test_tensor_constant(self):
        x = tf.keras.Input(())
        out = f(x, 3.0)
        model = tf.keras.Model(x, out)
        np.testing.assert_equal(self.evaluate(model(tf.constant([3.0, 4.0]))), [6, 7])

    def test_constant_tensor(self):
        x = tf.keras.Input(())
        out = f(3.0, x)
        model = tf.keras.Model(x, out)
        np.testing.assert_equal(self.evaluate(model(tf.constant([3.0, 4.0]))), [6, 7])

    def test_same_tensors(self):
        x = tf.keras.Input(())
        out = f(x, x)
        model = tf.keras.Model(x, out)
        np.testing.assert_equal(self.evaluate(model(tf.constant([3.0, 4.0]))), [6, 8])

    def test_different_tensors(self):
        x = tf.keras.Input(())
        y = tf.keras.Input(())
        out = f(x, y)
        model = tf.keras.Model([x, y], out)
        x = tf.constant([3, 4], tf.float32)
        y = tf.constant([5, 6], tf.float32)
        np.testing.assert_equal(self.evaluate(model((x, y))), [8, 10])

    def test_tensor_const_tensor(self):
        x = tf.keras.Input(())
        out = f(x, tf.convert_to_tensor(3.0))
        model = tf.keras.Model(x, out)
        np.testing.assert_equal(self.evaluate(model(tf.constant([3.0, 4.0]))), [6, 7])


if __name__ == "__main__":
    tf.test.main()
