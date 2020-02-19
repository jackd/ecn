import numpy as np
import tensorflow as tf
from ecn.ops import ragged


class name(tf.test.TestCase):

    def test_transpose_csr(self):
        i = [0, 1, 1, 2, 2, 2, 3, 5]
        j = [1, 0, 2, 1, 2, 3, 1, 4]
        values = tf.range(8)
        splits = ragged.ids_to_splits(i)

        actual_indices, actual_splits, actual_values = self.evaluate(
            ragged.transpose_csr(j, splits, values))
        # expected_j = [0, 1, 1, 1, 2, 2, 3, 4]
        expected_splits = [0, 1, 4, 6, 7, 8]
        np.testing.assert_equal(actual_splits, expected_splits)

        # ragged.col_sort(actual_indices, actual_splits)
        expected_i = [1, 0, 2, 3, 1, 2, 2, 5]
        np.testing.assert_equal(actual_indices, expected_i)
        np.testing.assert_equal(actual_values, [1, 0, 3, 6, 2, 4, 5, 7])


if __name__ == '__main__':
    tf.test.main()
