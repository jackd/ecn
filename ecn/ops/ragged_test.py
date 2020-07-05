import numpy as np
import tensorflow as tf

from ecn.ops import ragged


class RaggedOpsTest(tf.test.TestCase):
    def test_ragged_csr_transpose_empty_end(self):
        # grid = comp.Grid((7,))
        # link = grid.link((2,), (2,), (0,))

        indices = tf.constant([0, 1, 2, 3, 4, 5])
        splits = tf.constant([0, 2, 4, 6])
        values = tf.range(6)
        indices, splits, values = self.evaluate(
            ragged.transpose_csr(indices, splits, values, nrows_out=7)
        )
        np.testing.assert_equal(indices, [0, 0, 1, 1, 2, 2])
        np.testing.assert_equal(splits, [0, 1, 2, 3, 4, 5, 6, 6])

    def test_transpose_csr(self):
        i = [0, 1, 1, 2, 2, 2, 3, 5]
        j = [1, 0, 2, 1, 2, 3, 1, 4]
        values = tf.range(8)
        splits = ragged.ids_to_splits(i)

        actual_indices, actual_splits, actual_values = self.evaluate(
            ragged.transpose_csr(j, splits, values)
        )
        # expected_j = [0, 1, 1, 1, 2, 2, 3, 4]
        expected_splits = [0, 1, 4, 6, 7, 8]
        np.testing.assert_equal(actual_splits, expected_splits)

        # ragged.col_sort(actual_indices, actual_splits)
        expected_i = [1, 0, 2, 3, 1, 2, 2, 5]
        np.testing.assert_equal(actual_indices, expected_i)
        np.testing.assert_equal(actual_values, [1, 0, 3, 6, 2, 4, 5, 7])

    def test_transpose_csr_empty(self):
        indices = []
        splits = [0]
        values = []

        actual_indices, actual_splits, actual_values = self.evaluate(
            ragged.transpose_csr(indices, splits, values)
        )
        expected_indices = []
        expected_splits = [0]
        expected_values = []
        np.testing.assert_equal(actual_indices, expected_indices)
        np.testing.assert_equal(actual_splits, expected_splits)
        np.testing.assert_equal(actual_values, expected_values)


if __name__ == "__main__":
    tf.test.main()
