import unittest
import numpy as np
import ecn.np_utils.ragged as ragged
from scipy.sparse import coo_matrix


def ragged_to_sparse(indices, splits, values, shape):
    row_lengths = splits[1:] - splits[:-1]
    i = np.repeat(np.arange(row_lengths.size), row_lengths)
    return coo_matrix((values, (i, indices)), shape=shape)


class RaggedTest(unittest.TestCase):

    def test_lengths_to_ids(self):
        row_lengths = np.array([2, 3, 5])
        actual = ragged.lengths_to_ids(row_lengths)
        expected = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        np.testing.assert_equal(actual, expected)
        actual = ragged.ids_to_lengths(expected)
        np.testing.assert_equal(actual, row_lengths)

    def test_splits_to_lengths(self):
        splits = np.array([0, 2, 5, 10])
        actual = ragged.splits_to_lengths(splits)
        lengths = np.array([2, 3, 5])
        np.testing.assert_equal(actual, lengths)
        actual = ragged.lengths_to_splits(lengths)
        np.testing.assert_equal(actual, [0, 2, 5, 10])

    def test_col_sort(self):
        i = np.array([0, 1, 1, 2, 2, 2, 3, 5])
        j = np.array([1, 0, 2, 3, 2, 4, 1, 4])
        splits = ragged.ids_to_splits(i)
        ragged.col_sort(j, splits)
        np.testing.assert_equal(j, [1, 0, 2, 2, 3, 4, 1, 4])

    def test_transpose_csr(self):
        i = np.array([0, 1, 1, 2, 2, 2, 3, 5])
        j = np.array([1, 0, 2, 1, 2, 3, 1, 4])
        splits = ragged.ids_to_splits(i)

        actual_indices, actual_splits = ragged.transpose_csr(j, splits)
        # expected_j = [0, 1, 1, 1, 2, 2, 3, 4]
        expected_splits = [0, 1, 4, 6, 7, 8]
        np.testing.assert_equal(actual_splits, expected_splits)

        ragged.col_sort(actual_indices, actual_splits)
        expected_i = [1, 0, 2, 3, 1, 2, 2, 5]
        np.testing.assert_equal(actual_indices, expected_i)

    def test_mask_ragged_rows(self):
        np.random.seed(123)
        n_in = 100
        n_out = 10
        k = 10
        row_lengths = np.random.randint(0, k, size=(n_out,))
        row_lengths[[2, 5, n_out - 1]] = 0
        total = np.sum(row_lengths)
        indices = np.random.randint(0, n_in, size=(total,))
        splits = np.concatenate([[0], np.cumsum(row_lengths)])
        row_mask = row_lengths > 0

        values = np.random.uniform(size=(total,))
        rhs = np.random.uniform(size=(n_in,))

        sparse = ragged_to_sparse(indices, splits, values, shape=(n_out, n_in))
        expected = sparse @ rhs
        np.testing.assert_allclose(expected[np.logical_not(row_mask)], 0)
        expected = expected[row_mask]

        indices, splits = ragged.mask_rows(indices, splits, row_mask)
        n_out = splits.size - 1

        sparse = ragged_to_sparse(indices, splits, values, shape=(n_out, n_in))
        actual = sparse @ rhs
        np.testing.assert_allclose(actual, expected)


if __name__ == '__main__':
    unittest.main()
    # RaggedTest().test_transpose_csr()
