import unittest
import numpy as np
import ecn.np_utils.neighbors as neigh
from scipy.sparse import coo_matrix


def ragged_to_sparse(indices, splits, values, shape):
    row_lengths = splits[1:] - splits[:-1]
    i = np.repeat(np.arange(row_lengths.size), row_lengths)
    return coo_matrix((values, (i, indices)), shape=shape)


class NeighborsTest(unittest.TestCase):

    def test_reindex(self):
        np.random.seed(123)
        n_in = 100
        n_out = 10
        k = 10
        row_lengths = np.random.randint(0, k, size=(n_out,))
        total = np.sum(row_lengths)
        indices = np.random.randint(0, n_in, size=(total,))
        splits = np.concatenate([[0], np.cumsum(row_lengths)])

        values = np.random.uniform(size=(total,))
        rhs = np.random.uniform(size=(n_in,))

        sparse = ragged_to_sparse(indices, splits, values, shape=(n_out, n_in))
        expected = sparse @ rhs

        mask = np.zeros((n_in,), dtype=np.bool)
        mask[indices] = True
        if np.all(mask):
            raise RuntimeError(
                'test isn\'t going to work if all inputs are used.')
        n_in = np.count_nonzero(mask)

        # masked_indices, masked_splits = neigh.mask_ragged_cols(
        #     indices, splits, mask)
        ri = neigh.reindex_index(mask)
        indices = neigh.reindex(indices, ri)
        sparse = ragged_to_sparse(indices, splits, values, shape=(n_out, n_in))
        actual = sparse @ rhs[mask]
        np.testing.assert_equal(actual, expected)

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

        indices, splits = neigh.mask_ragged_rows(indices, splits, row_mask)
        n_out = splits.size - 1

        sparse = ragged_to_sparse(indices, splits, values, shape=(n_out, n_in))
        actual = sparse @ rhs
        np.testing.assert_allclose(actual, expected)


if __name__ == '__main__':
    # NeighborsTest().test_ravel_multi_index()
    unittest.main()
