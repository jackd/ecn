import unittest
import numpy as np
import ecn.np_utils.neighbors as neigh
import ecn.np_utils.grid as grid
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

    def test_compute_neighbors_1d_finite(self):
        in_times = np.array([0, 2, 3, 100], dtype=np.int64)
        in_coords = np.array([0, 3, 2, 1], dtype=np.int64)
        out_times = np.array([3, 3, 100], dtype=np.int64)
        out_coords = np.array([1, 3, 2], dtype=np.int64)
        neighbor_offsets = np.arange(-1, 2, dtype=np.int64)
        event_duration = 10
        spatial_buffer_size = 5
        partitions, indices, splits = neigh.compute_neighbors_1d(
            in_times=in_times,
            in_coords=in_coords,
            out_times=out_times,
            out_coords=out_coords,
            neighbor_offsets=neighbor_offsets,
            event_duration=event_duration,
            spatial_buffer_size=spatial_buffer_size,
        )
        expected_partitions = np.array([0, 2, 0, 1, 0])
        expected_indices = np.array([0, 2, 2, 1, 3])
        expected_splits = np.array([0, 2, 4, 5])

        np.testing.assert_equal(partitions, expected_partitions)
        np.testing.assert_equal(indices, expected_indices)
        np.testing.assert_equal(splits, expected_splits)

    def test_compute_neighbors_1d_infinite(self):
        in_times = np.array([0, 2, 3, 100], dtype=np.int64)
        in_coords = np.array([0, 3, 2, 1], dtype=np.int64)
        out_times = np.array([3, 3, 100], dtype=np.int64)
        out_coords = np.array([1, 3, 2], dtype=np.int64)
        neighbor_offsets = np.arange(-1, 2, dtype=np.int64)
        event_duration = None
        spatial_buffer_size = 5
        partitions, indices, splits = neigh.compute_neighbors_1d(
            in_times=in_times,
            in_coords=in_coords,
            out_times=out_times,
            out_coords=out_coords,
            neighbor_offsets=neighbor_offsets,
            event_duration=event_duration,
            spatial_buffer_size=spatial_buffer_size,
        )
        expected_partitions = np.array([0, 2, 0, 1, 0, 1, 2])
        expected_indices = np.array([0, 2, 2, 1, 3, 2, 1])
        expected_splits = np.array([0, 2, 4, 7])

        np.testing.assert_equal(partitions, expected_partitions)
        np.testing.assert_equal(indices, expected_indices)
        np.testing.assert_equal(splits, expected_splits)

    def test_neighborhood_args_as_1d(self):
        in_coords = np.array([
            [0, 0],
        ], dtype=np.int64)
        out_coords = np.array([
            [0, 0],
            [3, 3],
        ], dtype=np.int64)
        offsets = grid.neighbor_offsets((3, 3))
        in_coords, out_coords, offsets, size = neigh.neighborhood_args_as_1d(
            in_coords, out_coords, offsets)
        np.testing.assert_equal(in_coords, [7])
        np.testing.assert_equal(out_coords, [7, 28])
        np.testing.assert_equal(offsets, [-7, -6, -5, -1, 0, 1, 5, 6, 7])
        self.assertEqual(size, 36)

    def test_compute_neighbors_2d_finite(self):
        in_coords = np.array([
            [0, 0],
            [2, 3],
            [1, 1],
        ], dtype=np.int64)
        in_times = np.array([0, 2, 4], dtype=np.int64)
        out_coords = np.array([
            [0, 0],
            [2, 2],
        ], dtype=np.int64)
        out_times = np.array([3, 5])
        event_duration = None
        spatial_buffer_size = 4
        neighbor_offsets = grid.neighbor_offsets((3, 3))

        partitions, indices, splits = neigh.compute_neighbors(
            in_times=in_times,
            in_coords=in_coords,
            out_times=out_times,
            out_coords=out_coords,
            neighbor_offsets=neighbor_offsets,
            event_duration=event_duration,
            spatial_buffer_size=spatial_buffer_size,
        )

        np.testing.assert_equal(partitions, [4, 0, 5])
        np.testing.assert_equal(indices, [0, 2, 1])
        np.testing.assert_equal(splits, [0, 1, 3])

    def test_buffer_overflow(self):
        in_times = np.arange(10, dtype=np.int64)
        out_times = np.arange(5, 8, dtype=np.int64)
        in_coords = np.zeros((10,), dtype=np.int64)
        out_coords = np.zeros((3,), dtype=np.int64)
        neighbor_offsets = np.zeros((1,), dtype=np.int64)

        event_duration = None
        spatial_buffer_size = 4
        partitions, indices, splits = neigh.compute_neighbors_1d(
            in_times=in_times,
            in_coords=in_coords,
            out_times=out_times,
            out_coords=out_coords,
            neighbor_offsets=neighbor_offsets,
            event_duration=event_duration,
            spatial_buffer_size=spatial_buffer_size,
        )
        np.testing.assert_equal(partitions, np.zeros((9,), dtype=np.int64))
        np.testing.assert_equal(indices, [3, 4, 5, 4, 5, 6, 5, 6, 7])
        np.testing.assert_equal(splits, [0, 3, 6, 9])


if __name__ == '__main__':
    # NeighborsTest().test_compute_neighbors_1d_finite()
    # NeighborsTest().test_neighborhood_args_as_1d()
    # NeighborsTest().test_neighbor_offsets()
    unittest.main()
