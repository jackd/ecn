import unittest
import numpy as np
import ecn.np_utils.grid as grid


class GridTest(unittest.TestCase):

    def test_neighbor_offsets(self):
        actual = grid.neighbor_offsets(np.array((3,)))
        expected = [[-1], [0], [1]]
        np.testing.assert_equal(actual, expected)

        actual = grid.neighbor_offsets(np.array((2, 3)))
        expected = np.stack(np.meshgrid(np.arange(2),
                                        np.arange(3) - 1,
                                        indexing='ij'),
                            axis=-1).reshape((6, 2))
        np.testing.assert_equal(actual, expected)

        actual = grid.neighbor_offsets(np.array((2, 2)))
        np.testing.assert_equal(actual, [[0, 0], [0, 1], [1, 0], [1, 1]])
        actual = grid.neighbor_offsets(np.array((3, 3)))
        np.testing.assert_equal(actual, [
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, -1],
            [0, 0],
            [0, 1],
            [1, -1],
            [1, 0],
            [1, 1],
        ])

    # def test_strided_grid_neighbors(self):
    #     kwargs = dict(
    #         grid_starts=(-1, -1),
    #         grid_stops=(2, 2),
    #         strides=(2, 2),
    #         shape=(11, 11),
    #     )
    #     actual = np.array(
    #         tuple(grid.strided_grid_neighbors(coords=(3, 3), **kwargs)))
    #     np.testing.assert_equal(actual, [
    #         [5, 5],
    #         [5, 6],
    #         [5, 7],
    #         [6, 5],
    #         [6, 6],
    #         [6, 7],
    #         [7, 5],
    #         [7, 6],
    #         [7, 7],
    #     ])

    #     actual = np.array(
    #         tuple(grid.strided_grid_neighbors(coords=(3, 5), **kwargs)))
    #     np.testing.assert_equal(actual, [
    #         [5, 9],
    #         [5, 10],
    #         [6, 9],
    #         [6, 10],
    #         [7, 9],
    #         [7, 10],
    #     ])

    #     actual = np.array(
    #         tuple(grid.strided_grid_neighbors(coords=(0, 3), **kwargs)))
    #     np.testing.assert_equal(actual, [
    #         [0, 5],
    #         [0, 6],
    #         [0, 7],
    #         [1, 5],
    #         [1, 6],
    #         [1, 7],
    #     ])

    def test_ravel_multi_index(self):
        dims = (5, 6, 7)
        size = 100
        indices = tuple(np.random.randint(0, d, size=size) for d in dims)
        actual = grid.ravel_multi_index(indices, dims)
        expected = np.ravel_multi_index(indices, dims)
        np.testing.assert_equal(actual, expected)

    def test_ravel_multi_index_transpose(self):
        dims = (5, 6, 7)
        size = 100
        indices = tuple(np.random.randint(0, d, size=size) for d in dims)
        actual = grid.ravel_multi_index_transpose(np.stack(indices, axis=-1),
                                                  dims)
        expected = np.ravel_multi_index(indices, dims)
        np.testing.assert_equal(actual, expected)

    def test_unravel_index(self):
        dims = (5, 6, 7)
        size = 100
        indices = tuple(np.random.randint(0, d, size=size) for d in dims)
        ravelled = np.ravel_multi_index(indices, dims)
        # dims = (2, 3)
        # ravelled = np.array([1, 3, 5, 2])

        actual = grid.unravel_index(ravelled, dims)
        expected = np.stack(np.unravel_index(ravelled, dims), axis=0)
        np.testing.assert_equal(actual, expected)

    def test_unravel_index_transpose(self):
        dims = (5, 6, 7)
        size = 100
        indices = tuple(np.random.randint(0, d, size=size) for d in dims)
        ravelled = np.ravel_multi_index(indices, dims)
        # dims = (2, 3)
        # ravelled = np.array([1, 3, 5, 2])

        actual = grid.unravel_index_transpose(ravelled, dims)
        expected = np.stack(np.unravel_index(ravelled, dims), axis=-1)
        np.testing.assert_equal(actual, expected)

    def test_sparse_neighborhood(self):
        in_shape = np.array((4, 5), dtype=np.int64)
        kernel_shape = np.array((3, 3), dtype=np.int64)
        strides = np.array((2, 2), dtype=np.int64)
        padding = np.array((0, 0), dtype=np.int64)
        p, indices, splits, out_shape = grid.sparse_neighborhood(
            in_shape, kernel_shape, strides, padding=padding)
        np.testing.assert_equal(out_shape, (2, 2))
        np.testing.assert_equal(p, tuple(range(9)) * 2 + tuple(range(6)) * 2)
        np.testing.assert_equal(indices, [
            0, 1, 2, 5, 6, 7, 10, 11, 12, 2, 3, 4, 7, 8, 9, 12, 13, 14, 10, 11,
            12, 15, 16, 17, 12, 13, 14, 17, 18, 19
        ])
        np.testing.assert_equal(splits, [0, 9, 18, 24, 30])

    def test_sparse_neighborhood_padded(self):
        in_shape = np.array((4, 5), dtype=np.int64)
        kernel_shape = np.array((3, 3), dtype=np.int64)
        strides = np.array((2, 2), dtype=np.int64)
        padding = np.array((1, 1), dtype=np.int64)
        p, indices, splits, out_shape = grid.sparse_neighborhood(
            in_shape, kernel_shape, strides, padding=padding)
        np.testing.assert_equal(out_shape, (2, 3))
        np.testing.assert_equal(
            p, (4, 5, 7, 8, 3, 4, 5, 6, 7, 8, 3, 4, 6, 7, 1, 2, 4, 5, 7, 8) +
            tuple(range(9)) + (0, 1, 3, 4, 6, 7))
        np.testing.assert_equal(indices, [
            0, 1, 5, 6, 1, 2, 3, 6, 7, 8, 3, 4, 8, 9, 5, 6, 10, 11, 15, 16, 6,
            7, 8, 11, 12, 13, 16, 17, 18, 8, 9, 13, 14, 18, 19
        ])
        np.testing.assert_equal(splits, [0, 4, 10, 14, 20, 29, 35])


if __name__ == '__main__':
    unittest.main()
