import unittest
import numpy as np
import ecn.np_utils.grid as grid


class GridTest(unittest.TestCase):

    def test_strided_grid_neighbors(self):
        kwargs = dict(
            grid_starts=(-1, -1),
            grid_stops=(2, 2),
            strides=(2, 2),
            shape=(11, 11),
        )
        actual = np.array(
            tuple(grid.strided_grid_neighbors(coords=(3, 3), **kwargs)))
        np.testing.assert_equal(actual, [
            [5, 5],
            [5, 6],
            [5, 7],
            [6, 5],
            [6, 6],
            [6, 7],
            [7, 5],
            [7, 6],
            [7, 7],
        ])

        actual = np.array(
            tuple(grid.strided_grid_neighbors(coords=(3, 5), **kwargs)))
        np.testing.assert_equal(actual, [
            [5, 9],
            [5, 10],
            [6, 9],
            [6, 10],
            [7, 9],
            [7, 10],
        ])

        actual = np.array(
            tuple(grid.strided_grid_neighbors(coords=(0, 3), **kwargs)))
        np.testing.assert_equal(actual, [
            [0, 5],
            [0, 6],
            [0, 7],
            [1, 5],
            [1, 6],
            [1, 7],
        ])

    def test_ravel_multi_index(self):
        dims = (5, 6, 7)
        size = 100
        indices = tuple(np.random.randint(0, d, size=size) for d in dims)
        actual = grid.ravel_multi_index(indices, dims)
        expected = np.ravel_multi_index(indices, dims)
        np.testing.assert_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
