import numpy as np
import tensorflow as tf

import ecn.ops.grid as grid


class GridOpsTest(tf.test.TestCase):
    def test_ravel_multi_index_simple(self):
        dims = (5, 6, 7)
        size = 100
        indices = tuple(np.random.randint(0, d, size=size) for d in dims)
        expected = np.ravel_multi_index(indices, dims)

        indices = tf.constant(np.stack(indices, axis=-1), dtype=tf.int64)
        dims = tf.constant(dims, dtype=tf.int64)
        actual = self.evaluate(grid.ravel_multi_index(indices, dims))

        np.testing.assert_equal(actual, expected)

    def test_ravel_multi_index_int_dims(self):
        dims = 5
        size = 100
        indices = np.random.randint(0, dims, size=(3, size))
        expected = np.ravel_multi_index(tuple(indices), (dims, dims, dims))
        indices = tf.constant(indices.T, dtype=tf.int64)
        actual = self.evaluate(grid.ravel_multi_index(indices, dims))
        np.testing.assert_equal(actual, expected)

    def test_ravel_multi_index_axis(self):
        dims = 5
        size = 100
        indices = np.random.randint(0, dims, size=(3, size))
        expected = np.ravel_multi_index(tuple(indices), (dims, dims, dims))
        indices = tf.constant(indices, dtype=tf.int64)
        actual = self.evaluate(grid.ravel_multi_index(indices, dims, axis=0))
        np.testing.assert_equal(actual, expected)

    def test_unravel_index_transpose(self):
        dims = (5, 6, 7)
        size = 100
        indices = tuple(np.random.randint(0, d, size=size) for d in dims)
        ravelled = np.ravel_multi_index(indices, dims)
        # dims = (2, 3)
        # ravelled = np.array([1, 3, 5, 2])

        actual = self.evaluate(grid.unravel_index_transpose(ravelled, dims))
        expected = np.stack(np.unravel_index(ravelled, dims), axis=-1)
        np.testing.assert_equal(actual, expected)

    def test_base_grid_coords(self):
        np.testing.assert_equal(
            self.evaluate(grid.base_grid_coords((3, 4))),
            [
                [0, 0],
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [2, 0],
                [2, 1],
                [2, 2],
                [2, 3],
            ],
        )

    def test_grid_coords(self):
        coords, shape = self.evaluate(
            grid.grid_coords(in_shape=[5], kernel_shape=[3], strides=[1], padding=[0],)
        )
        np.testing.assert_equal(shape, (3,))
        np.testing.assert_equal(coords, np.expand_dims([0, 1, 2], axis=-1))

    def test_strided_grid_coords(self):
        coords, shape = self.evaluate(
            grid.grid_coords(in_shape=[5], kernel_shape=[3], strides=[2], padding=[0],)
        )
        np.testing.assert_equal(shape, (2,))
        np.testing.assert_equal(coords, np.expand_dims([0, 2], axis=-1))

    def test_padded_grid_coords(self):
        coords, shape = self.evaluate(
            grid.grid_coords(in_shape=[5], kernel_shape=[3], strides=[1], padding=[1],)
        )
        np.testing.assert_equal(shape, (5,))
        np.testing.assert_equal(coords, np.expand_dims([-1, 0, 1, 2, 3], axis=-1))

    def test_padded_strided_grid_coords(self):
        coords, shape = self.evaluate(
            grid.grid_coords(in_shape=[5], kernel_shape=[3], strides=[2], padding=[1],)
        )
        np.testing.assert_equal(shape, (3,))
        np.testing.assert_equal(coords, np.expand_dims([-1, 1, 3], axis=-1))

    def test_even_grid_coords(self):
        coords, shape = self.evaluate(
            grid.grid_coords(in_shape=[4], kernel_shape=[3], strides=[2], padding=[1],)
        )
        np.testing.assert_equal(shape, (2,))
        np.testing.assert_equal(coords, np.expand_dims([-1, 1], axis=-1))

    def test_sparse_neighborhood(self):
        in_shape = (4, 5)
        kernel_shape = (3, 3)
        strides = (2, 2)
        padding = (0, 0)
        p, indices, splits, out_shape = self.evaluate(
            grid.sparse_neighborhood(in_shape, kernel_shape, strides, padding=padding)
        )
        np.testing.assert_equal(out_shape, (1, 2))
        np.testing.assert_equal(p, tuple(range(9)) * 2)
        np.testing.assert_equal(
            indices, [0, 1, 2, 5, 6, 7, 10, 11, 12, 2, 3, 4, 7, 8, 9, 12, 13, 14]
        )
        np.testing.assert_equal(splits, [0, 9, 18])

    def test_sparse_neighborhood_1d(self):
        in_shape = (7,)
        kernel_shape = (3,)
        strides = (2,)
        padding = (0,)
        p, indices, splits, out_shape = self.evaluate(
            grid.sparse_neighborhood(in_shape, kernel_shape, strides, padding=padding)
        )
        np.testing.assert_equal(out_shape, (3,))
        np.testing.assert_equal(p, tuple(range(3)) * 3)
        np.testing.assert_equal(indices, [0, 1, 2, 2, 3, 4, 4, 5, 6])
        np.testing.assert_equal(splits, [0, 3, 6, 9])

        in_shape = (7,)
        kernel_shape = (2,)
        strides = (2,)
        padding = (0,)
        p, indices, splits, out_shape = self.evaluate(
            grid.sparse_neighborhood(in_shape, kernel_shape, strides, padding=padding)
        )
        np.testing.assert_equal(out_shape, (3,))
        np.testing.assert_equal(p, tuple(range(2)) * 3)
        np.testing.assert_equal(indices, [0, 1, 2, 3, 4, 5])
        np.testing.assert_equal(splits, [0, 2, 4, 6])

    def test_sparse_neighborhood_padded(self):
        in_shape = (4, 5)
        kernel_shape = (3, 3)
        strides = (2, 2)
        padding = (1, 1)
        p, indices, splits, out_shape = self.evaluate(
            grid.sparse_neighborhood(in_shape, kernel_shape, strides, padding=padding)
        )
        np.testing.assert_equal(out_shape, (2, 3))
        np.testing.assert_equal(
            p,
            (4, 5, 7, 8, 3, 4, 5, 6, 7, 8, 3, 4, 6, 7, 1, 2, 4, 5, 7, 8)
            + tuple(range(9))
            + (0, 1, 3, 4, 6, 7),
        )
        np.testing.assert_equal(
            indices,
            [
                0,
                1,
                5,
                6,
                1,
                2,
                3,
                6,
                7,
                8,
                3,
                4,
                8,
                9,
                5,
                6,
                10,
                11,
                15,
                16,
                6,
                7,
                8,
                11,
                12,
                13,
                16,
                17,
                18,
                8,
                9,
                13,
                14,
                18,
                19,
            ],
        )
        np.testing.assert_equal(splits, [0, 4, 10, 14, 20, 29, 35])

    def test_sparse_neighborhood_in_place(self):
        partitions, coords, splits = self.evaluate(
            grid.sparse_neighborhood_in_place((5,), (3,))
        )
        np.testing.assert_equal(partitions, [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
        np.testing.assert_equal(coords, [0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4])
        np.testing.assert_equal(splits, [0, 2, 5, 8, 11, 13])


if __name__ == "__main__":
    tf.test.main()
    # GridOpsTest().test_ravel_multi_index_axis()
