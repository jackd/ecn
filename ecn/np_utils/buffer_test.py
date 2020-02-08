import unittest
import numpy as np

import ecn.np_utils.buffer as bu


class BufferTest(unittest.TestCase):

    def test_discard_left(self):
        start_stop = np.array([3, 2])
        mod = 5
        bu.discard_left(start_stop, mod)
        np.testing.assert_equal(start_stop, [4, 2])
        bu.discard_left(start_stop, mod)
        np.testing.assert_equal(start_stop, [0, 2])

    def test_push_right(self):
        start_stop = np.array([3, 1])
        mod = 5
        values = np.array([2, -1, -1, 0, 1])
        bu.push_right(3, values, start_stop, mod)
        np.testing.assert_equal(start_stop, [3, 2])
        np.testing.assert_equal(values, [2, 3, -1, 0, 1])
        bu.push_right(4, values, start_stop, mod)
        np.testing.assert_equal(start_stop, [4, 3])
        np.testing.assert_equal(values, [2, 3, 4, 0, 1])
        bu.push_right(5, values, start_stop, mod)
        np.testing.assert_equal(start_stop, [0, 4])
        np.testing.assert_equal(values, [2, 3, 4, 5, 1])

    def test_indices(self):
        start_stop = np.array([3, 1])
        mod = 5
        indices = list(bu.indices(start_stop, mod))
        np.testing.assert_equal(indices, [3, 4, 0])
        start_stop = np.array([2, 4])
        indices = list(bu.indices(start_stop, mod))
        np.testing.assert_equal(indices, [2, 3])
        start_stop = np.array([2, 2])
        indices = list(bu.indices(start_stop, mod))
        np.testing.assert_equal(indices, [])


if __name__ == '__main__':
    unittest.main()
