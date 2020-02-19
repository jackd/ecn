import numpy as np
import unittest
from ecn.np_utils import spike
from ecn.np_utils import utils


class SpikeTest(unittest.TestCase):

    def test_global_spike(self):
        times = np.array([0, 1, 2, 3, 4, 100, 110, 111, 112, 113],
                         dtype=np.int64)
        actual = spike.global_spike_threshold(times, 5, 2, -2)
        np.testing.assert_equal(actual, [2, 112])

    def test_spike(self):
        t0 = np.array([0, 1, 2, 3, 4, 100, 110, 111, 112, 113], dtype=np.int64)
        t1 = t0[:8] + 2
        c0 = np.expand_dims(np.zeros_like(t0), axis=-1)
        c1 = np.expand_dims(np.ones_like(t1), axis=-1)

        times, coords = utils.merge(t0, c0, t1, c1)
        coords = np.squeeze(coords, axis=-1)
        kwargs = dict(decay_time=5, threshold=2, reset_potential=-2)
        actual_t, actual_c = spike.spike_threshold(
            times,
            coords,
            neigh_coords=np.array([0, 1]),
            neigh_splits=np.array([0, 1, 2]),
            **kwargs)

        actual_t0 = actual_t[actual_c == 0]
        actual_t1 = actual_t[actual_c == 1]

        expected_t0 = spike.global_spike_threshold(t0, **kwargs)
        expected_t1 = spike.global_spike_threshold(t1, **kwargs)
        np.testing.assert_equal(actual_t0, expected_t0)
        np.testing.assert_equal(actual_t1, expected_t1)


if __name__ == '__main__':
    unittest.main()
    # SpikeTest().test_spike()
