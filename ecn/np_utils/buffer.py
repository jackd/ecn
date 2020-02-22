import numpy as np
from numba import njit

IntArray = np.ndarray


@njit(inline='always')
def _push_index(k, start_stop, mod):
    out = start_stop[k] = (start_stop[k] + 1) % mod
    return out


@njit(inline='always')
def discard_left(start_stop, mod):
    return _push_index(0, start_stop, mod)


@njit(inline='always')
def push_right(value, values, start_stop, mod):
    values[start_stop[1]] = value
    i = _push_index(1, start_stop, mod)
    if i == start_stop[0]:
        _push_index(0, start_stop, mod)


@njit(inline='always')
def index(i, start, mod):
    return (start + i) % mod


@njit(inline='always')
def get_item(i, values, start, mod):
    return values[index(i, start, mod)]


@njit()
def indices(start_stop, mod):
    start, stop = start_stop
    dtype = start_stop.dtype
    if stop >= start:
        diff = stop - start
        out = np.empty((diff,), dtype=dtype)
        for i in range(diff):
            out[i] = start + i
        return out
    else:
        diff = mod - start
        out = np.empty((diff + stop,), dtype=dtype)
        for i in range(diff):
            out[i] = start + i
        for i in range(stop):
            out[i + diff] = i
        return out


@njit(inline='always')
def length(start_stop, mod):
    start, stop = start_stop
    return (stop - start) % mod


class CyclicBuffer(object):

    def __init__(self, values: np.ndarray, start_stop: IntArray):
        self._values = values
        self._start_stop = start_stop
        self._mod = self._values.shape[0]

    @property
    def max_length(self):
        return self._mod - 1

    def __len__(self):
        start, stop = self._start_stop
        return (stop - start) % self._mod

    def push_right(self, value):
        return push_right(value, self._values, self._start_stop, self._mod)

    def discard_left(self):
        return discard_left(self._start_stop, self._mod)

    def __getitem__(self, i):
        return self._values[index(i, self._start_stop[0], self._mod)]


if __name__ == '__main__':
    print(list(indices(np.array([5, 3]), 10)))
