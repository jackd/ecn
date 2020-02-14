"""Fails."""
import numpy as np
import numba as nb


@nb.njit()
def f(x):
    return np.zeros([xi * 2 for xi in x])


print(f((2, 3, 4)))
