"""Fails."""
import numba as nb
import numpy as np


@nb.njit()
def f(x):
    return np.zeros([xi * 2 for xi in x])


print(f((2, 3, 4)))
