import numpy as np
import numba as nb


@nb.njit(inline='always')
def double_length(x):
    return np.concatenate((x, np.empty_like(x)), axis=0)
