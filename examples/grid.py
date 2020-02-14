from ecn.np_utils import grid
import numpy as np
from scipy.sparse import csr_matrix

in_shape = np.array((4, 5), dtype=np.int64)
kernel_shape = np.array((3, 3), dtype=np.int64)
strides = np.array((2, 2), dtype=np.int64)
padding = np.array((1, 1), dtype=np.int64)

indices, splits, shape = grid.sparse_neighborhood(in_shape, kernel_shape,
                                                  strides, padding)
print(shape)
print(indices)
print(splits)
print(indices.shape)
print(splits.shape)

values = np.ones_like(indices)
mat = csr_matrix((values, indices, splits))

import matplotlib.pyplot as plt
plt.spy(mat)
plt.show()
