from ecn.ops import neighbors as _neigh_ops
from wtftf.meta import layered

compute_pooled_neighbors = layered(_neigh_ops.compute_pooled_neighbors)
compute_full_neighbors = layered(_neigh_ops.compute_full_neighbors)
compute_pointwise_neighbors = layered(_neigh_ops.compute_pointwise_neighbors)
compute_neighbors = layered(_neigh_ops.compute_neighbors)
