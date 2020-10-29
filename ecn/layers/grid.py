from ecn.ops import grid as _grid_ops
from wtftf.meta import layered

ravel_multi_index = layered(_grid_ops.ravel_multi_index)
unravel_index_transpose = layered(_grid_ops.unravel_index_transpose)
base_grid_coords = layered(_grid_ops.base_grid_coords)
output_shape = layered(_grid_ops.output_shape)
grid_coords = layered(_grid_ops.grid_coords)
shift_grid_coords = layered(_grid_ops.shift_grid_coords)
sparse_neighborhood = layered(_grid_ops.sparse_neighborhood)
sparse_neighborhood_in_place = layered(_grid_ops.sparse_neighborhood_in_place)
sparse_neighborhood_from_mask = layered(_grid_ops.sparse_neighborhood_from_mask)
sparse_neighborhood_from_mask_in_place = layered(
    _grid_ops.sparse_neighborhood_from_mask_in_place
)
