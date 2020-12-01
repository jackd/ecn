from wtftf.meta import layered

from ecn.ops import lif as _lif_ops

leaky_integrate_and_fire = layered(_lif_ops.leaky_integrate_and_fire)
spatial_leaky_integrate_and_fire = layered(_lif_ops.spatial_leaky_integrate_and_fire)
