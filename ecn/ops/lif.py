import functools
from typing import Tuple

import tensorflow as tf

from numba_stream import lif as _np_lif

IntTensor = tf.Tensor
FloatTensor = tf.Tensor


def leaky_integrate_and_fire(
    times: IntTensor,
    decay_time: int,
    threshold: float = 1.0,
    reset_potential: float = -1.0,
    max_out_events: int = -1,
) -> IntTensor:
    out_times = tf.numpy_function(
        functools.partial(
            _np_lif.leaky_integrate_and_fire,
            max_out_events=max_out_events,
            decay_time=decay_time,
            threshold=threshold,
            reset_potential=reset_potential,
        ),
        (times,),
        times.dtype,
    )
    out_times.set_shape((None,))
    return out_times


def spatial_leaky_integrate_and_fire(
    times: IntTensor,
    coords: IntTensor,
    grid_indices: IntTensor,
    grid_splits: IntTensor,
    decay_time: int,
    threshold: float = 1.0,
    reset_potential: float = -1.0,
    out_size: int = -1,
) -> Tuple[IntTensor, IntTensor]:
    assert isinstance(times, tf.Tensor)
    assert isinstance(coords, tf.Tensor)
    assert isinstance(grid_indices, tf.Tensor)
    assert isinstance(grid_splits, tf.Tensor)

    out_times, out_coords = tf.numpy_function(
        functools.partial(
            _np_lif.spatial_leaky_integrate_and_fire,
            decay_time=decay_time,
            threshold=threshold,
            reset_potential=reset_potential,
            out_size=out_size,
        ),
        (times, coords, grid_indices, grid_splits),
        (times.dtype, coords.dtype),
    )
    out_times.set_shape((None,))
    out_coords.set_shape((None,))
    return out_times, out_coords
