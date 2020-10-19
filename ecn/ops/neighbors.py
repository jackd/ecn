import functools
from typing import Optional, Tuple

import tensorflow as tf

from numba_stream import neighbors as _np_neigh

IntTensor = tf.Tensor
BoolTensor = tf.Tensor


def compute_pooled_neighbors(
    in_times: IntTensor,
    out_times: IntTensor,
    event_duration: int,
    max_neighbors: int = -1,
) -> Tuple[IntTensor, IntTensor]:
    # purely temporal
    indices, splits = tf.numpy_function(
        functools.partial(
            _np_neigh.compute_pooled_neighbors,
            event_duration=event_duration,
            max_neighbors=max_neighbors,
        ),
        [in_times, out_times],
        [in_times.dtype] * 2,
    )
    for t in indices, splits:
        t.set_shape((None,))
    return indices, splits


def compute_full_neighbors(
    in_times: IntTensor,
    in_coords: IntTensor,
    out_times: IntTensor,
    event_duration: int = -1,
    max_neighbors: int = -1,
) -> Tuple[IntTensor, IntTensor, IntTensor]:
    # flatten conv
    partitions, indices, splits = tf.numpy_function(
        functools.partial(
            _np_neigh.compute_full_neighbors,
            event_duration=event_duration,
            max_neighbors=max_neighbors,
        ),
        [in_times, in_coords, out_times],
        [in_times.dtype] * 3,
    )
    for t in partitions, indices, splits:
        t.set_shape((None,))
    return partitions, indices, splits


def compute_pointwise_neighbors(
    in_times: IntTensor,
    in_coords: IntTensor,
    out_times: IntTensor,
    out_coords: IntTensor,
    spatial_buffer_size: int,
    event_duration: Optional[int] = None,
) -> Tuple[IntTensor, IntTensor]:
    # 1x1 conv
    indices, splits = tf.numpy_function(
        functools.partial(
            _np_neigh.compute_pointwise_neighbors,
            event_duration=event_duration,
            spatial_buffer_size=spatial_buffer_size,
        ),
        [in_times, in_coords, out_times, out_coords],
        [in_times.dtype] * 2,
    )
    for t in indices, splits:
        t.set_shape((None,))
    return indices, splits


def compute_neighbors(
    in_times: IntTensor,
    in_coords: IntTensor,
    out_times: IntTensor,
    out_coords: IntTensor,
    grid_partitions: IntTensor,
    grid_indices: IntTensor,
    grid_splits: IntTensor,
    spatial_buffer_size: int,
    event_duration: Optional[int] = None,
) -> Tuple[IntTensor, IntTensor, IntTensor]:
    assert isinstance(in_times, tf.Tensor)
    assert isinstance(in_coords, tf.Tensor)
    assert isinstance(out_times, tf.Tensor)
    assert isinstance(out_coords, tf.Tensor)
    assert isinstance(grid_partitions, tf.Tensor)
    assert isinstance(grid_indices, tf.Tensor)
    assert isinstance(grid_splits, tf.Tensor)
    assert isinstance(spatial_buffer_size, int)
    if event_duration is not None:
        assert isinstance(event_duration, int)

    fn = functools.partial(
        _np_neigh.compute_neighbors,
        event_duration=event_duration,
        spatial_buffer_size=spatial_buffer_size,
    )
    partitions, indices, splits = tf.numpy_function(
        fn,
        (
            in_times,
            in_coords,
            out_times,
            out_coords,
            grid_partitions,
            grid_indices,
            grid_splits,
        ),
        (grid_partitions.dtype, grid_indices.dtype, grid_splits.dtype),
    )
    for t in (partitions, indices, splits):
        t.set_shape((None,))
    return partitions, indices, splits
