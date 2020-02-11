# import os
# os.environ['NUMBA_DISABLE_JIT'] = '1'
from typing import Tuple
import numpy as np
import numba as nb
import ecn.np_utils.spike as spike_ops
import ecn.np_utils.neighbors as neigh_ops

IntArray = np.ndarray

# class SpatialConvArgs(NamedTuple):
#     out_times: IntArray
#     out_coords: IntArray
#     indices: IntArray
#     splits: IntArray

# class GlobalConvArgs(NamedTuple):
#     out_times: IntArray
#     indices: IntArray
#     splits: IntArray


@nb.njit()
def preprocess_spatial_conv(times: IntArray,
                            coords: IntArray,
                            stride: int,
                            decay_time: int,
                            event_duration: int,
                            spatial_buffer_size: int,
                            threshold: float = 2.,
                            reset_potential: float = -1.):
    pooled_coords = coords // stride
    out_times, out_coords = spike_ops.spike_threshold(
        times=times,
        coords=pooled_coords,
        threshold=threshold,
        reset_potential=reset_potential,
        decay_time=decay_time)

    indices, splits = neigh_ops.compute_neighbors(
        in_times=times,
        in_coords=pooled_coords,
        out_times=out_times,
        out_coords=out_coords,
        event_duration=event_duration,
        spatial_buffer_size=spatial_buffer_size)
    return out_times, out_coords, indices, splits


@nb.njit()
def preprocess_global_conv(times: IntArray,
                           decay_time: int,
                           event_duration: int,
                           threshold: float = 2.,
                           reset_potential: float = -1.
                          ) -> Tuple[IntArray, IntArray, IntArray]:
    out_times = spike_ops.global_spike_threshold(
        times, decay_time, threshold=threshold, reset_potential=reset_potential)
    indices, splits = neigh_ops.compute_global_neighbors(
        in_times=times, out_times=out_times, event_duration=event_duration)
    return out_times, indices, splits


@nb.njit()
def _preprocess_network(times: IntArray,
                        coords: IntArray,
                        stride: int,
                        decay_time: int,
                        event_duration: int,
                        spatial_buffer_size: int,
                        num_layers: int,
                        threshold: float = 2.,
                        reset_potential: float = -1.):
    for _ in range(num_layers):
        out_times, out_coords, indices, splits = preprocess_spatial_conv(
            times=times,
            coords=coords,
            stride=stride,
            decay_time=decay_time,
            event_duration=event_duration,
            spatial_buffer_size=spatial_buffer_size,
            threshold=threshold,
            reset_potential=reset_potential)
        yield out_times, out_coords, indices, splits
        times = out_times
        coords = out_coords
        event_duration *= 2
        decay_time *= 2


# @nb.njit()
def preprocess_network(
        times: IntArray,
        coords: IntArray,
        stride: int,
        decay_time: int,
        event_duration: int,
        spatial_buffer_size: int,
        num_layers: int,
        threshold: float = 2.,
        reset_potential: float = -1.
):  # -> Tuple[Tuple[SpatialConvArgs, ...], GlobalConvArgs]:
    layer_args = list(
        _preprocess_network(
            times=times,
            coords=coords,
            stride=stride,
            decay_time=decay_time,
            event_duration=event_duration,
            spatial_buffer_size=spatial_buffer_size,
            num_layers=num_layers,
            threshold=threshold,
            reset_potential=reset_potential,
        ))
    times = layer_args[-1][0]
    time_factor = 2**num_layers
    decay_time *= time_factor
    event_duration *= time_factor
    global_args = preprocess_global_conv(
        times=times,
        decay_time=decay_time,
        event_duration=event_duration,
        threshold=threshold,
        reset_potential=reset_potential,
    )
    return layer_args, global_args


if __name__ == '__main__':
    from events_tfds.events.nmnist import NMNIST
    ds = NMNIST().as_dataset(split='train', as_supervised=True)
    for example, label in ds.take(1):
        coords = example['coords'].numpy()
        times = example['time'].numpy()
        preprocess_network(times,
                           coords,
                           stride=2,
                           decay_time=10000,
                           event_duration=10000 * 8,
                           spatial_buffer_size=32,
                           num_layers=3,
                           reset_potential=-2.)
        print('worked!')
