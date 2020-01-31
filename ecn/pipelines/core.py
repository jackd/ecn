from typing import Optional, Callable
import numpy as np
import tensorflow as tf
from kblocks.framework.problems.pipelines import DataPipeline
from typing import NamedTuple
from ecn.np_utils import spike

IntArray = np.ndarray
FloatArray = np.ndarray
AUTOTUNE = tf.data.experimental.AUTOTUNE


class CustomPipeline(DataPipeline):

    def __init__(self,
                 batch_size: int,
                 pre_cache_map: Optional[Callable] = None,
                 pre_batch_map: Optional[Callable] = None,
                 post_batch_map: Optional[Callable] = None,
                 cache_path: str = '',
                 shuffle_buffer: Optional[int] = None,
                 repeats: Optional[int] = None,
                 prefetch_buffer: int = AUTOTUNE,
                 num_parallel_calls: int = AUTOTUNE):

        self._pre_cache_map = pre_cache_map
        self._pre_batch_map = pre_batch_map
        self._post_batch_map = post_batch_map

        self._batch_size = batch_size
        self._cache_path = cache_path
        self._shuffle_buffer = shuffle_buffer
        self._repeats = repeats
        self._prefetch_buffer = prefetch_buffer
        self._num_parallel_calls = num_parallel_calls

    def get_config(self):
        return dict(
            pre_cache_map=self._pre_cache_map,
            pre_batch_map=self._post_batch_map,
            post_batch_map=self._post_batch_map,
            batch_size=self._batch_size,
            cache_path=self._cache_path,
            shuffle_buffer=self._shuffle_buffer,
            repeats=self._repeats,
            prefetch_buffer=self._prefetch_buffer,
            num_parallel_calls=self._num_parallel_calls,
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __call__(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        if self._pre_cache_map is not None:
            dataset = dataset.map(self._pre_cache_map, self._num_parallel_calls)
            dataset = dataset.cache(self._cache_path)

        if self._repeats != -1:
            dataset = dataset.repeat(self._repeats)
        if self._shuffle_buffer is not None:
            dataset = dataset.shuffle(self._shuffle_buffer)

        if self._pre_batch_map is not None:
            dataset = dataset.map(self._pre_batch_map, self._num_parallel_calls)

        dataset = dataset.batch(self.batch_size)

        if self._post_batch_map is not None:
            dataset = dataset.map(self._post_batch_map,
                                  self._num_parallel_calls)

        if self._prefetch_buffer is not None:
            dataset = dataset.prefetch(self._prefetch_buffer)
        return dataset


# class SpikeSpec(NamedTuple):
#     decay_time: int
#     max_out_events: int
#     threshold: float
#     reset_potential: float

# def spike_spec(decay_time: int,
#                max_out_events: int,
#                threshold: float = 2.,
#                reset_potential: float = -1.):
#     return SpikeSpec(decay_time, max_out_events, threshold, reset_potential)

# class NeighborsSpec(NamedTuple):
#     event_duration: int
#     spatial_buffer_size: int
#     max_neighbors: int

# class SpikeConvBuilder(object):

#     def __init__(self, stride, spike_spec: SpikeSpec,
#                  neighbors_spec: NeighborsSpec):
#         self.stride = stride
#         self.spike_spec = spike_spec
#         self.neighbors_spec = neighbors_spec

#     def pre_cache_map_np(self, in_times: IntArray, in_coords: IntArray):
#         stride = self.stride
#         if self.stride != 1:
#             coords = in_coords // stride
#         ss = self.spike_spec
#         out_times, out_coords, out_events = spike.spike_threshold(
#             in_times, coords, ss.decay_time, ss.max_out_events, ss.threshold,
#             ss.reset_potential)
