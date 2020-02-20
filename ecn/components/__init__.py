import contextlib
from typing import Tuple, Optional, Union, TypeVar, Generic, Dict
import numpy as np
import tensorflow as tf

from kblocks.ops.ragged import pre_batch_ragged, post_batch_ragged
from kblocks.ops import sparse as sparse_ops

from kblocks.ops import repeat as tf_repeat

from ecn.ops import spike as spike_ops
from ecn.ops import neighbors as neigh_ops

from ecn.layers import conv as conv_layers

from ecn.ops import ragged as ragged_ops
from ecn.ops import grid as grid_ops
from ecn import multi_graph as mg
import ecn.pub_sub as ps

BoolTensor = tf.Tensor
IntTensor = tf.Tensor
FloatTensor = tf.Tensor

BoolArray = np.ndarray
IntArray = np.ndarray

Lambda = tf.keras.layers.Lambda

# def Lambda(*args, **kwargs):
#     layer = tf.keras.layers.Lambda(*args, **kwargs)
#     if layer.name == 'lambda_6':
#         raise Exception()
#     return layer

T = TypeVar('T')

S0 = TypeVar('S', bound='Stream')
S1 = TypeVar('S', bound='Stream')

# def maybe_pad(values, diff):
#     padding = tf.maximum(diff, tf.zeros_like(diff))
#     return tf.pad(values, [[0, padding]])

#     # return tf.cond(diff > 0, lambda: tf.pad(values, [[0, diff]]),
#     #                lambda: values)


def pad_ragged(rt, padding):
    # assert (rt.ragged_rank == 1)
    flat_values = rt.values

    flat_values = tf.pad(rt.flat_values, [[0, padding]])
    starts, total = tf.split(rt.row_splits, [1, -1])
    row_splits = tf.concat((starts, total + padding), axis=0)

    # def if_true():
    #     flat_values = tf.pad(rt.flat_values, [[0, diff]])
    #     row_splits = tf.concat((rt.row_starts(), [batched_size]), axis=0)
    #     return flat_values, row_splits

    # def if_false():
    #     return rt.flat_values, rt.row_splits

    # flat_values, row_splits = tf.cond(diff > 0, if_true, if_false)
    return tf.RaggedTensor.from_row_splits(flat_values,
                                           row_splits,
                                           validate=False)


class Grid(object):

    def __init__(self, shape):
        self._static_shape = (None
                              if isinstance(shape, tf.Tensor) else tuple(shape))
        with mg.pre_batch_context():
            self._shape = tf.convert_to_tensor(shape, dtype=tf.int64)
        self._shape.shape.assert_has_rank(1)

    @property
    def static_shape(self) -> Optional[Tuple[int, ...]]:
        return self._static_shape

    @property
    def shape(self) -> IntTensor:
        return self._shape

    @property
    def ndims(self) -> int:
        return len(self._shape)

    def link(self, kernel_shape: Tuple[int, ...], strides,
             padding) -> 'GridNeighbors':
        # calculate the transpose and return it's transpose
        with mg.pre_batch_context():
            shape = self.static_shape or self.shape
            (partitions, indices, splits,
             out_shape) = grid_ops.sparse_neighborhood(shape, kernel_shape,
                                                       strides, padding)
            if not any(
                    isinstance(t, tf.Tensor)
                    for t in (strides, padding,
                              kernel_shape)) and self.static_shape is not None:
                out_shape = grid_ops.output_shape(np.array(self.static_shape),
                                                  np.array(kernel_shape),
                                                  np.array(strides),
                                                  np.array(padding))

            out_grid = Grid(out_shape)
            return GridNeighbors(out_grid, self, indices, splits, partitions,
                                 np.prod(kernel_shape, dtype=np.int64)).T

    def self_link(self, kernel_shape: Tuple[int, ...]) -> 'GridNeighbors':
        with mg.pre_batch_context():
            partitions, indices, splits = grid_ops.sparse_neighborhood_in_place(
                self.shape, kernel_shape)
            link = GridNeighbors(self, self, indices, splits, partitions,
                                 np.prod(kernel_shape, dtype=np.int64))
            link._transpose = link
        return link

    def partial_self_link(self, kernel_mask: BoolArray) -> 'GridNeighbors':
        with mg.pre_batch_context():
            num_partitions = np.count_nonzero(kernel_mask)
            (partitions, indices,
             splits) = grid_ops.sparse_neighborhood_from_mask_in_place(
                 self.shape, kernel_mask)
            link = GridNeighbors(self, self, indices, splits, partitions,
                                 num_partitions)
            link._transpose = link
            return link

    def ravel_indices(self, indices: IntTensor, axis=-1):
        indices.shape.assert_has_rank(2)
        with mg.pre_batch_context():
            return grid_ops.ravel_multi_index(indices, self._shape)

    def unravel_indices(self, indices: IntTensor, axis=-1):
        indices.shape.assert_has_rank(1)
        with mg.pre_batch_context():
            if axis == 0:
                return tf.unravel_index(indices, self._shape)
            else:
                return grid_ops.unravel_index_transpose(indices, self._shape)


class GridNeighbors(object):
    """Contains sparse connectedness information between two grids."""

    def __init__(self, in_grid: Grid, out_grid: Grid, indices: IntTensor,
                 splits: IntTensor, partitions: IntTensor, num_partitions: int):
        self._in_grid = in_grid
        self._partitions = partitions
        self._indices = indices
        self._splits = splits
        self._out_grid = out_grid
        self._num_partitions = num_partitions
        self._transpose = None

    @property
    def num_partitions(self) -> int:
        return self._num_partitions

    @property
    def transpose(self) -> 'GridNeighbors':
        if self._transpose is None:
            # create transpose
            with mg.pre_batch_context():
                indices, splits, partitions = ragged_ops.transpose_csr(
                    self.indices, self.splits, self.partitions)
                self._transpose = GridNeighbors(self._out_grid, self._in_grid,
                                                indices, splits, partitions,
                                                self.num_partitions)
                self._transpose._transpose = self
        return self._transpose

    @property
    def T(self):
        return self.transpose

    @property
    def in_grid(self) -> Grid:
        return self._in_grid

    @property
    def out_grid(self) -> Grid:
        return self._out_grid

    @property
    def indices(self) -> IntTensor:
        return self._indices

    @property
    def splits(self) -> IntTensor:
        return self._splits

    @property
    def partitions(self) -> IntTensor:
        return self._partitions


class Stream(object):
    _publisher = ps.Publisher()
    on_create: ps.Topic = _publisher.topic

    def __init__(self, times: IntTensor, min_mean_size: Optional[int] = None):
        mg.assert_is_pre_batch(times)
        self._min_mean_size = min_mean_size
        self._times = times
        with mg.pre_batch_context():
            self._example_size = tf.size(times, out_type=tf.int64)
        self._batched = False
        self._valid_size = None
        self._model_row_ends = None
        Stream._publisher.add(self)

    @property
    def batched(self) -> bool:
        return self._batched

    @property
    def example_size(self) -> IntTensor:
        return self._example_size

    @property
    def min_mean_size(self) -> Optional[int]:
        return self._min_mean_size

    @property
    def times(self) -> IntTensor:
        return self._times

    def _batch(self):
        if self._batched:
            return

        with mg.post_batch_context():
            self._row_lengths = mg.batch(self._example_size)
            self._row_splits = ragged_ops.lengths_to_splits(self._row_lengths)
            self._row_ends = self._row_splits[1:] - 1
            self._row_starts, total = tf.split(self._row_splits, [-1, 1],
                                               axis=0)
            self._batch_size = tf.size(self._row_lengths, out_type=tf.int64)
            self._valid_size = tf.squeeze(total, axis=0)

            if self._min_mean_size is None:
                self._batched_size = self._valid_size
                self._padding = None
            else:
                self._batched_size = tf.maximum(
                    self._batch_size * self.min_mean_size, self._valid_size)
                self._padding = self._batched_size - self._valid_size

        self._batched = True

    @property
    def row_splits(self) -> IntTensor:
        self._batch()
        return self._row_splits

    @property
    def row_lengths(self) -> IntTensor:
        self._batch()
        return self._row_lengths

    @property
    def row_starts(self) -> IntTensor:
        self._batch()
        return self._row_starts

    @property
    def batch_size(self) -> IntTensor:
        self._batch()
        return self._batch_size

    @property
    def batched_size(self) -> IntTensor:
        self._batch()
        return self._batched_size

    @property
    def padding(self) -> IntTensor:
        self._batch()
        return self._padding

    @property
    def valid_size(self) -> IntTensor:
        return self._valid_size

    def prepare_model_inputs(self, features) -> tf.RaggedTensor:
        mg.assert_is_pre_batch(features)
        with mg.pre_batch_context():
            features = pre_batch_ragged(features)
        with mg.post_batch_context():
            features = post_batch_ragged(mg.batch(features), validate=False)

            if self._min_mean_size is not None:
                features = pad_ragged(features, self.padding)
        features = mg.model_input(features)
        return features

    def prepare_labels(self, labels) -> Dict[str, tf.Tensor]:
        mg.assert_is_pre_batch(labels)
        labels = mg.batch(labels)
        ret = {'final': labels}
        with mg.post_batch_context():
            row_lengths = self.row_lengths
            labels = tf_repeat(labels, row_lengths, axis=0)
            if self._min_mean_size is not None:
                labels = tf.pad(labels, [[0, self.padding]])
            ret['stream'] = labels
        return ret

    def prepare_weights(self, weights=None) -> Dict[str, tf.Tensor]:
        ret = {}
        if weights is not None:
            mg.assert_is_pre_batch(weights)
            weights = mg.batch(weights)
            ret['final'] = weights
        else:
            weights = 1
        with mg.post_batch_context():
            row_lengths = self.row_lengths
            weights = weights / (self._batch_size * row_lengths)
            weights = tf_repeat(weights, row_lengths, axis=0)
            if self._min_mean_size is not None:
                weights = tf.pad(weights, [[0, self._padding]])
            ret['stream'] = weights
        return ret

    @property
    def model_row_ends(self):
        if self._model_row_ends is None:
            self._model_row_ends = mg.model_input(self._row_ends)
        return self._model_row_ends

    @classmethod
    def from_config(cls, config):
        return Stream(**config)

    def get_config(self):
        return dict(times=self.times, min_mean_size=self.min_mean_size)

    def _rebuild(self: T, **kwargs) -> T:
        config = self.get_config()
        config.update(kwargs)
        return self.__class__.from_config(config)

    def mask(self, mask: BoolTensor) -> 'Stream':
        mg.assert_is_pre_batch(mask)
        with mg.pre_batch_context():
            return self._rebuild(times=tf.boolean_mask(self.times, mask))

    def gather(self, indices: IntTensor) -> 'Stream':
        mg.assert_is_pre_batch(indices)
        with mg.pre_batch_context():
            return self._rebuild(times=tf.gather(self.times, indices))


class SpatialStream(Stream):

    def __init__(self,
                 grid: Union[Grid, IntTensor, IntArray, Tuple[int, ...]],
                 times: IntTensor,
                 coords: IntTensor,
                 min_mean_size: Optional[int] = None):
        assert (times.shape.ndims == 1)
        for t in (times, coords):
            mg.assert_is_pre_batch(t)

        self._grid = grid if isinstance(grid, Grid) else Grid(grid)
        self._times = times
        if coords.shape.ndims == 1:
            self._shaped_coords = None
            self._coords = coords
        elif coords.shape.ndims == 2:
            self._shaped_coords = coords
            self._coords = self._grid.ravel_indices(coords)
        else:
            raise ValueError('coords must be rank 1 or 2, got {}'.format(
                coords.shape))

        super().__init__(times=times, min_mean_size=min_mean_size)

    @classmethod
    def from_config(cls, config):
        return SpatialStream(**config)

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def coords(self) -> IntTensor:
        return self._coords

    @property
    def shaped_coords(self):
        if self._shaped_coords is None:
            self._shaped_coords = self.grid.unravel_indices(self.coords)
        return self._shaped_coords

    def get_config(self):
        config = super().get_config()
        config.update(dict(coords=self.coords, grid=self.grid))
        return config

    def mask(self, mask: BoolTensor) -> 'SpatialStream':
        mg.assert_is_pre_batch(mask)
        with mg.pre_batch_context():
            return self._rebuild(times=tf.boolean_mask(self.times, mask),
                                 coords=tf.boolean_mask(self.coords, mask))

    def gather(self, indices: IntTensor) -> 'SpatialStream':
        mg.assert_is_pre_batch(indices)
        with mg.pre_batch_context():
            return self._rebuild(times=tf.gather(self.times, indices),
                                 coords=tf.gather(self.coords, indices))


class Convolver(Generic[S0, S1]):
    _publisher = ps.Publisher()
    on_create: ps.Topic = _publisher.topic

    def __init__(self, in_stream: S0, out_stream: S1,
                 partitions: Optional[IntTensor], indices: IntTensor,
                 splits: IntTensor, decay_time: int, num_partitions: int):
        for t in indices, splits:
            mg.assert_is_pre_batch(t)
        if partitions is None:
            assert (num_partitions == 1)
        else:
            mg.assert_is_pre_batch(partitions)
        self._decay_time = decay_time
        self._in_stream = in_stream
        self._out_stream = out_stream
        self._partitions = partitions
        self._indices = indices
        self._splits = splits
        self._num_partitions = num_partitions

        self._batched_dts = None
        self._model_dts = None
        Convolver._publisher.add(self)

    def _rebuild(self, **kwargs) -> 'Convolver':
        params = dict(
            num_partitions=self._num_partitions,
            in_stream=self.in_stream,
            out_stream=self.out_stream,
            partitions=self.partitions,
            indices=self.indices,
            splits=self.splits,
            decay_time=self.decay_time,
        )
        params.update(kwargs)
        return Convolver(**params)

    def reindex(self, indices: IntTensor) -> 'Convolver':
        mg.assert_is_pre_batch(indices)
        with mg.pre_batch_context():
            return self._rebuild(indices=tf.gather(indices, self.indices))

    def gather(self, indices: IntTensor) -> 'Convolver':
        mg.assert_is_pre_batch(indices)

        with mg.pre_batch_context():
            values = tf.stack((self._partitions, self._indices), axis=-1)
            values, splits = ragged_ops.gather_rows(values, self._splits,
                                                    indices)
            partitions, indices = tf.unstack(values, axis=-1)

        return self._rebuild(partitions=partitions,
                             indices=indices,
                             splits=splits)

    @property
    def partitions(self) -> IntTensor:
        return self._partitions

    @property
    def indices(self) -> IntTensor:
        """Prebatch indices into in_stream used."""
        return self._indices

    @property
    def splits(self) -> IntTensor:
        return self._splits

    @property
    def in_stream(self) -> S0:
        return self._in_stream

    @property
    def out_stream(self) -> S1:
        return self._out_stream

    @property
    def decay_time(self) -> int:
        return self._decay_time

    @property
    def num_partitions(self):
        return self._num_partitions

    @property
    def batched(self):
        return self._batched_dts is not None

    @property
    def batched_dts(self):
        if not self.batched:
            num_partitions = self.num_partitions
            if num_partitions == 1:
                self._batched_dts = self._batch_single_partition()
            else:
                self._batched_dts = self._batch_multi_partition()
        return self._batched_dts

    def _batch_single_partition(self):

        assert (self.num_partitions == 1)
        in_stream = self._in_stream
        out_stream = self._out_stream
        with mg.pre_batch_context():
            rowids = ragged_ops.splits_to_ids(self._splits)
            dt = tf.cast(
                tf.gather(out_stream.times, rowids) -
                tf.gather(in_stream.times, self._indices),
                tf.float32) / self.decay_time
            indices = tf.stack((rowids, self._indices), axis=-1)
            # indices = tf.cast(indices, tf.int32)
            dt = tf.SparseTensor(indices, dt, (-1, -1))

        with mg.post_batch_context():
            dt = mg.batch(dt)
            batch_size = dt.dense_shape[0]

            dense_shape = tf.stack(
                (out_stream.batched_size, in_stream.batched_size), axis=0)

            # block diagonalize
            b, i, j = tf.unstack(dt.indices, axis=-1)
            i = i + tf.gather(out_stream.row_starts, b)
            j = j + tf.gather(in_stream.row_starts, b)

            dt = tf.SparseTensor(tf.stack((i, j), axis=-1), dt.values,
                                 dense_shape)
            return (dt,)

    def _batch_multi_partition(self):
        num_partitions = self.num_partitions

        assert (num_partitions > 1)
        assert (not self.batched)
        in_stream = self._in_stream
        out_stream = self._out_stream

        with mg.pre_batch_context():
            rowids = ragged_ops.splits_to_ids(self._splits)
            # don't divide dt here so we can stack later before partitioning
            dt = (tf.gather(out_stream.times, rowids) -
                  tf.gather(in_stream.times, self._indices))
            ijts = tf.dynamic_partition(
                tf.stack((rowids, self._indices, dt), axis=-1),
                tf.cast(self._partitions, tf.int32), num_partitions)
            dts = []
            for ijt in ijts:
                ij, dt = tf.split(ijt, [2, 1], axis=-1)
                dt = (tf.cast(tf.squeeze(dt, axis=-1), tf.float32) /
                      self._decay_time)
                # ij = tf.cast(ij, tf.int32)
                dts.append(tf.SparseTensor(ij, dt, (-1, -1)))

        with mg.post_batch_context():
            batched_dts = []
            dts = [mg.batch(dt) for dt in dts]
            # batch_size = dts[0].dense_shape[0]
            dense_shape = tf.stack(
                (out_stream.batched_size, in_stream.batched_size), axis=0)
            for dt in dts:
                # block diagonalize
                b, i, j = tf.unstack(dt.indices, axis=-1)
                i = i + tf.gather(out_stream.row_starts, b)
                j = j + tf.gather(in_stream.row_starts, b)
                # bij = tf.stack((b, i, j), axis=-1)
                bij = tf.stack((i, j), axis=-1)

                batched_dts.append(tf.SparseTensor(bij, dt.values, dense_shape))
            return tuple(batched_dts)

    @property
    def model_dts(self):
        if self._model_dts is None:
            # self._model_dts = tuple(
            #     tf.keras.layers.Lambda(sparse_ops.remove_leading_dim)(
            #         mg.model_input(dt)) for dt in self.batched_dts)
            self._model_dts = tuple(
                mg.model_input(dt) for dt in self.batched_dts)
        return self._model_dts

    def convolve(self, features: Optional[tf.Tensor], filters: int,
                 temporal_kernel_size: int, **kwargs):
        if features is not None:
            mg.assert_is_model_tensor(features)
        if self.num_partitions == 1:
            assert (len(self.model_dts) == 1)
            return conv_layers.temporal_event_conv(
                features=features,
                dt=self.model_dts[0],
                filters=filters,
                temporal_kernel_size=temporal_kernel_size,
                **kwargs)
        else:
            return conv_layers.spatio_temporal_event_conv(
                features=features,
                dt=self.model_dts,
                filters=filters,
                temporal_kernel_size=temporal_kernel_size,
                spatial_kernel_size=self.num_partitions,
                **kwargs)


def spike_threshold(stream: SpatialStream,
                    link: GridNeighbors,
                    decay_time: int,
                    threshold: float = 1.,
                    reset_potential: float = -1.,
                    min_mean_size: Optional[int] = None) -> SpatialStream:
    assert (stream.grid == link.in_grid)
    with mg.pre_batch_context():
        times, coords = spike_ops.spike_threshold(
            stream.times,
            stream.coords,
            link.indices,
            link.splits,
            decay_time=decay_time,
            threshold=threshold,
            reset_potential=reset_potential)
    return SpatialStream(link.out_grid,
                         times,
                         coords,
                         min_mean_size=min_mean_size)


def global_spike_threshold(stream: Stream,
                           decay_time: int,
                           threshold: float = 1.,
                           reset_potential: float = -1,
                           min_mean_size: Optional[int] = None) -> Stream:
    with mg.pre_batch_context():
        time = spike_ops.global_spike_threshold(stream.times,
                                                decay_time=decay_time,
                                                threshold=threshold,
                                                reset_potential=reset_potential)
    return Stream(time, min_mean_size=min_mean_size)


def spatio_temporal_convolver(grid_neighbors: GridNeighbors,
                              in_stream: SpatialStream,
                              out_stream: SpatialStream,
                              decay_time: int,
                              spatial_buffer_size: int,
                              max_decays: int = 4
                             ) -> Convolver[SpatialStream, SpatialStream]:
    assert (grid_neighbors.in_grid == in_stream.grid)
    assert (grid_neighbors.out_grid == out_stream.grid)
    grid_neighbors = grid_neighbors.T
    with mg.pre_batch_context():
        partitions, indices, splits = neigh_ops.compute_neighbors(
            in_stream.times, in_stream.coords, out_stream.times,
            out_stream.coords, grid_neighbors.partitions,
            grid_neighbors.indices, grid_neighbors.splits, spatial_buffer_size,
            decay_time * max_decays)
    return Convolver(
        num_partitions=grid_neighbors.num_partitions,
        in_stream=in_stream,
        out_stream=out_stream,
        partitions=partitions,
        indices=indices,
        splits=splits,
        decay_time=decay_time,
    )


def pointwise_convolver(in_stream,
                        out_stream,
                        decay_time: int,
                        spatial_buffer_size: int,
                        max_decays: int = 4
                       ) -> Convolver[SpatialStream, SpatialStream]:
    assert (in_stream.grid == out_stream.grid)
    with mg.pre_batch_context():
        indices, splits = neigh_ops.compute_pointwise_neighbors(
            in_stream.times,
            in_stream.coords,
            out_stream.times,
            out_stream.coords,
            event_duration=decay_time * max_decays,
            spatial_buffer_size=spatial_buffer_size)
    return Convolver(num_partitions=1,
                     in_stream=in_stream,
                     out_stream=out_stream,
                     partitions=None,
                     indices=indices,
                     splits=splits,
                     decay_time=decay_time)


def flatten_convolver(in_stream: SpatialStream,
                      out_stream: Stream,
                      decay_time: int,
                      max_decays: int = 4,
                      num_partitions: Optional[int] = None
                     ) -> Convolver[SpatialStream, Stream]:
    assert (not isinstance(out_stream, SpatialStream))

    if num_partitions is None:
        if in_stream.grid.static_shape is None:
            raise ValueError(
                'Either input_stream grid must be static or num_partitions must be '
                'provided')
        num_partitions_ = np.prod(in_stream.grid.static_shape)
    else:
        num_partitions_ = num_partitions
    with mg.pre_batch_context():
        partitions, indices, splits = neigh_ops.compute_full_neighbors(
            in_times=in_stream.times,
            in_coords=in_stream.coords,
            out_times=out_stream.times,
            event_duration=decay_time * max_decays,
        )
    return Convolver[SpatialStream, Stream](num_partitions=num_partitions_,
                                            in_stream=in_stream,
                                            out_stream=out_stream,
                                            partitions=partitions,
                                            indices=indices,
                                            splits=splits,
                                            decay_time=decay_time)


def temporal_convolver(in_stream: Stream,
                       out_stream: Stream,
                       decay_time: int,
                       max_decays: int = 4) -> Convolver[Stream, Stream]:
    assert (not isinstance(in_stream, SpatialStream))
    assert (not isinstance(out_stream, SpatialStream))
    with mg.pre_batch_context():
        indices, splits = neigh_ops.compute_pooled_neighbors(
            in_stream.times,
            out_stream.times,
            event_duration=max_decays * decay_time)
    return Convolver(num_partitions=1,
                     in_stream=in_stream,
                     out_stream=out_stream,
                     partitions=None,
                     indices=indices,
                     splits=splits,
                     decay_time=decay_time)


def stream_accumulator():
    return ps.accumulator(Stream.on_create)


def convolver_accumulator():
    return ps.accumulator(Convolver.on_create)
