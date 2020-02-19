from ecn.np_utils.buffer import indices
from typing import Tuple, Optional, Union
import numpy as np
import tensorflow as tf

from kblocks.ops.ragged import pre_batch_ragged, post_batch_ragged
from kblocks.ops import sparse as sparse_ops

from kblocks.extras.layers import sparse as sparse_layers
from kblocks.extras.layers.wrapper import as_lambda

from ..ops import spike as spike_ops
from ..ops import ragged as ragged_ops
from ..ops import neighbors as neigh_ops
from ..ops import grid as grid_ops
from .. import meta
from ..layers import conv as conv_layers

BoolTensor = tf.Tensor
IntTensor = tf.Tensor
FloatTensor = tf.Tensor
BuiltModels = meta.BuiltModels

Lambda = tf.keras.layers.Lambda

# def maybe_pad(values, diff):
#     return tf.cond(diff > 0, lambda: tf.pad(values, [[0, diff]]),
#                    lambda: values)


def maybe_pad_ragged(rt, min_mean_size):
    assert (rt.ragged_rank == 1)
    flat_values = rt.values
    min_size = rt.nrows() * min_mean_size
    valid_size = tf.shape(flat_values, out_type=min_size.dtype)[0]
    diff = min_size - valid_size

    def if_true():
        flat_values = tf.pad(rt.flat_values, [[0, diff]])
        row_splits = tf.concat((rt.row_starts(), [min_size]), axis=0)
        return flat_values, row_splits

    def if_false():
        return rt.flat_values, rt.row_splits

    flat_values, row_splits = tf.cond(diff > 0, if_true, if_false)
    return tf.RaggedTensor.from_row_splits(flat_values, row_splits), valid_size


class Grid(object):

    def __init__(self, shape: IntTensor):
        self._shape = tf.convert_to_tensor(shape, dtype=tf.int64)
        self._shape.shape.assert_has_rank(1)

    @property
    def shape(self):
        return self._shape

    @property
    def ndims(self):
        return len(self._shape)

    def link(self, kernel_shape: Tuple[int, ...], strides,
             padding) -> 'GridNeighbors':
        # calculate the transpose and return it's transpose
        partitions, indices, splits, out_shape = grid_ops.sparse_neighborhood(
            self.shape, kernel_shape, strides, padding)
        out_grid = Grid(out_shape)
        return GridNeighbors(out_grid, self, indices, splits, partitions,
                             np.prod(kernel_shape)).T

    def self_link(self, kernel_shape: Tuple[int, ...]) -> 'GridNeighbors':
        partitions, indices, splits = grid_ops.sparse_neighborhood_in_place(
            self.shape, kernel_shape)
        link = GridNeighbors(self, self, indices, splits, partitions,
                             np.prod(kernel_shape))
        link._transpose = link
        return link

    def ravel_indices(self, indices: IntTensor, axis=-1):
        indices.shape.assert_has_rank(2)
        return as_lambda(grid_ops.ravel_multi_index, indices, self._shape)

    def unravel_indices(self, indices: IntTensor, axis=-1):
        indices.shape.assert_has_rank(1)
        if axis == 0:
            return as_lambda(tf.unravel_index, indices, self._shape)
        else:
            return as_lambda(grid_ops.unravel_index_transpose, indices,
                             self._shape)


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

    def __init__(self,
                 grid: Union[Grid, IntTensor],
                 times: IntTensor,
                 coords: IntTensor,
                 min_mean_size: Optional[int] = None):
        assert (times.shape.ndims == 1)
        for t in (times, coords):
            meta.set_mark(t, BuiltModels.PRE_BATCH)

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

        self._min_mean_size = min_mean_size
        self._example_size = as_lambda(tf.size, times, out_type=tf.int64)

        self._batched = False
        self._batched_times = None
        self._valid_size = None

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
    def grid(self) -> Grid:
        return self._grid

    @property
    def coords(self) -> IntTensor:
        return self._coords

    @property
    def times(self) -> IntTensor:
        return self._times

    @property
    def shaped_coords(self):
        if self._shaped_coords is None:
            self._shaped_coords = self.grid.unravel_indices(self.coords)
        return self._shaped_coords

    def _batch(self):
        if self._batched:
            return
        self._row_lengths = meta.batch(self._example_size)
        self._row_splits = as_lambda(ragged_ops.lengths_to_splits,
                                     self._row_lengths)
        self._row_starts, total = as_lambda(tf.split,
                                            self._row_splits, [-1, 1],
                                            axis=0)
        self._batch_size = as_lambda(tf.size,
                                     self._row_starts,
                                     out_type=tf.int64)
        self._valid_size = as_lambda(tf.squeeze, total, axis=0)
        if self.min_mean_size is None:
            self._batched_size = self._valid_size
        else:
            self._batched_size = as_lambda(
                tf.maximum, self._batch_size * self.min_mean_size,
                self._valid_size)
        self._batched = True

    @property
    def row_splits(self) -> IntTensor:
        self._batch()
        return self._row_splits

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
    def valid_size(self) -> IntTensor:
        return self._valid_size

    def prepare_features(self, features) -> tf.RaggedTensor:
        meta.set_mark(features, BuiltModels.PRE_BATCH)
        features = as_lambda(post_batch_ragged,
                             meta.batch(as_lambda(pre_batch_ragged, features)))
        if self._min_mean_size is not None:
            features, valid_size = as_lambda(maybe_pad_ragged, features,
                                             self._min_mean_size)
            del valid_size
        return features

    def _rebuild(self, **kwargs):
        params = dict(
            grid=self.grid,
            times=self.times,
            coords=self.coords,
            min_mean_size=self.min_mean_size,
        )
        params.update(kwargs)
        return Stream(**params)

    def mask(self, mask: BoolTensor) -> 'Stream':
        meta.set_mark(mask, BuiltModels.PRE_BATCH)
        return self._rebuild(times=tf.boolean_mask(self.times, mask),
                             coords=tf.boolean_mask(self.coords, mask))

    # @property
    # def batched_times(self) -> tf.RaggedTensor:
    #     if self._batched_times is None:
    #         self._batched_times = post_batch_ragged(
    #             meta.batch(pre_batch_ragged(self._times)))
    #     return self._batched_times

    # @property
    # def model_times(self) -> tf.RaggedTensor:
    #     return self.model_times


class Convolver(object):

    def __init__(self, in_stream: Stream, out_stream: Stream,
                 partitions: IntTensor, indices: IntTensor, splits: IntTensor,
                 decay_time: int, num_partitions: int):
        for t in partitions, indices, splits:
            meta.set_mark(t, BuiltModels.PRE_BATCH)
        self._decay_time = decay_time
        self._in_stream = in_stream
        self._out_stream = out_stream
        self._partitions = partitions
        self._indices = indices
        self._splits = splits
        self._num_partitions = num_partitions

        self._batched_dts = None
        self._model_dts = None

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
        return Convolver(**kwargs)

    def reindex(self, indices: IntTensor) -> 'Convolver':
        meta.set_mark(indices, BuiltModels.PRE_BATCH)
        return self._rebuild(indices=tf.gather(indices, self.indices))

    def gather(self, indices: IntTensor) -> 'Convolver':
        values = tf.stack((self.partitions, self.indices), axis=-1)
        values, splits = as_lambda(neigh_ops.gather_ragged_rows, values,
                                   self.splits)
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
    def in_stream(self) -> Stream:
        return self._in_stream

    @property
    def out_stream(self) -> Stream:
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
            in_stream = self._in_stream
            out_stream = self._out_stream
            num_partitions = self.num_partitions
            rowids = ragged_ops.splits_to_ids(self._splits)
            dt = (tf.gather(out_stream.times, rowids) -
                  tf.gather(in_stream.times, self._indices))
            dt = as_lambda(
                tf.SparseTensor,
                tf.stack((rowids, self._indices, self._partitions), axis=-1),
                dt, (-1, -1, num_partitions))

            in_size = self._in_stream.batched_size
            out_size = self._out_stream.batched_size
            dt = meta.batch(dt)
            batch_size = dt.dense_shape[0]

            dense_shape = (batch_size, out_size, in_size)
            batched_dts = []

            # block diagonalize
            b, i, j, p = tf.unstack(sparse_layers.indices(dt), axis=0)
            i = i + tf.gather(self._out_stream.row_starts, b)
            j = j + tf.gather(self._in_stream.row_starts, b)

            # partition
            bijt = tf.dynamic_partition(
                tf.stack((b, i, j, sparse_layers.values(dt)), axis=-1), p,
                num_partitions)
            dense_shape = tf.stack((batch_size, out_size, in_size), axis=-1)
            batched_dts = []
            for partitioned in bijt:
                bij, dt = tf.split(partitioned, [3, 1])
                dt = tf.cast(tf.squeeze(dt, axis=0),
                             tf.float32) / self._decay_time
                batched_dts.append(
                    as_lambda(tf.SparseTensor, bij, dt, dense_shape))
            self._batched_dts = tuple(batched_dts)
        return self._batched_dts

    @property
    def model_dts(self):
        if self._model_dts is None:
            self._model_dts = tuple(
                Lambda(sparse_ops.remove_leading_dim)(meta.model_input(dt))
                for dt in self.batched_dts)
        return self._model_dts

    def convolve(self, features: FloatTensor, filters: int,
                 temporal_kernel_size: int, spatial_kernel_size: int, **kwargs):
        meta.set_mark(features, BuiltModels.TRAINED)
        return conv_layers.SpatioTemporalEventConv(
            filters=filters,
            tempoarl_kernel_size=temporal_kernel_size,
            spatial_kernel_size=spatial_kernel_size,
            **kwargs)([features, *self.model_dts])


def spike_threshold(stream: Stream, link: GridNeighbors, **kwargs) -> Stream:
    assert (stream.grid == link.in_grid)
    times, coords = as_lambda(spike_ops.spike_threshold, stream.times,
                              stream.coords, link.indices, link.splits,
                              **kwargs)
    return Stream(link.out_grid, times, coords)


def convolver(grid_neighbors: GridNeighbors,
              in_stream: Stream,
              out_stream: Stream,
              decay_time: int,
              spatial_buffer_size: int,
              max_decays: int = 4):
    assert (grid_neighbors.in_grid == in_stream.grid)
    assert (grid_neighbors.out_grid == out_stream.grid)
    grid_neighbors = grid_neighbors.T
    partitions, indices, splits = as_lambda(
        neigh_ops.compute_neighbors, in_stream.times, in_stream.coords,
        out_stream.times, out_stream.coords, grid_neighbors.partitions,
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
