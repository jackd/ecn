import functools
from typing import Generic, Optional, Tuple, TypeVar, Union

import numpy as np
import tensorflow as tf

import ecn.pub_sub as ps
import kblocks.extras.layers.sparse as sparse_layers
import meta_model.pipeline as pl
from composite_layers import ragged, sparse
from ecn.layers import conv as conv_layers
from ecn.layers import grid as grid_layers
from ecn.layers import lif as lif_layers
from ecn.layers import neighbors as neigh_layers
from ecn.layers import ragged as ragged_layers
from kblocks.keras import layers

BoolTensor = tf.Tensor
IntTensor = tf.Tensor
FloatTensor = tf.Tensor

BoolArray = np.ndarray
IntArray = np.ndarray

DTYPE = tf.int32

Lambda = tf.keras.layers.Lambda

T = TypeVar("T")

S0 = TypeVar("S0", bound="Stream")
S1 = TypeVar("S1", bound="Stream")


def _normalize_coords(coords, shape):
    """Normalize ravelled coordinates, maintaining aspect ratio."""
    shape = tf.convert_to_tensor(shape, dtype=coords.dtype)
    coords = grid_layers.unravel_index_transpose(coords, shape)
    coords = coords - shape // 2
    scale = tf.cast(tf.reduce_max(shape), tf.float32) / 2
    return tf.cast(coords, tf.float32) / scale


def _final_features(args):
    features, row_lengths, row_ends = args
    final = tf.gather(features, row_ends)
    return tf.where(tf.equal(row_lengths, 0), tf.zeros_like(final), final)


def to_nearest_power(x, base=2):
    x = tf.convert_to_tensor(x, dtype_hint=tf.int64)
    base = tf.convert_to_tensor(base, dtype_hint=x.dtype)
    assert x.dtype.is_integer
    return base ** tf.cast(
        tf.math.ceil(
            tf.math.log(tf.cast(x, tf.float32)) / tf.math.log(tf.cast(base, tf.float32))
        ),
        x.dtype,
    )


def maybe_cast(tensor, dtype):
    if tensor.dtype != dtype:
        return tf.cast(tensor, dtype)
    return tensor


@functools.wraps(tf.stack)
def tf_stack(values, axis=0, name=None):
    # fixed duplicate name in graph issues.
    if isinstance(values, tf.Tensor) and name is None:
        name = values[0].graph.unique_name("stack")
        return tf.stack(values, axis=axis, name=name)
    return tf.stack(values, axis=axis, name=name)


def pad_ragged(rt, padding, values=0):
    assert ragged.ragged_rank(rt) == 1
    flat_values = ragged.flat_values(rt)
    row_splits = ragged.row_splits(rt)
    padding_tensor = [[0, 0]] * (len(flat_values.shape))
    padding_tensor[0][1] = padding
    padding_tensor = tf.convert_to_tensor(padding_tensor)
    flat_values = tf.pad(flat_values, padding_tensor, constant_values=values)
    starts, total = tf.split(row_splits, [-1, 1])
    row_splits = tf.concat((starts, total + padding), axis=0)

    # def fn(args):
    #     flat_values, row_splits = args
    #     return tf.RaggedTensor.from_row_splits(
    #         flat_values, tf.cast(row_splits, tf.int64), validate=True
    #     )

    # return tf.keras.layers.Lambda(fn)([flat_values, row_splits])
    return ragged.from_row_splits(
        flat_values, tf.cast(row_splits, tf.int64), validate=False
    )


class Grid:
    def __init__(self, shape, dtype=DTYPE):
        self._dtype = dtype
        self._static_shape = tf.get_static_value(shape)
        if self._static_shape is not None:
            self._static_shape = tuple(int(i) for i in self._static_shape)

        if tf.is_tensor(shape):
            self._shape = maybe_cast(shape, dtype=dtype)
        else:
            self._shape = tf.convert_to_tensor(shape, dtype=dtype)
        self._size = tf.math.reduce_prod(self._shape)
        self._shape.shape.assert_has_rank(1)

    @property
    def size(self):
        return self._size

    @property
    def dtype(self):
        return self._dtype

    @property
    def static_shape(self) -> Optional[Tuple[int, ...]]:
        return self._static_shape

    @property
    def shape(self) -> IntTensor:
        return self._shape

    @property
    def ndims(self) -> int:
        return len(self._shape)

    def link(self, kernel_shape: Tuple[int, ...], strides, padding) -> "GridNeighbors":
        # calculate the transpose and return it's transpose
        (partitions, indices, splits, out_shape) = grid_layers.sparse_neighborhood(
            self.shape, kernel_shape, strides, padding
        )
        if (
            not any(tf.is_tensor(t) for t in (strides, padding, kernel_shape))
            and self.static_shape is not None
        ):
            out_shape = grid_layers.output_shape(
                np.array(self.static_shape),
                np.array(kernel_shape),
                np.array(strides),
                np.array(padding),
            )

        out_grid = Grid(out_shape)
        return GridNeighbors(
            out_grid,
            self,
            indices,
            splits,
            partitions,
            np.prod(kernel_shape, dtype=self.dtype.as_numpy_dtype),
        ).T

    def self_link(self, kernel_shape: Tuple[int, ...]) -> "GridNeighbors":
        partitions, indices, splits = grid_layers.sparse_neighborhood_in_place(
            self.shape, kernel_shape
        )
        link = GridNeighbors(
            self,
            self,
            indices,
            splits,
            partitions,
            np.prod(kernel_shape, dtype=self.dtype.as_numpy_dtype),
        )
        link._transpose = link  # pylint: disable=protected-access
        return link

    def partial_self_link(self, kernel_mask: BoolArray) -> "GridNeighbors":
        num_partitions = np.count_nonzero(kernel_mask)
        (
            partitions,
            indices,
            splits,
        ) = grid_layers.sparse_neighborhood_from_mask_in_place(self.shape, kernel_mask)
        link = GridNeighbors(self, self, indices, splits, partitions, num_partitions)
        link._transpose = link  # pylint: disable=protected-access
        return link

    def ravel_indices(self, indices: IntTensor, axis=-1):
        indices.shape.assert_has_rank(2)
        return grid_layers.ravel_multi_index(indices, self._shape, axis=axis)

    def unravel_indices(self, indices: IntTensor, axis=-1):
        indices.shape.assert_has_rank(1)
        if axis == 0:
            return tf.unravel_index(indices, self._shape)
        return grid_layers.unravel_index_transpose(indices, self._shape)


class GridNeighbors:
    """Contains sparse connectedness information between two grids."""

    def __init__(
        self,
        in_grid: Grid,
        out_grid: Grid,
        indices: IntTensor,
        splits: IntTensor,
        partitions: IntTensor,
        num_partitions: int,
    ):
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
    def transpose(self) -> "GridNeighbors":
        if self._transpose is None:
            # create transpose
            indices, splits, partitions = ragged_layers.transpose_csr(
                self.indices,
                self.splits,
                self.partitions,
                nrows_out=maybe_cast(self._out_grid.size, self.indices.dtype),
                validate=False,
            )
            self._transpose = GridNeighbors(
                self._out_grid,
                self._in_grid,
                indices,
                splits,
                partitions,
                self.num_partitions,
            )
            self._transpose._transpose = self  # pylint: disable=protected-access
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


class Stream:
    _publisher = ps.Publisher()
    on_create: ps.Topic = _publisher.topic

    def __init__(
        self,
        times: IntTensor,
        min_mean_size: Optional[int] = None,
        bucket_sizes: bool = False,
        dtype=DTYPE,
    ):
        self._dtype = dtype

        self._times = tf.cast(times, dtype)

        self._min_mean_size = min_mean_size
        self._batched = False
        self._size = None
        self._model_row_starts = None
        self._model_row_ends = None
        self._model_row_splits = None
        self._model_value_rowids = None
        self._model_row_lengths = None
        self._model_times = None
        self._model_batch_size = None
        self._model_valid_size = None

        self._bucket_sizes = bucket_sizes

        Stream._publisher.add(self)

    @property
    def dtype(self):
        return self._dtype

    @property
    def batched(self) -> bool:
        return self._batched

    @property
    def min_mean_size(self) -> Optional[int]:
        return self._min_mean_size

    @property
    def times(self) -> IntTensor:
        return self._times

    @property
    def size(self):
        if self._size is None:
            self._size = tf.squeeze(tf.shape(self._times, out_type=self.dtype), axis=0)
        return self._size

    @property
    def cached_times(self):
        self._batch()
        return self._cached_times

    def _batch(self):
        if self._batched:
            return

        times = pl.cache(self._times)
        self._cached_times = times

        times = pl.batch(times)
        self._row_splits = ragged.row_splits(times)
        self._valid_size = self._row_splits[-1]
        if self._bucket_sizes:
            self._batched_size = to_nearest_power(self._valid_size, 2)
            self._padding = self._batched_size - self._valid_size
            times = pad_ragged(times, self._padding, values=times.dtype.limits[1])
            self._row_splits = ragged.row_splits(times)
        else:
            self._batched_size = self._valid_size
            self._padding = None

        self._batched_times = ragged.flat_values(times)
        self._value_rowids = ragged.value_rowids(times)
        self._row_starts, total = tf.split(self._row_splits, [-1, 1], axis=0)
        self._valid_size = tf.squeeze(total, axis=0)

        _, row_ends = tf.split(self._row_splits, [1, -1])
        self._row_lengths = row_ends - self._row_starts
        self._row_ends = row_ends - 1  # used in model gather

        self._batch_size = tf.size(self._row_lengths, out_type=tf.int64)

        self._batched = True

    def frame_indices(self, t_starts, t_ends, num_frames: int, dtype=None):
        """
        Returns flat batched frame indices in t_start, t_end.

        Assumes t_starts <= t < t_ends

        Args:
            t_starts: [batch_size] ints, batched
            t_ends: [batch_size] ints, batched

        Returns:
            frame_indices: [batched_sized] ints.
        """
        if dtype is None:
            dtype = self.dtype
        if t_starts is not None:
            t_starts.shape.assert_has_rank(1)

        t_ends.shape.assert_has_rank(1)

        t = self.batched_times
        t_ends = tf.gather(t_ends, self.value_rowids)
        if t_starts is not None:
            t_starts = tf.gather(t_starts, self.value_rowids)
            t = t - t_starts
            t_ends = t_ends - t_starts

        t = num_frames * t / t_ends
        t = tf.cast(t, dtype)
        return t

    def batch_features(self, features):
        return ragged.from_row_splits(features, self.model_row_splits)

    def mean_features(self, features, batch_size):
        return tf.math.unsorted_segment_mean(
            features, self.model_value_rowids, num_segments=batch_size
        )

    @property
    def batched_times(self):
        self._batch()
        return self._batched_times

    @property
    def value_rowids(self):
        self._batch()
        return self._value_rowids

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

    @property
    def model_valid_size(self) -> IntTensor:
        if self._model_valid_size is not None:
            valid_size = tf.expand_dims(self.valid_size, axis=0)
            valid_size = pl.model_input(valid_size)
            self._model_valid_size = tf.squeeze(valid_size)
        return self._model_valid_size

    def prepare_model_inputs(self, features) -> tf.RaggedTensor:
        features = pl.batch(pl.cache(features))
        if self.padding is not None:
            features = pad_ragged(features, self.padding)
        features = pl.model_input(features)
        return features

    # def prepare_final_labels(self, labels) -> tf.Tensor:
    #     return pl.batch(labels)

    # def prepare_final_weights(self, weights):
    #     return pl.batch(weights)

    # def prepare_labels(self, labels) -> Dict[str, tf.Tensor]:
    #     labels = self.prepare_final_labels(labels)
    #     ret = {"final": labels}
    #     row_lengths = self.row_lengths
    #     labels = tf.repeat(labels, row_lengths, axis=0)
    #     # if self._min_mean_size is not None:
    #     if self.padding is not None:
    #         labels = tf.pad(labels, [[0, self.padding]])
    #     ret["stream"] = labels
    #     return ret

    # def prepare_weights(self, weights=None) -> Dict[str, tf.Tensor]:
    #     ret = {}
    #     if weights is None:
    #         weights = 1
    #     else:
    #         weights = self.prepare_final_weights(weights)
    #         ret["final"] = weights

    #     row_lengths = self.row_lengths
    #     weights = weights / (self.batch_size * row_lengths)
    #     weights = tf.repeat(weights, row_lengths, axis=0)
    #     # if self._min_mean_size is not None:
    #     if self.padding is not None:
    #         weights = tf.pad(weights, [[0, self.padding]])
    #     ret["stream"] = weights
    #     return ret

    def final_features(self, features):
        return Lambda(_final_features)(
            [features, self.model_row_lengths, self.model_row_splits]
        )

    @property
    def model_times(self):
        if self._model_times is None:
            self._model_times = pl.model_input(self.batched_times)
        return self._model_times

    @property
    def model_row_lengths(self):
        if self._model_row_lengths is None:
            splits = self.model_row_splits
            self._model_row_lengths = splits[1:] - splits[:-1]
        return self._model_row_lengths

    @property
    def model_row_starts(self):
        if self._model_row_starts is None:
            self._model_row_starts = self.model_row_splits[:-1]
        return self._model_row_starts

    @property
    def model_row_splits(self):
        if self._model_row_splits is None:
            self._model_row_splits = pl.model_input(self.row_splits)
        return self._model_row_splits

    @property
    def model_row_ends(self):
        if self._model_row_ends is None:
            # self._model_row_ends = pl.model_input(self._row_ends)
            self._model_row_ends = self.model_row_splits[1:] - 1
        return self._model_row_ends

    @property
    def model_value_rowids(self):
        if self._model_value_rowids is None:
            self._model_value_rowids = tf.ragged.row_splits_to_segment_ids(
                self.model_row_splits
            )
        return self._model_value_rowids

    @classmethod
    def from_config(cls, config):
        return Stream(**config)

    def get_config(self):
        return dict(times=self.times, min_mean_size=self.min_mean_size)

    def _rebuild(self: T, **kwargs) -> T:
        config = self.get_config()
        config.update(kwargs)
        return self.__class__.from_config(config)

    def mask(self, mask: BoolTensor) -> "Stream":
        return self._rebuild(times=tf.boolean_mask(self.times, mask))

    def gather(self, indices: IntTensor) -> "Stream":
        return self._rebuild(times=tf.gather(self.times, indices))

    def pool_features(
        self,
        t_end: IntTensor,
        features: FloatTensor,
        filters: int,
        temporal_kernel_size: int,
        num_decays=4,
        **kwargs
    ):
        batch_size = tf.size(t_end)
        value_rowids = self.model_value_rowids
        t_end = tf.gather(t_end, value_rowids)
        dt = tf.cast(num_decays * (t_end - self.model_times), tf.float32) / tf.cast(
            t_end, tf.float32
        )
        return conv_layers.TemporalEventPooling(
            filters=filters, temporal_kernel_size=temporal_kernel_size, **kwargs
        )([features, dt, value_rowids, batch_size])


class SpatialStream(Stream):
    def __init__(
        self,
        grid: Union[Grid, IntTensor, IntArray, Tuple[int, ...]],
        times: IntTensor,
        coords: IntTensor,
        bucket_sizes: bool = False,
        min_mean_size: Optional[int] = None,
        dtype: tf.DType = DTYPE,
    ):
        assert times.shape.ndims == 1

        coords = tf.cast(coords, dtype)

        self._grid = grid if isinstance(grid, Grid) else Grid(grid)
        if coords.shape.ndims == 1:
            self._shaped_coords = None
            self._coords = coords
        elif coords.shape.ndims == 2:
            self._shaped_coords = coords
            self._coords = self._grid.ravel_indices(coords)
        else:
            raise ValueError("coords must be rank 1 or 2, got {}".format(coords.shape))

        self._batched_coords = None
        self._model_coords = None

        super().__init__(
            times=times,
            bucket_sizes=bucket_sizes,
            min_mean_size=min_mean_size,
            dtype=dtype,
        )

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
    def batched_coords(self):
        if self._batched_coords is None:
            coords = pl.cache(self._coords)
            coords = pl.batch(coords)
            self._batched_coords = ragged.flat_values(coords)
            padding = self.padding
            if padding is not None:
                self._batched_coords = tf.pad(self._batched_coords, [[0, padding]])
        return self._batched_coords

    @property
    def model_coords(self):
        if self._model_coords is None:
            self._model_coords = pl.model_input(self.batched_coords)
        return self._model_coords

    def pool_features(
        self,
        t_end: IntTensor,
        features: FloatTensor,
        filters: int,
        temporal_kernel_size: int,
        num_decays=4,
        **kwargs
    ):
        coords = self.model_coords
        if self.grid.static_shape is not None:
            coords = Lambda(
                _normalize_coords, arguments=dict(shape=self.grid.static_shape)
            )(coords)
        else:
            raise NotImplementedError("TODO")
        features = features + layers.Dense(features.shape[-1])(coords)
        return super().pool_features(
            t_end, features, filters, temporal_kernel_size, num_decays, **kwargs
        )

    def voxelize(
        self, reduction, features, t_start, t_end, num_frames: int, batch_size=None
    ):
        static_shape = self.grid.static_shape
        assert static_shape is not None
        static_size = np.prod(static_shape)
        assert features.shape[-1] is not None
        num_frames = np.array(num_frames, dtype=np.int64)

        batch_index = self.value_rowids
        batched_coords = maybe_cast(self.batched_coords, batch_index.dtype)
        time = self.frame_indices(t_start, t_end, num_frames, dtype=batch_index.dtype)
        dims = tf.stack((self.batch_size, num_frames, static_size), axis=0)
        indices = tf.stack((batch_index, time, batched_coords), axis=0)
        indices = grid_layers.ravel_multi_index(indices, dims, axis=0)

        indices = pl.model_input(indices)
        if batch_size is None:
            features.shape.assert_has_rank(3)
            batch_size = tf.shape(features)[0]
            assert ragged.is_ragged(features)
            features = ragged.values(features)
        features.shape.assert_has_rank(2)
        if self.padding is not None:
            valid_size = self.model_valid_size
            features = features[:valid_size]
            indices = indices[:valid_size]
        features = reduction(
            features,
            indices,
            num_segments=batch_size * (num_frames * np.prod(static_shape)),
        )
        features = Lambda(
            tf.reshape,
            arguments=dict(shape=(-1, num_frames, *static_shape, features.shape[-1])),
        )(features)

        return features

    def mean_voxelize(self, features, t_start, t_end, num_frames: int, batch_size=None):
        return self.voxelize(
            tf.math.unsorted_segment_mean,
            features,
            t_start,
            t_end,
            num_frames,
            batch_size,
        )

    def max_voxelize(self, features, t_start, t_end, num_frames: int, batch_size=None):
        return self.voxelize(
            tf.math.unsorted_segment_max,
            features,
            t_start,
            t_end,
            num_frames,
            batch_size,
        )

    @property
    def shaped_coords(self):
        if self._shaped_coords is None:
            self._shaped_coords = self.grid.unravel_indices(self.coords)
        return self._shaped_coords

    def get_config(self):
        config = super().get_config()
        config.update(dict(coords=self.coords, grid=self.grid))
        return config

    def mask(self, mask: BoolTensor) -> "SpatialStream":
        return self._rebuild(
            times=tf.boolean_mask(self.times, mask),
            coords=tf.boolean_mask(self.coords, mask),
        )

    def gather(self, indices: IntTensor) -> "SpatialStream":
        return self._rebuild(
            times=tf.gather(self.times, indices),
            coords=tf.gather(self.coords, indices),
        )


class Convolver(Generic[S0, S1]):
    _publisher = ps.Publisher()
    on_create: ps.Topic = _publisher.topic

    def __init__(
        self,
        in_stream: S0,
        out_stream: S1,
        partitions: Optional[IntTensor],
        indices: IntTensor,
        splits: IntTensor,
        decay_time: int,
        num_partitions: int,
        dtype=DTYPE,
    ):
        self._dtype = dtype

        if partitions is None:
            assert num_partitions == 1

        indices = tf.cast(indices, dtype)
        splits = tf.cast(splits, dtype)
        if partitions is not None:
            partitions = tf.cast(partitions, dtype)

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

    @property
    def dtype(self):
        return self._dtype

    def _rebuild(self, **kwargs) -> "Convolver":
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

    def reindex(self, indices: IntTensor) -> "Convolver":
        return self._rebuild(indices=tf.gather(indices, self.indices))

    def gather(self, indices: IntTensor) -> "Convolver":
        values = tf_stack((self._partitions, self._indices), axis=-1)
        values, splits = ragged_layers.gather_rows(values, self._splits, indices)
        partitions, indices = tf.unstack(values, axis=-1)
        return self._rebuild(partitions=partitions, indices=indices, splits=splits)

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
            if self.num_partitions == 1:
                self._batched_dts = (
                    self._batch_single_partition(self._indices, self._splits),
                )
            else:
                self._batched_dts = self._batch_multi_partition()
        return self._batched_dts

    def _batch_single_partition(self, indices, splits) -> tf.SparseTensor:
        in_stream = self._in_stream
        out_stream = self._out_stream
        splits = maybe_cast(splits, self.dtype)
        indices = maybe_cast(indices, self.dtype)

        indices = pl.cache(indices)
        splits = pl.cache(splits)

        # with mg.pre_batch_context():
        #     i = maybe_cast(tf.ragged.row_splits_to_segment_ids(splits),
        #                    tf.int64)
        #     j = maybe_cast(indices, tf.int64)
        #     ij = tf_stack((i, j), axis=-1)
        #     values = tf.ones_like(i)
        #     dt = tf.SparseTensor(ij, values, (-1, -1))

        # dt = mg.batch(dt)
        # with mg.post_batch_context():
        #     b, i, j = tf.unstack(dt.indices, num=3, axis=-1)
        #     i = i + tf.gather(out_stream.row_starts, b)
        #     j = j + tf.gather(in_stream.row_starts, b)
        #     values = tf.cast(
        #         tf.gather(out_stream.batched_times, i) - tf.gather(
        #             in_stream.batched_times, j), tf.float32) / self.decay_time
        #     ij = tf_stack((i, j), axis=-1)
        #     dt = tf.SparseTensor(
        #         ij, values, (out_stream.batched_size, in_stream.batched_size))
        # return dt

        ragged_indices = ragged.from_row_splits(
            maybe_cast(indices, tf.int64), maybe_cast(splits, tf.int64), validate=False,
        )

        ragged_indices = pl.batch(ragged_indices)

        b, i, j = sparse_layers.ragged_to_sparse_indices(
            ragged_indices, in_stream.row_starts
        )
        del b
        dt = (
            tf.cast(
                tf.gather(out_stream.batched_times, i)
                - tf.gather(in_stream.batched_times, j),
                tf.float32,
            )
            / self.decay_time
        )
        dense_shape = tf_stack(
            (out_stream.batched_size, in_stream.batched_size), axis=0
        )
        ij = tf_stack((i, j), axis=-1)
        assert ij.dtype == tf.int64
        assert dense_shape.dtype == tf.int64
        dt = sparse.SparseTensor(ij, dt, dense_shape)
        return dt

    def _batch_multi_partition(self):
        num_partitions = self.num_partitions

        assert num_partitions > 1
        assert not self.batched

        # ##################################
        # ###### IMPL 0 START. WORKS #######
        # ##################################
        # dts = []

        # with mg.pre_cache_context():
        #     rowids = tf.ragged.row_splits_to_segment_ids(self._splits,
        #                                                  out_type=self.dtype)
        #     partitions = maybe_cast(self._partitions, tf.int32)
        #     ijs = tf.dynamic_partition(
        #         tf_stack((rowids, self._indices), axis=-1), partitions,
        #         num_partitions)

        #     # components = []
        #     for ij in ijs:
        #         i, j = tf.unstack(ij, num=2, axis=-1)
        #         indices = tf.RaggedTensor.from_value_rowids(
        #             j, i, nrows=self.out_stream.size, validate=False)
        #         dts.append(
        #             self._batch_single_partition(indices.values,
        #                                          indices.row_splits))
        # return tuple(dts)
        # ##################################
        # ########## IMPL 0 END ############
        # ##################################

        ##################################
        ###### IMPL 1 START. WORKS #######
        ##################################

        components = []

        rowids = tf.ragged.row_splits_to_segment_ids(self._splits, out_type=self.dtype)
        partitions = maybe_cast(self._partitions, tf.int32)

        ijs = tf.dynamic_partition(
            tf_stack((rowids, self._indices), axis=-1, name="multi_partition_stack"),
            partitions,
            num_partitions,
        )

        for ij in ijs:
            ij.shape.assert_has_rank(2)
            assert ij.shape[1] == 2
            i, j = tf.unstack(ij, num=2, axis=-1)
            indices = ragged.from_value_rowids(
                j, i, nrows=maybe_cast(self.out_stream.size, i.dtype), validate=False,
            )

            components.append((ragged.values(indices), ragged.row_splits(indices)))

        all_ragged_indices = [
            pl.batch(
                ragged.from_row_splits(
                    tf.cast(pl.cache(v), tf.int64),
                    tf.cast(pl.cache(rs), tf.int64),
                    validate=False,
                )
            )
            for v, rs in components
        ]

        in_stream = self.in_stream
        out_stream = self.out_stream

        # ragged to sparse
        counts = []
        all_b = []
        all_i = []
        all_j = []
        for ragged_indices in all_ragged_indices:
            b = tf.ragged.row_splits_to_segment_ids(
                ragged.row_splits(ragged_indices), out_type=self.dtype
            )
            ragged_indices = ragged.values(ragged_indices)
            b = tf.repeat(b, ragged.row_lengths(ragged_indices), axis=0)
            # sparse indices must eventually be int64
            i = tf.ragged.row_splits_to_segment_ids(
                ragged.row_splits(ragged_indices), out_type=tf.int64
            )
            j = tf.cast(ragged.values(ragged_indices), tf.int64)
            counts.append(ragged.row_splits(ragged_indices)[-1])
            # total = tf.split(ragged_indices.row_splits, [-1, 1])[1]
            # counts.append(tf.squeeze(total, axis=0))
            all_b.append(b)
            all_i.append(i)
            all_j.append(j)

        # concat for efficient dt / block diagonalizing.
        cat_b = tf.concat(all_b, axis=0)
        cat_i = tf.concat(all_i, axis=0)
        cat_j = tf.concat(all_j, axis=0)

        # block diagonalize
        # skip i offset since it was automatically done in ragged batching
        cat_j = cat_j + tf.gather(in_stream.row_starts, cat_b)

        cat_dt = (
            tf.cast(
                tf.gather(out_stream.batched_times, cat_i)
                - tf.gather(in_stream.batched_times, cat_j),
                tf.float32,
            )
            / self.decay_time
        )
        cat_ij = tf_stack((cat_i, cat_j), axis=-1)

        dense_shape = tf_stack(
            (out_stream.batched_size, in_stream.batched_size), axis=0
        )
        # tf.SparseTensor indices and dense_shape must be int64
        if dense_shape.dtype != tf.int64:
            dense_shape = tf.cast(dense_shape, tf.int64)

        dts = tf.split(cat_dt, counts)
        ijs = tf.split(cat_ij, counts)

        return tuple(
            sparse.SparseTensor(ij, dt, dense_shape) for ij, dt in zip(ijs, dts)
        )
        ##################################
        ########## IMPL 1 END ############
        ##################################

    @property
    def model_dts(self):
        if self._model_dts is None:
            self._model_dts = tuple(pl.model_input(dt) for dt in self.batched_dts)
        return self._model_dts

    def convolve(
        self,
        features: Optional[tf.Tensor],
        filters: int,
        temporal_kernel_size: int,
        **kwargs
    ):
        if self.num_partitions == 1:
            assert len(self.model_dts) == 1
            return conv_layers.temporal_event_conv(
                features=features,
                dt=self.model_dts[0],
                filters=filters,
                temporal_kernel_size=temporal_kernel_size,
                **kwargs
            )
        else:
            return conv_layers.spatio_temporal_event_conv(
                features=features,
                dt=self.model_dts,
                filters=filters,
                temporal_kernel_size=temporal_kernel_size,
                spatial_kernel_size=self.num_partitions,
                **kwargs
            )


def spatial_leaky_integrate_and_fire(
    stream: SpatialStream,
    link: GridNeighbors,
    decay_time: int,
    threshold: float = 1.0,
    reset_potential: float = -1.0,
    bucket_sizes: bool = False,
    min_mean_size: Optional[int] = None,
) -> SpatialStream:
    assert stream.grid == link.in_grid
    times, coords = lif_layers.spatial_leaky_integrate_and_fire(
        stream.times,
        stream.coords,
        link.indices,
        link.splits,
        decay_time=decay_time,
        threshold=threshold,
        reset_potential=reset_potential,
    )
    return SpatialStream(
        link.out_grid,
        times,
        coords,
        bucket_sizes=bucket_sizes,
        min_mean_size=min_mean_size,
        dtype=stream.dtype,
    )


def leaky_integrate_and_fire(
    stream: Stream,
    decay_time: int,
    threshold: float = 1.0,
    reset_potential: float = -1,
    bucket_sizes: bool = False,
    min_mean_size: Optional[int] = None,
) -> Stream:
    time = lif_layers.leaky_integrate_and_fire(
        stream.times,
        decay_time=decay_time,
        threshold=threshold,
        reset_potential=reset_potential,
    )
    return Stream(
        time, min_mean_size=min_mean_size, dtype=stream.dtype, bucket_sizes=bucket_sizes
    )


def spatio_temporal_convolver(
    grid_neighbors: GridNeighbors,
    in_stream: SpatialStream,
    out_stream: SpatialStream,
    decay_time: int,
    spatial_buffer_size: int,
    max_decays: int = 4,
) -> Convolver[SpatialStream, SpatialStream]:
    assert grid_neighbors.in_grid == in_stream.grid
    assert grid_neighbors.out_grid == out_stream.grid
    grid_neighbors = grid_neighbors.T
    partitions, indices, splits = neigh_layers.compute_neighbors(
        in_stream.times,
        in_stream.coords,
        out_stream.times,
        out_stream.coords,
        grid_neighbors.partitions,
        grid_neighbors.indices,
        grid_neighbors.splits,
        spatial_buffer_size,
        decay_time * max_decays,
    )

    return Convolver(
        num_partitions=grid_neighbors.num_partitions,
        in_stream=in_stream,
        out_stream=out_stream,
        partitions=partitions,
        indices=indices,
        splits=splits,
        decay_time=decay_time,
    )


def pointwise_convolver(
    in_stream,
    out_stream,
    decay_time: int,
    spatial_buffer_size: int,
    max_decays: int = 4,
) -> Convolver[SpatialStream, SpatialStream]:
    assert in_stream.grid == out_stream.grid
    indices, splits = neigh_layers.compute_pointwise_neighbors(
        in_stream.times,
        in_stream.coords,
        out_stream.times,
        out_stream.coords,
        event_duration=decay_time * max_decays,
        spatial_buffer_size=spatial_buffer_size,
    )
    return Convolver(
        num_partitions=1,
        in_stream=in_stream,
        out_stream=out_stream,
        partitions=None,
        indices=indices,
        splits=splits,
        decay_time=decay_time,
    )


def flatten_convolver(
    in_stream: SpatialStream,
    out_stream: Stream,
    decay_time: int,
    max_decays: int = 4,
    num_partitions: Optional[int] = None,
) -> Convolver[SpatialStream, Stream]:
    assert not isinstance(out_stream, SpatialStream)

    if num_partitions is None:
        if in_stream.grid.static_shape is None:
            raise ValueError(
                "Either input_stream grid must be static or num_partitions must be "
                "provided"
            )
        num_partitions_ = np.prod(in_stream.grid.static_shape)
    else:
        num_partitions_ = num_partitions
    partitions, indices, splits = neigh_layers.compute_full_neighbors(
        in_times=in_stream.times,
        in_coords=in_stream.coords,
        out_times=out_stream.times,
        event_duration=decay_time * max_decays,
    )
    return Convolver[SpatialStream, Stream](
        num_partitions=num_partitions_,
        in_stream=in_stream,
        out_stream=out_stream,
        partitions=partitions,
        indices=indices,
        splits=splits,
        decay_time=decay_time,
    )


def temporal_convolver(
    in_stream: Stream, out_stream: Stream, decay_time: int, max_decays: int = 4
) -> Convolver[Stream, Stream]:
    indices, splits = neigh_layers.compute_pooled_neighbors(
        in_stream.times, out_stream.times, event_duration=max_decays * decay_time
    )
    return Convolver(
        num_partitions=1,
        in_stream=in_stream,
        out_stream=out_stream,
        partitions=None,
        indices=indices,
        splits=splits,
        decay_time=decay_time,
    )


def stream_accumulator():
    return ps.accumulator(Stream.on_create)


def convolver_accumulator():
    return ps.accumulator(Convolver.on_create)
