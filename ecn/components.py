import functools
from typing import Generic, Optional, Tuple, TypeVar, Union

import kblocks.extras.layers.sparse as sparse_layers
import meta_model.pipeline as pl
import numpy as np
import tensorflow as tf
from kblocks.keras import layers
from wtftf.meta import memoized_property
from wtftf.ragged import RaggedStructure, is_ragged
from wtftf.ragged import layers as ragged_wrappers
from wtftf.ragged import ragged_rank
from wtftf.sparse import layers as sparse_wrappers

import ecn.pub_sub as ps
from ecn.layers import conv as conv_layers
from ecn.layers import grid as grid_layers
from ecn.layers import lif as lif_layers
from ecn.layers import neighbors as neigh_layers
from ecn.layers import ragged as ragged_layers

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


def maybe_cast(tensor, dtype):
    if tensor.dtype != dtype:
        return tf.cast(tensor, dtype)
    return tensor


def ragged_components(rt):
    assert ragged_rank(rt) == 1
    return ragged_wrappers.values(rt), ragged_wrappers.row_splits(rt)


@functools.wraps(tf.stack)
def tf_stack(values, axis=0, name=None):
    # fixed duplicate name in graph issues.
    if isinstance(values, tf.Tensor) and name is None:
        name = values[0].graph.unique_name("stack")
        return tf.stack(values, axis=axis, name=name)
    return tf.stack(values, axis=axis, name=name)


class Grid:
    """
    A regular ND grid used for linking to other ND grids.

    These are essential parts of GridNeighbors (links between grids) which determine
    which pixels of an input grid influence pixels of an output grid. For example,
    an unpadded 5x5 input grid with a spatial kernel size of 3x3 will have an output
    grid of shape 3x3. Output pixel (i, j) will be neighbors with input pixels
    (i:i+3, j:j+3). GridNeighbors themselves use ravelled indices (indices into the
    flattened array), i.e. pixel (1, 0) in the input grid would have a GridNeighbors
    index of 5.
    """

    def __init__(self, shape, dtype=DTYPE):
        self._dtype = dtype
        self._static_shape = tf.get_static_value(shape)
        if self._static_shape is not None:
            self._static_shape = tuple(int(i) for i in self._static_shape)

        if tf.is_tensor(shape):
            self._shape = maybe_cast(shape, dtype=dtype)
        else:
            self._shape = tf.convert_to_tensor(shape, dtype=dtype)
        self._shape.shape.assert_has_rank(1)

    @memoized_property
    def size(self):
        return tf.math.reduce_prod(self._shape)

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
        transposed = GridNeighbors(
            out_grid,
            self,
            indices,
            splits,
            partitions,
            np.prod(kernel_shape, dtype=self.dtype.as_numpy_dtype),
        )
        return transposed.T

    def _self_link(
        self, partitions, indices, splits, num_partitions: int
    ) -> "GridNeighbors":
        link = GridNeighbors(self, self, indices, splits, partitions, num_partitions)
        link._transpose = link  # pylint: disable=protected-access
        return link

    def self_link(self, kernel_shape: Tuple[int, ...]) -> "GridNeighbors":
        partitions, indices, splits = grid_layers.sparse_neighborhood_in_place(
            self.shape, kernel_shape
        )
        num_partitions = np.prod(kernel_shape, dtype=self.dtype.as_numpy_dtype)
        return self._self_link(partitions, indices, splits, num_partitions)

    def partial_self_link(self, kernel_mask: BoolArray) -> "GridNeighbors":
        (
            partitions,
            indices,
            splits,
        ) = grid_layers.sparse_neighborhood_from_mask_in_place(self.shape, kernel_mask)
        num_partitions = np.count_nonzero(kernel_mask)
        return self._self_link(partitions, indices, splits, num_partitions)

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

    def __init__(self, times: IntTensor, dtype=DTYPE):
        self._dtype = dtype
        self._times = tf.cast(times, dtype)
        self._batched_structure = None
        Stream._publisher.add(self)

    @property
    def dtype(self):
        return self._dtype

    @property
    def times(self) -> IntTensor:
        return self._times

    @memoized_property
    def size(self):
        return tf.squeeze(tf.shape(self._times, out_type=self.dtype), axis=0)

    @memoized_property
    def cached_times(self):
        return pl.cache(self._times)

    def _batch(self):
        if self._batched_structure is None:
            times = pl.batch(self.cached_times)
            self._batched_structure = RaggedStructure.from_tensor(times)
            self._batched_times = ragged_wrappers.flat_values(times)

    @property
    def batched_times(self):
        self._batch()
        return self._batched_times

    @property
    def batched_structure(self) -> RaggedStructure:
        self._batch()
        return self._batched_structure

    @memoized_property
    def model_structure(self):
        return RaggedStructure.from_row_splits(self.batched_structure.row_splits)

    @memoized_property
    def model_times(self):
        return pl.model_input(self.batched_times)

    def frame_indices(
        self,
        t_starts: Optional[IntTensor],
        t_ends: IntTensor,
        num_frames: int,
        dtype: tf.DType = None,
    ):
        """
        Returns flat batched frame indices in t_start, t_end.

        Assumes t_starts <= self.batched_times < t_ends

        This is the batched version of
        ```
        (t - t_starts) / (t_ends - t_starts) * num_frames
        ```
        where `t` is the times of this stream.

        Args:
            t_starts (optional): [batch_size] ints, batched
            t_ends: [batch_size] ints, batched
            num_frames: number of frames.

        Returns:
            frame_indices: [batched_sized] ints in [0, num_frames).
        """
        if dtype is None:
            dtype = self.dtype
        if t_starts is not None:
            t_starts.shape.assert_has_rank(1)

        t_ends.shape.assert_has_rank(1)

        t = self.batched_times

        value_rowids = self.batched_structure.value_rowids
        if t_starts is not None:
            t_ends = t_ends - t_starts
            t = t - tf.gather(t_starts, value_rowids)
        t = t / tf.gather(t_ends, value_rowids)
        t = num_frames * t
        t = tf.cast(t, dtype)
        return t

    @classmethod
    def from_config(cls, config):
        return Stream(**config)

    def get_config(self):
        return dict(times=self.times)

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
        value_rowids = self.model_structure.value_rowids
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

        super().__init__(times=times, dtype=dtype)

    @classmethod
    def from_config(cls, config):
        return SpatialStream(**config)

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def coords(self) -> IntTensor:
        return self._coords

    @memoized_property
    def batched_coords(self):
        coords = pl.cache(self._coords)
        coords = pl.batch(coords)
        return ragged_wrappers.flat_values(coords)

    @memoized_property
    def model_coords(self):
        return pl.model_input(self.batched_coords)

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
        static_size = np.prod(static_shape, dtype=np.int64)
        assert features.shape[-1] is not None
        num_frames = np.array(num_frames, dtype=np.int64)

        batch_index = self.batched_structure.value_rowids
        batched_coords = maybe_cast(self.batched_coords, batch_index.dtype)
        time = self.frame_indices(t_start, t_end, num_frames, dtype=batch_index.dtype)
        dims = tf.stack(
            (
                maybe_cast(self.batched_structure.nrows, tf.int64),
                num_frames,
                static_size,
            ),
            axis=0,
        )
        indices = tf.stack((batch_index, time, batched_coords), axis=0)
        indices = grid_layers.ravel_multi_index(indices, dims, axis=0)

        indices = pl.model_input(indices)
        if batch_size is None:
            features.shape.assert_has_rank(3)
            batch_size = tf.shape(features)[0]
            assert is_ragged(features)
            features = ragged_wrappers.values(features)
        features.shape.assert_has_rank(2)
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
        Convolver._publisher.add(self)

    @property
    def dtype(self):
        return self._dtype

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

    @memoized_property
    def batched_dts(self) -> Tuple[tf.SparseTensor, ...]:
        if self.num_partitions == 1:
            return (self._batch_single_partition(self._indices, self._splits),)
        return self._batch_multi_partition()

    def _batch_single_partition(self, indices, splits) -> tf.SparseTensor:
        in_stream = self._in_stream
        out_stream = self._out_stream
        splits = maybe_cast(splits, self.dtype)
        indices = maybe_cast(indices, self.dtype)

        indices = pl.cache(indices)
        splits = pl.cache(splits)

        ragged_indices = ragged_wrappers.from_row_splits(
            maybe_cast(indices, tf.int64), maybe_cast(splits, tf.int64)
        )

        ragged_indices = pl.batch(ragged_indices)

        b, i, j = sparse_layers.ragged_to_sparse_indices(
            ragged_indices, in_stream.batched_structure.row_starts
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
            (
                out_stream.batched_structure.total_size,
                in_stream.batched_structure.total_size,
            ),
            axis=0,
        )
        ij = tf_stack((i, j), axis=-1)
        assert ij.dtype == tf.int64
        assert dense_shape.dtype == tf.int64
        dt = sparse_wrappers.SparseTensor(maybe_cast(ij, tf.int64), dt, dense_shape)
        return dt

    def _batch_multi_partition(self) -> Tuple[tf.SparseTensor, ...]:
        num_partitions = self.num_partitions

        assert num_partitions > 1

        components = []

        rowids = tf.ragged.row_splits_to_segment_ids(self._splits, out_type=self.dtype)
        partitions = maybe_cast(self._partitions, tf.int32)

        ijs = tf.dynamic_partition(
            tf_stack((rowids, self._indices), axis=-1, name="multi_partition_stack"),
            partitions,
            num_partitions,
        )

        # sorted transpose
        for ij in ijs:
            ij.shape.assert_has_rank(2)
            assert ij.shape[1] == 2
            # num required in tf-nightly (2.5)
            i, j = tf.unstack(ij, num=2, axis=-1)
            indices = ragged_wrappers.from_value_rowids(
                j, i, nrows=maybe_cast(self.out_stream.size, i.dtype)
            )

            components.append(ragged_components(indices))

        all_ragged_indices = [
            pl.batch(
                ragged_wrappers.from_row_splits(
                    tf.cast(pl.cache(v), tf.int64), tf.cast(pl.cache(rs), tf.int64)
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
                ragged_wrappers.row_splits(ragged_indices), out_type=self.dtype
            )
            ragged_indices = ragged_wrappers.values(ragged_indices)
            b = tf.repeat(b, ragged_wrappers.row_lengths(ragged_indices), axis=0)
            # sparse indices must eventually be int64
            i = tf.ragged.row_splits_to_segment_ids(
                ragged_wrappers.row_splits(ragged_indices), out_type=tf.int64
            )
            j = tf.cast(ragged_wrappers.values(ragged_indices), tf.int64)
            counts.append(ragged_wrappers.row_splits(ragged_indices)[-1])
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
        cat_j = cat_j + tf.gather(in_stream.batched_structure.row_starts, cat_b)

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
            (
                maybe_cast(out_stream.batched_structure.total_size, tf.int64),
                maybe_cast(in_stream.batched_structure.total_size, tf.int64),
            ),
            axis=0,
        )
        # tf.SparseTensor indices and dense_shape must be int64
        if dense_shape.dtype != tf.int64:
            dense_shape = tf.cast(dense_shape, tf.int64)

        dts = tf.split(cat_dt, counts)
        ijs = tf.split(cat_ij, counts)

        return tuple(
            sparse_wrappers.SparseTensor(maybe_cast(ij, tf.int64), dt, dense_shape)
            for ij, dt in zip(ijs, dts)
        )

    @memoized_property
    def model_dts(self):
        return tuple(pl.model_input(dt) for dt in self.batched_dts)

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
    return SpatialStream(link.out_grid, times, coords, dtype=stream.dtype)


def leaky_integrate_and_fire(
    stream: Stream,
    decay_time: int,
    threshold: float = 1.0,
    reset_potential: float = -1,
) -> Stream:
    time = lif_layers.leaky_integrate_and_fire(
        stream.times,
        decay_time=decay_time,
        threshold=threshold,
        reset_potential=reset_potential,
    )
    return Stream(time, dtype=stream.dtype)


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
    """
    Get a convolver that convolves over all spatial dimensions.

    This is conceptually similar to applying a 3x3 image convolution without padding
    to a 3x3 image. While the output has no spatial dimension (or is 1x1), it is
    distinctly different to global pooling.

    Args:
        in_stream: input spatial stream.
        out_stream: output temporal stream.
        decay_time: time-scale influencing speed of exponential decay.
        max_decays: number of `decay_time` durations before temporal neighborhood is
            truncated.
        num_partitions: number of pixels in the input stream. This is computed from
            `in_stream.grid.static_shape`. If that is not known, a ValueError is raised.

    Returns:
        A Convolver.
    """
    assert not isinstance(out_stream, SpatialStream)

    if num_partitions is None:
        if in_stream.grid.static_shape is None:
            raise ValueError(
                "Either input_stream grid must be static or num_partitions must be "
                "provided"
            )
        num_partitions = np.prod(in_stream.grid.static_shape)
    partitions, indices, splits = neigh_layers.compute_full_neighbors(
        in_times=in_stream.times,
        in_coords=in_stream.coords,
        out_times=out_stream.times,
        event_duration=decay_time * max_decays,
    )
    return Convolver[SpatialStream, Stream](
        num_partitions=num_partitions,
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
    """
    Get a convolver between two temporal streams.

    Args:
        in_stream: input stream.
        out_stream: output stream.
        decay_time: time-scale influencing speed of exponential decay.
        max_decays: number of `decay_time` durations before temporal neighborhood is
            truncated.

    Returns:
        A temporal Convolver.
    """
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
    """
    Context manager that accumulates streams as they are created.

    This is mostly useful for debugging. See `ecn/vis.py` for example usage.

    Example usage:
    ```python

    with stream_accumulator() as streams:
        input_stream = Stream(...)
        output_stream = leaky_integrate_and_fire(stream, ...)
    assert streams == [input_stream, output_stream]

    Returns:
        A context manager that yields a list which is appended to as `Stream`s are
        created.
    ```

    See also: `convolver_accumulator`.
    """
    return ps.accumulator(Stream.on_create)


def convolver_accumulator():
    """
    Context manager that accumulates convolvers as they are created.

    This is mostly useful for debugging. See `ecn/vis.py` for example usage.

    Returns:
        A context manager that yields a list which is appended to as `Convolver`s are
        created.

    See also: `stream_accumulator`.
    """
    return ps.accumulator(Convolver.on_create)
