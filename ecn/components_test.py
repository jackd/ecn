import functools
import itertools

import meta_model.pipeline as pl
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from meta_model.batchers import RaggedBatcher

import ecn.components as comp

Lambda = tf.keras.layers.Lambda


def feature_inputs(features, features_type="none"):
    if features_type == "none":
        return None
    if features_type == "binary":
        features = tf.split(features, [1, -1], axis=1)[0]
        features = tf.squeeze(features, axis=1) > 0.5
    elif features_type == "float":
        pass
    else:
        raise ValueError(
            '`features_type` must be "none", "binary" or "float",'
            f" got {features_type}"
        )
    out = pl.model_input(pl.batch(pl.cache(features)))
    out = Lambda(lambda x: tf.identity(x.values))(out)
    return out


FEATURE_TYPES = (
    "none",
    "binary",
    "float",
)

INT_TYPES = (
    tf.int32,
    tf.int64,
)

INT_AND_FEATURE_TYPES = tuple(itertools.product(INT_TYPES, FEATURE_TYPES))


class ComponentsTest(tf.test.TestCase, parameterized.TestCase):
    def _test_batched_simple(self, dataset, build_fn):
        batcher = RaggedBatcher(batch_size=1)
        pipeline, model = pl.build_pipelined_model(
            build_fn, dataset.element_spec, batcher=batcher
        )
        assert model is not None
        single = (
            dataset.map(pipeline.pre_cache_map_func())
            .map(pipeline.pre_batch_map_func())
            .apply(tf.data.experimental.dense_to_ragged_batch(1))
            .map(pipeline.post_batch_map_func())
        )
        double = (
            dataset.map(pipeline.pre_cache_map_func())
            .map(pipeline.pre_batch_map_func())
            .apply(tf.data.experimental.dense_to_ragged_batch(2))
            .map(pipeline.post_batch_map_func())
        )

        def as_tuple(x):
            return x if isinstance(x, tuple) else (x,)

        single_out = []
        single_labels = []
        single_weights = []
        for features, labels, weights in single:
            out = model(features)
            single_out.append(as_tuple(out))
            single_labels.append(as_tuple(labels))
            single_weights.append(as_tuple(weights))

        single_out = tf.nest.map_structure(
            lambda *args: tf.concat(args, axis=0).numpy(), *single_out
        )
        single_labels = tf.nest.map_structure(
            lambda *args: tf.concat(args, axis=0).numpy(), *single_labels
        )
        single_weights = tf.nest.map_structure(
            lambda *args: tf.concat(args, axis=0).numpy(), *single_weights
        )

        double_out = None
        double_labels = None
        double_weights = None
        for features, labels, weights in double:
            assert double_out is None
            (double_out, double_labels, double_weights) = tf.nest.map_structure(
                lambda x: as_tuple(x.numpy()), (model(features), labels, weights)
            )

        assert_allclose = functools.partial(np.testing.assert_allclose, rtol=1e-4)
        tf.nest.map_structure(assert_allclose, single_out, double_out)
        tf.nest.map_structure(assert_allclose, single_labels, double_labels)
        tf.nest.map_structure(assert_allclose, single_weights, double_weights)

    @parameterized.parameters(*INT_AND_FEATURE_TYPES)
    def test_batched_global_conv(self, dtype, feature_type):
        def build_fn(kwargs):
            times = kwargs["times"]
            features = kwargs["features"]
            stream = comp.Stream(times, dtype=dtype)
            conv = comp.temporal_convolver(stream, stream, decay_time=1, max_decays=4)
            out = conv.convolve(feature_inputs(features, feature_type), 2, 2)
            return out, (), ()

        data = [
            dict(
                times=np.arange(10, dtype=np.int64),
                features=np.random.uniform(size=(10, 3)).astype(np.float32),
            ),
            dict(
                times=np.arange(9, dtype=np.int64),
                features=np.random.uniform(size=(9, 3)).astype(np.float32),
            ),
        ]

        dataset = tf.data.Dataset.from_generator(
            lambda: data,
            {"times": tf.int64, "features": tf.float32},
            {"times": (None,), "features": (None, 3)},
        )

        self._test_batched_simple(dataset, build_fn)

    @parameterized.parameters(*INT_AND_FEATURE_TYPES)
    def test_batched_1d_conv(self, dtype, feature_type):
        def build_fn(kwargs):
            coords = kwargs["coords"]
            times = kwargs["times"]
            features = kwargs["features"]
            grid = comp.Grid((2,), dtype=dtype)
            link = grid.self_link((2,))
            stream = comp.SpatialStream(grid, times, coords, dtype=dtype)

            conv = comp.spatio_temporal_convolver(
                link, stream, stream, spatial_buffer_size=5, decay_time=2, max_decays=4
            )

            features = feature_inputs(features, feature_type)
            out = conv.convolve(features, 2, 2)
            return out, (), ()

        data = [
            {
                "times": np.arange(10, dtype=np.int64),
                "coords": np.expand_dims(
                    np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), axis=-1
                ),
                "features": np.random.uniform(size=(10, 3)).astype(np.float32),
            },
            {
                "times": np.arange(9, dtype=np.int64),
                "coords": np.expand_dims(
                    np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]), axis=-1
                ),
                "features": np.random.uniform(size=(9, 3)).astype(np.float32),
            },
        ]

        dataset = tf.data.Dataset.from_generator(
            lambda: data,
            {"times": tf.int64, "coords": tf.int64, "features": tf.float32},
            {"times": (None,), "coords": (None, 1), "features": (None, 3)},
        )

        self._test_batched_simple(dataset, build_fn)

    @parameterized.parameters(*INT_AND_FEATURE_TYPES)
    def test_lif_conv(self, dtype, feature_type):
        def build_fn(kwargs):
            times = kwargs["times"]
            features = kwargs["features"]
            stream = comp.Stream(times, dtype=dtype)
            out_stream = comp.leaky_integrate_and_fire(stream, 2)
            conv = comp.temporal_convolver(
                stream, out_stream, decay_time=2, max_decays=4
            )

            out = conv.convolve(feature_inputs(features, feature_type), 2, 2)
            return out, (), ()

        data = [
            dict(
                times=np.arange(10, dtype=np.int64),
                features=np.random.uniform(size=(10, 3)).astype(np.float32),
            ),
            dict(
                times=np.arange(9, dtype=np.int64),
                features=np.random.uniform(size=(9, 3)).astype(np.float32),
            ),
        ]

        dataset = tf.data.Dataset.from_generator(
            lambda: data,
            dict(times=tf.int64, features=tf.float32),
            dict(times=(None,), features=(None, 3)),
        )

        self._test_batched_simple(dataset, build_fn)

    @parameterized.parameters(*INT_AND_FEATURE_TYPES)
    def test_batched_1d_lif_conv(self, dtype, feature_type):
        def build_fn(kwargs):
            coords = kwargs["coords"]
            times = kwargs["times"]
            features = kwargs["features"]
            grid = comp.Grid((2,), dtype=dtype)
            link = grid.self_link((2,))
            stream = comp.SpatialStream(grid, times, coords, dtype=dtype)
            out_stream = comp.spatial_leaky_integrate_and_fire(stream, link, 2)

            conv = comp.spatio_temporal_convolver(
                link,
                stream,
                out_stream,
                spatial_buffer_size=5,
                decay_time=2,
                max_decays=4,
            )

            features = feature_inputs(features, feature_type)
            # out = features
            out = conv.convolve(features, 2, 2)

            return out, (), ()

        data = [
            {
                "times": np.arange(10, dtype=np.int64),
                "coords": np.expand_dims(
                    np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), axis=-1
                ),
                "features": np.random.uniform(size=(10, 3)).astype(np.float32),
            },
            {
                "times": np.arange(9, dtype=np.int64),
                "coords": np.expand_dims(
                    np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]), axis=-1
                ),
                "features": np.random.uniform(size=(9, 3)).astype(np.float32),
            },
        ]

        dataset = tf.data.Dataset.from_generator(
            lambda: data,
            {
                "times": tf.int64,
                "coords": tf.int64,
                "features": tf.float32,
            },
            {
                "times": (None,),
                "coords": (None, 1),
                "features": (None, 3),
            },
        )

        self._test_batched_simple(dataset, build_fn)

    @parameterized.parameters(*INT_AND_FEATURE_TYPES)
    def test_batched_1d_lif_conv_big(self, dtype, feature_type):
        # np.random.seed(123)  # passes
        np.random.seed(124)  # fails
        grid_size = 7

        def build_fn(kwargs):
            coords = kwargs["coords"]
            times = kwargs["times"]
            features = kwargs["features"]
            grid = comp.Grid((grid_size,), dtype=dtype)
            link = grid.link((2,), (2,), (0,))
            stream = comp.SpatialStream(grid, times, coords, dtype=dtype)
            out_stream = comp.spatial_leaky_integrate_and_fire(stream, link, 2)
            features = feature_inputs(features, feature_type)

            conv = comp.spatio_temporal_convolver(
                link,
                stream,
                out_stream,
                spatial_buffer_size=5,
                decay_time=2,
                max_decays=4,
            )

            # features = Lambda(
            #     lambda x: tf.identity(x.values))(conv.model_dts[1])
            features = Lambda(lambda x: tf.identity(x.values))(conv.model_dts[1])
            # features = conv.convolve(features, 2, 2)

            return features, (), ()

        data = [
            {
                "times": np.arange(10, dtype=np.int64),
                "coords": np.random.randint(grid_size, size=(10, 1)),
                "features": np.random.uniform(size=(10, 3)).astype(np.float32),
            },
            {
                "times": np.arange(9, dtype=np.int64),
                "coords": np.random.randint(grid_size, size=(9, 1)),
                "features": np.random.uniform(size=(9, 3)).astype(np.float32),
            },
        ]

        dataset = tf.data.Dataset.from_generator(
            lambda: data,
            {
                "times": tf.int64,
                "coords": tf.int64,
                "features": tf.float32,
            },
            {
                "times": (None,),
                "coords": (None, 1),
                "features": (None, 3),
            },
        )

        self._test_batched_simple(dataset, build_fn)

    @parameterized.parameters(*INT_AND_FEATURE_TYPES)
    def test_batched_1d_lif_conv_chained(self, dtype, feature_type):
        grid_size = 13
        t0 = 200

        def build_fn(kwargs):
            coords = kwargs["coords"]
            times = kwargs["times"]
            features = kwargs["features"]
            grid = comp.Grid((grid_size,), dtype=dtype)
            link = grid.link((3,), (2,), (1,))
            stream = comp.SpatialStream(grid, times, coords, dtype=dtype)
            out_stream = comp.spatial_leaky_integrate_and_fire(stream, link, t0)
            features = feature_inputs(features, feature_type)

            conv = comp.spatio_temporal_convolver(
                link,
                stream,
                out_stream,
                spatial_buffer_size=5,
                decay_time=t0,
                max_decays=4,
            )

            features = conv.convolve(features, 2, 2, activation="relu")

            stream = out_stream
            link = stream.grid.self_link((3,))
            out_stream = comp.spatial_leaky_integrate_and_fire(stream, link, 2 * t0)
            conv = comp.spatio_temporal_convolver(
                link,
                stream,
                out_stream,
                spatial_buffer_size=5,
                decay_time=2 * t0,
                max_decays=4,
            )

            features = conv.convolve(features, 2, 2)

            return features, (), ()

        data = [
            {
                "times": np.arange(100, dtype=np.int64),
                "coords": np.random.randint(grid_size, size=(100, 1)),
                "features": np.random.uniform(size=(100, 3)).astype(np.float32),
            },
            {
                "times": np.arange(90, dtype=np.int64),
                "coords": np.random.randint(grid_size, size=(90, 1)),
                "features": np.random.uniform(size=(90, 3)).astype(np.float32),
            },
        ]

        dataset = tf.data.Dataset.from_generator(
            lambda: data,
            {
                "times": tf.int64,
                "coords": tf.int64,
                "features": tf.float32,
            },
            {
                "times": (None,),
                "coords": (None, 1),
                "features": (None, 3),
            },
        )

        self._test_batched_simple(dataset, build_fn)

    @parameterized.parameters(*INT_AND_FEATURE_TYPES)
    def test_batched_global_1d_lif_conv(self, dtype, feature_type):
        def build_fn(kwargs):
            coords = kwargs["coords"]
            times = kwargs["times"]
            features = kwargs["features"]
            grid = comp.Grid((2,), dtype=dtype)
            stream = comp.SpatialStream(grid, times, coords, dtype=dtype)
            out_stream = comp.leaky_integrate_and_fire(stream, 2)

            conv = comp.flatten_convolver(
                stream, out_stream, decay_time=2, max_decays=4
            )

            features = feature_inputs(features, feature_type)
            # out = features
            out = conv.convolve(features, 2, 2)
            out = tf.keras.layers.BatchNormalization()(out)

            return out, (), ()

        data = (
            {
                "times": np.arange(10, dtype=np.int64),
                "coords": np.expand_dims(
                    np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), axis=-1
                ),
                "features": np.random.uniform(size=(10, 3)).astype(np.float32),
            },
            {
                "times": np.arange(9, dtype=np.int64),
                "coords": np.expand_dims(
                    np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]), axis=-1
                ),
                "features": np.random.uniform(size=(9, 3)).astype(np.float32),
            },
        )

        dataset = tf.data.Dataset.from_generator(
            lambda: data,
            {"times": tf.int64, "coords": tf.int64, "features": tf.float32},
            {"times": (None,), "coords": (None, 1), "features": (None, 3)},
        )

        self._test_batched_simple(dataset, build_fn)


if __name__ == "__main__":
    tf.test.main()
    # ComponentsTest().test_batched_1d_conv0()
    # ComponentsTest().test_batched_1d_lif_conv_chained0()
    # ComponentsTest().test_batched_1d_conv0()
