# import os

# os.environ["NUMBA_DISABLE_JIT"] = "1"

import functools

import numpy as np
import tensorflow as tf

import ecn.components as comp
import kblocks.multi_graph as mg

Lambda = tf.keras.layers.Lambda


def assert_eager_variable_creator(next_creator, **kwargs):
    print(f"checking, {tf.executing_eagerly()}")
    assert tf.executing_eagerly()
    variable = next_creator(**kwargs)
    print(variable)
    return variable


def get_cached_dataset(source, num_examples=2):
    out = source.get_dataset("train").take(num_examples).cache()
    for _ in out:
        pass
    return out


def feature_inputs(features, stream, features_type="none"):
    if features_type == "none":
        return None
    if features_type == "binary":
        with mg.pre_cache_context():
            features = tf.split(features, [1, -1], axis=1)[0]
            features = tf.squeeze(features, axis=1) > 0.5
    elif features_type == "float":
        pass
    else:
        raise ValueError(
            '`features_type` must be "none", "binary" or "float",'
            f" got {features_type}"
        )
    out = stream.prepare_model_inputs(features)
    out = Lambda(lambda x: tf.identity(x.values))(out)
    return out


FEATURE_TYPES = (
    # "none",
    # "binary",
    "float",
)

INT_TYPES = (
    # tf.int32,
    tf.int64,
)


class ComponentsTest(tf.test.TestCase):
    def _test_batched_simple(self, dataset, build_fn):
        with tf.variable_creator_scope(assert_eager_variable_creator):
            built = mg.build_multi_graph(build_fn, dataset.element_spec)
        model = built.trained_model
        assert model is not None
        single = (
            dataset.map(built.pre_cache_map)
            .map(built.pre_batch_map)
            .batch(1)
            .map(built.post_batch_map)
        )
        double = (
            dataset.map(built.pre_cache_map)
            .map(built.pre_batch_map)
            .batch(2)
            .map(built.post_batch_map)
        )

        def as_tuple(x):
            return x if isinstance(x, tuple) else (x,)

        single_out = []
        single_labels = []
        single_weights = []
        # ensure no batch norm effects
        with tf.keras.backend.learning_phase_scope(False):
            for features, labels, weights in single:
                out = model(features)
                single_out.append(as_tuple(out))
                single_labels.append(as_tuple(labels))
                single_weights.append(as_tuple(weights))

            for (si,) in single_out:
                print(si.shape)

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

    def test_batched_global_conv(self):
        def build_fn(times, features, dtype, feature_type):
            stream = comp.Stream(times, dtype=dtype)
            conv = comp.temporal_convolver(stream, stream, decay_time=1, max_decays=4)
            out = conv.convolve(feature_inputs(features, stream, feature_type), 2, 2)
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

        for dtype in INT_TYPES:
            for feature_type in FEATURE_TYPES:
                self._test_batched_simple(
                    dataset,
                    functools.partial(build_fn, dtype=dtype, feature_type=feature_type),
                )

    # def test_batched_1d_conv(self):
    #     def build_fn(coords, times, features, dtype, feature_type):
    #         grid = comp.Grid((2,), dtype=dtype)
    #         link = grid.self_link((2,))
    #         stream = comp.SpatialStream(grid, times, coords, dtype=dtype)

    #         conv = comp.spatio_temporal_convolver(
    #             link, stream, stream, spatial_buffer_size=5, decay_time=2, max_decays=4
    #         )

    #         features = feature_inputs(features, stream, feature_type)
    #         out = conv.convolve(features, 2, 2)
    #         return out, (), ()

    #     data = [
    #         {
    #             "times": np.arange(10, dtype=np.int64),
    #             "coords": np.expand_dims(
    #                 np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), axis=-1
    #             ),
    #             "features": np.random.uniform(size=(10, 3)).astype(np.float32),
    #         },
    #         {
    #             "times": np.arange(9, dtype=np.int64),
    #             "coords": np.expand_dims(
    #                 np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]), axis=-1
    #             ),
    #             "features": np.random.uniform(size=(9, 3)).astype(np.float32),
    #         },
    #     ]

    #     dataset = tf.data.Dataset.from_generator(
    #         lambda: data,
    #         {"times": tf.int64, "coords": tf.int64, "features": tf.float32},
    #         {"times": (None,), "coords": (None, 1), "features": (None, 3)},
    #     )

    #     for dtype in INT_TYPES:
    #         for feature_type in FEATURE_TYPES:
    #             self._test_batched_simple(
    #                 dataset,
    #                 functools.partial(build_fn, dtype=dtype, feature_type=feature_type),
    #             )

    # def test_global_spike_conv(self):
    #     def build_fn(times, features, dtype, feature_type):
    #         stream = comp.Stream(times, dtype=dtype)
    #         out_stream = comp.global_spike_threshold(stream, 2)
    #         conv = comp.temporal_convolver(
    #             stream, out_stream, decay_time=2, max_decays=4
    #         )

    #         out = conv.convolve(feature_inputs(features, stream, feature_type), 2, 2)
    #         return out, (), ()

    #     data = [
    #         dict(
    #             times=np.arange(10, dtype=np.int64),
    #             features=np.random.uniform(size=(10, 3)).astype(np.float32),
    #         ),
    #         dict(
    #             times=np.arange(9, dtype=np.int64),
    #             features=np.random.uniform(size=(9, 3)).astype(np.float32),
    #         ),
    #     ]

    #     dataset = tf.data.Dataset.from_generator(
    #         lambda: data,
    #         dict(times=tf.int64, features=tf.float32),
    #         dict(times=(None,), features=(None, 3)),
    #     )

    #     for dtype in INT_TYPES:
    #         for feature_type in FEATURE_TYPES:
    #             self._test_batched_simple(
    #                 dataset,
    #                 functools.partial(build_fn, dtype=dtype, feature_type=feature_type),
    #             )

    # def test_batched_1d_spike_conv(self):
    #     def build_fn(coords, times, features, dtype=tf.int64, feature_type="none"):
    #         grid = comp.Grid((2,), dtype=dtype)
    #         link = grid.self_link((2,))
    #         stream = comp.SpatialStream(grid, times, coords, dtype=dtype)
    #         out_stream = comp.spike_threshold(stream, link, 2)

    #         conv = comp.spatio_temporal_convolver(
    #             link,
    #             stream,
    #             out_stream,
    #             spatial_buffer_size=5,
    #             decay_time=2,
    #             max_decays=4,
    #         )

    #         features = feature_inputs(features, stream, feature_type)
    #         # out = features
    #         out = conv.convolve(features, 2, 2)

    #         return out, (), ()

    #     data = [
    #         {
    #             "times": np.arange(10, dtype=np.int64),
    #             "coords": np.expand_dims(
    #                 np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), axis=-1
    #             ),
    #             "features": np.random.uniform(size=(10, 3)).astype(np.float32),
    #         },
    #         {
    #             "times": np.arange(9, dtype=np.int64),
    #             "coords": np.expand_dims(
    #                 np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]), axis=-1
    #             ),
    #             "features": np.random.uniform(size=(9, 3)).astype(np.float32),
    #         },
    #     ]

    #     dataset = tf.data.Dataset.from_generator(
    #         lambda: data,
    #         {
    #             "times": tf.int64,
    #             "coords": tf.int64,
    #             "features": tf.float32,
    #         },
    #         {
    #             "times": (None,),
    #             "coords": (None, 1),
    #             "features": (None, 3),
    #         },
    #     )

    #     for dtype in INT_TYPES:
    #         for feature_type in FEATURE_TYPES:
    #             self._test_batched_simple(
    #                 dataset,
    #                 functools.partial(build_fn, dtype=dtype, feature_type=feature_type),
    #             )

    # def test_batched_1d_spike_conv_big(self):
    #     # np.random.seed(123)  # passes
    #     np.random.seed(124)  # fails
    #     grid_size = 7

    #     def build_fn(coords, times, features, dtype=tf.int64, feature_type="none"):
    #         grid = comp.Grid((grid_size,), dtype=dtype)
    #         link = grid.link((2,), (2,), (0,))
    #         stream = comp.SpatialStream(grid, times, coords, dtype=dtype)
    #         out_stream = comp.spike_threshold(stream, link, 2)
    #         features = feature_inputs(features, stream, feature_type)

    #         conv = comp.spatio_temporal_convolver(
    #             link,
    #             stream,
    #             out_stream,
    #             spatial_buffer_size=5,
    #             decay_time=2,
    #             max_decays=4,
    #         )

    #         # features = Lambda(
    #         #     lambda x: tf.identity(x.values))(conv.model_dts[1])
    #         features = Lambda(lambda x: tf.identity(x.values))(conv.model_dts[1])
    #         # features = conv.convolve(features, 2, 2)

    #         return features, (), ()

    #     data = [
    #         {
    #             "times": np.arange(10, dtype=np.int64),
    #             "coords": np.random.randint(grid_size, size=(10, 1)),
    #             "features": np.random.uniform(size=(10, 3)).astype(np.float32),
    #         },
    #         {
    #             "times": np.arange(9, dtype=np.int64),
    #             "coords": np.random.randint(grid_size, size=(9, 1)),
    #             "features": np.random.uniform(size=(9, 3)).astype(np.float32),
    #         },
    #     ]

    #     dataset = tf.data.Dataset.from_generator(
    #         lambda: data,
    #         {
    #             "times": tf.int64,
    #             "coords": tf.int64,
    #             "features": tf.float32,
    #         },
    #         {
    #             "times": (None,),
    #             "coords": (None, 1),
    #             "features": (None, 3),
    #         },
    #     )

    #     for dtype in INT_TYPES:
    #         for feature_type in FEATURE_TYPES:
    #             self._test_batched_simple(
    #                 dataset,
    #                 functools.partial(build_fn, dtype=dtype, feature_type=feature_type),
    #             )

    # def test_batched_1d_spike_conv_chained(self):
    #     grid_size = 13
    #     t0 = 200

    #     def build_fn(coords, times, features, dtype=tf.int64, feature_type="none"):
    #         grid = comp.Grid((grid_size,), dtype=dtype)
    #         link = grid.link((3,), (2,), (1,))
    #         stream = comp.SpatialStream(grid, times, coords, dtype=dtype)
    #         out_stream = comp.spike_threshold(stream, link, t0)
    #         features = feature_inputs(features, stream, feature_type)

    #         conv = comp.spatio_temporal_convolver(
    #             link,
    #             stream,
    #             out_stream,
    #             spatial_buffer_size=5,
    #             decay_time=t0,
    #             max_decays=4,
    #         )

    #         features = conv.convolve(features, 2, 2, activation="relu")

    #         stream = out_stream
    #         # link = stream.grid.link((3,), (2,), (0))
    #         link = stream.grid.self_link((3,))
    #         out_stream = comp.spike_threshold(stream, link, 2 * t0)
    #         conv = comp.spatio_temporal_convolver(
    #             link,
    #             stream,
    #             out_stream,
    #             spatial_buffer_size=5,
    #             decay_time=2 * t0,
    #             max_decays=4,
    #         )

    #         features = conv.convolve(features, 2, 2)

    #         return features, (), ()

    #     data = [
    #         {
    #             "times": np.arange(100, dtype=np.int64),
    #             "coords": np.random.randint(grid_size, size=(100, 1)),
    #             "features": np.random.uniform(size=(100, 3)).astype(np.float32),
    #         },
    #         {
    #             "times": np.arange(90, dtype=np.int64),
    #             "coords": np.random.randint(grid_size, size=(90, 1)),
    #             "features": np.random.uniform(size=(90, 3)).astype(np.float32),
    #         },
    #     ]

    #     dataset = tf.data.Dataset.from_generator(
    #         lambda: data,
    #         {
    #             "times": tf.int64,
    #             "coords": tf.int64,
    #             "features": tf.float32,
    #         },
    #         {
    #             "times": (None,),
    #             "coords": (None, 1),
    #             "features": (None, 3),
    #         },
    #     )

    #     for dtype in INT_TYPES:
    #         for feature_type in FEATURE_TYPES:
    #             self._test_batched_simple(
    #                 dataset,
    #                 functools.partial(build_fn, dtype=dtype, feature_type=feature_type),
    #             )

    # def test_batched_global_1d_spike_conv(self):
    #     def build_fn(coords, times, features, dtype=tf.int64, feature_type="none"):
    #         grid = comp.Grid((2,), dtype=dtype)
    #         stream = comp.SpatialStream(grid, times, coords, dtype=dtype)
    #         out_stream = comp.global_spike_threshold(stream, 2)

    #         conv = comp.flatten_convolver(
    #             stream, out_stream, decay_time=2, max_decays=4
    #         )

    #         features = feature_inputs(features, stream, feature_type)
    #         # out = features
    #         out = conv.convolve(features, 2, 2)
    #         out = tf.keras.layers.BatchNormalization()(out)

    #         return out, (), ()

    #     data = [
    #         {
    #             "times": np.arange(10, dtype=np.int64),
    #             "coords": np.expand_dims(
    #                 np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), axis=-1
    #             ),
    #             "features": np.random.uniform(size=(10, 3)).astype(np.float32),
    #         },
    #         {
    #             "times": np.arange(9, dtype=np.int64),
    #             "coords": np.expand_dims(
    #                 np.array([0, 0, 0, 0, 0, 1, 1, 1, 1]), axis=-1
    #             ),
    #             "features": np.random.uniform(size=(9, 3)).astype(np.float32),
    #         },
    #     ]

    #     dataset = tf.data.Dataset.from_generator(
    #         lambda: data,
    #         {
    #             "times": tf.int64,
    #             "coords": tf.int64,
    #             "features": tf.float32,
    #         },
    #         {
    #             "times": (None,),
    #             "coords": (None, 1),
    #             "features": (None, 3),
    #         },
    #     )

    #     for dtype in INT_TYPES:
    #         for feature_type in FEATURE_TYPES:
    #             self._test_batched_simple(
    #                 dataset,
    #                 functools.partial(build_fn, dtype=dtype, feature_type=feature_type),
    #             )

    # def test_to_nearest_power(self):
    #     self.assertEqual(self.evaluate(comp.to_nearest_power(2)), 2)
    #     self.assertEqual(self.evaluate(comp.to_nearest_power(3)), 4)
    #     self.assertEqual(self.evaluate(comp.to_nearest_power(7)), 8)
    #     self.assertEqual(self.evaluate(comp.to_nearest_power(8)), 8)
    #     self.assertEqual(self.evaluate(comp.to_nearest_power(9)), 16)


if __name__ == "__main__":
    tf.test.main()
