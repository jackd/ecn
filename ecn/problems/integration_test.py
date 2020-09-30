"""Tests of pipelined source + model. These take a while."""
import functools

import numpy as np
import tensorflow as tf

import ecn.problems.builders as builders
import ecn.problems.sources as sources
import ecn.problems.utils as utils
from kblocks.framework.sources import DataSource, DelegatingSource, Split

Lambda = tf.keras.layers.Lambda


class DeterministicSource(DelegatingSource):
    def __init__(self, base: DataSource, num_examples=2):
        super().__init__(base=base)
        self._num_examples = num_examples

    def get_dataset(self, split: Split) -> tf.data.Dataset:
        return super().get_dataset(split).take(self._num_examples)

    def epoch_length(self, split: Split) -> int:
        return self._num_examples


class IntegrationTest(tf.test.TestCase):
    def _test_batched_trainable(self, base_source, build_fn):
        source = DeterministicSource(base_source, num_examples=2)

        single = utils.multi_graph_trainable(
            build_fn, source, 1, cache_managers=None, repeats=1, compiler=None
        )
        double = utils.multi_graph_trainable(
            build_fn, source, 2, cache_managers=None, repeats=1, compiler=None
        )
        model = single.model
        outs_single = []
        labels_single = []
        weights_single = []
        with tf.keras.backend.learning_phase_scope(False):
            for features, labels, weights in single.source.get_dataset("train"):
                outs_single.append(model(features))
                labels_single.append(labels)
                weights_single.append(weights)
            assert len(outs_single) == 2
            outs_single = tf.nest.map_structure(
                lambda *args: tf.concat(args, axis=0).numpy(), *outs_single
            )
            labels_single = tf.nest.map_structure(
                lambda *args: tf.concat(args, axis=0).numpy(), *labels_single
            )
            weights_single = tf.nest.map_structure(
                lambda *args: tf.concat(args, axis=0).numpy(), *weights_single
            )

            outs_double = None
            labels_double = None
            weights_double = None
            for features, labels, weights in double.source.get_dataset("train"):
                assert outs_double is None  # ensure only loops once
                (outs_double, labels_double, weights_double) = tf.nest.map_structure(
                    lambda x: x.numpy(), (model(features), labels, weights)
                )
            assert outs_double is not None  # ensure only loops once

        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-4, atol=1e-4
        )
        tf.nest.map_structure(assert_allclose, labels_single, labels_double)
        # stream weights are divided by batch size
        assert_allclose(weights_double["stream"] * 2, weights_single["stream"])
        tf.nest.map_structure(assert_allclose, outs_single, outs_double)

    def test_simple_multi_graph(self):
        base_source = sources.nmnist_source(shuffle_files=False)
        build_fn = functools.partial(
            builders.simple_multi_graph,
            hidden_units=(16,),
            filters0=4,
            kt0=2,
            static_sizes=False,
        )
        self._test_batched_trainable(base_source, build_fn)

    def test_inception(self):
        base_source = sources.nmnist_source(shuffle_files=False)
        build_fn = functools.partial(
            builders.inception_multi_graph,
            hidden_units=(16,),
            filters0=4,
            kt0=2,
            static_sizes=False,
        )

        self._test_batched_trainable(base_source, build_fn)


if __name__ == "__main__":
    tf.test.main()
