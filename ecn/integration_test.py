"""Tests of pipelined source + model. These take a while."""
import functools

import numpy as np
import tensorflow as tf

import ecn.builders as builders
import ecn.sources as sources
from kblocks.framework.batchers import RaggedBatcher
from kblocks.framework.meta_model import meta_model_trainable
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

        single = meta_model_trainable(
            build_fn, source, RaggedBatcher(1), cache_managers=None, compiler=None,
        )
        double = meta_model_trainable(
            build_fn, source, RaggedBatcher(2), cache_managers=None, compiler=None,
        )
        model = single.model
        outs_single = []
        labels_single = []
        with tf.keras.backend.learning_phase_scope(False):
            for features, labels in single.source.get_dataset("train"):
                outs_single.append(model(features))
                labels_single.append(labels)
            assert len(outs_single) == 2
            outs_single = tf.nest.map_structure(
                lambda *args: tf.concat(args, axis=0).numpy(), *outs_single
            )
            labels_single = tf.nest.map_structure(
                lambda *args: tf.concat(args, axis=0).numpy(), *labels_single
            )

            outs_double = None
            labels_double = None
            for (features, labels,) in double.source.get_dataset("train"):
                assert outs_double is None  # ensure only loops once
                (outs_double, labels_double) = tf.nest.map_structure(
                    lambda x: x.numpy(), (model(features), labels)
                )
            assert outs_double is not None  # ensure only loops once

        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-4, atol=1e-4
        )
        tf.nest.map_structure(assert_allclose, labels_single, labels_double)
        tf.nest.map_structure(assert_allclose, outs_single, outs_double)

    def test_inception_vox_pooling(self):
        base_source = sources.nmnist_source(shuffle_files=False)
        build_fn = functools.partial(
            builders.inception_vox_pooling, hidden_units=(16,), filters0=4, kt0=2,
        )

        self._test_batched_trainable(base_source, build_fn)


if __name__ == "__main__":
    tf.test.main()
