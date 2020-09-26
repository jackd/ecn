from typing import Optional, Tuple

import gin
import numpy as np
import tensorflow as tf

from ecn.ops import augment as augment_ops
from ecn.ops.augment import MaybeBool
from kblocks.framework.sources import DataSource, DelegatingSource, Split

MaybeBool = augment_ops.MaybeBool
DEFAULT_ROTATE_LIMITS = (-np.pi / 8, np.pi / 8)
AUTOTUNE = tf.data.experimental.AUTOTUNE


@gin.configurable(module="ecn.sources")
class Augmented2DSource(DelegatingSource):
    def __init__(
        self,
        base: DataSource,
        flip_ud: MaybeBool = False,
        flip_lr: MaybeBool = False,
        flip_time: MaybeBool = 0.5,
        rotate_limits: Optional[Tuple[float, float]] = DEFAULT_ROTATE_LIMITS,
        num_parallel_calls: int = AUTOTUNE,
    ):
        super().__init__(base=base)
        self._rotate_limits = rotate_limits
        self._num_parallel_calls = num_parallel_calls
        self._map_params = dict(
            flip_ud=flip_ud,
            flip_lr=flip_lr,
            flip_time=flip_time,
            rotate_limits=rotate_limits,
        )

    def get_dataset(self, split: Split):
        dataset = super().get_dataset(split)
        if split == "train":

            def map_fn(features, labels, weights=None):
                time = features["time"]
                coords = features["coords"]
                polarity = features["polarity"]
                time, coords, polarity, _ = augment_ops.augment(
                    time,
                    coords,
                    polarity,
                    grid_shape=self.meta["grid_shape"],
                    **self._map_params
                )

                features.update(dict(time=time, coords=coords, polarity=polarity))
                return (
                    (features, labels)
                    if weights is None
                    else (features, labels, weights)
                )

            dataset = dataset.map(map_fn, self._num_parallel_calls)
        return dataset
