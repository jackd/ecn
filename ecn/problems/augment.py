from typing import Tuple, Dict, Any, Optional
import itertools
import gin
import tensorflow as tf
import numpy as np
from kblocks.framework.sources import DataSource
from kblocks.framework.sources import Split
from ecn.ops import augment as augment_ops

DEFAULT_ROTATE_LIMITS = (-np.pi / 8, np.pi / 8)
AUTOTUNE = tf.data.experimental.AUTOTUNE


@gin.configurable(module='ecn.sources')
class Augmented2DSource(DataSource):

    def __init__(
            self,
            base_source: DataSource,
            flip_ud: bool = False,
            flip_lr: bool = False,
            flip_time: bool = True,
            rotate_limits: Optional[
                Tuple[float, float]] = DEFAULT_ROTATE_LIMITS,
            cycle_length: int = 1,
            block_length: int = 32,
            num_parallel_calls: int = 1,
    ):
        self._base_source = base_source
        keys = []
        values = []
        maybes = (('flip_ud', flip_ud), ('flip_lr', flip_lr), ('flip_time',
                                                               flip_time))
        for k, v in maybes:
            if v:
                keys.append(k)
                values.append((False, True))
        combinations = tuple(zip(*itertools.product(*values)))
        self._num_combinations = len(combinations[0])
        self._param_sets = {k: np.array(v) for k, v in zip(keys, combinations)}
        self._rotate_limits = rotate_limits
        self._block_length = block_length
        self._num_parallel_calls = num_parallel_calls
        self._cycle_length = cycle_length

    @property
    def meta(self) -> Dict[str, Any]:
        return self._base_source.meta

    def get_dataset(self, split: Split):
        base_dataset = self._base_source.get_dataset(split)
        if split == 'train':
            aug_ds = tf.data.Dataset.from_tensor_slices(self._param_sets)

            def params_map_fn(kwargs):

                def base_map_fn(features, labels, weights=None):
                    time = features['time']
                    coords = features['coords']
                    polarity = features['polarity']
                    time, coords, polarity, _ = augment_ops.augment(
                        time,
                        coords,
                        polarity,
                        grid_shape=self._base_source.meta['grid_shape'],
                        rotate_limits=self._rotate_limits,
                        **kwargs)

                    features.update(
                        dict(time=time, coords=coords, polarity=polarity))
                    return ((features, labels) if weights is None else
                            (features, labels, weights))

                return base_dataset.map(base_map_fn, self._num_parallel_calls)

            return aug_ds.interleave(params_map_fn,
                                     cycle_length=self._cycle_length,
                                     block_length=self._block_length)

        else:
            return base_dataset

    def examples_per_epoch(self, split: Split):
        return self._base_source.examples_per_epoch(split)


if __name__ == '__main__':
    from ecn.problems import sources
    base_source = sources.nmnist_source()

    aug_source = Augmented2DSource(
        base_source,
        flip_time=True,
        block_length=1,
        rotate_limits=(-np.pi, np.pi),  # exaggerate
    )
    for source in (base_source, aug_source):
        print(source.examples_per_epoch('train'))
        print(source.examples_per_epoch('validation'))

    for example in aug_source.get_dataset('train'):
        sources.vis_example(example)
