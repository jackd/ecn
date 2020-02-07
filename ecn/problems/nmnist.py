import tensorflow as tf
from kblocks.framework.problems.tfds import TfdsProblem
from events_tfds.events.nmnist import NMNIST

import gin


@gin.configurable(module='ecn')
def nmnist(pipeline):
    return TfdsProblem(
        NMNIST(),
        split_map={'validation': 'test'},
        pipeline=pipeline,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=(tf.keras.metrics.SparseCategoricalAccuracy(),))


@gin.configurable(module='ecn')
def vis_nmnist_example(features, labels, weights=None):
    import numpy as np
    from events_tfds.vis.image import as_frames
    import events_tfds.vis.anim as anim
    coords = features['coords']
    time = features['time']
    polarity = features['polarity'].numpy()

    coords = (coords - tf.reduce_min(coords, axis=0)).numpy()
    print(np.max(coords, axis=0))
    time = (time - tf.reduce_min(time)).numpy()
    # coords[:, 0] = np.max(coords[:, 0]) - coords[:, 0]
    coords[:, 1] = np.max(coords[:, 1]) - coords[:, 1]

    frames = as_frames(coords, time, polarity, num_frames=20)
    anim.animate_frames(frames)


if __name__ == '__main__':
    from kblocks.framework.problems.pipelines import BasePipeline
    pipeline = BasePipeline(8)
    dataset = nmnist(pipeline).get_base_dataset('train')
    for features, labels in dataset:
        vis_nmnist_example(features, labels)
