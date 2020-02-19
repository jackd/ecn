import gin
import tensorflow as tf

from kblocks.framework.sources import TfdsSource

from events_tfds.events.nmnist import NMNIST
from events_tfds.events.nmnist import NUM_CLASSES
from events_tfds.events.nmnist import GRID_SHAPE


@gin.configurable(module='ecn.nmnist')
def nmnist_source():
    return TfdsSource(NMNIST(),
                      split_map={'validation': 'test'},
                      meta=dict(num_classes=NUM_CLASSES, grid_shape=GRID_SHAPE))


@gin.configurable(module='ecn.nmnist')
def vis_nmnist_example(example):
    import numpy as np
    from events_tfds.vis.image import as_frames
    import events_tfds.vis.anim as anim
    features, label = example
    coords = features['coords']
    time = features['time']
    polarity = features['polarity'].numpy()
    print(label.numpy())

    coords = (coords - tf.reduce_min(coords, axis=0)).numpy()
    print(np.max(coords, axis=0))
    time = (time - tf.reduce_min(time)).numpy()

    frames = as_frames(coords, time, polarity, num_frames=20)
    anim.animate_frames(frames)


if __name__ == '__main__':
    dataset = nmnist_source().get_dataset('train')
    for example in dataset:
        vis_nmnist_example(example)
