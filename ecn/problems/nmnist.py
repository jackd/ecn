import tensorflow as tf
from kblocks.framework.problems.tfds import TfdsProblem
from events_tfds.events.nmnist import NMNIST
from events_tfds.events.nmnist import NUM_CLASSES

import gin


@gin.configurable(module='ecn.problems')
def nmnist(pipeline):
    return TfdsProblem(
        NMNIST(),
        split_map={'validation': 'test'},
        pipeline=pipeline,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=(tf.keras.metrics.SparseCategoricalAccuracy(),),
        outputs_spec=(tf.TensorSpec(shape=(None, NUM_CLASSES),
                                    dtype=tf.float32))
        # loss=(
        #     tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #     None,  # final
        # ),
        # metrics=(
        #     (tf.keras.metrics.SparseCategoricalAccuracy(name='stream_acc'),),
        #     (tf.keras.metrics.SparseCategoricalAccuracy(name='final_acc'),),
        # ),
        # outputs_spec=(tf.TensorSpec(shape=(None, NUM_CLASSES),
        #                             dtype=tf.float32),) * 2

        # loss={
        #     'stream':
        #         tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #     'final':
        #         None
        # },
        # metrics={
        #     'stream': (tf.keras.metrics.SparseCategoricalAccuracy(),),
        #     'final': (tf.keras.metrics.SparseCategoricalAccuracy(),),
        # },
        # outputs_spec={
        #     'stream':
        #         tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32),
        #     'final':
        #         tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32),
        # }
    )


@gin.configurable(module='ecn.problems')
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
