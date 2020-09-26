import gin
import tensorflow as tf

from kblocks.framework.trainable import Trainable


@gin.configurable()
def check_sizes(trainable: Trainable, split="train"):
    source = trainable.source
    dataset = source.get_dataset(split).map(lambda *args: tf.nest.flatten(args))
    # total = source.epoch_length(split)
    for args in dataset:
        print([a.shape for a in args])
