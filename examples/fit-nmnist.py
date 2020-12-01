import functools
import os

import gin
import numpy as np
import shape_tfds.shape.modelnet  # pylint: disable=unused-import
import tensorflow as tf
import tensorflow_datasets as tfds
from events_tfds.events import nmnist
from kblocks.data import dense_to_ragged_batch
from kblocks.extras.callbacks import PrintLogger, ReduceLROnPlateauModule
from kblocks.models import compiled
from kblocks.trainables import trainable_fit
from kblocks.trainables.meta_models import build_meta_model_trainable

from ecn.builders.vox_pool import inception_vox_pooling
from ecn.ops.augment import augment_event_dataset

BackupAndRestore = tf.keras.callbacks.experimental.BackupAndRestore
AUTOTUNE = tf.data.experimental.AUTOTUNE
os.environ["TF_DETERMINISTIC_OPS"] = "1"

gin.parse_config(
    """\
import kblocks.keras.layers
import kblocks.keras.regularizers
import ecn.layers.conv

tf.keras.layers.Dense.kernel_regularizer = %l2_reg
tf.keras.layers.Conv3D.kernel_regularizer = %l2_reg
ecn.layers.EventConvBase.kernel_regularizer = %l2_reg

l2_reg = @tf.keras.regularizers.l2()
tf.keras.regularizers.l2.l = 4e-5
"""
)

batch_size = 32
run = 0
problem_dir = "/tmp/ecn/nmnist"
experiment_dir = os.path.join(problem_dir, f"run-{run:04d}")
epochs = 500
reduce_lr_patience = 10
reduce_lr_factor = 0.2
shuffle_buffer = 1024
seed = 0
data_seed = 0

tf.random.set_seed(seed)  # used for weight initialization


train_map_func = functools.partial(
    augment_event_dataset,
    flip_time=0.5,
    rotate_limits=(-np.pi / 8, np.pi / 8),
    grid_shape=nmnist.GRID_SHAPE,
)

validation_map_func = None

train_ds = tfds.load(
    name="nmnist",
    shuffle_files=True,
    split="train",
    as_supervised=True,
    read_config=tfds.core.utils.read_config.ReadConfig(
        shuffle_seed=0, shuffle_reshuffle_each_iteration=True
    ),
)

validation_ds = tfds.load(
    name="nmnist", shuffle_files=False, split="test", as_supervised=True
)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [
    tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
    tf.keras.metrics.SparseCategoricalAccuracy(),
]
optimizer = tf.keras.optimizers.Adam()


monitor_kwargs = dict(monitor="sparse_categorical_accuracy", mode="max")
model_callbacks = [
    ReduceLROnPlateauModule(
        patience=reduce_lr_patience, factor=reduce_lr_factor, **monitor_kwargs
    )
]

trainable = build_meta_model_trainable(
    meta_model_func=functools.partial(
        inception_vox_pooling,
        grid_shape=nmnist.GRID_SHAPE,
        num_classes=nmnist.NUM_CLASSES,
        num_levels=3,
        vox_start=0,
        filters0=32,
    ),
    train_dataset=train_ds,
    validation_dataset=validation_ds,
    batcher=dense_to_ragged_batch(batch_size=batch_size),
    shuffle_buffer=shuffle_buffer,
    compiler=functools.partial(
        compiled, loss=loss, metrics=metrics, optimizer=optimizer
    ),
    cache_factory=None,  # functools.partial(snapshot, compression="GZIP"),
    cache_dir=None,  # os.path.join(problem_dir, "cache"),
    cache_repeats=None,  # num_cache_repeats,
    train_augment_func=train_map_func,
    validation_augment_func=validation_map_func,
    callbacks=model_callbacks,
    seed=data_seed,
)

logging_callbacks = [
    PrintLogger(),
    tf.keras.callbacks.TensorBoard(os.path.join(experiment_dir, "logs")),
    BackupAndRestore(os.path.join(experiment_dir, "backup")),
]

fit = trainable_fit(
    trainable=trainable,
    callbacks=logging_callbacks,
    epochs=epochs,
    experiment_dir=experiment_dir,
)

fit.run()
