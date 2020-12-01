import functools

import tensorflow_datasets as tfds

from ecn import vis
from ecn.ops.augment import augment_event_dataset
from events_tfds.events import asl_dvs

reverse_xy = False

# name = "nmnist"
# aug_kwargs = dict(flip_time=0.5)

# name = "cifar10_dvs"
# aug_kwargs = dict(flip_time=0.5, flip_ud=True)
# reverse_xy = True

# name = "ncaltech101"
# aug_kwargs = dict(flip_time=0.5, flip_lr=True)

name = "asl_dvs"
aug_kwargs = dict(grid_shape=asl_dvs.GRID_SHAPE, flip_ud=True, flip_time=0.5)

dataset = tfds.load(name, split="train", as_supervised=True)

for example in dataset.map(functools.partial(augment_event_dataset, **aug_kwargs)):
    vis.vis_example2d(example, num_frames=16, fps=4, reverse_xy=reverse_xy)
