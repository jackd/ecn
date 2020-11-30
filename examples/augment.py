import functools

import tensorflow_datasets as tfds

from ecn import vis
from ecn.ops.augment import augment_event_dataset
from events_tfds.events import asl_dvs

reverse_xy = False

# base_source = sources.nmnist_source()
# aug_kwargs = dict(flip_time=0.5)

# base_source = sources.cifar10_dvs_source()
# aug_kwargs = dict(flip_time=0.5, flip_ud=True)
# reverse_xy = True

# base_source = sources.ncaltech101_source()
# aug_kwargs = dict(flip_time=0.5, flip_lr=True)

dataset = tfds.load("asl_dvs", split="train", as_supervised=True)
aug_kwargs = dict(grid_shape=asl_dvs.GRID_SHAPE, flip_ud=True, flip_time=0.5)

for example in dataset.map(functools.partial(augment_event_dataset, **aug_kwargs)):
    vis.vis_example2d(example, num_frames=16, fps=4, reverse_xy=reverse_xy)
