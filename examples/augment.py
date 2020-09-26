# import numpy as np
from ecn.problems import augment, sources

# from events_tfds.events.cifar10_dvs import CLASSES
# base_source = sources.nmnist_source()
# base_source = sources.cifar10_dvs_source()
# base_source = sources.ncaltech101_source()
base_source = sources.ncars_source()
CLASSES = None

aug_source = augment.Augmented2DSource(
    base_source,
    flip_time=0.5,
    # flip_lr=True,
    flip_ud=True,
    rotate_limits=None
    # rotate_limits=(-np.pi, np.pi),  # exaggerate
)
for source in (base_source, aug_source):
    print(source.epoch_length("train"))
    print(source.epoch_length("validation"))

for example in aug_source.get_dataset("train"):
    sources.vis_example(
        example,
        # reverse_xy=True,
        # flip_up_down=True,
        num_frames=1,
        fps=4,
        class_names=CLASSES,
    )
