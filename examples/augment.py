# import numpy as np
from ecn.problems import augment, sources

# base_source = sources.nmnist_source()
# aug_kwargs = dict(flip_time=0.5)
# vis_kwargs = dict(num_frames=8, fps=2)

# base_source = sources.cifar10_dvs_source()
# aug_kwargs = dict(flip_time=0.5, flip_ud=True)
# vis_kwargs = dict(num_frames=16, fps=4, reverse_xy=True)

base_source = sources.ncaltech101_source()
aug_kwargs = dict(flip_time=0.5, flip_ud=True)
vis_kwargs = dict(num_frames=16, fps=4, reverse_xy=False)

# base_source = sources.ncars_source()
# aug_kwargs = dict(flip_time=False, flip_ud=True)

aug_source = augment.Augmented2DSource(
    base_source,
    **aug_kwargs,
    # rotate_limits=(-np.pi, np.pi),  # exaggerate
)
for source in (base_source, aug_source):
    print(source.epoch_length("train"))
    print(source.epoch_length("validation"))

for example in aug_source.get_dataset("train"):
    sources.vis_example(example, **vis_kwargs)
