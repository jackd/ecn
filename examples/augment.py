from ecn import sources, vis

reverse_xy = False
flip_up_down = False

# base_source = sources.nmnist_source()
# aug_kwargs = dict(flip_time=0.5)

# base_source = sources.cifar10_dvs_source()
# aug_kwargs = dict(flip_time=0.5, flip_ud=True)
# reverse_xy = True

# base_source = sources.ncaltech101_source()
# aug_kwargs = dict(flip_time=0.5, flip_lr=True)

# base_source = sources.ncars_source()
# aug_kwargs = dict(flip_time=False, flip_ud=True)

base_source = sources.asl_dvs_source()
aug_kwargs = dict()
flip_up_down = True


aug_source = sources.Augmented2DSource(
    base_source,
    **aug_kwargs,
    # rotate_limits=(-np.pi, np.pi),  # exaggerate
)
for source in (base_source, aug_source):
    print(source.epoch_length("train"))
    print(source.epoch_length("validation"))

for example in aug_source.get_dataset("train"):
    vis.vis_example(
        example, num_frames=16, fps=4, reverse_xy=reverse_xy, flip_up_down=flip_up_down
    )
