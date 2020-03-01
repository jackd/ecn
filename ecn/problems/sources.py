import gin
import tensorflow as tf
from kblocks.framework.sources import TfdsSource


@gin.configurable(module='ecn.sources')
def ncars_source(**kwargs):
    from events_tfds.events import ncars
    return TfdsSource(ncars.Ncars(),
                      split_map={'validation': 'test'},
                      meta=dict(grid_shape=ncars.GRID_SHAPE, num_classes=2),
                      **kwargs)


@gin.configurable(module='ecn.sources')
def nmnist_source(**kwargs):
    from events_tfds.events import nmnist
    return TfdsSource(nmnist.NMNIST(),
                      split_map={'validation': 'test'},
                      meta=dict(num_classes=nmnist.NUM_CLASSES,
                                grid_shape=nmnist.GRID_SHAPE),
                      **kwargs)


@gin.configurable(module='ecn.sources')
def mnist_dvs_source(scale=16, train_percent=90, **kwargs):
    from events_tfds.events import mnist_dvs
    config = {
        4: mnist_dvs.SCALE4,
        8: mnist_dvs.SCALE8,
        16: mnist_dvs.SCALE16,
    }[scale]
    builder = mnist_dvs.MnistDVS(config=config)
    if kwargs.get('download_and_prepare', True):
        builder.download_and_prepare()
    examples_per_epoch = builder.info.splits['train'].num_examples

    return TfdsSource(
        builder,
        split_map={
            'train': f'train[:{train_percent}%]',
            'validation': f'train[{train_percent}%:]'
        },
        examples_per_epoch={
            'train': int(examples_per_epoch * train_percent / 100),
            'validation': int(examples_per_epoch * (1 - train_percent / 100))
        },
        meta=dict(num_classes=mnist_dvs.NUM_CLASSES,
                  grid_shape=mnist_dvs.GRID_SHAPE),
        **kwargs)


@gin.configurable(module='ecn.sources')
def ncaltech101_source(train_percent=90, **kwargs):
    from events_tfds.events import ncaltech101
    builder = ncaltech101.Ncaltech101()
    if kwargs.get('download_and_prepare', True):
        builder.download_and_prepare()
    examples_per_epoch = builder.info.splits['train'].num_examples

    return TfdsSource(
        builder,
        split_map={
            'train': f'train[:{train_percent}%]',
            'validation': f'train[{train_percent}%:]'
        },
        examples_per_epoch={
            'train': int(examples_per_epoch * train_percent / 100),
            'validation': int(examples_per_epoch * (1 - train_percent / 100))
        },
        meta=dict(num_classes=ncaltech101.NUM_CLASSES,
                  grid_shape=ncaltech101.GRID_SHAPE),
        **kwargs)


@gin.configurable(module='ecn.sources')
def cifar10_dvs_source(train_percent=90, **kwargs):
    from events_tfds.events import cifar10_dvs
    builder = cifar10_dvs.Cifar10DVS()
    if kwargs.get('download_and_prepare', True):
        builder.download_and_prepare()
    examples_per_epoch = builder.info.splits['train'].num_examples

    return TfdsSource(
        builder,
        split_map={
            'train': f'train[:{train_percent}%]',
            'validation': f'train[{train_percent}%:]'
        },
        examples_per_epoch={
            'train': int(examples_per_epoch * train_percent / 100),
            'validation': int(examples_per_epoch * (1 - train_percent / 100))
        },
        meta=dict(num_classes=cifar10_dvs.NUM_CLASSES,
                  grid_shape=cifar10_dvs.GRID_SHAPE),
        **kwargs)


@gin.configurable(module='ecn.sources')
def asl_dvs_source(train_percent=80, **kwargs):
    from events_tfds.events import asl_dvs
    builder = asl_dvs.AslDvs()
    if kwargs.get('download_and_prepare', True):
        builder.download_and_prepare()
    examples_per_epoch = builder.info.splits['train'].num_examples

    return TfdsSource(
        builder,
        split_map={
            'train': f'train[:{train_percent}%]',
            'validation': f'train[{train_percent}%:]'
        },
        examples_per_epoch={
            'train': int(examples_per_epoch * train_percent / 100),
            'validation': int(examples_per_epoch * (1 - train_percent / 100))
        },
        meta=dict(num_classes=asl_dvs.NUM_CLASSES,
                  grid_shape=asl_dvs.GRID_SHAPE),
        **kwargs)


@gin.configurable(module='ecn.sources')
def ntidigits_source():
    from events_tfds.events import ntidigits
    builder = ntidigits.Ntidigits()

    return TfdsSource(builder,
                      split_map={'validation': 'test'},
                      meta=dict(num_classes=ntidigits.NUM_CLASSES,
                                grid_shape=(ntidigits.NUM_CHANNELS,)))


@gin.configurable(module='ecn.sources')
def vis_example(example,
                num_frames=20,
                fps=4,
                reverse_xy=False,
                flip_up_down=False,
                class_names=None):
    from events_tfds.vis.image import as_frames
    import events_tfds.vis.anim as anim
    features, label = example
    coords = features['coords']
    time = features['time']
    time = (time - tf.reduce_min(time)).numpy()
    polarity = features['polarity'].numpy()
    if class_names is not None:
        print(class_names[label.numpy()])
    else:
        print(label.numpy())

    coords = (coords - tf.reduce_min(coords, axis=0)).numpy()
    print(f'{time.shape[0]} events over {time[-1] - time[0]} dt')
    if reverse_xy:
        coords = coords[:, -1::-1]
    frames = as_frames(coords,
                       time,
                       polarity,
                       num_frames=num_frames,
                       flip_up_down=flip_up_down)
    anim.animate_frames(frames, fps=fps)


if __name__ == '__main__':
    source = cifar10_dvs_source()
    vis_kwargs = {'reverse_xy': True, 'flip_up_down': True}
    # source, vis_kwargs = nmnist_source2(), {}

    print('number of examples:')
    for split in ('train', 'validation'):
        print('{:20}: {}'.format(split, source.examples_per_epoch(split)))
    for example in source.get_dataset('train'):
        vis_example(example, **vis_kwargs)
