# pylint: disable=import-outside-toplevel
import gin

from kblocks.framework.sources import TfdsSource


@gin.configurable(module="ecn.sources")
def ncars_source(**kwargs):
    from events_tfds.events import ncars

    return TfdsSource(
        ncars.Ncars(),
        split_map={"validation": "test"},
        meta=dict(grid_shape=ncars.GRID_SHAPE, num_classes=1),
        **kwargs,
    )


@gin.configurable(module="ecn.sources")
def nmnist_source(**kwargs):
    from events_tfds.events import nmnist

    return TfdsSource(
        nmnist.NMNIST(),
        split_map={"validation": "test"},
        meta=dict(num_classes=nmnist.NUM_CLASSES, grid_shape=nmnist.GRID_SHAPE),
        **kwargs,
    )


@gin.configurable(module="ecn.sources")
def mnist_dvs_source(scale=16, train_percent=90, **kwargs):
    from events_tfds.events import mnist_dvs

    config = {4: mnist_dvs.SCALE4, 8: mnist_dvs.SCALE8, 16: mnist_dvs.SCALE16,}[scale]
    builder = mnist_dvs.MnistDVS(config=config)
    if kwargs.get("download_and_prepare", True):
        builder.download_and_prepare()

    return TfdsSource(
        builder,
        split_map={
            "train": f"train[:{train_percent}%]",
            "validation": f"train[{train_percent}%:]",
        },
        meta=dict(num_classes=mnist_dvs.NUM_CLASSES, grid_shape=mnist_dvs.GRID_SHAPE),
        **kwargs,
    )


@gin.configurable(module="ecn.sources")
def ncaltech101_source(train_percent=90, **kwargs):
    from events_tfds.events import ncaltech101

    builder = ncaltech101.Ncaltech101()
    if kwargs.get("download_and_prepare", True):
        builder.download_and_prepare()

    return TfdsSource(
        builder,
        split_map={
            "train": f"train[:{train_percent}%]",
            "validation": f"train[{train_percent}%:]",
        },
        meta=dict(
            num_classes=ncaltech101.NUM_CLASSES, grid_shape=ncaltech101.GRID_SHAPE
        ),
        **kwargs,
    )


@gin.configurable(module="ecn.sources")
def cifar10_dvs_source(train_percent=90, **kwargs):
    from events_tfds.events import cifar10_dvs

    builder = cifar10_dvs.Cifar10DVS()
    if kwargs.get("download_and_prepare", True):
        builder.download_and_prepare()

    return TfdsSource(
        builder,
        split_map={
            "train": f"train[:{train_percent}%]",
            "validation": f"train[{train_percent}%:]",
        },
        meta=dict(
            num_classes=cifar10_dvs.NUM_CLASSES, grid_shape=cifar10_dvs.GRID_SHAPE
        ),
        **kwargs,
    )


@gin.configurable(module="ecn.sources")
def asl_dvs_source(train_percent=80, **kwargs):
    from events_tfds.events import asl_dvs

    builder = asl_dvs.AslDvs()
    if kwargs.get("download_and_prepare", True):
        builder.download_and_prepare()

    return TfdsSource(
        builder,
        split_map={
            "train": f"train[:{train_percent}%]",
            "validation": f"train[{train_percent}%:]",
        },
        meta=dict(num_classes=asl_dvs.NUM_CLASSES, grid_shape=asl_dvs.GRID_SHAPE),
        **kwargs,
    )


@gin.configurable(module="ecn.sources")
def ntidigits_source():
    from events_tfds.events import ntidigits

    builder = ntidigits.Ntidigits()

    return TfdsSource(
        builder,
        split_map={"validation": "test"},
        meta=dict(
            num_classes=ntidigits.NUM_CLASSES, grid_shape=(ntidigits.NUM_CHANNELS,)
        ),
    )
