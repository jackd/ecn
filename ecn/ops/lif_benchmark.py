"""Script investigating performance cost of tf wrapping."""
import tensorflow as tf

import ecn.ops.lif as tf_lif
import numba_stream.lif as np_lif
from ecn.benchmark_utils import BenchmarkManager
from events_tfds.events.nmnist import NMNIST


def get_data():

    ds = NMNIST().as_dataset(split="train", as_supervised=True)
    for example, _ in ds.take(1):
        coords = example["coords"]
        times = example["time"]
        polarity = example["polarity"]
        return coords, times, polarity
    raise RuntimeError("Empty dataset??")


if __name__ == "__main__":
    coords, times, polarity = get_data()
    del coords, polarity

    kwargs = dict(decay_time=10000, threshold=2, reset_potential=-2.0)

    manager = BenchmarkManager()

    manager.benchmark(times=times.numpy(), name="np", **kwargs)(
        np_lif.leaky_integrate_and_fire
    )
    with tf.device("/cpu:0"):
        manager.benchmark(times=times, name="np_wrapped", **kwargs)(
            tf_lif.leaky_integrate_and_fire
        )
    manager.run_benchmarks(50, 100)
