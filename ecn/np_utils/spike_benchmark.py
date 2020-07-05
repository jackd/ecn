import tensorflow as tf

import ecn.np_utils.spike as np_spike
import ecn.ops.spike as tf_spike
from ecn.benchmark_utils import BenchmarkManager


def get_data():
    from events_tfds.events.nmnist import NMNIST

    ds = NMNIST().as_dataset(split="train", as_supervised=True)
    for example, _ in ds.take(1):
        coords = example["coords"]
        times = example["time"]
        polarity = example["polarity"]
        return coords, times, polarity
    raise RuntimeError("Empty dataset??")


coords, times, polarity = get_data()

kwargs = dict(decay_time=10000, threshold=2, reset_potential=-2.0)

manager = BenchmarkManager()

manager.benchmark(times=times.numpy(), name="np", **kwargs)(
    np_spike.global_spike_threshold
)
with tf.device("/cpu:0"):
    # manager.benchmark(times=times, name='tf',
    #                   **kwargs)(tf_spike.global_spike_threshold_tf)
    manager.benchmark(times=times, name="np_wrapped", **kwargs)(
        tf_spike.global_spike_threshold
    )
    manager.run_benchmarks(50, 100)

# manager = BenchmarkManager()
# manager.benchmark(times=times.numpy(),
#                   coords=coords.numpy(),
#                   name='spatial_np',
#                   **kwargs)(np_spike.spike_threshold)
# manager.benchmark(times=times, coords=coords, name='spatial_np',
#                   **kwargs)(tf_spike.spike_threshold)
# manager.run_benchmarks(50, 100)
