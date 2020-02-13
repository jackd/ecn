from ecn.benchmark_utils import benchmark
from ecn.benchmark_utils import run_benchmarks
import ecn.np_utils.preprocess as pp


def get_data():
    from events_tfds.events.nmnist import NMNIST
    ds = NMNIST().as_dataset(split='train', as_supervised=True)
    for example, _ in ds.take(1):
        coords = example['coords'].numpy()
        times = example['time'].numpy()
        polarity = example['polarity'].numpy()
        return coords, times, polarity
    raise RuntimeError('Empty dataset??')


coords, times, polarity = get_data()

benchmark(times=times,
          coords=coords,
          polarity=polarity,
          stride=2,
          decay_time=10000,
          event_duration=10000 * 8,
          spatial_buffer_size=32,
          num_layers=3,
          reset_potential=-2.)(pp.preprocess_network_trimmed)

run_benchmarks(50, 100)
