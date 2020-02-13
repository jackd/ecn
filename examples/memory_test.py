import tensorflow as tf
from ecn.pipelines.scnn import pre_map
from ecn.ops import spike as spike_ops_tf
from ecn.np_utils import spike as spike_ops_np
# from ecn.np_utils import preprocess as pp
from ecn.ops import preprocess_tf as pp
# from ecn.ops import preprocess as pp
from events_tfds.events.nmnist import NMNIST
from tqdm import tqdm

ds = NMNIST().as_dataset(split='train', as_supervised=True)

kwargs = dict(stride=2,
              num_layers=3,
              decay_time=10000,
              event_duration=40000,
              threshold=2.,
              reset_potential=-2.,
              spatial_buffer_size=32)

# for events, _ in tqdm(ds):
# pp.preprocess_global_conv(
# events['time'].numpy(),
# decay_time=10000,
# event_duration=40000,
# threshold=2.,
# reset_potential=-2.)

# pp.preprocess_spatial_conv(events['time'].numpy(),
#                            events['coords'].numpy(),
#                            stride=2,
#                            decay_time=10000,
#                            event_duration=40000,
#                            threshold=2.,
#                            reset_potential=-2.,
#                            spatial_buffer_size=32)

# times = events['time'].numpy()
# coords = events['coords'].numpy()


def map_fn(events, labels):
    return tuple(
        tf.nest.flatten(
            pp.preprocess_network_trimmed(events['time'], events['coords'],
                                          events['polarity'], **kwargs)))


# ds = ds.map(pre_map)
# ds = ds.map(map_fn)

for events, labels in tqdm(ds):
    times = events['time']
    coords = events['coords']
    polarity = events['polarity']

    # times = times.numpy()
    # coords = coords.numpy()
    # polarity = polarity.numpy()

    # pp.preprocess_spatial_conv(times,
    #                            coords,
    #                            stride=2,
    #                            decay_time=10000,
    #                            event_duration=40000,
    #                            threshold=2.,
    #                            reset_potential=-2.,
    #                            spatial_buffer_size=32)
    # pp.preprocess_network(
    #     times,
    #     coords,
    #     # polarity,
    #     **kwargs,
    # )
    pp.preprocess_network_trimmed(
        times,
        coords,
        polarity,
        **kwargs,
    )
