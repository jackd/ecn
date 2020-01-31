import numpy as np
import tensorflow as tf
from ecn.np_utils import spike
from ecn.np_utils import neighbors as neigh
# from ecn.np_utils import conv
from events_tfds.events.nmnist import NMNIST
from events_tfds.events.utils import as_frames
import events_tfds.events.anim as anim

stride = 2

dataset = NMNIST().as_dataset(split='train', as_supervised=True)
frame_kwargs = dict(num_frames=20)

for events, label in dataset:
    decay_time = 10000
    event_duration = 30000
    spatial_buffer_size = 8
    max_out_events = 2048
    max_neighbors = 2048

    coords = events['coords']
    time = events['time'].numpy()
    polarity = events['polarity'].numpy()

    print('dt = {}'.format(np.max(time) - np.min(time)))

    coords = (coords - tf.reduce_min(coords, axis=0)).numpy()
    img_data = [as_frames(coords, time, polarity, **frame_kwargs)]
    sizes = [time.size]
    spike_kwargs = dict(threshold=2., reset_potential=-2.)

    for i in range(3):
        print('-------------')
        print('---LAYER {}---'.format(i))

        out_time, out_coords, out_events = spike.spike_threshold(
            time,
            coords // 2,
            decay_time,
            max_out_events,
            **spike_kwargs,
        )
        print('events {}: {} / {}'.format(i, out_events, max_out_events))
        out_time = out_time[:out_events]
        out_coords = out_coords[:out_events]
        sizes.append(out_events)

        indices, splits, _ = neigh.compute_neighbors(
            time,
            coords,
            out_time,
            out_coords,
            stride,
            event_duration,
            spatial_buffer_size,
            max_neighbors,
        )
        print('max_neighbors {}: {} / {}'.format(i, splits[:, :, -1].flatten(),
                                                 max_neighbors))

        decay_time *= 2
        event_duration *= 2
        max_out_events //= 4
        max_neighbors //= 2
        spatial_buffer_size *= 2

        time = out_time
        coords = out_coords

        img_data.append(as_frames(coords, time, **frame_kwargs))

    global_times, global_events = spike.global_spike_threshold(
        time, max_out_events, decay_time, **spike_kwargs)
    print('-------------')
    print('---GLOBAL----')
    print('events: {} / {}'.format(global_events, max_out_events))
    global_times = global_times[:global_events]
    global_indices, global_splits, _ = neigh.compute_global_neighbors(
        time,
        global_times,
        event_duration=event_duration,
        max_neighbors=max_neighbors)
    print('max_neighbors: {} / {}'.format(global_splits[-1], max_neighbors))

    coords = np.zeros((global_events, 2), dtype=np.int64)
    img_data.append(as_frames(coords, global_times, **frame_kwargs))
    sizes.append(global_events)
    print(sizes)

    anim.animate_frames_multi(*img_data, fps=4)
