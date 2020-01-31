import numpy as np
import tensorflow as tf
from ecn.np_utils.spike import spike_threshold, global_spike_threshold
from events_tfds.events.nmnist import NMNIST
from events_tfds.events.utils import as_frames
import events_tfds.events.anim as anim

decay_time = 10000

dataset = NMNIST().as_dataset(split='train', as_supervised=True)
frame_kwargs = dict(num_frames=20)

for events, label in dataset:
    coords = events['coords']
    time = events['time'].numpy()
    polarity = events['polarity'].numpy()

    coords = (coords - tf.reduce_min(coords, axis=0)).numpy()
    img_data = [as_frames(coords, time, polarity, **frame_kwargs)]
    sizes = [time.size]
    for _ in range(3):
        coords //= 2

        time, coords, out_events = spike_threshold(time, coords, decay_time,
                                                   coords.shape[0])
        time = time[:out_events]
        coords = coords[:out_events]
        sizes.append(out_events)
        img_data.append(as_frames(coords, time, **frame_kwargs))

    global_times, global_events = global_spike_threshold(
        time, sizes[-1], decay_time)
    global_times = global_times[:global_events]
    coords = np.zeros((global_events, 2), dtype=np.int64)
    img_data.append(as_frames(coords, global_times, **frame_kwargs))
    sizes.append(global_events)
    print(sizes)

    anim.animate_frames_multi(*img_data, fps=4)
