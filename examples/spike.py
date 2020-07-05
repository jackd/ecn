import numpy as np
import tensorflow as tf

import events_tfds.vis.anim as anim
from ecn.np_utils.spike import global_spike_threshold, spike_threshold
from events_tfds.events.nmnist import NMNIST
from events_tfds.vis.image import as_frames

decay_time = 10000

dataset = NMNIST().as_dataset(split="train", as_supervised=True)
frame_kwargs = dict(num_frames=20)
stride = 3

for events, label in dataset:
    coords = events["coords"]
    time = events["time"].numpy()
    polarity = events["polarity"].numpy()

    coords = (coords - tf.reduce_min(coords, axis=0)).numpy()
    img_data = [as_frames(coords, time, polarity, **frame_kwargs)]
    sizes = [time.size]
    for _ in range(3):
        coords //= stride

        time, coords = spike_threshold(time, coords, decay_time, coords.shape[0])

        sizes.append(time.size)
        img_data.append(as_frames(coords, time, **frame_kwargs))

    global_times, global_events = global_spike_threshold(time, sizes[-1], decay_time)
    global_times = global_times[:global_events]
    coords = np.zeros((global_events, 2), dtype=np.int64)
    img_data.append(as_frames(coords, global_times, **frame_kwargs))
    sizes.append(global_events)
    print(sizes)

    anim.animate_frames_multi(*img_data, fps=4)
