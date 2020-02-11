import tensorflow as tf
import numpy as np
from ecn.np_utils.spike import spike_threshold
from ecn.np_utils.partition import temporal_partition
from ecn.np_utils.neighbors import compute_global_neighbors, compute_neighbors
from events_tfds.events.nmnist import NMNIST

stride = 2
decay_time = 50000
event_duration = decay_time * 20


def vis(in_times,
        in_coords,
        polarity,
        out_times,
        out_coords,
        index_values,
        index_splits,
        frames=(2, 4),
        stride=2):
    import matplotlib.pyplot as plt
    num_frames = np.prod(frames)
    stop_times = np.linspace(0, in_times[-1], num_frames + 1)[1:]
    in_partition = temporal_partition(in_times, stop_times)
    out_partition = temporal_partition(out_times, stop_times)

    out_coords = (out_coords + 0.5) * stride - 0.5

    in_ids = np.split(np.arange(in_times.size), in_partition)
    out_ids = np.split(np.arange(out_times.size), out_partition)

    fig, axs = plt.subplots(*frames)
    axs = axs.flatten()
    del fig

    # polarity = polarity.astype(np.uint32)
    for (iid, oid, ax) in zip(in_ids, out_ids, axs):
        # neighborhood
        out_index = oid[0]
        for dy in range(stride):
            for dx in range(stride):
                start, end = index_splits[dy, dx, out_index:out_index + 2]
                in_indices = index_values[dy, dx, start:end]
                ox, oy = out_coords[out_index]
                for i in in_indices:
                    ix, iy = in_coords[i]
                    ax.plot([ox, ix], [oy, iy], color='black')

        p = polarity[iid]
        not_p = np.logical_not(p)
        xi, yi = in_coords[iid].T
        xo, yo = out_coords[oid].T
        # in events
        ax.scatter(xi[p], yi[p], c='red', s=5)
        ax.scatter(xi[not_p], yi[not_p], c='green', s=5)
        # out events
        ax.scatter(xo, yo, c='black', s=5)

    plt.show()


dataset = NMNIST().as_dataset(split='train', as_supervised=True)
frame_kwargs = dict(num_frames=20)

for events, label in dataset:
    coords = events['coords']
    time = events['time']
    polarity = events['polarity'].numpy()

    coords = (coords - tf.reduce_min(coords, axis=0)).numpy()
    time = (time - tf.reduce_min(time)).numpy()
    out_times, out_coords = spike_threshold(time, coords // stride, decay_time,
                                            coords.shape[0])
    index_values, index_lengths = compute_neighbors(
        time,
        coords,
        out_times,
        out_coords,
        event_duration,
        spatial_buffer_size=10,
    )
    vis(time, coords, polarity, out_times, out_coords, index_values,
        index_lengths)
