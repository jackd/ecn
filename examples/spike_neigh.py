import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

import numpy as np
import tensorflow as tf
from ecn.np_utils import spike
from ecn.np_utils import neighbors as neigh
# from ecn.np_utils import conv
from events_tfds.events.nmnist import NMNIST
from events_tfds.vis.image import as_frames
import events_tfds.vis.anim as anim

from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def vis_adjacency(indices, split):
    plt.figure()
    ax = plt.gca()
    row_lengths = splits[1:] - splits[:-1]
    i = np.repeat(np.arange(row_lengths.size), row_lengths, axis=0)
    j = indices
    sp = coo_matrix((np.ones(i.shape, dtype=np.bool), (i, j)))
    ax.spy(sp, markersize=1)
    # ax.imshow(sp, cmap='gray')


def vis_graph(coords, time, out_coords, out_time, indices, splits, n=10):
    splits = splits[:n + 1]
    out_coords = out_coords[:n]
    out_time = out_time[:n]
    indices = indices[:splits[-1]]
    in_indices = list(set(indices))

    row_lengths = splits[1:] - splits[:-1]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(coords[in_indices, 0],
               coords[in_indices, 1],
               time[in_indices],
               color='black')
    ax.scatter(out_coords[:, 0], out_coords[:, 1], out_time, color='blue')

    j = indices
    x = coords[j, 0]
    y = coords[j, 1]
    t = time[j]
    in_xyz = np.stack((x, y, t), axis=-1)

    out_x = np.repeat(out_coords[:, 0], row_lengths, axis=0)
    out_y = np.repeat(out_coords[:, 1], row_lengths, axis=0)
    out_t = np.repeat(out_time, row_lengths, axis=0)
    out_xyz = np.stack((out_x, out_y, out_t), axis=-1)

    segs = np.stack((in_xyz, out_xyz), axis=-2)
    assert (segs.shape[1:] == (2, 3))

    ax.add_collection(Line3DCollection(segs))


stride = 2

dataset = NMNIST().as_dataset(split='train', as_supervised=True)
frame_kwargs = dict(num_frames=20)

for events, label in dataset:
    decay_time = 10000
    event_duration = 30000000
    spatial_buffer_size = 32
    max_out_events = 2048

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
        pooled_coords = coords // 2

        out_time, out_coords, out_events = spike.spike_threshold(
            time,
            pooled_coords,
            decay_time=decay_time,
            max_out_events=max_out_events,
            **spike_kwargs,
        )
        print('events {}: {} / {}'.format(i, out_events, max_out_events))
        out_time = out_time[:out_events]
        out_coords = out_coords[:out_events]
        sizes.append(out_events)

        indices, splits = neigh.compute_neighbors(
            time,
            pooled_coords,
            out_time,
            out_coords,
            event_duration=event_duration,
            spatial_buffer_size=spatial_buffer_size,
        )
        indices = indices[:splits[-1]]

        vis_graph(coords,
                  time, (out_coords + 0.5) * stride - 0.5,
                  out_time,
                  indices,
                  splits,
                  n=10)
        vis_adjacency(indices, splits)
        plt.show()

        print('neighbors {}: {}'.format(i, indices.size))

        decay_time *= 2
        event_duration *= 2
        max_out_events //= 4
        spatial_buffer_size //= 2

        time = out_time
        coords = out_coords

        img_data.append(as_frames(coords, time, **frame_kwargs))

    global_times, global_events = spike.global_spike_threshold(
        time,
        max_out_events=max_out_events,
        decay_time=decay_time,
        **spike_kwargs,
    )

    print('-------------')
    print('---GLOBAL----')
    print('events: {} / {}'.format(global_events, max_out_events))
    global_times = global_times[:global_events]
    max_neighbors = 32
    global_indices, global_splits, _ = neigh.compute_global_neighbors(
        time,
        global_times,
        event_duration=event_duration,
        max_neighbors=max_neighbors)
    global_indices = global_indices[:global_splits[-1]]
    print('max_neighbors: {} / {}'.format(global_splits[-1], max_neighbors))

    coords = np.zeros((global_events, 2), dtype=np.int64)
    img_data.append(as_frames(coords, global_times, **frame_kwargs))
    sizes.append(global_events)
    print(sizes)

    anim.animate_frames_multi(*img_data, fps=4)
