# import os
# os.environ['NUMBA_DISABLE_JIT'] = '1'

import numpy as np
import tensorflow as tf
from ecn.np_utils import spike
from ecn.np_utils import neighbors as neigh
from ecn.np_utils import grid
import ecn.np_utils.ragged as ragged
# from ecn.np_utils import conv
from events_tfds.events.nmnist import NMNIST
from events_tfds.vis.image import as_frames
import events_tfds.vis.anim as anim

from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def vis_adjacency(indices, split, in_times, out_times, decay_time):
    _, (ax0, ax1, ax2) = plt.subplots(3, 1)
    row_lengths = splits[1:] - splits[:-1]
    i = np.repeat(np.arange(row_lengths.size), row_lengths, axis=0)
    j = indices
    values = np.exp(-(out_times[i] - in_times[j]) / decay_time)
    sp = coo_matrix((values, (i, j)))
    ax0.spy(sp, markersize=1)
    ax1.imshow(sp.todense(), cmap='gray_r')

    mask = values > 1e-2
    print('mean significant adjacency: {}'.format(np.mean(mask)))
    sp = coo_matrix((values[mask], (i[mask], j[mask])))
    ax2.spy(sp, markersize=1)
    # ax1.spy(sp)


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
all_event_counts = []

for events, label in dataset.take(100):
    decay_time = 7500
    event_duration = decay_time * 4
    spatial_buffer_size = 32
    in_shape = np.array((34, 34))

    coords = events['coords'].numpy()
    time = events['time'].numpy()
    polarity = events['polarity'].numpy()

    print('{} events over {} dt'.format(time.size, np.max(time) - np.min(time)))

    coords_1d = grid.ravel_multi_index_transpose(coords, in_shape)
    img_data = [as_frames(coords, time, polarity, **frame_kwargs)]
    sizes = [time.size]
    spike_kwargs = dict(threshold=2., reset_potential=-2.)
    neigh_kwargs = dict(kernel_shape=np.array((3, 3)),
                        strides=np.array((2, 2)),
                        padding=np.array((0, 0)))

    for i in range(2):
        print('-------------')
        print('---LAYER {}---'.format(i))
        p, neigh_indices, splits, out_shape = grid.sparse_neighborhood(
            in_shape, **neigh_kwargs)
        neigh_indices_T, splits_T = ragged.transpose_csr(neigh_indices, splits)

        out_time, out_coords_1d = spike.spike_threshold_1d(
            time,
            coords_1d,
            decay_time=decay_time,
            neigh_indices=neigh_indices_T,
            neigh_splits=splits_T,
            **spike_kwargs,
        )
        out_coords = grid.unravel_index_transpose(out_coords_1d, out_shape)
        out_events = out_time.size
        print('events {}: {}'.format(i, out_events))
        sizes.append(out_events)

        # partitions, indices, splits = neigh.compute_neighbors_nd(
        #     time,
        #     coords.copy(),
        #     out_time,
        #     out_coords * stride,
        #     neighbor_offsets=neigh.neighbor_offsets((stride, stride)),
        #     event_duration=event_duration,
        #     spatial_buffer_size=spatial_buffer_size,
        # )
        # mask = np.zeros((time.size,), dtype=np.bool)
        # mask[indices] = True
        # ri = neigh.reindex_index(mask)
        # indices0 = neigh.reindex(indices, ri)
        # print('events: ', np.count_nonzero(mask), time.size)

        # vis_graph(coords,
        #           time,
        #           out_coords * stride + 0.5,
        #           out_time,
        #           indices,
        #           splits,
        #           n=10)
        # vis_adjacency(indices, splits, time, out_time, decay_time)
        # plt.show()

        # print('neighbors {}: {}'.format(i, indices.size))

        # decay_time *= 2
        # event_duration *= 2
        # spatial_buffer_size //= 2

        # # in place
        # partitions, indices, splits = neigh.compute_neighbors_nd(
        #     out_time,
        #     out_coords.copy(),
        #     out_time,
        #     out_coords.copy(),
        #     neighbor_offsets=neigh.neighbor_offsets((3, 3)),
        #     event_duration=event_duration,
        #     spatial_buffer_size=spatial_buffer_size)

        # vis_graph(out_coords,
        #           out_time,
        #           out_coords,
        #           out_time,
        #           indices,
        #           splits,
        #           n=10)
        # vis_adjacency(indices, splits, out_time, out_time, decay_time)
        # plt.show()

        time = out_time
        coords_1d = out_coords_1d
        in_shape = out_shape

        img_data.append(as_frames(out_coords, time, **frame_kwargs))

    global_times = spike.global_spike_threshold(
        time,
        decay_time=decay_time,
        **spike_kwargs,
    )

    print('-------------')
    print('---GLOBAL----')
    global_events = global_times.size
    print('events: {}'.format(global_events))
    max_neighbors = 32
    global_indices, global_splits = neigh.compute_global_neighbors(
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
    all_event_counts.append(sizes)

    anim.animate_frames_multi(*img_data, fps=4)

print(np.mean(all_event_counts, axis=0))
