"""Attempts to visualize event connectedness."""
import collections

import events_tfds.vis.anim as anim
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from events_tfds.events.nmnist import NMNIST
from events_tfds.vis.image import as_frames
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.sparse import coo_matrix

from ecn.ops import grid, lif
from ecn.ops import neighbors as neigh
from ecn.ops import ragged


def vis_adjacency(indices, splits, in_times, out_times, decay_time):
    _, (ax0, ax1, ax2) = plt.subplots(3, 1)
    row_lengths = splits[1:] - splits[:-1]
    i = np.repeat(np.arange(row_lengths.size), row_lengths, axis=0)
    j = indices
    values = np.exp(-(out_times[i] - in_times[j]) / decay_time)
    sp = coo_matrix((values, (i, j)))
    ax0.spy(sp, markersize=1)
    ax1.imshow(sp.todense(), cmap="gray_r")

    mask = values > 1e-2
    print("mean significant adjacency: {}".format(np.mean(mask)))
    sp = coo_matrix((values[mask], (i[mask], j[mask])))
    ax2.spy(sp, markersize=1)
    # ax1.spy(sp)


def vis_graph(coords, time, out_coords, out_time, indices, splits, n=10):
    splits = splits[: n + 1]
    out_coords = out_coords[:n]
    out_time = out_time[:n]
    indices = indices[: splits[-1]]
    in_indices = list(set(indices))

    row_lengths = splits[1:] - splits[:-1]

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.scatter(
        coords[in_indices, 0], coords[in_indices, 1], time[in_indices], color="black"
    )
    ax.scatter(out_coords[:, 0], out_coords[:, 1], out_time, color="blue")

    j = indices
    x = coords[j, 0]
    y = coords[j, 1]
    t = time[j]
    in_xyz = np.stack((x, y, t), axis=-1)

    out_x = np.repeat(out_coords[:, 0], row_lengths, axis=0)
    out_y = np.repeat(out_coords[:, 1], row_lengths, axis=0)
    out_t = np.repeat(out_time, row_lengths, axis=0)
    out_xyz = np.stack((out_x, out_y, out_t), axis=-1)
    # print(out_xyz[:, :2] - in_xyz[:, :2])

    segs = np.stack((in_xyz, out_xyz), axis=-2)
    assert segs.shape[1:] == (2, 3)

    ax.add_collection(Line3DCollection(segs))


ds_kwargs = dict(
    strides=np.array((2, 2)),
    kernel_shape=np.array((5, 5)),
    padding=np.array((2, 2)),
)
ip_kwargs = dict(
    strides=np.array((1, 1)),
    kernel_shape=np.array((3, 3)),
    padding=np.array((1, 1)),
)

GRID_SHAPE = (34, 34)
NUM_LEVELS = 3
lif_kwargs = dict(
    reset_potential=-1.0,
)
DECAY_TIME = 12500
NUM_FRAMES = 20
FPS = 4
SPATIAL_BUFFER = 32

ConvSpec = collections.namedtuple("ConvSpec", ["kernel_shape", "strides", "padding"])

SPECS = (
    ConvSpec((3, 3), (1, 1), (0, 0)),
    ConvSpec((5, 5), (2, 2), (2, 2)),
    ConvSpec((5, 5), (2, 2), (2, 2)),
)

THRESHOLDS = (1.0, 2.0, 2.0)


def process_events(times, coords, polarity):
    print("{} events over {} ms".format(times.shape[0], (times[-1] - times[0]).numpy()))
    decay_time = DECAY_TIME
    spatial_buffer_size = SPATIAL_BUFFER

    event_sizes = [times.shape[0]]
    img_data = [
        as_frames(
            coords.numpy(),
            [0, 1] if times.shape[0] == 0 else times.numpy(),
            polarity.numpy(),
            num_frames=NUM_FRAMES,
            shape=GRID_SHAPE[-1::-1],
        )
    ]
    neigh_sizes = []
    mean_neigh_sizes = []

    shape = tf.constant(GRID_SHAPE, dtype=tf.int64)
    shaped_in_coords = coords
    coords = grid.ravel_multi_index(coords, shape)

    for spec, thresh in zip(SPECS, THRESHOLDS):
        partitions, indices, splits, out_shape = grid.sparse_neighborhood(shape, *spec)

        # resample
        indices_T, splits_T, _ = ragged.transpose_csr(indices, splits, partitions)
        out_times, out_coords = lif.spatial_leaky_integrate_and_fire(
            times,
            coords,
            grid_indices=indices_T,
            grid_splits=splits_T,
            decay_time=decay_time,
            threshold=thresh,
            out_size=tf.math.reduce_prod(out_shape).numpy(),
            **lif_kwargs
        )
        event_sizes.append(out_times.shape[0])

        shaped_out_coords = grid.unravel_index_transpose(out_coords, out_shape)
        img_data.append(
            as_frames(
                shaped_out_coords.numpy(),
                [0, 1] if times.shape[0] == 0 else out_times.numpy(),
                num_frames=NUM_FRAMES,
                shape=out_shape.numpy()[-1::-1],
            )
        )

        # resample conv
        neigh_part, neigh_indices, neigh_splits = neigh.compute_neighbors(
            in_times=times,
            in_coords=coords,
            out_times=out_times,
            out_coords=out_coords,
            grid_indices=indices,
            grid_partitions=partitions,
            grid_splits=splits,
            event_duration=decay_time * 4,
            spatial_buffer_size=spatial_buffer_size,
        )
        del neigh_part

        # neigh_indices, neigh_splits = ragged.mask_rows(neigh_indices,
        #                                                neigh_splits,
        #                                                neigh_part == 1)

        neigh_sizes.append(neigh_indices.shape[0])
        mean_neigh_sizes.append(neigh_indices.shape[0] / out_times.shape[0])

        # vis_adjacency(neigh_indices.numpy(), neigh_splits.numpy(),
        #               times.numpy(), out_times.numpy(), decay_time)
        vis_graph(
            shaped_in_coords.numpy(),
            times.numpy(),
            grid.shift_grid_coords(shaped_out_coords, *spec).numpy(),
            out_times.numpy(),
            neigh_indices.numpy(),
            neigh_splits.numpy(),
            n=10,
        )

        # in-place conv
        partitions, indices, splits = grid.sparse_neighborhood_in_place(
            out_shape, (3, 3)
        )
        neigh_part, neigh_indices, neigh_splits = neigh.compute_neighbors(
            out_times,
            out_coords,
            out_times,
            out_coords,
            grid_partitions=partitions,
            grid_indices=indices,
            grid_splits=splits,
            event_duration=decay_time * 8,
            spatial_buffer_size=spatial_buffer_size,
        )

        neigh_sizes.append(neigh_indices.shape[0])
        mean_neigh_sizes.append(neigh_indices.shape[0] / out_times.shape[0])

        # vis_adjacency(neigh_indices.numpy(), neigh_splits.numpy(),
        #               times.numpy(), out_times.numpy(), decay_time * 2)

        vis_graph(
            shaped_out_coords.numpy(),
            out_times.numpy(),
            shaped_out_coords.numpy(),
            #   grid.shift_grid_coords(shaped_out_coords, (3, 3), (1, 1),
            #                          (1, 1)).numpy(),
            out_times.numpy(),
            neigh_indices.numpy(),
            neigh_splits.numpy(),
            n=10,
        )

        plt.show()
        decay_time *= 2
        shape = out_shape
        coords = out_coords
        shaped_in_coords = shaped_out_coords
        times = out_times

    print("event sizes: ", event_sizes)
    print("neigh sizes: ", neigh_sizes)
    print("mean degree: ", " ".join(["{:.2f}".format(s) for s in mean_neigh_sizes]))
    anim.animate_frames_multi(*img_data, fps=FPS)


shape = GRID_SHAPE
for events, label in NMNIST().as_dataset(split="train", as_supervised=True):
    times = events["time"]
    coords = events["coords"]
    polarity = events["polarity"]
    process_events(events["time"], events["coords"], events["polarity"])
