from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.sparse import coo_matrix

import events_tfds.vis.anim as anim
from ecn import components as comp
from events_tfds.events.nmnist import GRID_SHAPE, NMNIST
from events_tfds.vis.image import as_frames
from multi_graph import DebugBuilderContext


def vis_adjacency(convolver: comp.Convolver):
    # _, (ax0, ax1, ax2) = plt.subplots(3, 1)
    _, (ax0, ax1) = plt.subplots(1, 2)
    splits = convolver.splits.numpy()
    indices = convolver.indices.numpy()
    out_times = convolver.out_stream.times.numpy()
    in_times = convolver.in_stream.times.numpy()
    decay_time = convolver.decay_time
    row_lengths = splits[1:] - splits[:-1]
    i = np.repeat(np.arange(row_lengths.size), row_lengths, axis=0)
    j = indices
    values = np.exp(-(out_times[i] - in_times[j]) / decay_time)
    sp = coo_matrix((values, (i, j)))
    ax0.spy(sp, markersize=1)
    dense = sp.todense()
    ax1.imshow(dense, cmap="gray_r")
    print("Density: {}".format(values.size / np.prod(dense.shape)))


def vis_graph(convolver: comp.Convolver[comp.SpatialStream, comp.SpatialStream], n=10):
    stride = convolver.in_stream.grid.shape // convolver.out_stream.grid.shape
    splits = convolver.splits.numpy()
    indices = convolver.indices.numpy()
    out_coords = (convolver.out_stream.shaped_coords * stride).numpy()
    out_times = convolver.out_stream.times.numpy()
    in_coords = convolver.in_stream.shaped_coords.numpy()
    in_times = convolver.in_stream.times.numpy()

    splits = splits[: n + 1]
    out_coords = out_coords[:n]
    out_times = out_times[:n]
    indices = indices[: splits[-1]]
    in_indices = list(set(indices))

    row_lengths = splits[1:] - splits[:-1]

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.scatter(
        in_coords[in_indices, 0],
        in_coords[in_indices, 1],
        in_times[in_indices],
        color="black",
    )
    ax.scatter(out_coords[:, 0], out_coords[:, 1], out_times, color="blue")

    j = indices
    x = in_coords[j, 0]
    y = in_coords[j, 1]
    t = in_times[j]
    in_xyz = np.stack((x, y, t), axis=-1)

    out_x = np.repeat(out_coords[:, 0], row_lengths, axis=0)
    out_y = np.repeat(out_coords[:, 1], row_lengths, axis=0)
    out_t = np.repeat(out_times, row_lengths, axis=0)
    out_xyz = np.stack((out_x, out_y, out_t), axis=-1)

    segs = np.stack((in_xyz, out_xyz), axis=-2)
    assert segs.shape[1:] == (2, 3)

    ax.add_collection(Line3DCollection(segs))


def vis_streams(stream0, *streams, polarity=None):
    img_data = [
        as_frames(
            stream0.shaped_coords.numpy(),
            [0, 1] if stream0.times.shape[0] == 0 else stream0.times.numpy(),
            None if polarity is None else polarity.numpy(),
            num_frames=NUM_FRAMES,
            shape=stream0.grid.static_shape,
        )
    ]
    for stream in streams[:-1]:
        img_data.append(
            as_frames(
                stream.shaped_coords.numpy(),
                stream.times.numpy(),
                shape=stream.grid.static_shape,
                num_frames=NUM_FRAMES,
            )
        )
    glob = streams[-1]
    times = glob.times.numpy()
    img_data.append(
        as_frames(
            np.zeros((times.size, 2), dtype=np.int64),
            times,
            shape=[1, 1],
            num_frames=NUM_FRAMES,
        )
    )
    anim.animate_frames_multi(*img_data, fps=FPS)


DECAY_TIME = 10000
NUM_FRAMES = 20
FPS = 4
SPATIAL_BUFFER = 32
LIF_KWARGS = dict(reset_potential=-2.0, threshold=1.1)
# LIF_KWARGS = {}


def process_example(events, label):
    del label
    decay_time = DECAY_TIME
    with DebugBuilderContext():
        coords = events["coords"]
        times = events["time"]
        grid = comp.Grid(GRID_SHAPE)
        link = grid.link((3, 3), (1, 1), (0, 0))

        in_stream = comp.SpatialStream(grid, times, coords)
        out_stream = comp.spatial_leaky_integrate_and_fire(
            in_stream, link, decay_time=decay_time, **LIF_KWARGS
        )
        streams: List[comp.Stream] = [in_stream, out_stream]

        convolver = comp.spatio_temporal_convolver(
            link,
            in_stream,
            out_stream,
            decay_time=decay_time,
            spatial_buffer_size=SPATIAL_BUFFER,
        )
        convolvers: List[List[Any]] = [[None, convolver]]

        in_stream = out_stream
        del out_stream

        for _ in range(2):
            # in place
            decay_time *= 2
            link = in_stream.grid.self_link((3, 3))
            ip_convolver = comp.spatio_temporal_convolver(
                link,
                in_stream,
                in_stream,
                decay_time=decay_time,
                spatial_buffer_size=SPATIAL_BUFFER,
            )

            # link = in_stream.grid.link((5, 5), (2, 2), (2, 2))
            link = in_stream.grid.link((3, 3), (2, 2), (1, 1))
            out_stream = comp.spatial_leaky_integrate_and_fire(
                in_stream, link, decay_time=decay_time, **LIF_KWARGS
            )
            streams.append(out_stream)

            ds_convolver = comp.spatio_temporal_convolver(
                link,
                in_stream,
                out_stream,
                decay_time=decay_time,
                spatial_buffer_size=SPATIAL_BUFFER,
            )
            convolvers.append([ip_convolver, ds_convolver])
            in_stream = out_stream
            del out_stream

        decay_time *= 2
        global_stream = comp.leaky_integrate_and_fire(
            in_stream, decay_time=decay_time, **LIF_KWARGS
        )
        streams.append(global_stream)
        decay_time *= 2
        flat_convolver = comp.flatten_convolver(in_stream, global_stream, decay_time)
        temporal_convolver = comp.temporal_convolver(
            global_stream, global_stream, decay_time
        )
        convolvers.append([flat_convolver, temporal_convolver])

        return tf.stack([tf.size(stream.times) for stream in streams])


out = []
batch_size = 32
for i, (events, label) in enumerate(
    NMNIST().as_dataset(split="train", as_supervised=True)
):
    sizes = process_example(events, label)
    out.append(sizes)
    if (i + 1) % batch_size == 0:
        out = tf.stack(out, axis=-1)
        print(tf.reduce_mean(out, axis=-1))
        out = []
