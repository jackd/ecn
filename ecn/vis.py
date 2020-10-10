import gin
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import logging

import ecn.components as comp
import events_tfds.vis.anim as anim
import multi_graph as mg
from events_tfds.vis.image import as_frame, as_frames
from kblocks.framework.sources import DataSource


@gin.configurable(module="ecn.vis")
def vis_streams(
    build_fn,
    base_source: DataSource,
    num_frames=20,
    fps=4,
    group_size=1,
    skip_vis=False,
):

    builder = mg.MultiGraphBuilder(batch_size=1)
    with builder:
        with comp.stream_accumulator() as streams:
            inputs = tf.nest.map_structure(
                builder.pre_cache_input, base_source.element_spec
            )
            logging.info("Building multi graph...")
            build_fn(*inputs, **base_source.meta)
            logging.info("Successfully built!")
        outputs = []
        static_shapes = []
        for stream in streams:
            if isinstance(stream, comp.SpatialStream):
                outputs.append((stream.times, stream.shaped_coords))
                static_shapes.append(stream.grid.static_shape)
            else:
                outputs.append((stream.times,))
                static_shapes.append(None)

        has_polarity = "polarity" in inputs[0]
        if has_polarity:
            polarity = inputs[0]["polarity"]
            outputs = (tuple(outputs), polarity)
        else:
            outputs = tuple(outputs)

    flat_inputs = tf.nest.flatten(inputs, expand_composites=True)
    fn = mg.subgraph(
        flat_inputs[0].graph.as_graph_def(add_shapes=True),
        flat_inputs,
        tf.nest.flatten(outputs, expand_composites=True),
    )

    def map_fn(*args, **kwargs):
        flat_in = tf.nest.flatten((args, kwargs), expand_composites=True)
        flat_out = fn(*flat_in)
        return tf.nest.pack_sequence_as(outputs, flat_out, expand_composites=True)

    del builder
    dataset = base_source.get_dataset("train").map(map_fn)
    sizes = np.zeros((group_size, len(streams)), dtype=np.int64)
    for i, example in enumerate(dataset):
        if has_polarity:
            streams, polarity = example
            polarity = polarity.numpy()
        else:
            streams = example
            polarity = None
        img_data = []
        ii = i % group_size
        sizes[ii] = [s[0].shape[0] for s in streams]
        if ii == group_size - 1:
            print(f"{i}: sizes: {np.mean(sizes, axis=0).astype(np.int64)}")
            if group_size > 1:
                print(f"{i}: stds : {np.std(sizes, axis=0).astype(np.int64)}")
        if skip_vis:
            continue
        for i, (stream_data, shape) in enumerate(zip(streams, static_shapes)):
            if len(stream_data) == 1:
                assert shape is None
                shape = [1, 1]
                (times,) = stream_data
                coords = np.zeros((times.shape[0], 2), dtype=np.int64)
            else:
                times, coords = stream_data
                coords = coords.numpy()

            if times.shape[0] > 0:
                img_data.append(
                    as_frames(
                        coords,
                        times.numpy(),
                        num_frames=num_frames,
                        polarity=polarity[: coords.shape[0]] if i == 0 else None,
                        shape=shape,
                    )
                )

        anim.animate_frames_multi(*img_data, fps=fps)


@gin.configurable(module="ecn.vis")
def vis_streams1d(build_fn, base_source: DataSource):

    # colors = np.array([
    #     [0, 0, 0],
    #     [255, 0, 0],
    #     [0, 255, 0],
    #     [0, 0, 255],
    #     [255, 255, 0],
    #     [255, 0, 255],
    #     [0, 255, 255],
    # ],
    #                   dtype=np.uint8)

    builder = mg.MultiGraphBuilder(batch_size=1)
    with builder:
        with comp.stream_accumulator() as streams:
            inputs = tf.nest.map_structure(
                builder.pre_cache_input, base_source.element_spec
            )
            logging.info("Building multi graph...")
            build_fn(*inputs, **base_source.meta)
            logging.info("Successfully built!")
        outputs = []
        static_shapes = []
        for stream in streams:
            if isinstance(stream, comp.SpatialStream):
                outputs.append((stream.times, stream.shaped_coords))
                static_shapes.append(stream.grid.static_shape)
            else:
                outputs.append((stream.times,))
                static_shapes.append((1,))

        outputs = tuple(outputs)

    flat_inputs = tf.nest.flatten(inputs, expand_composites=True)
    fn = mg.subgraph(
        flat_inputs[0].graph.as_graph_def(add_shapes=True),
        flat_inputs,
        tf.nest.flatten(outputs, expand_composites=True),
    )

    def map_fn(*args, **kwargs):
        flat_in = tf.nest.flatten((args, kwargs), expand_composites=True)
        flat_out = fn(*flat_in)
        out = tf.nest.pack_sequence_as(outputs, flat_out, expand_composites=True)
        return out

    del builder
    dataset = base_source.get_dataset("train").map(map_fn)
    print("num_channels: {}".format([shape[0] for shape in static_shapes]))
    for streams in dataset:
        print("Sizes: {}".format([s[0].shape[0] for s in streams]))
        xys = []
        for (stream_data, shape) in zip(streams, static_shapes):
            if len(stream_data) == 1:
                (times,) = stream_data
                coords = 0.5 * np.ones((times.shape[0],), dtype=np.int64)
                xys.append((coords, times))
            else:
                size = shape[0]
                times, coords = (x.numpy() for x in stream_data)
                coords = np.squeeze(coords, axis=-1)
                assert np.max(coords) < size
                xys.append(((coords + 0.5) / size, times))
        plt.figure()
        for i, xy in enumerate(xys):
            plt.scatter(*xy, s=(i + 1))
        plt.show()


def _vis_single_adjacency(indices, splits, in_times, out_times, decay_time, ax0, ax1):
    from scipy.sparse import coo_matrix

    if indices.size == 0:
        return
    print(f"{in_times.size} -> {out_times.size} events")
    print(f"{indices.size} edges")
    row_lengths = splits[1:] - splits[:-1]
    i = np.repeat(np.arange(row_lengths.size), row_lengths, axis=0)
    j = indices
    values = np.exp(-(out_times[i] - in_times[j]) / decay_time)
    sp = coo_matrix((values, (i, j)))
    print(f"Sparsity: {sp.nnz / np.prod(sp.shape) * 100}%")
    ax0.spy(sp, markersize=0.5)
    ax0.set_xticks([])

    ax1.hist(values)


@gin.configurable(module="ecn.vis")
def vis_adjacency(build_fn, base_source: DataSource):
    builder = mg.MultiGraphBuilder(batch_size=1)
    with builder:
        with comp.stream_accumulator() as streams:
            with comp.convolver_accumulator() as convolvers:
                inputs = tf.nest.map_structure(
                    builder.pre_cache_input, base_source.element_spec
                )
                logging.info("Building multi graph...")
                build_fn(*inputs, **base_source.meta)
                logging.info("Successfully built!")
        stream_indices = {s: i for i, s in enumerate(streams)}
        outputs = []
        decay_times = []
        stream_indices = tuple(
            (stream_indices[c.in_stream], stream_indices[c.out_stream])
            for c in convolvers
        )
        for c in convolvers:
            outputs.append((c.indices, c.splits, c.in_stream.times, c.out_stream.times))
            decay_times.append(c.decay_time)

    flat_inputs = tf.nest.flatten(inputs, expand_composites=True)
    fn = mg.subgraph(
        flat_inputs[0].graph.as_graph_def(add_shapes=True),
        flat_inputs,
        tf.nest.flatten(outputs, expand_composites=True),
    )

    def map_fn(*args, **kwargs):
        flat_in = tf.nest.flatten((args, kwargs), expand_composites=True)
        flat_out = fn(*flat_in)
        return tf.nest.pack_sequence_as(outputs, flat_out, expand_composites=True)

    del builder
    dataset = base_source.get_dataset("train").map(map_fn)
    num_convolvers = len(convolvers)
    for convolvers in dataset:
        convolvers = tf.nest.map_structure(lambda c: c.numpy(), convolvers)
        _, axes = plt.subplots(2, num_convolvers)
        if num_convolvers == 1:
            axes = np.expand_dims(axes, axis=-1)
        for i, (conv_data, decay_time, indices) in enumerate(
            zip(convolvers, decay_times, stream_indices)
        ):
            s0, s1 = indices
            title = f"Conv {s0} -> {s1}"
            ax0, ax1 = axes[:, i]
            _vis_single_adjacency(*conv_data, decay_time=decay_time, ax0=ax0, ax1=ax1)
            ax0.title.set_text(title)

        plt.show()


@gin.configurable(module="ecn.vis")
def vis_example(
    example,
    num_frames=20,
    fps=4,
    reverse_xy=False,
    flip_up_down=False,
    class_names=None,
):

    features, label = example
    coords = features["coords"]
    time = features["time"]
    time = (time - tf.reduce_min(time)).numpy()
    polarity = features["polarity"].numpy()
    if class_names is not None:
        print(class_names[label.numpy()])
    else:
        print(label.numpy())

    coords = (coords - tf.reduce_min(coords, axis=0)).numpy()
    print(f"{time.shape[0]} events over {time[-1] - time[0]} dt")
    if reverse_xy:
        coords = coords[:, -1::-1]

    if num_frames == 1:
        frame = as_frame(coords, polarity)
        plt.imshow(frame)
        plt.show()
    else:
        frames = as_frames(
            coords, time, polarity, num_frames=num_frames, flip_up_down=flip_up_down
        )
        anim.animate_frames(frames, fps=fps)


if __name__ == "__main__":
    import functools
    from ecn import builders, sources

    build_fn = functools.partial(
        builders.inception_vox_pooling,
        num_levels=3,
        vox_start=0,
        # reset_potential=-1.0,
        # threshold=0.75,
        # decay_time=1000,
        # max_events=16000
        # initial_pooling=2,
        # max_events=300000,
        # decay_time_expansion_rate=np.sqrt(2),
        # num_levels=6,
        # initial_pooling=2
        # decay_time_expansion_rate=2
        #     #  num_levels=3,
        #     #  vox_start=0,
    )
    source = sources.nmnist_source()
    # source = sources.mnist_dvs_source()
    # source = sources.cifar10_dvs_source()

    # build_fn = builders.inception_vox_pooling
    # source = sources.cifar10_dvs_source()

    # source = sources.asl_dvs_source()
    # source = sources.ncars_source()
    # source = sources.ncaltech101_source()
    # build_fn = functools.partial(
    # builders.inception_multi_graph_v2,
    # builders.inception_flash_multi_graph,
    # decay_time=2000,
    # reset_potential=-1.
    # )
    # group_size = 1
    # build_fn = builders.simple1d_graph
    # source = sources.ntidigits_source()

    # multi_graph_trainable(build_fn, source, batch_size=2)
    # print('built successfully')

    # vis_streams1d(build_fn, source)
    # vis_streams(build_fn, source)
    # vis_streams(build_fn, source, group_size=group_size, skip_vis=True)
    vis_adjacency(build_fn, source)
    #     from ecn.nmnist import simple_multi_graph
    #     from ecn.nmnist import nmnist_source
    #     trainable = multi_grpah_trainable(simple_multi_graph,
    #                                       nmnist_source(),
    #                                       batch_size=16,
    #                                       compiler=compile_stream_classifier)
    #     source = trainable.source
    #     for example in source:
    #         pass
    # source_fn = sources.nmnist_source
    # benchmark_source(source_fn, build_fn)
