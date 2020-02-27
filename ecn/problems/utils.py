from ecn.problems.builders import simple1d_graph
from typing import Optional, Callable
from absl import logging
import functools
import gin
import tensorflow as tf
import numpy as np
from kblocks.framework.sources import DataSource
from kblocks.framework.pipelines import BasePipeline
from kblocks.framework.sources import PipelinedSource
from kblocks.framework.trainable import Trainable
from ecn import multi_graph as mg
from ecn import components as comp
# from kblocks.keras import losses
# from kblocks.keras import metrics
# from kblocks.keras import optimizers

losses = tf.keras.losses
metrics = tf.keras.metrics
optimizers = tf.keras.optimizers


@gin.configurable(module='ecn.utils')
def compile_stream_classifier(model: tf.keras.Model,
                              optimizer=None,
                              target='final'):
    if optimizer is None:
        optimizer = optimizers.Adam()
    if target == 'final':
        loss_weights = {'final': 1.0, 'stream': 0.0}
    else:
        loss_weights = {'final': 0.0, 'stream': 1.0}
    model.compile(
        loss={
            'final':
                losses.SparseCategoricalCrossentropy(from_logits=True,
                                                     name='final_xe'),
            'stream':
                losses.SparseCategoricalCrossentropy(from_logits=True,
                                                     reduction='sum',
                                                     name='stream_xe')
        },
        metrics={
            'final': metrics.SparseCategoricalAccuracy(name='acc'),
            'stream': metrics.SparseCategoricalAccuracy(name='acc')
        },
        loss_weights=loss_weights,
        optimizer=optimizer,
    )


@gin.configurable(module='ecn.utils')
def compile_classifier(model: tf.keras.Model, optimizer=None):
    if optimizer is None:
        optimizer = optimizers.Adam()
    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[metrics.SparseCategoricalAccuracy(name='acc')],
                  optimizer=optimizer)


@gin.configurable(module='ecn.utils')
def compile_binary_classifier(model: tf.keras.Model, optimizer=None):
    if optimizer is None:
        optimizer = optimizers.Adam()
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  metrics=[metrics.BinaryAccuracy(name='acc')],
                  optimizer=optimizer)


@gin.configurable(module='ecn.utils')
def compile_stream_binary_classifier(model: tf.keras.Model,
                                     optimizer=None,
                                     target='final'):
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()
    if target == 'final':
        loss_weights = {'final': 1.0, 'stream': 0.0}
    else:
        loss_weights = {'final': 0.0, 'stream': 1.0}
    model.compile(
        loss={
            'final':
                losses.BinaryCrossentropy(from_logits=True, name='final_xe'),
            'stream':
                losses.BinaryCrossentropy(from_logits=True,
                                          reduction='sum',
                                          name='stream_xe')
        },
        metrics={
            'final': metrics.BinaryAccuracy(name='acc'),
            'stream': metrics.BinaryAccuracy(name='acc')
        },
        loss_weights=loss_weights,
        optimizer=optimizer,
    )


@gin.configurable(module='ecn.utils')
def get_cache_name(problem_id, model_id):
    return f'{problem_id}_{model_id}'


@gin.configurable(module='ecn.utils')
def multi_graph_trainable(build_fn: Callable,
                          base_source: DataSource,
                          batch_size: int,
                          compiler=compile_stream_classifier,
                          model_dir: Optional[str] = None,
                          **pipeline_kwargs):
    logging.info('Building multi graph...')
    built = mg.build_multi_graph(
        functools.partial(build_fn, **base_source.meta),
        base_source.example_spec, batch_size)
    logging.info('Successfully built!')

    pipeline = BasePipeline(batch_size,
                            pre_cache_map=built.pre_cache_map,
                            pre_batch_map=built.pre_batch_map,
                            post_batch_map=built.post_batch_map,
                            **pipeline_kwargs)
    source = PipelinedSource(base_source, pipeline)
    model = built.trained_model
    if compiler is not None:
        compiler(model)
    return Trainable(source, model, model_dir)


def vis_streams(build_fn,
                base_source: DataSource,
                num_frames=20,
                fps=4,
                group_size=1,
                skip_vis=False):
    from events_tfds.vis.image import as_frames
    import events_tfds.vis.anim as anim

    builder = mg.MultiGraphBuilder(batch_size=1)
    with builder:
        with comp.stream_accumulator() as streams:
            inputs = tf.nest.map_structure(builder.pre_cache_input,
                                           base_source.example_spec)
            logging.info('Building multi graph...')
            build_fn(*inputs, **base_source.meta)
            logging.info('Successfully built!')
        outputs = []
        static_shapes = []
        for stream in streams:
            if isinstance(stream, comp.SpatialStream):
                outputs.append((stream.times, stream.shaped_coords))
                static_shapes.append(stream.grid.static_shape)
            else:
                outputs.append((stream.times,))
                static_shapes.append(None)

        has_polarity = 'polarity' in inputs[0]
        if has_polarity:
            outputs = (tuple(outputs), inputs[0]['polarity'])
        else:
            outputs = tuple(outputs)

    flat_inputs = tf.nest.flatten(inputs, expand_composites=True)
    fn = mg.subgraph(flat_inputs[0].graph.as_graph_def(add_shapes=True),
                     flat_inputs,
                     tf.nest.flatten(outputs, expand_composites=True))

    def map_fn(*args, **kwargs):
        flat_in = tf.nest.flatten((args, kwargs), expand_composites=True)
        flat_out = fn(*flat_in)
        return tf.nest.pack_sequence_as(outputs,
                                        flat_out,
                                        expand_composites=True)

    del builder
    dataset = base_source.get_dataset('train').map(map_fn)
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
            print(f'{i}: sizes: {np.mean(sizes, axis=0).astype(np.int64)}')
            if group_size > 1:
                print(f'{i}: stds : {np.std(sizes, axis=0).astype(np.int64)}')
        if skip_vis:
            continue
        for i, (stream_data, shape) in enumerate(zip(streams, static_shapes)):
            if len(stream_data) == 1:
                assert (shape is None)
                shape = [1, 1]
                times, = stream_data
                coords = np.zeros((times.shape[0], 2), dtype=np.int64)
            else:
                times, coords = stream_data
                coords = coords.numpy()

            if times.shape[0] > 0:
                img_data.append(
                    as_frames(coords,
                              times.numpy(),
                              num_frames=num_frames,
                              polarity=polarity if i == 0 else None,
                              shape=shape))

        anim.animate_frames_multi(*img_data, fps=fps)


def vis_streams1d(build_fn, base_source: DataSource):
    import matplotlib.pyplot as plt
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
            inputs = tf.nest.map_structure(builder.pre_cache_input,
                                           base_source.example_spec)
            logging.info('Building multi graph...')
            build_fn(*inputs, **base_source.meta)
            logging.info('Successfully built!')
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
    fn = mg.subgraph(flat_inputs[0].graph.as_graph_def(add_shapes=True),
                     flat_inputs,
                     tf.nest.flatten(outputs, expand_composites=True))

    def map_fn(*args, **kwargs):
        flat_in = tf.nest.flatten((args, kwargs), expand_composites=True)
        flat_out = fn(*flat_in)
        out = tf.nest.pack_sequence_as(outputs,
                                       flat_out,
                                       expand_composites=True)
        return out

    del builder
    dataset = base_source.get_dataset('train').map(map_fn)
    print('num_channels: {}'.format([shape[0] for shape in static_shapes]))
    for streams in dataset:
        print('Sizes: {}'.format([s[0].shape[0] for s in streams]))
        xys = []
        for (stream_data, shape) in zip(streams, static_shapes):
            if len(stream_data) == 1:
                times, = stream_data
                coords = 0.5 * np.ones((times.shape[0],), dtype=np.int64)
                xys.append((coords, times))
            else:
                size = shape[0]
                times, coords = (x.numpy() for x in stream_data)
                coords = np.squeeze(coords, axis=-1)
                assert (np.max(coords) < size)
                xys.append(((coords + 0.5) / size, times))
        plt.figure()
        for i, xy in enumerate(xys):
            plt.scatter(*xy, s=(i + 1))
        plt.show()


def _vis_single_adjacency(indices, splits, in_times, out_times, decay_time, ax0,
                          ax1):
    from scipy.sparse import coo_matrix
    print(f'{in_times.size} -> {out_times.size} events')
    print(f'{indices.size} edges')
    row_lengths = splits[1:] - splits[:-1]
    i = np.repeat(np.arange(row_lengths.size), row_lengths, axis=0)
    j = indices
    values = np.exp(-(out_times[i] - in_times[j]) / decay_time)
    sp = coo_matrix((values, (i, j)))
    print(f'Sparsity: {sp.nnz / np.prod(sp.shape) * 100}%')
    ax0.spy(sp, markersize=0.5)
    ax0.set_xticks([])

    ax1.hist(values)


def vis_adjacency(build_fn, base_source: DataSource):
    import matplotlib.pyplot as plt
    builder = mg.MultiGraphBuilder(batch_size=1)
    with builder:
        with comp.stream_accumulator() as streams:
            with comp.convolver_accumulator() as convolvers:
                inputs = tf.nest.map_structure(builder.pre_cache_input,
                                               base_source.example_spec)
                logging.info('Building multi graph...')
                build_fn(*inputs, **base_source.meta)
                logging.info('Successfully built!')
        stream_indices = {s: i for i, s in enumerate(streams)}
        outputs = []
        decay_times = []
        stream_indices = tuple(
            (stream_indices[c.in_stream], stream_indices[c.out_stream])
            for c in convolvers)
        for c in convolvers:
            outputs.append(
                (c.indices, c.splits, c.in_stream.times, c.out_stream.times))
            decay_times.append(c.decay_time)

    flat_inputs = tf.nest.flatten(inputs, expand_composites=True)
    fn = mg.subgraph(flat_inputs[0].graph.as_graph_def(add_shapes=True),
                     flat_inputs,
                     tf.nest.flatten(outputs, expand_composites=True))

    def map_fn(*args, **kwargs):
        flat_in = tf.nest.flatten((args, kwargs), expand_composites=True)
        flat_out = fn(*flat_in)
        return tf.nest.pack_sequence_as(outputs,
                                        flat_out,
                                        expand_composites=True)

    del builder
    dataset = base_source.get_dataset('train').map(map_fn)
    num_convolvers = len(convolvers)
    for convolvers in dataset:
        convolvers = tf.nest.map_structure(lambda c: c.numpy(), convolvers)
        _, axes = plt.subplots(2, num_convolvers)
        if num_convolvers == 1:
            axes = np.expand_dims(axes, axis=-1)
        for i, (conv_data, decay_time, indices) in enumerate(
                zip(convolvers, decay_times, stream_indices)):
            s0, s1 = indices
            title = f'Conv {s0} -> {s1}'
            ax0, ax1 = axes[:, i]
            _vis_single_adjacency(*conv_data,
                                  decay_time=decay_time,
                                  ax0=ax0,
                                  ax1=ax1)
            ax0.title.set_text(title)

        plt.show()


def benchmark_source(source_fn, build_fn, take=1000, batch_size=32):
    base_source = source_fn(examples_per_epoch={
        'train': take,
        'validation': batch_size,
    })
    trainable = multi_graph_trainable(
        build_fn,
        base_source,
        batch_size=batch_size,
        cache_dir='/tmp/ecn_tests/benchmark_source',
        clear_cache=True,
        use_cache=True)
    trainable.fit(10, validation_freq=20)


if __name__ == '__main__':
    from ecn.problems import builders
    from ecn.problems import sources
    # from ecn.problems import augment

    # source = sources.ncars_source()
    # build_fn = builders.ncars_inception_graph

    # source = sources.ntidigits_source()
    # build_fn = functools.partial(builders.simple1d_half_graph,
    #                              #  kernel_size=5,
    #                              #  threshold=1.0
    #                             )

    # build_fn = builders.simple_multi_graph
    # build_fn = builders.inception_multi_graph
    # source = sources.mnist_dvs_source()
    # source = augment.Augmented2DSource(source)
    # build_fn = functools.partial(builders.inception128_multi_graph,
    #                              threshold=1.5,
    #                              decay_time=10000,
    #                              start_mean_size=131072,
    #                              hidden_mean_sizes=(16384, 4096, 1024, 128),
    #                              final_mean_size=32)

    # source = sources.cifar10_dvs_source()
    # build_fn = builders.inception128_multi_graph

    # build_fn = functools.partial(builders.inception_pooling, threshold=1.1)
    # source = sources.asl_dvs_source()

    build_fn = functools.partial(builders.inception_pooling, num_levels=4)
    source = sources.ncaltech101_source()
    # build_fn = functools.partial(
    # builders.inception_multi_graph_v2,
    # builders.inception_flash_multi_graph,
    # decay_time=2000,
    # reset_potential=-1.
    # )
    group_size = 1
    # build_fn = builders.simple1d_graph
    # source = sources.ntidigits_source()

    # multi_graph_trainable(build_fn, source, batch_size=2)
    # print('built successfully')

    # vis_streams1d(build_fn, source)
    # vis_streams(build_fn, source)
    vis_streams(build_fn, source, group_size=group_size, skip_vis=True)
    # vis_adjacency(build_fn, source)
    #     from ecn.problems.nmnist import simple_multi_graph
    #     from ecn.problems.nmnist import nmnist_source
    #     trainable = multi_grpah_trainable(simple_multi_graph,
    #                                       nmnist_source(),
    #                                       batch_size=16,
    #                                       compiler=compile_stream_classifier)
    #     source = trainable.source
    #     for example in source:
    #         pass
    # source_fn = sources.nmnist_source
    # benchmark_source(source_fn, build_fn)
