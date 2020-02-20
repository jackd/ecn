# import os
# os.environ['NUMBA_DISABLE_JIT'] = '1'
from typing import Optional, Callable
from absl import logging
import functools
import gin
import tensorflow as tf
import numpy as np
from kblocks.framework.pipelines import BasePipeline
from kblocks.framework.sources import DataSource
from kblocks.framework.sources import PipelinedSource
from kblocks.framework.trainable import Trainable
from ecn import multi_graph as mg
from ecn import components as comp
losses = tf.keras.losses
metrics = tf.keras.metrics


@gin.configurable(module='ecn.utils')
def compile_stream_classifier(model: tf.keras.Model,
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
def multi_graph_trainable(build_fn: Callable,
                          base_source: DataSource,
                          batch_size: int,
                          compiler=compile_stream_classifier,
                          use_cache=False,
                          model_dir: Optional[str] = None,
                          **pipeline_kwargs):

    logging.info('Building multi graph...')
    built = mg.build_multi_graph(
        functools.partial(build_fn, **base_source.meta),
        base_source.example_spec, batch_size)
    logging.info('Successfully built!')

    if use_cache:
        pre_cache_map = built.pre_batch_map
        pre_batch_map = None
    else:
        pre_cache_map = None
        pre_batch_map = built.pre_batch_map

    pipeline = BasePipeline(batch_size,
                            pre_cache_map=pre_cache_map,
                            pre_batch_map=pre_batch_map,
                            post_batch_map=built.post_batch_map,
                            **pipeline_kwargs)
    source = PipelinedSource(base_source, pipeline)
    model = built.trained_model
    compiler(model)
    return Trainable(source, model, model_dir)


def vis_streams(build_fn, base_source: DataSource, num_frames=20, fps=4):
    from events_tfds.vis.image import as_frames
    import events_tfds.vis.anim as anim

    builder = mg.MultiGraphBuilder(base_source.example_spec, batch_size=1)
    with builder:
        with comp.stream_accumulator() as streams:
            inputs = builder._pre_batch_builder._inputs
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

        outputs = (tuple(outputs), inputs[0]['polarity'])
        fn = mg.subgraph(
            tf.nest.flatten(inputs,
                            expand_composites=True)[0].graph.as_graph_def(),
            inputs, outputs)

    del builder
    dataset = base_source.get_dataset('train').map(fn)
    for example in dataset:
        streams, polarity = example
        img_data = []
        print('Sizes: {}'.format([s[0].shape[0] for s in streams]))
        for i, (stream_data, shape) in enumerate(zip(streams, static_shapes)):
            if shape is None:
                shape = [1, 1]
                times, = stream_data
                coords = np.zeros((times.shape[0], 2), dtype=np.int64)
            else:
                times, coords = stream_data
                coords = coords.numpy()

            # img_data.append(
            #     as_frames(coords,
            #               times.numpy(),
            #               num_frames=num_frames,
            #               polarity=polarity.numpy() if i == 0 else None,
            #               shape=shape))

        # anim.animate_frames_multi(*img_data, fps=fps)
<<<<<<< HEAD


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
    builder = mg.MultiGraphBuilder(base_source.example_spec, batch_size=1)
    with builder:
        with comp.stream_accumulator() as streams:
            with comp.convolver_accumulator() as convolvers:
                inputs = builder._pre_batch_builder._inputs
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

        fn = mg.subgraph(
            tf.nest.flatten(inputs, expand_composites=True)[0], inputs, outputs)

    del builder
    dataset = base_source.get_dataset('train').map(fn)
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


if __name__ == '__main__':
    from ecn.problems import builders
    from ecn.problems import sources

    # build_fn = builders.simple_multi_graph
    build_fn = builders.inception_multi_graph
    source = sources.nmnist_source()
    # build_fn = builders.inception128_multi_graph
    # source = sources.cifar10_dvs_source()

    multi_graph_trainable(build_fn, source, batch_size=2)
    print('built successfully')

    # vis_streams(build_fn, source)
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
=======


if __name__ == '__main__':
    # from ecn.problems.builders import simple_multi_graph
    from ecn.problems.builders import inception_multi_graph
    from ecn.problems.nmnist import nmnist_source
    vis_streams(inception_multi_graph, nmnist_source())
>>>>>>> 6a7d4ccf87cee374ba18762d648637f28e18e8be