from typing import Optional, Callable
from absl import logging
import functools
import gin
import tensorflow as tf
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
                          compiler,
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
    import numpy as np
    from events_tfds.vis.image import as_frames
    import events_tfds.vis.anim as anim

    builder = mg.MultiGraphBuilder(base_source.example_spec, batch_size=1)
    with builder:
        with comp.stream_accumulator() as streams:
            inputs = builder._pre_batch_builder._inputs
            logging.info('Building multi graph...')
            builder.build(*build_fn(*inputs, **base_source.meta))
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

        outputs = tuple(outputs) + (inputs[0]['polarity'],)
        fn = mg.subgraph(outputs[0][0].graph.as_graph_def(), inputs, outputs)

    del builder
    dataset = base_source.get_dataset('train').map(fn)
    for example in dataset:
        *streams, polarity = example
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

            img_data.append(
                as_frames(coords,
                          times.numpy(),
                          num_frames=num_frames,
                          polarity=polarity.numpy() if i == 0 else None,
                          shape=shape))

        anim.animate_frames_multi(*img_data, fps=fps)


if __name__ == '__main__':
    from ecn.problems.builders import simple_multi_graph
    from ecn.problems.nmnist import nmnist_source
    vis_streams(simple_multi_graph, nmnist_source())
#     from ecn.problems.nmnist import simple_multi_graph
#     from ecn.problems.nmnist import nmnist_source
#     trainable = multi_grpah_trainable(simple_multi_graph,
#                                       nmnist_source(),
#                                       batch_size=16,
#                                       compiler=compile_stream_classifier)
#     source = trainable.source
#     for example in source:
#         pass
