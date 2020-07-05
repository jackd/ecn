import functools
from typing import Callable, Optional, Union

import gin
import tensorflow as tf

import kerastuner as kt
from ecn.multi_graph import build_multi_graph
from ecn.multi_graph.core import BuiltMultiGraph
from ecn.problems import builders
from kblocks.framework.pipelines import BasePipeline
from kblocks.framework.sources import DataSource, PipelinedSource
from kblocks.framework.tune import tune
from kblocks.keras.tuner import Hyperband


@gin.configurable(module="ecn.tune")
def tuned_simple1d_half_graph(
    features,
    labels,
    weights=None,
    hp: Optional[kt.HyperParameters] = None,
    num_classes=11,
    grid_shape=(64,),
):
    return builders.simple1d_half_graph(
        features,
        labels,
        weights,
        filters0=hp.Choice("filters0", [8, 16, 32, 64], default=16),
        kt0=hp.Choice("kt0", [2, 4, 8], default=4),
        # activation=hp.Choice('activation', ['relu', 'selu', 'softplus']),
        # use_batch_norm=hp.Boolean('use_batch_norm', default=True),
        dropout_rate=hp.Float("dropout_rate", 0.25, 0.9),
        num_classes=num_classes,
        grid_shape=grid_shape,
    )


@gin.configurable(module="ecn.tune")
def tuned_sgd(hp):
    return tf.keras.optimizers.SGD(
        learning_rate=hp.Float("learning_rate", 1e-4, 1.0, sampling="log"),
        momentum=hp.Float("momentum", 0.5, 0.99),
        nesterov=hp.Boolean("nesterov", True),
    )


@gin.configurable(module="ecn.tune")
def tuned_adam(hp):
    return tf.keras.optimizers.Adam(
        learning_rate=hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
    )


@gin.configurable(module="ecn.tune")
def tune_multi_graph(
    build_fn: Callable,
    base_source: DataSource,
    compiler: Callable,
    max_epochs,
    batch_size,
    objective,
    tuner_fn=Hyperband,
    optimizer: Optional[Union[tf.keras.optimizers.Optimizer, Callable]] = None,
    **pipeline_kwargs
):
    def build_multi_graph_hyper(hp) -> BuiltMultiGraph:
        return build_multi_graph(
            functools.partial(build_fn, hp=hp, **base_source.meta,),
            base_source.example_spec,
            batch_size,
        )

    def build_model(hp):
        built = build_multi_graph_hyper(hp)
        model = built.trained_model
        if callable(optimizer):
            kwargs = dict(optimizer=optimizer(hp))
        elif optimizer is None:
            kwargs = {}
        else:
            kwargs = dict(optimizer=optimizer)
        compiler(model, **kwargs)
        return model

    hp = kt.HyperParameters()
    built = build_multi_graph_hyper(hp)

    pipeline = BasePipeline(
        batch_size,
        pre_cache_map=built.pre_cache_map,
        pre_batch_map=built.pre_batch_map,
        post_batch_map=built.post_batch_map,
        **pipeline_kwargs,
    )
    source = PipelinedSource(base_source, pipeline, meta={})
    return tune(build_model, source, objective, max_epochs, tuner_fn)
