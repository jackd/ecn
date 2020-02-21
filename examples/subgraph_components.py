from typing import Callable
import functools
import timeit
import tensorflow as tf
import numpy as np
import numba

with tf.Graph().as_default() as graph:
    x = tf.range(10, dtype=tf.int64)
    y = tf.RaggedTensor.from_row_splits(x, [0, 3, 7, 10])
    # y = tf.SparseTensor(tf.expand_dims(x, axis=-1), tf.range(10), [10])

graph_def = graph.as_graph_def(add_shapes=True)


def subgraph(graph_def, inputs, outputs) -> Callable:
    """
    Extract a subgraph from the given graph_def as a `@tf.function`ed callable.

    Args:
        graph_def: a `GraphDef`, like from `tf.Graph.as_graph_def`.
        inputs: structure of inputs - either `TensorLike`s or the
            corresponding op names, like from `tensor.op.name`.
        outputs: structure of `TensorLike`s or strings corresponding to
            tensor names, like from `tensor.name`.
        learning_phase: Optional tensor which takes on the value of value from
            `tf.keras.backend.learning_phase()`.

    Returns:
        A callable which maps f(*inputs) to tensors in the same structure as
            `outputs`.
    """
    input_op_names = tuple(
        t if isinstance(t, str) else t.op.name
        for t in tf.nest.flatten(inputs, expand_composites=True))
    output_names = tuple(
        t if isinstance(t, str) else t.name
        for t in tf.nest.flatten(outputs, expand_composites=True))

    @tf.function()
    def graph_fn(*args, learning_phase=None, **kwargs):
        assert (len(args) == 0 or len(kwargs) == 0)
        if len(kwargs) == 0:
            if len(args) == 1:
                args, = args
            tf.nest.assert_same_structure(args, inputs)
        else:
            tf.nest.assert_same_structure(kwargs, inputs)
        input_map = dict(
            zip(input_op_names,
                tf.nest.flatten((args, kwargs), expand_composites=True)))
        out = tf.graph_util.import_graph_def(graph_def,
                                             input_map=input_map,
                                             return_elements=output_names)
        return out
        # return tf.nest.pack_sequence_as(outputs, out, expand_composites=True)

    return graph_fn


f = subgraph(graph_def, x, y)
print(f(tf.range(10, dtype=tf.int64) * 3))
