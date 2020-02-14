from typing import Optional, Callable, Dict, Tuple
import tensorflow as tf
from kblocks.tf_typing import TensorLike, TensorLikeSpec
from . import core
from . import marks as _marks

BuiltModels = _marks.BuiltModels


class GraphMarks(_marks.Marks):

    def __init__(self):
        base: Dict[tf.Tensor, str] = {}
        super().__init__(base)

    def _inputs(self, x: TensorLike) -> Tuple[tf.Tensor, ...]:
        if isinstance(x, tf.Tensor):
            return tuple(x.op.inputs)
        elif isinstance(x, tf.SparseTensor):
            return tuple(x.op.inputs) + (x.indices, x.values, x.dense_shape)
        elif isinstance(x, tf.RaggedTensor):
            return (x.flat_values, *x.nested_row_splits)
        else:
            raise TypeError('Invalid type of x, {}'.format(x))


class MultiGraphBuilderContext(core.MetaBuilderContext):

    def __init__(self, inputs_spec, batch_size: Optional[int] = None):
        super().__init__(batch_size=batch_size)
        self._marks = GraphMarks()
        self._graph: Optional[tf.Graph] = None
        self._pre_batch_inputs = tf.nest.map_structure(self._pre_batch_input,
                                                       inputs_spec)
        self._pre_batch_outputs = []
        self._post_batch_inputs = []
        self._post_batch_features = []
        self._model_inputs = []

        # self._learning_phase = None

    # def learning_phase(self) -> tf.Tensor:
    #     if self._learning_phase is None:
    #         self._learning_phase = tf.keras.backend.placeholder(
    #             shape=(), dtype=tf.bool, name='learning_phase')
    #     return self._learning_phase

    def get_mark(self, x: TensorLike):
        return self._marks[x]

    def set_mark(self, x: TensorLike, mark: str):
        self._marks[x] = mark

    def _validate_graph(self, x: core.TensorLike):
        g = x.graph
        if self._graph is None:
            self._graph = g
        elif g is not self._graph:
            raise ValueError('Tensor {} not in current graph.'.format(x))

    def _pre_batch_input(self, spec: TensorLikeSpec) -> TensorLike:
        if isinstance(spec, tf.TensorSpec):
            out = tf.keras.backend.placeholder(shape=spec.shape,
                                               dtype=spec.dtype)
        elif isinstance(spec, tf.SparseTensorSpec):
            out = tf.keras.backend.placeholder(shape=spec.shape,
                                               dtype=spec.dtype,
                                               sparse=True)
        elif isinstance(spec, tf.RaggedTensorSpec):
            raise NotImplementedError('TODO')
        else:
            raise TypeError(
                'Invalid type for spec: must core.be TensorSpecLike, got {}'.
                format(spec))
        self._validate_graph(out)
        self._marks[out] = BuiltModels.PRE_BATCH
        return out

    def batch(self, tensor: core.TensorLike):
        self._validate_graph(tensor)
        out = tf.keras.Input(shape=tensor.shape,
                             batch_size=self.batch_size,
                             ragged=isinstance(tensor, tf.RaggedTensor),
                             sparse=isinstance(tensor, tf.SparseTensor))
        self._pre_batch_outputs.append(tensor)
        self._post_batch_inputs.append(out)
        self._marks[tensor] = BuiltModels.PRE_BATCH
        self._marks[out] = BuiltModels.POST_BATCH
        return out

    def model_input(self, tensor: core.TensorLike, name: Optional[str] = None):
        self._validate_graph(tensor)
        assert (tensor.shape[0] == self.batch_size)
        out = tf.keras.Input(shape=tensor.shape[1:],
                             batch_size=tensor.shape[0],
                             ragged=isinstance(tensor, tf.RaggedTensor),
                             sparse=isinstance(tensor, tf.SparseTensor),
                             name=name)
        self._post_batch_features.append(tensor)
        self._model_inputs.append(out)
        self._marks[tensor] = BuiltModels.POST_BATCH
        self._marks[out] = BuiltModels.TRAINED
        return out

    def build(self, model_outputs, labels, weights=None) -> 'BuiltMultiGraph':
        if self._graph is None:
            raise RuntimeError('Cannot build: not inputs added anywhere')
        trained_model = tf.keras.Model(self._model_inputs, model_outputs)
        features = tuple(self._post_batch_features)
        if weights is None:
            post_batch_outputs = (features, labels)
        else:
            post_batch_outputs = (features, labels, weights)
        return BuiltMultiGraph(
            self._graph.as_graph_def(add_shapes=True),
            self._pre_batch_inputs,
            pre_batch_outputs=self._pre_batch_outputs,
            post_batch_inputs=self._post_batch_inputs,
            post_batch_outputs=post_batch_outputs,
            #   learning_phase=self._learning_phase,
            trained_model=trained_model)


def subgraph(
        graph_def,
        inputs,
        outputs,
        # learning_phase=None,
) -> Callable:
    """
    Extract a subgraph from the given graph_def as a `@tf.function`ed callable.

    Args:
        graph_def: a `GraphDef`, like from `tf.Graph.as_graph_def`.
        inputs: iterable of inputs - either `core.TensorLike`s or the corresponding
            op names, like from `tensor.op.name`.
        outputs: structure of `core.TensorLike`s or strings corresponding to tensor
            names, like from `tensor.name`.

    Returns:
        A callable which maps f(*inputs) to tensors in the same structure as
            `outputs`.
    """
    input_op_names = tuple(
        t if isinstance(t, str) else t.op.name for t in tf.nest.flatten(inputs))
    output_names = tuple(
        t if isinstance(t, str) else t.name for t in tf.nest.flatten(outputs))

    # if learning_phase is not None and not isinstance(learning_phase, str):
    #     learning_phase = learning_phase.op.name

    @tf.function
    def f(*input_args):
        # tf.nest.assert_same_structure(input_args, inputs)
        input_args = tf.nest.flatten(input_args)
        input_map = dict(zip(input_op_names, input_args))
        # if learning_phase is not None:
        #     lr = tf.keras.backend.learning_phase()
        #     input_map[learning_phase] = tf.cast(lr, tf.bool)
        out = tf.graph_util.import_graph_def(graph_def,
                                             input_map=input_map,
                                             return_elements=output_names)
        return tf.nest.pack_sequence_as(outputs, out)

    return f


class BuiltMultiGraph(object):

    def __init__(
            self,
            graph_def,
            pre_batch_inputs,
            pre_batch_outputs,
            post_batch_inputs,
            post_batch_outputs,
            # learning_phase,
            trained_model: tf.keras.Model,
    ):
        self._trained_model = trained_model
        self._pre_batch_fn = subgraph(
            graph_def,
            pre_batch_inputs,
            pre_batch_outputs,
            # learning_phase,
        )
        self._post_batch_fn = subgraph(
            graph_def,
            post_batch_inputs,
            post_batch_outputs,
            # learning_phase,
        )

    @property
    def pre_batch_map(self) -> Callable:
        return self._pre_batch_fn

    @property
    def post_batch_map(self) -> Callable:
        return self._post_batch_fn

    def rebuild_model(self, input_tensors=None,
                      clone_function=None) -> tf.keras.Model:
        self._trained_model = tf.keras.models.clone_model(
            self._trained_model,
            input_tensors=input_tensors,
            clone_function=clone_function)
        return self._trained_model

    @property
    def trained_model(self) -> tf.keras.Model:
        return self._trained_model


def build_multi_graph_graph(build_fn: Callable,
                            inputs_spec,
                            batch_size: Optional[int] = None
                           ) -> BuiltMultiGraph:
    if tf.executing_eagerly():
        raise RuntimeError(
            'Cannot call `build_multi_graph_graph` in eager mode. '
            'Either open a graph context or use `build_multi_graph_eager` or'
            '`build_multi_graph`.')
    with MultiGraphBuilderContext(inputs_spec) as ctx:
        inputs = ctx._pre_batch_inputs
        if isinstance(inputs,
                      (dict, tf.Tensor, tf.RaggedTensor, tf.SparseTensor)):
            inputs = inputs,

        # defensively copy structure
        inputs = tf.nest.map_structure(lambda x: x, inputs)
        fn_outputs = build_fn(*inputs)
    built = ctx.build(*fn_outputs)
    return built


def build_multi_graph_eager(build_fn: Callable,
                            inputs_spec,
                            batch_size: Optional[int] = None
                           ) -> BuiltMultiGraph:
    if not tf.executing_eagerly():
        raise RuntimeError(
            'Cannot call `build_multi_graph_eager` in graph mode. '
            'Either open a graph context or use `build_multi_graph_graph` or'
            '`build_multi_graph`.')
    with tf.Graph().as_default():
        built = build_multi_graph_graph(build_fn, inputs_spec, batch_size)
    built.rebuild_model()
    return built


def build_multi_graph(build_fn: Callable,
                      inputs_spec,
                      batch_size: Optional[int] = None) -> BuiltMultiGraph:
    fn = (build_multi_graph_eager
          if tf.executing_eagerly() else build_multi_graph_graph)
    return fn(build_fn, inputs_spec, batch_size)
