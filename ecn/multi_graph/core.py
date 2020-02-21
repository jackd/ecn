from typing import TypeVar, Tuple, Union, Iterable
import abc
from kblocks.tf_typing import TensorLike, TensorLikeSpec
from typing import Callable, Optional
import tensorflow as tf

T = TypeVar('T')
NameLike = Union[str, tf.Tensor]


def _spec_to_placeholder(spec):
    if isinstance(spec, tf.TensorSpec):
        return tf.keras.backend.placeholder(shape=spec.shape, dtype=spec.dtype)
    elif isinstance(spec, tf.SparseTensorSpec):
        return tf.keras.backend.placeholder(shape=spec.shape,
                                            dtype=spec.dtype,
                                            sparse=True)
    elif isinstance(spec, tf.RaggedTensorSpec):
        return tf.keras.backend.placeholder(shape=spec._shape,
                                            dtype=spec._dtype,
                                            ragged=True)
    else:
        raise TypeError(
            'Invalid type for spec: must be TensorSpecLike, got {}'.format(
                spec))


def _spec_to_input(spec):
    return tf.keras.Input(shape=spec.shape[1:],
                          batch_size=spec.shape[0],
                          ragged=isinstance(spec, tf.RaggedTensorSpec),
                          sparse=isinstance(spec, tf.SparseTensorSpec),
                          dtype=spec.dtype)


def _batched_placeholder_like(x, batch_size=None):
    shape = (batch_size, *x.shape)
    return tf.keras.backend.placeholder(shape=shape,
                                        dtype=x.dtype,
                                        sparse=isinstance(x, tf.SparseTensor),
                                        ragged=isinstance(x, tf.RaggedTensor))


def _placeholder_like(x):
    return tf.keras.backend.placeholder(shape=x.shape,
                                        dtype=x.dtype,
                                        sparse=isinstance(x, tf.SparseTensor),
                                        ragged=isinstance(x, tf.RaggedTensor))


def flatten_inputs(fn, input_structure, expand_composites=False):
    """
    Change the input interface of the given function.

    Args:
        fn: function with signature `fn(*args)`.
        input_structure: different input signature
        expand_composites: used in `tf.nest.flatten`.

    Returns:
        function with signature `out_fn(inputs)`, where `inputs` must have the
            same structure as `input_structure` according to
            `tf.nest.assert_same_structure`.
    """

    def flat_fn(*inputs):
        tf.nest.assert_same_structure(inputs,
                                      input_structure,
                                      expand_composites=expand_composites)
        flat_args = tf.nest.flatten(inputs, expand_composites=expand_composites)
        return fn(*flat_args)

    return flat_fn


def repack_outputs(fn, output_structure, expand_composites=False):
    """
    Change the output interface of a given function.

    Args:
        fn: function with signature `fn(*args, **kwargs) -> Sequence`
        output_structure: desired output structure.
        expand_composites: whether outputs of `fn` have composites that should
            be reduced via `tf.nest.pack_sequence_as`.

    Returns:
        function with signature 'out_fn(*args, **kwargs) -> outupts`, where
            outputs has structure of `output_structure`.
    """

    def flat_fn(*args, **kwargs):
        out = fn(*args, **kwargs)
        return tf.nest.pack_sequence_as(output_structure,
                                        out,
                                        expand_composites=expand_composites)

    return flat_fn


def subgraph(graph_def, inputs, outputs) -> Callable:
    """
    Extract a subgraph from the given graph_def as a `@tf.function`ed callable.

    Args:
        graph_def: a `GraphDef`, like from `tf.Graph.as_graph_def`.
        inputs: tensors or op names as inputs.
        outputs: tensor or tensor names of outputs.

    Returns:
        A callable which maps f(*inputs) to a list of tensors given in outputs.
    """
    input_op_names = tuple(
        t if isinstance(t, str) else t.op.name
        for t in tf.nest.flatten(inputs, expand_composites=True))
    output_names = tuple(
        t if isinstance(t, str) else t.name
        for t in tf.nest.flatten(outputs, expand_composites=True))

    @tf.function()
    def graph_fn(*args, **kwargs):
        args = tf.nest.flatten((args, kwargs), expand_composites=True)
        if len(args) != len(input_op_names):
            raise ValueError(
                f'Expected {len(input_op_names)} args, got {len(args)}: {args}')
        assert (len(args) == len(input_op_names))
        input_map = dict(zip(input_op_names, args))
        flat_out = tf.graph_util.import_graph_def(graph_def,
                                                  input_map=input_map,
                                                  return_elements=output_names)
        return tf.nest.pack_sequence_as(outputs,
                                        flat_out,
                                        expand_composites=True)

    return graph_fn


class GraphBuilder(object):

    def __init__(self):
        self._graph = tf.Graph()
        self._outputs = []
        self._inputs = []
        self._ctxs = []

    @property
    def graph(self):
        return self._graph

    def _validate_graph(self, x, name='x'):
        if isinstance(x, tf.RaggedTensor):
            return self._validate_graph(x.flat_values, name=name)
        elif isinstance(x, tf.SparseTensor):
            return self._validate_graph(x.indices, name=name)
        if x.graph is not self._graph:
            raise ValueError(
                'x is from a different graph - cannot add as input')

    def __enter__(self: T) -> T:
        ctx = self.graph.as_default()
        self._ctxs.append(ctx)
        ctx.__enter__()
        return self

    def __exit__(self, type, value, traceback):
        ctx = self._ctxs.pop()
        ctx.__exit__(type, value, traceback)

    def input(self, spec: TensorLikeSpec) -> TensorLike:
        with self._graph.as_default():
            out = _spec_to_placeholder(spec)
            self._inputs.append(out)
        return out

    def add_input(self, x: TensorLike, key: Optional[str] = None) -> None:
        self._validate_graph(x, 'input')
        self._inputs.append(x)

    def add_output(self, x) -> None:
        self._validate_graph(x, 'output')
        self._outputs.append(x)

    def build(self,
              inputs_structure=None,
              extra_outputs: Optional[Iterable[TensorLike]] = None) -> Callable:
        inputs = self._inputs
        if inputs_structure is not None:
            inputs = tf.nest.pack_sequence_as(inputs_structure,
                                              inputs,
                                              expand_composites=True)

        outputs = self.outputs
        if extra_outputs is not None:
            for x in tf.nest.flatten(extra_outputs, expand_composites=True):
                self._validate_graph(x)
            outputs = (outputs, *extra_outputs)

        return subgraph(self.graph.as_graph_def(add_shapes=True), inputs,
                        outputs)

    @property
    def outputs(self) -> Tuple[TensorLike, ...]:
        return tuple(self._outputs)

    @property
    def inputs(self) -> Tuple[TensorLike, ...]:
        return tuple(self._inputs)


class ModelBuilder(object):

    def __init__(self, batch_size=None):
        self._inputs = []
        self._outputs = []
        self._batch_size = batch_size

    def input(self, spec: TensorLikeSpec) -> TensorLike:
        out = _spec_to_input(spec)
        self._inputs.append(out)
        return out

    def input_like(self, x: TensorLike, name=None) -> TensorLike:
        out = tf.keras.Input(shape=x.shape[1:],
                             batch_size=x.shape[0],
                             ragged=isinstance(x, tf.RaggedTensor),
                             sparse=isinstance(x, tf.SparseTensor),
                             dtype=x.dtype,
                             name=name)
        self._inputs.append(out)
        return out

    def add_input(self, x: TensorLike):
        if not tf.keras.backend.is_keras_tensor(x):
            raise ValueError('input must be a valid keras tensor')
        if not isinstance(x._keras_history.layer, tf.keras.layers.InputLayer):
            raise ValueError(
                'inputs must come directly from a keras `InputLayer`')
        self._inputs.append(x)

    # def add_output(self, x: TensorLike) -> None:
    #     if base_layer_utils.needs_keras_history(x):
    #         base_layer_utils.create_keras_history(x)
    #     self._outputs.append(x)

    def build(self, outputs) -> tf.keras.Model:
        if isinstance(outputs, (list, tuple)) and len(outputs) == 1:
            outputs, = outputs
        return tf.keras.Model(self._inputs, outputs)

    @property
    def outputs(self) -> Tuple[TensorLike, ...]:
        return tuple(self._outputs)

    @property
    def inputs(self) -> Tuple[TensorLike, ...]:
        return tuple(self._inputs)


class BuiltMultiGraph(object):

    def __init__(self, pre_cache_map: Callable, pre_batch_map: Callable,
                 post_batch_map: Callable, trained_model: tf.keras.Model):
        self._pre_cache_map = pre_cache_map
        self._pre_batch_map = pre_batch_map
        self._post_batch_map = post_batch_map
        self._trained_model = trained_model

    @property
    def pre_cache_map(self):
        return self._pre_cache_map

    @property
    def pre_batch_map(self):
        return self._pre_batch_map

    @property
    def post_batch_map(self):
        return self._post_batch_map

    @property
    def trained_model(self):
        return self._trained_model


class MultiGraphContext(object):
    _stack = []

    @staticmethod
    def get_default() -> 'MultiGraphContext':
        if len(MultiGraphContext._stack) == 0:
            raise RuntimeError('No MultiGraphContext contexts open.')
        return MultiGraphContext._stack[-1]

    def __enter__(self) -> 'MultiGraphContext':
        MultiGraphContext._stack.append(self)
        return self

    def __exit__(self, type, value, traceback):
        top = MultiGraphContext._stack.pop()
        assert (top is self)

    @abc.abstractmethod
    def pre_cache_context(self):
        raise NotImplementedError('Abstract method')

    @abc.abstractmethod
    def pre_batch_context(self):
        raise NotImplementedError('Abstract method')

    @abc.abstractmethod
    def post_batch_context(self):
        raise NotImplementedError('Abstract method')

    @abc.abstractmethod
    def is_pre_cache(self, x: TensorLike):
        raise NotImplementedError()

    @abc.abstractmethod
    def is_pre_batch(self, x: TensorLike):
        raise NotImplementedError()

    @abc.abstractmethod
    def is_post_batch(self, x: TensorLike):
        raise NotImplementedError()

    def assert_is_pre_cache(self, x: TensorLike) -> None:
        pass

    def assert_is_pre_batch(self, x: TensorLike) -> None:
        pass

    def assert_is_post_batch(self, x: TensorLike) -> None:
        pass

    def assert_is_model_tensor(self, x: TensorLike) -> None:
        pass

    @abc.abstractmethod
    def cache(self, x: TensorLike) -> TensorLike:
        raise NotImplementedError('Abstract method')

    @abc.abstractmethod
    def batch(self, x: TensorLike) -> TensorLike:
        raise NotImplementedError('Abstract method')

    @abc.abstractmethod
    def model_input(self, x: TensorLike, name=None) -> TensorLike:
        raise NotImplementedError('Abstract method')


class MultiGraphBuilder(MultiGraphContext):

    def __init__(self, batch_size: Optional[int] = None):
        self._pre_cache_builder = GraphBuilder()
        self._pre_batch_builder = GraphBuilder()
        self._post_batch_builder = GraphBuilder()
        self._model_builder = ModelBuilder(batch_size=batch_size)
        self._batch_size = batch_size

    @property
    def pre_cache_graph(self) -> tf.Graph:
        return self._pre_cache_builder.graph

    @property
    def pre_batch_graph(self) -> tf.Graph:
        return self._pre_batch_builder.graph

    @property
    def post_batch_graph(self) -> tf.Graph:
        return self._post_batch_builder.graph

    def pre_cache_context(self):
        return self.pre_cache_graph.as_default()

    def pre_batch_context(self):
        return self.pre_batch_graph.as_default()

    def post_batch_context(self):
        return self.post_batch_graph.as_default()

    def is_pre_cache(self, x: TensorLike) -> bool:
        return hasattr(x, 'graph') and x.graph is self.pre_cache_graph

    def is_pre_batch(self, x: TensorLike) -> bool:
        return hasattr(x, 'graph') and x.graph is self.pre_batch_graph

    def is_post_batch(self, x: TensorLike) -> bool:
        return hasattr(x, 'graph') and x.graph is self.post_batch_graph

    def assert_is_pre_cache(self, x: TensorLike) -> None:
        if x.graph is not self.pre_cache_graph:
            raise ValueError('x is not part of pre_cache_graph')

    def assert_is_pre_batch(self, x: TensorLike) -> None:
        if x.graph is not self.pre_batch_graph:
            raise ValueError('x is not part of pre_batch_graph')

    def assert_is_post_batch(self, x: TensorLike) -> None:
        if x.graph is not self.post_batch_graph:
            raise ValueError('x is not part of post_batch_graph')

    def assert_is_model_tensor(self, x: TensorLike) -> None:
        if x.graph is self.pre_batch_graph:
            raise ValueError('x is part of pre_batch_graph')
        elif x.graph is self.post_batch_graph:
            raise ValueError('x is part of post_batch_graph')

    def pre_cache_input(self, spec: TensorLikeSpec) -> tf.Tensor:
        with self.pre_cache_context():
            return self._pre_cache_builder.input(spec)

    def cache(self, x: tf.Tensor):
        assert (isinstance(x, tf.Tensor))
        self.assert_is_pre_cache(x)
        self._pre_cache_builder.add_output(x)
        with self.pre_batch_context():
            out = _placeholder_like(x)
            self._pre_batch_builder.add_input(out)
        assert (x.shape is not None)
        if len(self._pre_cache_builder._inputs) == 3:
            raise Exception()
        return out

    def batch(self, x: TensorLike) -> TensorLike:
        if isinstance(x,
                      tf.Tensor) and x.shape.ndims > 0 and x.shape[0] is None:
            raise ValueError('Cannot batch tensor with unknown first dimension')
        self._pre_batch_builder.add_output(x)
        with self._post_batch_builder:
            out = _batched_placeholder_like(x)
        self._post_batch_builder.add_input(out)
        return out

    def model_input(self, x, name=None) -> TensorLike:
        self._post_batch_builder.add_output(x)
        return self._model_builder.input_like(x, name=name)

    def build(self, model_outputs, labels, weights=None,
              inputs_structure=None) -> BuiltMultiGraph:

        rest = (labels,) if weights is None else (labels, weights)

        return BuiltMultiGraph(
            self._pre_cache_builder.build(inputs_structure=inputs_structure),
            self._pre_batch_builder.build(),
            self._post_batch_builder.build(extra_outputs=rest),
            trained_model=self._model_builder.build(model_outputs))


get_default = MultiGraphContext.get_default


def is_pre_cache(x):
    return get_default().is_pre_cache(x)


def is_pre_batch(x):
    return get_default().is_pre_batch(x)


def is_post_batch(x):
    return get_default().is_post_batch(x)


def pre_cache_context():
    return get_default().pre_cache_context()


def pre_batch_context():
    return get_default().pre_batch_context()


def post_batch_context():
    return get_default().post_batch_context()


def assert_is_pre_cache(x: TensorLike):
    return get_default().assert_is_pre_cache(x)


def assert_is_pre_batch(x: TensorLike):
    return get_default().assert_is_pre_batch(x)


def assert_is_post_batch(x: TensorLike):
    return get_default().assert_is_post_batch(x)


def assert_is_model_tensor(x: TensorLike):
    return get_default().assert_is_model_tensor(x)


def cache(x: TensorLike) -> TensorLike:
    return get_default().cache(x)


def batch(x: TensorLike) -> TensorLike:
    return get_default().batch(x)


def model_input(x: TensorLike, name=None) -> TensorLike:
    return get_default().model_input(x, name=name)


def build_multi_graph(
        build_fn,
        inputs_spec,
        batch_size: Optional[int] = None,
) -> BuiltMultiGraph:
    builder = MultiGraphBuilder(batch_size)
    with builder:
        inputs = tf.nest.map_structure(builder.pre_cache_input, inputs_spec)
        if isinstance(inputs, dict):
            args = build_fn(**inputs)
        elif isinstance(inputs, tf.Tensor):
            args = build_fn(inputs)
        else:
            args = build_fn(*inputs)

        if len(args) == 2:
            model_outputs, labels = args
            weights = None
        else:
            model_outputs, labels, weights = args
        return builder.build(model_outputs, labels, weights, inputs_spec)
