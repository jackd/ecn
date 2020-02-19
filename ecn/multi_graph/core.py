from typing import TypeVar
import abc
from kblocks.tf_typing import TensorLike, TensorLikeSpec
from typing import Callable, Optional
import tensorflow as tf
from tensorflow.python.keras.engine import base_layer_utils  # pylint: disable=no-name-in-module,import-error

T = TypeVar('T')


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


def _batched_spec(spec, batch_size):
    if isinstance(spec, tf.TensorSpec):
        return tf.TensorSpec(shape=(1, *spec.shape), dtype=spec.dtype)
    else:
        raise NotImplementedError('TODO')


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


def _batched_input_like(x, batch_size=None):
    return tf.keras.Input(shape=x.shape,
                          dtype=x.dtype,
                          sparse=isinstance(x, tf.SparseTensor),
                          ragged=isinstance(x, tf.RaggedTensor),
                          batch_size=batch_size)


def subgraph(graph_def, inputs, outputs, learning_phase=None) -> Callable:
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

    if learning_phase is not None and not isinstance(learning_phase, str):
        learning_phase_name = learning_phase.op.name
    else:
        learning_phase_name = None

    @tf.function()
    def graph_fn(*args, learning_phase=None, **kwargs):
        assert (len(args) == 0 or len(kwargs) == 0)
        if len(kwargs) == 0:
            if len(args) == 1:
                args, = args
            tf.nest.assert_same_structure(args, inputs, expand_composites=True)
        else:
            tf.nest.assert_same_structure(kwargs,
                                          inputs,
                                          expand_composites=True)
        input_map = dict(
            zip(input_op_names,
                tf.nest.flatten((args, kwargs), expand_composites=True)))
        if learning_phase_name is not None and learning_phase is not None:
            input_map[learning_phase_name] = learning_phase
        out = tf.graph_util.import_graph_def(graph_def,
                                             input_map=input_map,
                                             return_elements=output_names)
        return tf.nest.pack_sequence_as(outputs, out, expand_composites=True)

    if learning_phase is None:
        return graph_fn

    def wrapped_fn(*args, **kwargs):
        kwargs['learning_phase'] = tf.keras.backend.learning_phase()
        return graph_fn(*args, **kwargs)

    return wrapped_fn


class StructuredModel(object):

    def __init__(self, inputs, outputs, model=None, clone_on_call=False):
        if model is None:
            model = tf.keras.Model(tf.nest.flatten(inputs),
                                   tf.nest.flatten(outputs))
        self._model = model
        self._inputs = inputs
        self._outputs = outputs
        self._clone_on_call = clone_on_call

    def __call__(self, *args, **kwargs):
        model = self._model
        if len(args) == 0:
            tf.nest.assert_same_structure(self._inputs, kwargs)
        elif len(kwargs) == 0:
            tf.nest.assert_same_structure(self._inputs, args)
        else:
            raise ValueError('one of args or kwargs must be empty')
        args = tf.nest.flatten((args, kwargs))
        if self._clone_on_call:
            out = tf.keras.models.clone_model(model, input_tensors=args).outputs
        else:
            out = model(args)
        return tf.nest.pack_sequence_as(self._outputs, out)

    @property
    def model(self):
        return self._model

    def clone(self, clone_function=None, clone_on_call=None):
        if clone_on_call is None:
            clone_on_call = self._clone_on_call
        return StructuredModel(
            self._inputs,
            self._outputs,
            tf.keras.models.clone_model(clone_function=clone_function),
            clone_on_call=clone_on_call)


class GraphBuilder(object):

    def __init__(self, inputs_spec=None):
        self._graph = tf.Graph()
        self._disallow_inputs = False
        self._outputs = []
        self._learning_phase = None
        self._ctxs = []
        if inputs_spec is None:
            self._inputs = []
        else:
            with self:
                self._inputs = tf.nest.map_structure(_spec_to_placeholder,
                                                     inputs_spec)
            self._disallow_inputs = True

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
        if self._disallow_inputs:
            raise ValueError('Cannot add inputs')
        with self._graph.as_default():
            out = _spec_to_placeholder(spec)
            self._inputs.append(out)
        return out

    def add_input(self, x) -> None:
        if self._disallow_inputs:
            raise ValueError('Cannot add inputs')
        self._validate_graph(x, 'input')
        self._inputs.append(x)

    def add_output(self, x) -> None:
        self._validate_graph(x, 'output')
        self._outputs.append(x)

    @property
    def learning_phase(self) -> tf.Tensor:
        if self._learning_phase is None:
            with self:
                self._learning_phase = tf.keras.backend.placeholder(
                    shape=(), dtype=tf.bool)
        return self._learning_phase

    def build(self, extra_outputs=None) -> Callable:
        if extra_outputs is None:
            outputs = tuple(self._outputs)
        else:
            outputs = (tuple(self._outputs), *extra_outputs)
        if self._disallow_inputs:
            inputs = self._inputs
        else:
            inputs = tuple(self._inputs)
        return subgraph(self.graph.as_graph_def(),
                        inputs,
                        outputs,
                        learning_phase=self._learning_phase)


class GraphModelBuilder(GraphBuilder):

    def __init__(self, inputs_spec=None):
        self._graph = tf.Graph()
        self._disallow_inputs = False
        self._outputs = []
        self._learning_phase = None
        self._ctxs = []
        if inputs_spec is None:
            self._inputs = []
        else:
            with self:
                self._inputs = tf.nest.map_structure(_spec_to_input,
                                                     inputs_spec)
            self._disallow_inputs = True

    def input(self, spec: TensorLikeSpec) -> TensorLike:
        if self._disallow_inputs:
            raise ValueError('Cannot add inputs')
        with self:
            out = _spec_to_input(spec)
            self._inputs.append(out)
            assert (out.dtype == spec.dtype)
        return out

    def add_output(self, x: TensorLike) -> None:
        with self:
            if base_layer_utils.needs_keras_history(x):
                base_layer_utils.create_keras_history(x)
            return super().add_output(x)

    def build(self, extra_outputs=None) -> StructuredModel:
        if extra_outputs is None:
            outputs = tuple(self._outputs)
        else:
            outputs = (tuple(self._outputs), *extra_outputs)

        if self._disallow_inputs:
            inputs = self._inputs
        else:
            inputs = tuple(self._inputs)

        with self:
            return StructuredModel(inputs, outputs, clone_on_call=True)


# class UnbatchedGraphModelBuilder(GraphModelBuilder):

#     def input(self, spec: TensorLikeSpec) -> TensorLike:
#         with self:
#             return tf.squeeze(super().input(_batched_spec(spec, 1)))

#     def build(self, extra_outputs=None) -> Callable:
#         fn = super().build(extra_outputs)

#         def out_fn(*args, **kwargs):
#             (args, kwargs) = tf.nest.map_structure(
#                 functools.partial(tf.expand_dims, axis=0), (args, kwargs))
#             return fn(*args, **kwargs)

#         return out_fn


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


class BuiltMultiGraph(object):

    def __init__(self, pre_batch_map: Callable, post_batch_map: Callable,
                 trained_model: tf.keras.Model):
        self._pre_batch_map = pre_batch_map
        self._post_batch_map = post_batch_map
        self._trained_model = trained_model

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
    def pre_batch_context(self):
        raise NotImplementedError('Abstract method')

    @abc.abstractmethod
    def post_batch_context(self):
        raise NotImplementedError('Abstract method')

    def assert_is_pre_batch(self, x: TensorLike) -> None:
        pass

    def assert_is_post_batch(self, x: TensorLike) -> None:
        pass

    def assert_is_model_tensor(self, x: TensorLike) -> None:
        pass

    @abc.abstractmethod
    def batch(self, x: TensorLike) -> TensorLike:
        raise NotImplementedError('Abstract method')

    @abc.abstractmethod
    def model_input(self, x: TensorLike, name=None) -> TensorLike:
        raise NotImplementedError('Abstract method')


class MultiGraphBuilder(MultiGraphContext):

    def __init__(self, inputs_spec, batch_size: Optional[int] = None):
        self._pre_batch_builder = GraphBuilder(inputs_spec=inputs_spec)
        self._post_batch_builder = GraphBuilder()
        self._model_builder = ModelBuilder(batch_size=batch_size)
        self._batch_size = batch_size

    @property
    def pre_batch_graph(self) -> tf.Graph:
        return self._pre_batch_builder.graph

    @property
    def post_batch_graph(self) -> tf.Graph:
        return self._post_batch_builder.graph

    def pre_batch_context(self):
        return self.pre_batch_graph.as_default()

    def post_batch_context(self):
        return self.post_batch_graph.as_default()

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

    def batch(self, x: TensorLike) -> TensorLike:
        self._pre_batch_builder.add_output(x)
        with self._post_batch_builder:
            out = _batched_placeholder_like(x)
        self._post_batch_builder.add_input(out)
        return out

    def model_input(self, x, name=None) -> TensorLike:
        self._post_batch_builder.add_output(x)
        return self._model_builder.input_like(x, name=name)

    def build(self, model_outputs, labels, weights=None) -> BuiltMultiGraph:
        rest = (labels,) if weights is None else (labels, weights)
        trained_model = self._model_builder.build(model_outputs)
        return BuiltMultiGraph(self._pre_batch_builder.build(),
                               self._post_batch_builder.build(rest),
                               trained_model=trained_model)


def _pre_batch_map(model: StructuredModel):

    def f(*args, **kwargs):
        args, kwargs = tf.nest.map_structure(
            lambda x: tf.expand_dims(x, axis=0), (args, kwargs))
        return model(*args, **kwargs)

    return f


def _validate_untrainable(model, name):
    if model.trainable_variables:
        raise ValueError('{} cannot contain trainable variables'.format(name))


class MultiModelBuilder(MultiGraphBuilder):

    def __init__(self, inputs_spec, batch_size: Optional[int] = None):
        self._inputs_spec = inputs_spec
        inputs_spec = tf.nest.map_structure(lambda x: _batched_spec(x, 1),
                                            inputs_spec)
        self._pre_batch_builder = GraphModelBuilder(inputs_spec)
        self._post_batch_builder = GraphModelBuilder()
        self._model_builder = ModelBuilder(batch_size=batch_size)
        self._batch_size = batch_size

    def build(self, model_outputs, labels, weights=None) -> BuiltMultiGraph:
        pre_batch_model = self._pre_batch_builder.build()
        _validate_untrainable(pre_batch_model.model, 'pre_batch_model')
        rest = (labels,) if weights is None else (labels, weights)
        post_batch_model = self._post_batch_builder.build(rest)
        _validate_untrainable(post_batch_model.model, 'post_batch_model')
        trained_model = self._model_builder.build(model_outputs)
        return BuiltMultiGraph(_pre_batch_map(pre_batch_model),
                               post_batch_model,
                               trained_model=trained_model)

    def batch(self, x: TensorLike) -> TensorLike:
        self._pre_batch_builder.add_output(x)
        with self._post_batch_builder:
            out = _batched_input_like(x)
        self._post_batch_builder.add_input(out)
        return out


get_default = MultiGraphContext.get_default


def pre_batch_context():
    return get_default().pre_batch_context()


def post_batch_context():
    return get_default().post_batch_context()


def assert_is_pre_batch(x: TensorLike):
    return get_default().assert_is_pre_batch(x)


def assert_is_post_batch(x: TensorLike):
    return get_default().assert_is_post_batch(x)


def assert_is_model_tensor(x: TensorLike):
    return get_default().assert_is_model_tensor(x)


def batch(x: TensorLike) -> TensorLike:
    return get_default().batch(x)


def model_input(x: TensorLike, name=None) -> TensorLike:
    return get_default().model_input(x, name=name)


# def build(model_outputs, labels, weights=None) -> BuiltMultiGraph:
#     return get_default().build(model_outputs, labels=labels, weights=weights)


def build_multi_model(build_fn, inputs_spec, batch_size: Optional[int] = None):
    builder = MultiModelBuilder(inputs_spec, batch_size)
    with builder:
        inputs = builder._pre_batch_builder._inputs
        with builder.pre_batch_context():
            inputs = tf.nest.map_structure(lambda x: tf.squeeze(x, axis=0),
                                           inputs)
        return builder.build(*build_fn(*inputs))


def build_multi_graph(
        build_fn,
        inputs_spec,
        batch_size: Optional[int] = None,
) -> BuiltMultiGraph:
    builder = MultiGraphBuilder(inputs_spec, batch_size)
    with builder:
        inputs = builder._pre_batch_builder._inputs
        return builder.build(*build_fn(*inputs))


if __name__ == '__main__':

    builder = GraphBuilder()
    with builder:
        x = builder.input(tf.TensorSpec(shape=(), dtype=tf.float32))
        y = builder.input(tf.TensorSpec(shape=(), dtype=tf.float32))
        # # where and using the value as a float works fine
        # y = x + tf.cast(builder.learning_phase, tf.float32)
        z = tf.where(tf.expand_dims(builder.learning_phase, axis=-1), x, y)
        # cond is broken
        # https://github.com/tensorflow/tensorflow/issues/36809
        # z = tf.cond(builder.learning_phase, lambda: x, lambda: y)
    builder.add_output(z)

    fn = builder.build()

    with tf.keras.backend.learning_phase_scope(False):
        print(fn(2., 3.))
    with tf.keras.backend.learning_phase_scope(True):
        print(fn(2., 3.))
