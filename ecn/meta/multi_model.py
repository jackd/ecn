from typing import Optional, Callable
import tensorflow as tf
from ecn.meta import core


class MultiModelBuilderContext(core.MetaBuilderContext):

    def __init__(self, inputs_spec, batch_size: Optional[int] = None):
        super().__init__(batch_size=batch_size)
        self._pre_batch_inputs = tf.nest.map_structure(self._pre_batch_input,
                                                       inputs_spec)

        self._pre_batch_outputs = []
        self._post_batch_inputs = []
        self._post_batch_features = []
        self._model_inputs = []

    def _pre_batch_input(self, spec: core.TensorSpecLike):
        return tf.keras.Input(shape=spec.shape,
                              batch_size=1,
                              dtype=spec.dtype,
                              sparse=isinstance(spec, tf.SparseTensorSpec),
                              ragged=isinstance(spec, tf.RaggedTensorSpec))

    def batch(self, tensor: core.TensorLike):
        out = tf.keras.Input(shape=tensor.shape,
                             batch_size=self.batch_size,
                             dtype=tensor.dtype,
                             sparse=isinstance(tensor, tf.SparseTensor),
                             ragged=isinstance(tensor, tf.RaggedTensor))
        self._pre_batch_outputs.append(tensor)
        self._post_batch_inputs.append(out)
        return out

    def model_input(self, tensor: core.TensorLike, name: Optional[str] = None):
        if self.batch_size is not None:
            assert (tensor.shape[0] == self.batch_size)
        inp = tf.keras.Input(
            shape=tensor.shape[1:],
            batch_size=self.batch_size,
            dtype=tensor.dtype,
            sparse=isinstance(tensor, tf.SparseTensor),
            ragged=isinstance(tensor, tf.RaggedTensor),
        )
        self._post_batch_features.append(tensor)
        self._model_inputs.append(inp)
        return inp

    def build(self, model_outputs, labels, weights=None) -> 'BuiltMultiModel':
        features = tuple(self._post_batch_features)
        if weights is None:
            post_batch_outputs = (features, labels)
        else:
            post_batch_outputs = (features, labels, weights)
        trained_model = tf.keras.Model(self._model_inputs, model_outputs)
        return BuiltMultiModel(self._pre_batch_inputs, self._pre_batch_outputs,
                               self._post_batch_inputs, post_batch_outputs,
                               trained_model)


def _pre_batch_model_fn(inputs, outputs):
    model = tf.keras.Model(tf.nest.flatten(inputs), tf.nest.flatten(outputs))

    def f(*args, **kwargs):
        args = tf.nest.flatten((args, kwargs))
        args = [tf.expand_dims(a, axis=0) for a in args]
        flat_outputs = model(args)
        return tf.nest.pack_sequence_as(outputs, flat_outputs)

    return f


def _post_batch_model_fn(inputs, outputs):
    model = tf.keras.Model(tf.nest.flatten(inputs), tf.nest.flatten(outputs))

    def f(*args, **kwargs):
        args = tf.nest.flatten((args, kwargs))
        flat_outputs = model(args)
        return tf.nest.pack_sequence_as(outputs, flat_outputs)

    return f


class BuiltMultiModel(object):

    def __init__(self, pre_batch_inputs, pre_batch_outputs, post_batch_inputs,
                 post_batch_outputs, trained_model: tf.keras.Model):
        self._pre_batch_fn = _pre_batch_model_fn(pre_batch_inputs,
                                                 pre_batch_outputs)
        self._post_batch_fn = _post_batch_model_fn(post_batch_inputs,
                                                   post_batch_outputs)
        self._trained_model = trained_model

    @property
    def pre_batch_map(self) -> Callable:
        return self._pre_batch_fn

    @property
    def post_batch_map(self) -> Callable:
        return self._post_batch_fn

    @property
    def trained_model(self) -> tf.keras.Model:
        return self._trained_model


def build_multi_model(build_fn: Callable,
                      inputs_spec,
                      batch_size: Optional[int] = None) -> BuiltMultiModel:
    with MultiModelBuilderContext(inputs_spec, batch_size=batch_size) as ctx:
        inputs = tf.nest.map_structure(
            lambda x: tf.keras.layers.Lambda(tf.squeeze, arguments=dict(axis=0))
            (x), ctx._pre_batch_inputs)
        if isinstance(inputs,
                      (dict, tf.Tensor, tf.RaggedTensor, tf.SparseTensor)):
            inputs = inputs,

        fn_outputs = build_fn(*inputs)
    built = ctx.build(*fn_outputs)
    return built
