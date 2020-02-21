raise NotImplementedError('deprecated')
# from typing import Optional
# import tensorflow as tf
# from tensorflow.python.keras.engine import base_layer_utils  # pylint: disable=no-name-in-module,import-error
# from kblocks.tf_typing import TensorLike
# from kblocks.tf_typing import TensorLikeSpec
# from . import core

# def _batched_input_like(x, batch_size=None):
#     return tf.keras.Input(shape=x.shape,
#                           dtype=x.dtype,
#                           sparse=isinstance(x, tf.SparseTensor),
#                           ragged=isinstance(x, tf.RaggedTensor),
#                           batch_size=batch_size)

# def _batched_spec(spec, batch_size):
#     if isinstance(spec, tf.TensorSpec):
#         return tf.TensorSpec(shape=(1, *spec.shape), dtype=spec.dtype)
#     else:
#         raise NotImplementedError('TODO')

# class StructuredModel(object):

#     def __init__(self, inputs, outputs, model=None, clone_on_call=False):
#         if model is None:
#             model = tf.keras.Model(tf.nest.flatten(inputs),
#                                    tf.nest.flatten(outputs))
#         self._model = model
#         self._inputs = inputs
#         self._outputs = outputs
#         self._clone_on_call = clone_on_call

#     def __call__(self, *args, **kwargs):
#         model = self._model
#         if len(args) == 0:
#             tf.nest.assert_same_structure(self._inputs, kwargs)
#         elif len(kwargs) == 0:
#             tf.nest.assert_same_structure(self._inputs, args)
#         else:
#             raise ValueError('one of args or kwargs must be empty')
#         args = tf.nest.flatten((args, kwargs))
#         if self._clone_on_call:
#             out = tf.keras.models.clone_model(model, input_tensors=args).outputs
#         else:
#             out = model(args)
#         return tf.nest.pack_sequence_as(self._outputs, out)

#     @property
#     def model(self):
#         return self._model

#     def clone(self, clone_function=None, clone_on_call=None):
#         if clone_on_call is None:
#             clone_on_call = self._clone_on_call
#         return StructuredModel(
#             self._inputs,
#             self._outputs,
#             tf.keras.models.clone_model(clone_function=clone_function),
#             clone_on_call=clone_on_call)

# class GraphModelBuilder(core.GraphBuilder):

#     def __init__(self, inputs_spec=None):
#         self._graph = tf.Graph()
#         self._disallow_inputs = False
#         self._outputs = []
#         self._learning_phase = None
#         self._ctxs = []
#         if inputs_spec is None:
#             self._inputs = []
#         else:
#             with self:
#                 self._inputs = tf.nest.map_structure(core._spec_to_input,
#                                                      inputs_spec)
#             self._disallow_inputs = True

#     def input(self, spec: TensorLikeSpec) -> TensorLike:
#         if self._disallow_inputs:
#             raise ValueError('Cannot add inputs')
#         with self:
#             out = core._spec_to_input(spec)
#             self._inputs.append(out)
#             assert (out.dtype == spec.dtype)
#         return out

#     def add_output(self, x: TensorLike) -> None:
#         with self:
#             if base_layer_utils.needs_keras_history(x):
#                 base_layer_utils.create_keras_history(x)
#             return super().add_output(x)

#     def build(self, extra_outputs=None) -> StructuredModel:
#         if extra_outputs is None:
#             outputs = tuple(self._outputs)
#         else:
#             outputs = (tuple(self._outputs), *extra_outputs)

#         if self._disallow_inputs:
#             inputs = self._inputs
#         else:
#             inputs = tuple(self._inputs)

#         with self:
#             return StructuredModel(inputs, outputs, clone_on_call=True)

# def _pre_batch_map(model: StructuredModel):

#     def f(*args, **kwargs):
#         args, kwargs = tf.nest.map_structure(
#             lambda x: tf.expand_dims(x, axis=0), (args, kwargs))
#         return model(*args, **kwargs)

#     return f

# def _validate_untrainable(model, name):
#     if model.trainable_variables:
#         raise ValueError('{} cannot contain trainable variables'.format(name))

# class MultiModelBuilder(core.MultiGraphBuilder):

#     def __init__(self, inputs_spec, batch_size: Optional[int] = None):
#         self._inputs_spec = inputs_spec
#         inputs_spec = tf.nest.map_structure(lambda x: _batched_spec(x, 1),
#                                             inputs_spec)
#         self._pre_batch_builder = GraphModelBuilder(inputs_spec)
#         self._post_batch_builder = GraphModelBuilder()
#         self._model_builder = core.ModelBuilder(batch_size=batch_size)
#         self._batch_size = batch_size

#     def build(self, model_outputs, labels,
#               weights=None) -> core.BuiltMultiGraph:
#         pre_batch_model = self._pre_batch_builder.build()
#         _validate_untrainable(pre_batch_model.model, 'pre_batch_model')
#         rest = (labels,) if weights is None else (labels, weights)
#         post_batch_model = self._post_batch_builder.build(rest)
#         _validate_untrainable(post_batch_model.model, 'post_batch_model')
#         trained_model = self._model_builder.build(model_outputs)
#         return core.BuiltMultiGraph(_pre_batch_map(pre_batch_model),
#                                     post_batch_model,
#                                     trained_model=trained_model)

#     def batch(self, x: TensorLike) -> TensorLike:
#         self._pre_batch_builder.add_output(x)
#         with self._post_batch_builder:
#             out = _batched_input_like(x)
#         self._post_batch_builder.add_input(out)
#         return out

# def build_multi_model(build_fn, inputs_spec, batch_size: Optional[int] = None):
#     builder = MultiModelBuilder(inputs_spec, batch_size)
#     with builder:
#         inputs = builder._pre_batch_builder._inputs
#         with builder.pre_batch_context():
#             inputs = tf.nest.map_structure(lambda x: tf.squeeze(x, axis=0),
#                                            inputs)
#         return builder.build(*build_fn(*inputs))
