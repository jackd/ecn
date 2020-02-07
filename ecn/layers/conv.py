import tensorflow as tf
import abc
from ecn.ops import conv as conv_ops

FloatTensor = tf.Tensor
IntTensor = tf.Tensor

layers = tf.keras.layers
initializers = tf.keras.initializers
regularizers = tf.keras.regularizers
constraints = tf.keras.constraints
activations = tf.keras.activations


class ConvBase(layers.Layer):

    def __init__(self,
                 filters: int,
                 temporal_kernel_size: int,
                 is_complex: bool = False,
                 activation=None,
                 use_bias: bool = True,
                 decay_initializer='ones',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 decay_regularizer=None,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 decay_constraint='non_neg',
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ConvBase, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        if is_complex:
            raise NotImplementedError('TODO')
        self.filters = filters
        self.temporal_kernel_size = temporal_kernel_size
        self.is_complex = is_complex
        self.use_bias = use_bias
        self.activation = activations.get(activation)

        self.decay_initializer = initializers.get(decay_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.decay_regularizer = regularizers.get(decay_regularizer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.decay_constraint = constraints.get(decay_constraint)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True

    def get_config(self):
        config = {
            'filters':
                self.filters,
            'temporal_kernel_size':
                self.temporal_kernel_size,
            'is_complex':
                self.is_complex,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'decay_initializer':
                initializers.serialize(self.decay_initializer),
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'decay_regularizer':
                regularizers.serialize(self.decay_regularizer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'decay_constraint':
                constraints.serialize(self.decay_constraint),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint)
        }
        base_config = super(ConvBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @abc.abstractmethod
    def call(self, inputs):
        raise NotImplementedError

    @abc.abstractmethod
    def _kernel_shape(self, input_shape):
        raise NotImplementedError

    def _finalize(self, outputs):
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def build(self, input_shape):
        if self.built:
            return

        self.decay = self.add_weight('decay',
                                     shape=(self.temporal_kernel_size,),
                                     initializer=self.decay_initializer,
                                     regularizer=self.decay_regularizer,
                                     constraint=self.decay_constraint)
        self.kernel = self.add_weight('kernel',
                                      shape=self._kernel_shape(input_shape),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      dtype=self.dtype,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight('bias',
                                        shape=[self.filters],
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        dtype=self.dtype,
                                        trainable=True)
        else:
            self.bias = None
        super(ConvBase, self).build(input_shape)


class EventConv(ConvBase):

    def _kernel_shape(self, input_shape):
        filters_in = input_shape[0][-1]
        k = input_shape[3][0]
        return (k, self.termporal_kernel_size, filters_in, self.filters)

    def call(self, inputs):
        in_features, in_times, out_times, sparse_indices = inputs
        features = conv_ops.event_conv(
            in_features=in_features,
            in_times=in_times,
            out_times=out_times,
            sparse_indices=sparse_indices,
            kernel=self.kernel,
            decay=self.decay,
        )
        return self._finalize(features)


class GlobalEventConv(ConvBase):

    def _kernel_shape(self, input_shape):
        filters_in = input_shape[0][-1]
        return (self.temporal_kernel_size, filters_in, self.filters)

    def call(self, inputs):
        in_features, in_times, out_times, sparse_indices = inputs
        features = conv_ops.global_event_conv(
            in_features=in_features,
            in_times=in_times,
            out_times=out_times,
            sparse_indices=sparse_indices,
            kernel=self.kernel,
            decay=self.decay,
        )
        return self._finalize(features)


def event_conv(in_features: FloatTensor, in_times: FloatTensor,
               out_times: FloatTensor, sparse_indices: IntTensor, filters: int,
               decay_time: float, **kwargs):
    return EventConv(filters=filters, decay_time=decay_time, **kwargs)(
        [in_features, in_times, out_times, sparse_indices])


def global_event_conv(in_features: FloatTensor, in_times: FloatTensor,
                      out_times: FloatTensor, sparse_indices: IntTensor,
                      filters: int, decay_time: float, **kwargs):
    return GlobalEventConv(filters=filters, decay_time=decay_time, **kwargs)(
        [in_features, in_times, out_times, sparse_indices])
