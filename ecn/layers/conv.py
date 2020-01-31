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


class ConvolutionBase(layers.Layer):

    def __init__(self,
                 filters: int,
                 decay_time: float,
                 activation=None,
                 use_bias: bool = True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ConvolutionBase, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.filters = int(filters)
        self.decay_time = float(decay_time)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True

    def get_config(self):
        config = {
            'filters':
                self.filters,
            'decay_time':
                self.decay_time,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint)
        }
        base_config = super(ConvolutionBase, self).get_config()
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


class EventConv(ConvolutionBase):

    def _kernel_shape(self, input_shape):
        filters_in = input_shape[0][-1]
        k = input_shape[3][0]
        return (k, filters_in, self.filters)

    def call(self, inputs):
        in_features, in_times, out_times, sparse_indices, valid_lengths = inputs
        features = conv_ops.event_conv(
            in_features=in_features,
            in_times=in_times,
            out_times=out_times,
            sparse_indices=sparse_indices,
            valid_lengths=valid_lengths,
            kernel=self.kernel,
            decay_time=self.decay_time,
        )
        return self._finalize(features)


class GlobalEventConv(ConvolutionBase):

    def _kernel_shape(self, input_shape):
        filters_in = input_shape[0][-1]
        return (filters_in, self.filters)

    def call(self, inputs):
        in_features, in_times, out_times, sparse_indices, valid_length = inputs
        features = conv_ops.global_event_conv(
            in_features=in_features,
            in_times=in_times,
            out_times=out_times,
            sparse_indices=sparse_indices,
            valid_length=valid_length,
            kernel=self.kernel,
            decay_time=self.decay_time,
        )
        return self._finalize(features)


def event_conv(in_features: FloatTensor, in_times: FloatTensor,
               out_times: FloatTensor, sparse_indices: IntTensor,
               valid_lengths: IntTensor, filters: int, decay_time: float,
               **kwargs):
    return EventConv(filters=filters, decay_time=decay_time, **kwargs)(
        [in_features, in_times, out_times, sparse_indices, valid_lengths])


def global_event_conv(in_features: FloatTensor, in_times: FloatTensor,
                      out_times: FloatTensor, sparse_indices: IntTensor,
                      valid_length: IntTensor, filters: int, decay_time: float,
                      **kwargs):
    return GlobalEventConv(filters=filters, decay_time=decay_time, **kwargs)(
        [in_features, in_times, out_times, sparse_indices, valid_length])
