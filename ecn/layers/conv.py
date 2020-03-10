from typing import Callable, Optional, Iterable, Union
import tensorflow as tf
import abc
import gin
from ecn.ops import conv as conv_ops

FloatTensor = tf.Tensor
IntTensor = tf.Tensor

layers = tf.keras.layers
initializers = tf.keras.initializers
regularizers = tf.keras.regularizers
constraints = tf.keras.constraints
activations = tf.keras.activations


def _complex(real, imag=None):
    return real if imag is None else tf.complex(real, imag)


@gin.configurable(module='ecn.layers')
class EventConvBase(layers.Layer):

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
                 imag_decay_initializer='zeros',
                 imag_kernel_initializer='glorot_uniform',
                 imag_bias_initializer='zeros',
                 imag_decay_regularizer=None,
                 imag_kernel_regularizer=None,
                 imag_bias_regularizer=None,
                 imag_decay_constraint=None,
                 imag_kernel_constraint=None,
                 imag_bias_constraint=None,
                 **kwargs):
        super(EventConvBase, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.filters = filters
        self.temporal_kernel_size = temporal_kernel_size
        self.is_complex = is_complex
        self.use_bias = use_bias
        self.activation: Optional[Callable] = activations.get(activation)
        if self.activation is not None:
            if not callable(self.activation):
                raise ValueError('activation {} is not callable'.format(
                    self.activation))

        self.decay_initializer = initializers.get(decay_initializer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.decay_regularizer = regularizers.get(decay_regularizer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.decay_constraint = constraints.get(decay_constraint)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.imag_decay_initializer = initializers.get(imag_decay_initializer)
        self.imag_kernel_initializer = initializers.get(imag_kernel_initializer)
        self.imag_bias_initializer = initializers.get(imag_bias_initializer)

        self.imag_decay_regularizer = regularizers.get(imag_decay_regularizer)
        self.imag_kernel_regularizer = regularizers.get(imag_kernel_regularizer)
        self.imag_bias_regularizer = regularizers.get(imag_bias_regularizer)

        self.imag_decay_constraint = constraints.get(imag_decay_constraint)
        self.imag_kernel_constraint = constraints.get(imag_kernel_constraint)
        self.imag_bias_constraint = constraints.get(imag_bias_constraint)

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
                constraints.serialize(self.bias_constraint),
            'imag_decay_initializer':
                initializers.serialize(self.imag_decay_initializer),
            'imag_kernel_initializer':
                initializers.serialize(self.imag_kernel_initializer),
            'imag_bias_initializer':
                initializers.serialize(self.imag_bias_initializer),
            'imag_decay_regularizer':
                regularizers.serialize(self.imag_decay_regularizer),
            'imag_kernel_regularizer':
                regularizers.serialize(self.imag_kernel_regularizer),
            'imag_bias_regularizer':
                regularizers.serialize(self.imag_bias_regularizer),
            'imag_decay_constraint':
                constraints.serialize(self.imag_decay_constraint),
            'imag_kernel_constraint':
                constraints.serialize(self.imag_kernel_constraint),
            'imag_bias_constraint':
                constraints.serialize(self.imag_bias_constraint)
        }
        base_config = super(EventConvBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @abc.abstractmethod
    def call(self, inputs):
        raise NotImplementedError

    @abc.abstractmethod
    def _kernel_shape(self, input_shape):
        raise NotImplementedError

    @abc.abstractmethod
    def _decay_shape(self, input_shape):
        raise NotImplementedError

    def _finalize(self, outputs):
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def build(self, input_shape):
        if self.built:
            return

        decay_shape = self._decay_shape(input_shape)
        self.decay_real = self.add_weight('decay_real',
                                          shape=decay_shape,
                                          initializer=self.decay_initializer,
                                          regularizer=self.decay_regularizer,
                                          constraint=self.decay_constraint)

        kernel_shape = self._kernel_shape(input_shape)
        self.kernel_real = self.add_weight('kernel_real',
                                           shape=kernel_shape,
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint,
                                           dtype=self.dtype,
                                           trainable=True)
        if self.use_bias:
            self.bias_real = self.add_weight('bias_real',
                                             shape=[self.filters],
                                             initializer=self.bias_initializer,
                                             regularizer=self.bias_regularizer,
                                             constraint=self.bias_constraint,
                                             dtype=self.dtype,
                                             trainable=True)
        else:
            self.bias_real = None

        if self.is_complex:
            self.decay_imag = self.add_weight(
                'decay_imag',
                shape=decay_shape,
                initializer=self.imag_decay_initializer,
                regularizer=self.imag_decay_regularizer,
                constraint=self.imag_decay_constraint)

            self.kernel_imag = self.add_weight(
                'kernel_imag',
                shape=kernel_shape,
                initializer=self.imag_kernel_initializer,
                regularizer=self.imag_kernel_regularizer,
                constraint=self.imag_kernel_constraint,
                dtype=self.dtype,
                trainable=True)
            if self.use_bias:
                self.bias_imag = self.add_weight(
                    'bias_imag',
                    shape=[self.filters],
                    initializer=self.imag_bias_initializer,
                    regularizer=self.imag_bias_regularizer,
                    constraint=self.imag_bias_constraint,
                    dtype=self.dtype,
                    trainable=True)
        else:
            self.decay_imag = None
            self.kernel_imag = None
            self.bias_imag = None
        super(EventConvBase, self).build(input_shape)

    @property
    def decay(self):
        return _complex(self.decay_real, self.decay_imag)

    @property
    def kernel(self):
        return _complex(self.kernel_real, self.kernel_imag)

    @property
    def bias(self):
        return _complex(self.bias_real, self.bias_imag)


@gin.configurable(module='ecn.layers')
class SpatioTemporalEventConv(EventConvBase):

    def __init__(self,
                 filters: int,
                 temporal_kernel_size: int,
                 spatial_kernel_size: Optional[int] = None,
                 combine: str = 'unstack',
                 **kwargs):
        if spatial_kernel_size is not None:
            spatial_kernel_size = int(spatial_kernel_size)
        self.combine = combine
        self.spatial_kernel_size = spatial_kernel_size
        super(SpatioTemporalEventConv,
              self).__init__(filters, temporal_kernel_size, **kwargs)

    def _validate_kernel_size(self, dt_shape):
        if isinstance(dt_shape, tf.TensorShape):
            sk = dt_shape[-1]
        else:
            sk = len(dt_shape)

        if sk is None:
            if self.spatial_kernel_size is None:
                raise ValueError(
                    'spatial_kernel_size not defined in constructor and not '
                    'inferable from static input sizes.')
        elif self.spatial_kernel_size is None:
            self.spatial_kernel_size = sk
        elif self.spatial_kernel_size != sk:
            raise ValueError(
                'spatial_kernel_size {} is not consistent with that inferred '
                'from input static shapes, {}'.format(self.spatial_kernel_size,
                                                      sk))
        assert (self.spatial_kernel_size is not None)
        assert (isinstance(self.spatial_kernel_size, int))

    def _kernel_shape(self, input_shape):
        filters_in = input_shape[0][-1]
        self._validate_kernel_size(input_shape[1:])
        # _spatial_size(input_shape)
        return (self.spatial_kernel_size, self.temporal_kernel_size, filters_in,
                self.filters)

    def _decay_shape(self, input_shape):
        self._validate_kernel_size(input_shape[1:])
        return (self.spatial_kernel_size, self.temporal_kernel_size)

    def get_config(self):
        config = super(SpatioTemporalEventConv, self).get_config()
        config['spatial_kernel_size'] = self.spatial_kernel_size
        config['combine'] = self.combine
        return config

    def call(self, inputs):
        features, *dt = inputs
        if len(dt) == 1:
            dt, = dt
        features = conv_ops.spatio_temporal_event_conv(
            features=features,
            dt=dt,
            kernel=self.kernel,
            decay=self.decay,
            combine=self.combine,
        )
        return self._finalize(features)


@gin.configurable(module='ecn.layers')
class FeaturelessSpatioTemporalEventConv(SpatioTemporalEventConv):

    def _kernel_shape(self, input_shape):
        self._validate_kernel_size(input_shape)
        return (self.spatial_kernel_size, self.temporal_kernel_size,
                self.filters)

    def _decay_shape(self, input_shape):
        self._validate_kernel_size(input_shape)
        return (self.spatial_kernel_size, self.temporal_kernel_size)

    def call(self, inputs):
        dt = inputs
        if len(dt) == 1:
            dt, = dt
        features = conv_ops.featureless_spatio_temporal_event_conv(
            dt=dt,
            kernel=self.kernel,
            decay=self.decay,
        )
        return self._finalize(features)


@gin.configurable(module='ecn.layers')
class BinarySpatioTemporalEventConv(SpatioTemporalEventConv):

    def _kernel_shape(self, input_shape):
        assert (len(input_shape[0]) == 1)
        self._validate_kernel_size(input_shape[1:])
        return (self.spatial_kernel_size, 2 * self.temporal_kernel_size,
                self.filters)

    def _decay_shape(self, input_shape):
        self._validate_kernel_size(input_shape[1:])
        return (self.spatial_kernel_size, self.temporal_kernel_size)

    def call(self, inputs):
        features, *dt = inputs
        if len(dt) == 1:
            dt, = dt
        features = conv_ops.binary_spatio_temporal_event_conv(
            features=features,
            dt=dt,
            kernel=self.kernel,
            decay=self.decay,
        )
        return self._finalize(features)


@gin.configurable(module='ecn.layers')
class FeaturelessTemporalEventConv(EventConvBase):

    def _kernel_shape(self, input_shape):
        return (self.temporal_kernel_size, self.filters)

    def _decay_shape(self, input_shape):
        return (self.temporal_kernel_size,)

    def call(self, inputs):
        dt = inputs
        features = conv_ops.featureless_temporal_event_conv(
            dt=dt,
            kernel=self.kernel,
            decay=self.decay,
        )
        return self._finalize(features)


@gin.configurable(module='ecn.layers')
class BinaryTemporalEventConv(EventConvBase):

    def _kernel_shape(self, input_shape):
        return (2 * self.temporal_kernel_size, self.filters)

    def _decay_shape(self, input_shape):
        return (self.temporal_kernel_size,)

    def call(self, inputs):
        features, dt = inputs
        features = conv_ops.binary_temporal_event_conv(
            features=features,
            dt=dt,
            kernel=self.kernel,
            decay=self.decay,
        )
        return self._finalize(features)


@gin.configurable(module='ecn.layers')
class TemporalEventConv(EventConvBase):

    def __init__(self, *args, **kwargs):
        self.combine = kwargs.pop('combine', 'unstack')
        super().__init__(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config['combine'] = self.combine
        return config

    def _kernel_shape(self, input_shape):
        filters_in = input_shape[0][-1]
        return (self.temporal_kernel_size, filters_in, self.filters)

    def _decay_shape(self, input_shape):
        return (self.temporal_kernel_size,)

    def call(self, inputs):
        features, dt = inputs
        features = conv_ops.temporal_event_conv(
            features=features,
            dt=dt,
            kernel=self.kernel,
            decay=self.decay,
            combine=self.combine,
        )
        return self._finalize(features)


@gin.configurable(module='ecn.layers')
class TemporalEventPooling(TemporalEventConv):

    def call(self, inputs):
        features, dt, value_rowids, batch_size = inputs
        features = conv_ops.temporal_event_pooling(features=features,
                                                   dt=dt,
                                                   value_rowids=value_rowids,
                                                   batch_size=batch_size,
                                                   kernel=self.kernel,
                                                   decay=self.decay)
        return self._finalize(features)


def spatio_temporal_event_conv(
        features: Optional[tf.Tensor],
        dt: Union[tf.SparseTensor, Iterable[tf.SparseTensor]],
        filters: int,
        temporal_kernel_size: int,
        **kwargs,
) -> FloatTensor:
    kwargs.update(
        dict(filters=filters, temporal_kernel_size=temporal_kernel_size))
    if features is None:
        return FeaturelessSpatioTemporalEventConv(**kwargs)(dt)
    inputs = [features, dt] if isinstance(dt,
                                          tf.SparseTensor) else [features, *dt]
    if features.dtype.is_bool:
        return BinarySpatioTemporalEventConv(**kwargs)(inputs)
    else:
        return SpatioTemporalEventConv(**kwargs)(inputs)


def temporal_event_conv(
        features: Optional[tf.Tensor],
        dt: Union[tf.SparseTensor, Iterable[tf.SparseTensor]],
        filters: int,
        temporal_kernel_size: int,
        **kwargs,
) -> FloatTensor:
    kwargs.update(
        dict(filters=filters, temporal_kernel_size=temporal_kernel_size))
    if features is None:
        return FeaturelessTemporalEventConv(**kwargs)(dt)
    inputs = [features, dt] if isinstance(dt,
                                          tf.SparseTensor) else [features, *dt]
    if features.dtype.is_bool:
        return BinaryTemporalEventConv(**kwargs)(inputs)
    else:
        return TemporalEventConv(**kwargs)(inputs)


if __name__ == '__main__':
    import numpy as np
    f_in = 7
    n_in = 11
    n_out = 5
    n_e = 23
    sk = 3
    is_complex = True
    neighs = []
    for _ in range(sk):
        n = np.random.randint(n_e) + 1
        i = tf.random.uniform((n,), minval=0, maxval=n_out, dtype=tf.int64)
        j = tf.random.uniform((n,), minval=0, maxval=n_in, dtype=tf.int64)
        neighs.append(
            tf.SparseTensor(tf.stack((i, j), axis=-1),
                            tf.random.uniform((n,), dtype=tf.float32),
                            [n_out, n_in]))

    filters = 2
    tk = 5
    layer = SpatioTemporalEventConv(filters, tk, sk, is_complex=is_complex)
    features = tf.random.normal((n_in, f_in))
    layer([features, *neighs])
    print('SpatioTemporalEventConv successfully built')

    # layer = TemporalEventConv(2, 3)
    # ij = tf.stack((i, j), axis=-1)
    # neigh = tf.SparseTensor(ij, tf.random.uniform((n_e,), dtype=tf.float32),
    #                         (n_out, n_in))
    # layer([features, neigh])
    # print('TemporalEventConv successfully built')
