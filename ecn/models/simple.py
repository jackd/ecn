import functools
import gin
import tensorflow as tf
from kblocks import spec
from kblocks.keras import layers
from kblocks.extras.layers import ragged as ragged_layers
import kblocks.ops.sparse as sparse_ops
from ecn.layers import conv as conv_layers

Lambda = tf.keras.layers.Lambda


def activate(x, activation='relu', use_batch_norm=True, dropout_prob=None):
    if dropout_prob is not None:
        x = layers.Dropout(dropout_prob)(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    activation = tf.keras.activations.get(activation)
    if activation is not None:
        x = activation(x)
    return x


@gin.configurable(module='ecn.models',
                  blacklist=['inputs_spec', 'outputs_spec'])
def simple_ecn(inputs_spec,
               outputs_spec,
               filters0=32,
               temporal_kernel_size=4,
               activation='relu',
               dropout_prob=0.5,
               use_batch_norm=True,
               final_only=False):
    inputs = tf.nest.map_structure(spec.to_input, inputs_spec)
    flat_inputs = tf.nest.flatten(inputs)

    filters = filters0
    features = inputs.pop('features')
    assert (isinstance(features, tf.RaggedTensor))
    features = ragged_layers.values(features)
    spatial_neighs = inputs.pop('spatial_neighs')
    global_neigh = inputs.pop('global_neigh')
    final_indices = inputs.pop('final_indices')

    if inputs:
        raise ValueError('Unused inputs: {}'.format(inputs))

    activation = functools.partial(activate,
                                   activation=activation,
                                   use_batch_norm=use_batch_norm,
                                   dropout_prob=dropout_prob)

    if dropout_prob > 0:
        features = layers.Dropout(dropout_prob)(features)

    for neigh in spatial_neighs:
        # neigh = Lambda(sparse_ops.remove_dim, arguments=dict(axis=0))(neigh)
        # neigh = sparse_ops.remove_dim(neigh)
        features = conv_layers.SpatialEventConv(
            filters, temporal_kernel_size)([features, *neigh])
        features = activate(features)
        # features = layers.Dense(filters * 4)(features)
        # features = activate(features)
        # features = layers.Dense(filters)(features)
        # features = activate(features)
        filters *= 2
    # global_neigh = Lambda(sparse_ops.remove_dim,
    #                       arguments=dict(axis=0))(global_neigh)
    features = conv_layers.GlobalEventConv(
        filters, temporal_kernel_size)([features, global_neigh])
    features = activate(features)

    num_classes = outputs_spec.shape[-1]
    logits_stream = layers.Dense(num_classes, name='stream')(features)
    if final_only:
        logits_final = Lambda(lambda args: tf.gather(*args),
                              name='final')([logits_stream, final_indices])
        outputs = logits_final
    else:
        outputs = logits_stream
    # outputs = (logits_stream, logits_final)
    return tf.keras.Model(flat_inputs, outputs)
