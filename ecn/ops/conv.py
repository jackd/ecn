from typing import Iterable, Union
import tensorflow as tf
import kblocks.ops.sparse as sparse_ops

BoolTensor = tf.Tensor
IntTensor = tf.Tensor
FloatTensor = tf.Tensor


def global_event_conv(features: FloatTensor,
                      dt: tf.SparseTensor,
                      kernel: FloatTensor,
                      decay: FloatTensor,
                      skip_dynamic_checks=False) -> FloatTensor:
    """
    Global event convolution.

    Documentation uses the following:
        n_in: number of input events
        n_out: number of output events
        f_in: number of input features per input event
        f_out: number of output features per output event
        tk: temporal kernel size
        E: number of edges

    Args:
        in_features: [n_in, f_in] float tensor of input event features.
        dt: Sparse tensor with non-negative time differences for values,
            `dense shape` [n_out, n_in].
        kernel: [tk, f_in, f_out] kernel weights.
        decay: [tk] non-negative decay weights in units per-time.

    Returns:
        [n_out, f_out] output features.
    """
    # arg checking
    features.shape.assert_has_rank(2)
    if not skip_dynamic_checks:
        n_in = dt.dense_shape[-1]
        tf.assert_equal(
            tf.shape(features, out_type=getattr(n_in, 'dtype', tf.int64))[0],
            n_in)
    if dt.dense_shape.shape[0] == 3:
        # remove batch dim
        dt = sparse_ops.remove_dim(dt, axis=0)

    assert (dt.dense_shape.shape[0] == 2)
    kernel.shape.assert_has_rank(3)
    assert (dt.dense_shape.shape[0] == 2)
    assert (decay.shape[0] == kernel.shape[0])
    assert (features.shape[1] == kernel.shape[1])
    assert (dt.dtype.is_floating)
    assert (kernel.dtype.is_floating)
    assert (decay.dtype.is_floating)

    # implementation start
    sparse_values = dt.values
    sparse_indices = dt.indices
    dense_shape = dt.dense_shape
    sparse_values = tf.exp(-tf.expand_dims(decay, axis=-1) * sparse_values)

    def map_fn(kernel, sparse_values):
        st = tf.SparseTensor(sparse_indices, sparse_values, dense_shape)
        out_features = tf.sparse.sparse_dense_matmul(st, features)
        return tf.matmul(out_features, kernel)

    kernel = tf.unstack(kernel, axis=0)
    sparse_values = tf.unstack(sparse_values, axis=0)
    features = tf.add_n([map_fn(k, sv) for k, sv in zip(kernel, sparse_values)])
    return features


def spatial_event_conv(features: FloatTensor,
                       dt: Union[tf.SparseTensor, Iterable[tf.SparseTensor]],
                       kernel: FloatTensor, decay: FloatTensor) -> FloatTensor:
    """
    Event convolution.

    Documentation uses the following:
        n_in: number of input events
        n_out: number of output events
        f_in: number of input features per input event
        f_out: number of output features per output event
        sk: number of spatial elements of the kernel. E.g. a 2x2 conv has k=4
        tk: number of temporal elements of the kernel.

    Args:
        features: [n_in, f_in] float tensor of input event features.
        dt: rank-3 `SpareTensor` with `dense_shape` [u, n_out, n_in] and values
            of non-negative time differences.
        kernel: [sk, tk, f_in, f_out] kernel weights.
        decay: [sk, tk] decay weights in units per-time.

    Returns:
        [n_out, f_out] output features.
    """
    decay.shape.assert_has_rank(2)
    assert (decay.dtype.is_floating)

    sk = decay.shape[0]
    assert (sk is not None)
    kernel.shape.assert_has_rank(4)
    assert (kernel.shape[2] == features.shape[1])
    assert (kernel.shape[0] == sk)
    assert (kernel.dtype.is_floating)

    assert (kernel.shape[:2] == decay.shape)

    # arg checking
    if isinstance(dt, tf.SparseTensor):
        # assert (dt.dtype.is_floating)
        # dt.shape.assert_has_rank(3)
        assert (getattr(dt, 'dtype').is_floating)
        shape = getattr(dt, 'shape')
        if shape.ndims == 4:
            # remove batch dim
            dt = sparse_ops.remove_dim(dt)
        else:
            shape.assert_has_rank(3)  # make pyright shut up
        dt_ = sparse_ops.unstack(dt, num_partitions=sk)
    elif isinstance(dt, tf.Tensor):
        raise ValueError('dense dt not supported')
    elif hasattr(dt, '__iter__'):
        dt_ = []
        for d in dt:
            assert (isinstance(d, tf.SparseTensor))
            shape = d.shape
            if shape.ndims == 3:
                # remove batch dim
                d = sparse_ops.remove_dim(d)
            else:
                shape.assert_has_rank(2)
            assert (d.dtype.is_floating)
            dt_.append(d)
    else:
        raise ValueError('Unrecognized dt type {}'.format(dt))
    sk = len(dt_)
    features.shape.assert_has_rank(2)

    # implementation start
    kernel = tf.unstack(kernel, axis=0)
    decay = tf.unstack(decay, axis=0)
    terms = [
        global_event_conv(features, *args, skip_dynamic_checks=True)
        for args in zip(dt_, kernel, decay)
    ]
    return tf.add_n(terms)
