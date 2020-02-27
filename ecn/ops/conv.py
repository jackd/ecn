from typing import Sequence, Union
import tensorflow as tf
import kblocks.ops.sparse as sparse_ops

BoolTensor = tf.Tensor
IntTensor = tf.Tensor
FloatTensor = tf.Tensor


def as_complex_tensor(x: tf.Tensor):
    return x if x.dtype.is_complex else tf.complex(x, tf.zeros_like(x))


def as_complex(x):
    if isinstance(x, tf.Tensor):
        return as_complex_tensor(x)
    elif x.dtype.is_complex:
        return x
    elif isinstance(x, tf.SparseTensor):
        return tf.SparseTensor(x.indices, as_complex_tensor(x.values),
                               x.dense_shape)
    elif isinstance(x, tf.RaggedTensor):
        return tf.ragged.map_flat_values(as_complex_tensor, x)
    else:
        raise TypeError(f'Unrecognized type for x, {x}')


def _validate_dtype(x):
    assert (x.dtype.is_floating or x.dtype.is_complex)


def complex_split(x):
    if isinstance(x, tf.Tensor):
        return tf.math.real(x), tf.math.imag(x)
    elif isinstance(x, tf.SparseTensor):
        real = tf.SparseTensor(x.indices, tf.math.real(x.values), x.dense_shape)
        imag = tf.SparseTensor(x.indices, tf.math.imag(x.values), x.dense_shape)
        return real, imag
    elif isinstance(x, tf.RaggedTensor):
        return (tf.ragged.map_flat_values(tf.math.real, x),
                tf.ragged.map_flat_values(tf.math.imag, x))
    else:
        raise TypeError(f'Unrecognized tensor type for x, {x}')


def sparse_dense_matmul(sp_a: tf.SparseTensor, b: tf.Tensor):
    # gradients aren't supported for complex sparse dense multiplication
    if sp_a.dtype.is_complex:
        assert (b.dtype.is_complex)
        ar, ai = complex_split(sp_a)
        br, bi = complex_split(b)

        b0_cat = tf.concat((br, bi), axis=-1)
        b1_cat = tf.concat((-bi, br), axis=-1)

        t0 = tf.sparse.sparse_dense_matmul(ar, b0_cat)
        t1 = tf.sparse.sparse_dense_matmul(ai, b1_cat)
        total = t0 + t1
        real, imag = tf.split(total, 2, axis=-1)
        return tf.complex(real, imag)
    else:
        return tf.sparse.sparse_dense_matmul(sp_a, b)


def featureless_temporal_event_conv(dt: tf.SparseTensor, kernel: FloatTensor,
                                    decay: FloatTensor) -> FloatTensor:
    """
    Global event convolution on inputs.

    Documentation uses the following:
        n_out: number of output events
        f_out: number of output features per output event
        tk: temporal kernel size
        E: number of edges

    Args:
        dt: Sparse tensor with non-negative time differences for values,
            `dense_shape == [n_out, n_in]`.
        kernel: [tk, f_out] kernel weights
        decay: [tk] non-negative decay weights in units per-time.

    Returns:
        [n_out, f_out] output features.
    """
    # arg checking
    tf.debugging.assert_non_negative(dt.values)
    if decay.dtype.is_complex:
        assert (kernel.dtype.is_complex)
        dt = as_complex(dt)
    kt = decay.shape[0]
    assert (kt is not None)

    kernel.shape.assert_has_rank(2)
    decay.shape.assert_has_rank(1)
    assert (kernel.shape[0] == decay.shape[0])

    assert (isinstance(dt, tf.SparseTensor))
    dt.shape.assert_has_rank(2)
    tf.debugging.assert_non_negative(
        tf.math.real(decay) if decay.dtype.is_complex else decay)
    values = tf.exp(-tf.expand_dims(decay, axis=0) *
                    tf.expand_dims(dt.values, axis=-1))  # [E, kt]
    i, j = tf.unstack(dt.indices, axis=-1)
    del j
    n_out = dt.dense_shape[0]
    row_sum = tf.math.unsorted_segment_sum(values, i, num_segments=n_out)
    return tf.matmul(row_sum, kernel)


def binary_temporal_event_conv(features: BoolTensor,
                               dt: tf.SparseTensor,
                               kernel: FloatTensor,
                               decay: FloatTensor,
                               validate: bool = True) -> FloatTensor:
    """
    Global event convolution on binary inputs.

    Documentation uses the following:
        n_in: number of input events
        n_out: number of output events
        f_out: number of output features per output event
        tk: temporal kernel size
        E: number of edges

    Args:
        features: [n_in] bool tensor of input events.
        dt: Sparse tensor with non-negative time differences for values,
            `dense_shape == [n_out, n_in]`.
        kernel: [2*tk, f_out] kernel weights
        decay: [tk] non-negative decay weights in units per-time.

    Returns:
        [n_out, f_out] output features.
    """
    # arg checking
    tf.debugging.assert_non_negative(dt.values)
    if decay.dtype.is_complex:
        assert (kernel.dtype.is_complex)
        dt = as_complex(dt)
    features.shape.assert_has_rank(1)
    kt = decay.shape[0]
    assert (kt is not None)

    if validate:
        n_in = dt.dense_shape[-1]
        tf.assert_equal(
            tf.shape(features, out_type=getattr(n_in, 'dtype', tf.int64))[0],
            n_in)
    assert (features.dtype == tf.bool)
    kernel.shape.assert_has_rank(2)
    decay.shape.assert_has_rank(1)
    assert (kernel.shape[0] == 2 * decay.shape[0])

    assert (isinstance(dt, tf.SparseTensor))
    dt.shape.assert_has_rank(2)
    tf.debugging.assert_non_negative(
        tf.math.real(decay) if decay.dtype.is_complex else decay)
    values = tf.exp(-tf.expand_dims(decay, axis=0) *
                    tf.expand_dims(dt.values, axis=-1))  # [E, kt]
    i, j = tf.unstack(dt.indices, axis=-1)
    features = tf.gather(features, j)
    segments = i * 2 + tf.cast(features, i.dtype)
    n_out = dt.dense_shape[0]
    # n_out * (2 * tk)
    row_sum = tf.math.unsorted_segment_sum(values,
                                           segments,
                                           num_segments=n_out * 2)
    row_sum = tf.reshape(row_sum, (-1, 2 * kt))
    return tf.matmul(row_sum, kernel)


def temporal_event_pooling(features: FloatTensor, dt: FloatTensor,
                           value_rowids: IntTensor, batch_size: IntTensor,
                           kernel: FloatTensor, decay: FloatTensor):
    """
    Equivalent to temporal_event_conv when there is a single event at the end.

    Args:
        features: [n_in, f_in]
        dt: [n_in]
        value_rowids: [n_in]
        batch_size: []
        kernel: [tk, f_in, f_out]
        decay: [tk]
    """
    features.shape.assert_has_rank(2)
    assert (features.dtype.is_floating)
    dt.shape.assert_has_rank(1)
    assert (dt.dtype.is_floating)
    value_rowids.shape.assert_has_rank(1)
    assert (value_rowids.dtype.is_integer)
    kernel.shape.assert_has_rank(3)
    decay.shape.assert_has_rank(1)
    tk, f_in, f_out = kernel.shape
    assert (decay.shape[0] == tk)
    assert (features.shape[1] == f_in)

    decayed_dt = tf.exp(-tf.expand_dims(decay, axis=0) *
                        tf.expand_dims(dt, axis=1))  # [E, tk]
    left = tf.expand_dims(features, axis=1) * tf.expand_dims(decayed_dt,
                                                             axis=-1)
    # left is now [E, tk, f_in]
    left = tf.math.unsorted_segment_sum(left, value_rowids, batch_size)
    # now [batch_size, tk, f_in]
    left = tf.reshape(left, (batch_size, tk * f_in))
    kernel = tf.reshape(kernel, (tk * f_in, f_out))
    return tf.matmul(left, kernel)


def temporal_event_conv(features: FloatTensor,
                        dt: tf.SparseTensor,
                        kernel: FloatTensor,
                        decay: FloatTensor,
                        validate: bool = True) -> FloatTensor:
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
        features: [n_in, f_in] float tensor of input event features.
        dt: Sparse tensor with non-negative time differences for values,
            `dense_shape == [n_out, n_in]`.
        kernel: [tk, f_in, f_out] kernel weights.
        decay: [tk] non-negative decay weights in units per-time.

    Returns:
        [n_out, f_out] output features.
    """
    # arg checking
    tf.debugging.assert_non_negative(dt.values)
    if decay.dtype.is_complex:
        assert (kernel.dtype.is_complex)
        dt = as_complex(dt)
        features = as_complex(features)
    features.shape.assert_has_rank(2)
    if validate:
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
    _validate_dtype(dt)
    _validate_dtype(kernel)
    _validate_dtype(decay)

    # implementation start
    sparse_values = dt.values
    sparse_indices = dt.indices
    dense_shape = dt.dense_shape
    tf.debugging.assert_non_negative(
        tf.math.real(decay) if decay.dtype.is_complex else decay)
    sparse_values = tf.exp(-tf.expand_dims(decay, axis=-1) * sparse_values)

    def map_fn(kernel, sparse_values):
        st = tf.SparseTensor(sparse_indices, sparse_values, dense_shape)
        # out_features = tf.sparse.sparse_dense_matmul(st, features)
        out_features = sparse_dense_matmul(st, features)
        return tf.matmul(out_features, kernel)

    kernel = tf.unstack(kernel, axis=0)
    sparse_values = tf.unstack(sparse_values, axis=0)
    features = tf.add_n([map_fn(k, sv) for k, sv in zip(kernel, sparse_values)])
    return features


def featureless_spatio_temporal_event_conv(
        dt: Union[tf.SparseTensor, Sequence[tf.SparseTensor]],
        kernel: FloatTensor, decay: FloatTensor) -> FloatTensor:
    """
    Event convolution.

    Documentation uses the following:
        n_out: number of output events
        f_out: number of output features per output event
        sk: number of spatial elements of the kernel. E.g. a 2x2 conv has k=4
        tk: number of temporal elements of the kernel.

    Args:
        dt: rank-3 `SpareTensor` with `dense_shape` [sk, n_out, n_in] and values
            of non-negative time differences.
        kernel: [sk, tk, f_out] kernel weights.
        decay: [sk, tk] decay weights in units per-time.

    Returns:
        [n_out, f_out] output features.
    """
    decay.shape.assert_has_rank(2)
    if decay.dtype.is_complex:
        assert (kernel.dtype.is_complex)
    kernel.shape.assert_has_rank(3)
    _validate_dtype(decay)
    sk = decay.shape[0]
    assert (sk is not None)

    assert (kernel.shape[0] == sk)
    _validate_dtype(kernel)
    assert (kernel.shape[0] == decay.shape[0])
    assert (kernel.shape[1] == decay.shape[1])
    dt_ = _split_spatial_dt(dt, sk)

    # implementation start
    kernel = tf.unstack(kernel, axis=0)
    decay = tf.unstack(decay, axis=0)
    terms = [
        featureless_temporal_event_conv(*args)
        for args in zip(dt_, kernel, decay)
    ]
    return tf.add_n(terms)


def binary_spatio_temporal_event_conv(
        features: BoolTensor,
        dt: Union[tf.SparseTensor, Sequence[tf.SparseTensor]],
        kernel: FloatTensor, decay: FloatTensor) -> FloatTensor:
    """
    Event convolution.

    Documentation uses the following:
        n_in: number of input events
        n_out: number of output events
        f_out: number of output features per output event
        sk: number of spatial elements of the kernel. E.g. a 2x2 conv has k=4
        tk: number of temporal elements of the kernel.

    Args:
        features: [n_in] float tensor of input event features.
        dt: rank-3 `SpareTensor` with `dense_shape` [sk, n_out, n_in] and values
            of non-negative time differences.
        kernel: [sk, 2*tk, f_out] kernel weights.
        decay: [sk, tk] decay weights in units per-time.

    Returns:
        [n_out, f_out] output features.
    """
    decay.shape.assert_has_rank(2)
    if decay.dtype.is_complex:
        assert (kernel.dtype.is_complex)
    kernel.shape.assert_has_rank(3)
    _validate_dtype(decay)
    sk = decay.shape[0]
    assert (sk is not None)

    assert (kernel.shape[0] == sk)
    _validate_dtype(kernel)
    assert (kernel.shape[0] == decay.shape[0])
    assert (kernel.shape[1] == 2 * decay.shape[1])
    features.shape.assert_has_rank(1)
    assert (features.dtype.is_bool)

    dt_ = _split_spatial_dt(dt, sk)

    # implementation start
    kernel = tf.unstack(kernel, axis=0)
    decay = tf.unstack(decay, axis=0)
    terms = [
        binary_temporal_event_conv(features, *args, validate=False)
        for args in zip(dt_, kernel, decay)
    ]
    return tf.add_n(terms)


def spatio_temporal_event_conv(
        features: FloatTensor,
        dt: Union[tf.SparseTensor, Sequence[tf.SparseTensor]],
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
        dt: rank-3 `SpareTensor` with `dense_shape` [sk, n_out, n_in] and values
            of non-negative time differences.
        kernel: [sk, tk, f_in, f_out] kernel weights.
        decay: [sk, tk] decay weights in units per-time.

    Returns:
        [n_out, f_out] output features.
    """
    if decay.dtype.is_complex:
        assert (kernel.dtype.is_complex)
        features = as_complex(features)
    decay.shape.assert_has_rank(2)
    kernel.shape.assert_has_rank(4)
    _validate_dtype(decay)
    sk = decay.shape[0]
    assert (sk is not None)

    assert (kernel.shape[0] == sk)
    _validate_dtype(kernel)

    assert (kernel.shape[:2] == decay.shape)
    assert (kernel.shape[-2] == features.shape[-1])
    features.shape.assert_has_rank(2)

    dt_ = _split_spatial_dt(dt, sk)

    # implementation start
    kernel = tf.unstack(kernel, axis=0)
    decay = tf.unstack(decay, axis=0)
    terms = [
        temporal_event_conv(features, *args, validate=False)
        for args in zip(dt_, kernel, decay)
    ]
    return tf.add_n(terms)


def _split_spatial_dt(dt: Union[tf.SparseTensor, Sequence[tf.SparseTensor]],
                      sk: int):
    # arg checking
    if isinstance(dt, tf.SparseTensor):
        # dt.shape.assert_has_rank(3)
        _validate_dtype(dt)
        shape = getattr(dt, 'shape')
        if shape.ndims == 4:
            # remove batch dim
            dt = sparse_ops.remove_leading_dim(dt)
        else:
            shape.assert_has_rank(3)
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
                d = sparse_ops.remove_leading_dim(d)
            else:
                shape.assert_has_rank(2)
            _validate_dtype(d)
            dt_.append(d)
    else:
        raise ValueError('Unrecognized dt type {}'.format(dt))
    return dt_
