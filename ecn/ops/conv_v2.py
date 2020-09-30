import tensorflow as tf

from . import conv as _base_ops

FloatTensor = tf.Tensor

as_complex = _base_ops.as_complex
sparse_dense_matmul = _base_ops.sparse_dense_matmul
_validate_dtype = _base_ops._validate_dtype  # pylint: disable=protected-access

temporal_event_conv = _base_ops.temporal_event_conv


def spatial_event_conv(
    features: FloatTensor,
    sp: tf.SparseTensor,
    kernel: FloatTensor,
    use_csr: bool = False,
):
    """
    Spatial event convolution.

    Args:
        features: [n_in, f_in]
        sp: [n_out * sk, n_in] SparseTensor with dense_shape and values
            given by decayed dts.
        kernel: [sk, f_in, f_out]

    Returns:
        [n_out, f_out] float tensor.
    """
    sp.shape.assert_has_rank(2)
    features.shape.assert_has_rank(2)
    kernel.shape.assert_has_rank(3)
    sk, f_in, f_out = kernel.shape
    assert all(d is not None for d in (sk, f_in, f_out))
    assert features.shape[1] == f_in
    n_out = -1

    if use_csr:
        x = _base_ops.csr_matmul(sp, features)
    else:
        x = tf.sparse.sparse_dense_matmul(sp, features)
    x = tf.reshape(x, (n_out, sk * f_in))
    kernel = tf.reshape(kernel, (sk * f_in, f_out))
    x = tf.matmul(x, kernel)
    return x


def spatio_temporal_event_conv(
    features: FloatTensor,
    dt: tf.SparseTensor,
    kernel: FloatTensor,
    decay: FloatTensor,
    use_csr: bool = False,
) -> FloatTensor:
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
        dt: [n_out, sk, n_in] `SpareTensor` with values non-negative time
            difference values.
        kernel: [tk, sk, f_in, f_out] kernel weights.
        decay: [tk, sk] decay weights in units per-time.

    Returns:
        [n_out, f_out] output features.
    """
    if decay.dtype.is_complex:
        assert kernel.dtype.is_complex
        features = as_complex(features)
    decay.shape.assert_has_rank(2)
    kernel.shape.assert_has_rank(4)
    _validate_dtype(decay)
    tk, sk = decay.shape
    assert tk is not None
    assert sk is not None

    _validate_dtype(kernel)

    assert kernel.shape[:2] == (tk, sk)
    assert kernel.shape[-2] == features.shape[-1]
    features.shape.assert_has_rank(2)

    # implementation start
    n_out = dt.dense_shape[0]
    n_in = dt.dense_shape[2]
    # no, s, ni = tf.unstack(dt.indices, axis=-1)
    s = tf.gather(dt.indices, 1, axis=-1)
    dt = tf.sparse.reshape(dt, (n_out * sk, n_in))
    kernels = tf.unstack(kernel, axis=0)
    decay = tf.gather(decay, s, axis=1)
    values = tf.exp(-decay * tf.expand_dims(dt.values, axis=0))
    values = tf.unstack(values, axis=0)

    terms = []
    for k, v in zip(kernels, values):
        sp = tf.SparseTensor(dt.indices, v, dt.dense_shape)
        terms.append(spatial_event_conv(features, sp, k, use_csr=use_csr))

    return tf.add_n(terms)
