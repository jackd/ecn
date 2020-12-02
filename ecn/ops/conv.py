from typing import Sequence, Union

import kblocks.ops.sparse as sparse_ops
import tensorflow as tf

# @tf.RegisterGradient("SparseTensorToCSRSparseMatrix")
# def _SparseTensorToCSRSparseMatrixGrad(op, grad):
#     """Gradient for sparse_tensor_to_csr_sparse_matrix op."""
#     # NOTE: both these return dense gradients...
#     _, values, dense_shape = sparse_lib.csr_sparse_matrix_to_sparse_tensor(
#         grad, type=op.get_attr("T")
#     )

#     # row_ptrs, col_inds, values = sparse_lib.csr_sparse_matrix_components(
#     #     grad, index=tf.zeros((), dtype=tf.int32), type=op.get_attr("T")
#     # )

#     i, j = tf.unstack(op.inputs[0], axis=-1)
#     flat_indices = i * dense_shape[1] + j
#     values = tf.gather(values, flat_indices, axis=0)
#     return (None, values, None)


BoolTensor = tf.Tensor
IntTensor = tf.Tensor
FloatTensor = tf.Tensor


def map_reduce_sum(
    map_fn, inputs, out_shape, out_type, parallel_iterations=None, method="unstack"
):
    if method == "fold":
        init = tf.zeros(out_shape, dtype=out_type)
        return tf.foldl(lambda acc, args: acc + map_fn(args), inputs, init)
    if method == "map":
        return tf.reduce_sum(
            tf.map_fn(
                map_fn, inputs, parallel_iterations=parallel_iterations, dtype=out_type
            ),
            axis=0,
        )
    if method == "vmap":
        return tf.reduce_sum(tf.vectorized_map(map_fn, inputs), axis=0)
    if method == "unstack":
        inputs = (tf.unstack(i, axis=0) for i in inputs)
        return tf.add_n([map_fn(args) for args in zip(*inputs)])

    options = ("fold", "map", "vmap", "unstack")
    raise ValueError(f"Invalid method {method}: must be one of {options}")


def as_complex_tensor(x: tf.Tensor):
    return x if x.dtype.is_complex else tf.complex(x, tf.zeros_like(x))


def as_complex(x):
    if isinstance(x, tf.Tensor):
        return as_complex_tensor(x)
    if x.dtype.is_complex:
        return x
    if isinstance(x, tf.SparseTensor):
        return tf.SparseTensor(x.indices, as_complex_tensor(x.values), x.dense_shape)
    if isinstance(x, tf.RaggedTensor):
        return tf.ragged.map_flat_values(as_complex_tensor, x)
    raise TypeError(f"Unrecognized type for x, {x}")


def _validate_dtype(x):
    assert x.dtype.is_floating or x.dtype.is_complex


def complex_split(x):
    if isinstance(x, tf.Tensor):
        return tf.math.real(x), tf.math.imag(x)
    if isinstance(x, tf.SparseTensor):
        real = tf.SparseTensor(x.indices, tf.math.real(x.values), x.dense_shape)
        imag = tf.SparseTensor(x.indices, tf.math.imag(x.values), x.dense_shape)
        return real, imag
    if isinstance(x, tf.RaggedTensor):
        return (
            tf.ragged.map_flat_values(tf.math.real, x),
            tf.ragged.map_flat_values(tf.math.imag, x),
        )
    raise TypeError(f"Unrecognized tensor type for x, {x}")


@tf.custom_gradient
def _csr_matmul(indices: tf.Tensor, values: tf.Tensor, dense_shape, b: tf.Tensor):
    try:
        # pylint: disable=import-outside-toplevel
        from tensorflow.python.ops.linalg.sparse import sparse as sparse_lib

        # pylint: enable=import-outside-toplevel
    except ImportError as e:
        raise ImportError("use_csr requires tensorflow >= 2.3") from e
    st = tf.SparseTensor(indices, values, dense_shape)
    csr_m = sparse_lib.CSRSparseMatrix(st)
    out = sparse_lib.matmul(csr_m, b)

    def grad(dy):
        rows, cols = tf.unstack(indices, axis=-1)
        parts_a = tf.gather(dy, rows, axis=0)
        parts_b = tf.gather(b, cols, axis=0)

        a_values_grad = tf.math.reduce_sum(parts_a * parts_b, axis=1)
        b_grad = sparse_lib.matmul(csr_m, dy, adjoint_a=True)
        return (None, a_values_grad, None, b_grad)

    return out, grad


def csr_matmul(sp_a: tf.SparseTensor, b: tf.Tensor) -> tf.Tensor:
    return _csr_matmul(sp_a.indices, sp_a.values, sp_a.dense_shape, b)


def sparse_dense_matmul(sp_a: tf.SparseTensor, b: tf.Tensor, use_csr: bool = False):
    matmul = csr_matmul if use_csr else tf.sparse.sparse_dense_matmul
    # gradients aren't supported for complex sparse dense multiplication
    if sp_a.dtype.is_complex:
        assert b.dtype.is_complex
        ar, ai = complex_split(sp_a)
        br, bi = complex_split(b)

        b0_cat = tf.concat((br, bi), axis=-1)
        b1_cat = tf.concat((-bi, br), axis=-1)

        t0 = matmul(ar, b0_cat)
        t1 = matmul(ai, b1_cat)
        total = t0 + t1
        real, imag = tf.split(total, 2, axis=-1)
        return tf.complex(real, imag)

    return matmul(sp_a, b)


def featureless_temporal_event_conv(
    dt: tf.SparseTensor, kernel: FloatTensor, decay: FloatTensor
) -> FloatTensor:
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
    if decay.dtype.is_complex:
        assert kernel.dtype.is_complex
        dt = as_complex(dt)
    kt = decay.shape[0]
    assert kt is not None

    kernel.shape.assert_has_rank(2)
    decay.shape.assert_has_rank(1)
    assert kernel.shape[0] == decay.shape[0]

    assert (
        isinstance(dt, tf.SparseTensor)
        or tf.keras.backend.is_keras_tensor(dt)
        and isinstance(dt.type_spec, tf.SparseTensorSpec)
    )
    dt.shape.assert_has_rank(2)
    values = tf.exp(
        -tf.expand_dims(decay, axis=0) * tf.expand_dims(dt.values, axis=-1)
    )  # [E, kt]
    i, j = tf.unstack(dt.indices, axis=-1)
    del j
    n_out = dt.dense_shape[0]
    row_sum = tf.math.unsorted_segment_sum(values, i, num_segments=n_out)
    return tf.matmul(row_sum, kernel)


def binary_temporal_event_conv(
    features: BoolTensor,
    dt: tf.SparseTensor,
    kernel: FloatTensor,
    decay: FloatTensor,
    validate: bool = True,
) -> FloatTensor:
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
    if decay.dtype.is_complex:
        assert kernel.dtype.is_complex
        dt = as_complex(dt)
    features.shape.assert_has_rank(1)
    kt = decay.shape[0]
    assert kt is not None

    if validate:
        n_in = dt.dense_shape[-1]
        tf.assert_equal(
            tf.shape(features, out_type=getattr(n_in, "dtype", tf.int64))[0], n_in
        )
    assert features.dtype == tf.bool
    kernel.shape.assert_has_rank(2)
    decay.shape.assert_has_rank(1)
    assert kernel.shape[0] == 2 * decay.shape[0]

    assert isinstance(dt, tf.SparseTensor)
    dt.shape.assert_has_rank(2)
    values = tf.exp(
        -tf.expand_dims(decay, axis=0) * tf.expand_dims(dt.values, axis=-1)
    )  # [E, kt]
    i, j = tf.unstack(dt.indices, axis=-1)
    features = tf.gather(features, j)
    segments = i * 2 + tf.cast(features, i.dtype)
    n_out = dt.dense_shape[0]
    row_sum = tf.math.unsorted_segment_sum(values, segments, num_segments=n_out * 2)
    row_sum = tf.reshape(row_sum, (-1, 2 * kt))
    return tf.matmul(row_sum, kernel)


def temporal_event_pooling(
    features: FloatTensor,
    dt: FloatTensor,
    value_rowids: IntTensor,
    batch_size: IntTensor,
    kernel: FloatTensor,
    decay: FloatTensor,
):
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
    assert features.dtype.is_floating
    dt.shape.assert_has_rank(1)
    assert dt.dtype.is_floating
    value_rowids.shape.assert_has_rank(1)
    assert value_rowids.dtype.is_integer
    kernel.shape.assert_has_rank(3)
    decay.shape.assert_has_rank(1)
    tk, f_in, f_out = kernel.shape
    assert decay.shape[0] == tk
    assert features.shape[1] == f_in

    decayed_dt = tf.exp(
        -tf.expand_dims(decay, axis=0) * tf.expand_dims(dt, axis=1)
    )  # [E, tk]
    left = tf.expand_dims(features, axis=1) * tf.expand_dims(decayed_dt, axis=-1)
    # left is now [E, tk, f_in]
    left = tf.math.unsorted_segment_sum(left, value_rowids, batch_size)
    # now [batch_size, tk, f_in]
    left = tf.reshape(left, (batch_size, tk * f_in))
    kernel = tf.reshape(kernel, (tk * f_in, f_out))
    return tf.matmul(left, kernel)


def temporal_event_conv(
    features: FloatTensor,
    dt: tf.SparseTensor,
    kernel: FloatTensor,
    decay: FloatTensor,
    validate: bool = True,
    combine: str = "unstack",
    use_csr: bool = False,
) -> FloatTensor:
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
    if decay.dtype.is_complex:
        assert kernel.dtype.is_complex
        dt = as_complex(dt)
        features = as_complex(features)
    features.shape.assert_has_rank(2)
    if validate:
        n_in = dt.dense_shape[-1]
        tf.assert_equal(
            tf.shape(features, out_type=getattr(n_in, "dtype", tf.int64))[0], n_in
        )
    if dt.dense_shape.shape[0] == 3:
        # remove batch dim
        dt = sparse_ops.remove_dim(dt, axis=0)
    assert dt.dense_shape.shape[0] == 2
    kernel.shape.assert_has_rank(3)
    assert dt.dense_shape.shape[0] == 2
    assert decay.shape[0] == kernel.shape[0]
    assert features.shape[1] == kernel.shape[1]
    _validate_dtype(dt)
    _validate_dtype(kernel)
    _validate_dtype(decay)

    # implementation start
    sparse_values = dt.values
    sparse_indices = dt.indices
    dense_shape = dt.dense_shape
    sparse_values = tf.exp(-tf.expand_dims(decay, axis=-1) * sparse_values)

    def map_fn(args):
        kernel, sparse_values = args
        st = tf.SparseTensor(sparse_indices, sparse_values, dense_shape)
        out_features = sparse_dense_matmul(st, features, use_csr=use_csr)
        return tf.matmul(out_features, kernel)

    f_out = kernel.shape[-1]
    n_out = dt.dense_shape[0]
    out_shape = (n_out, f_out)
    return map_reduce_sum(
        map_fn, (kernel, sparse_values), out_shape, kernel.dtype, method=combine
    )


def featureless_spatio_temporal_event_conv(
    dt: Union[tf.SparseTensor, Sequence[tf.SparseTensor]],
    kernel: FloatTensor,
    decay: FloatTensor,
) -> FloatTensor:
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
        assert kernel.dtype.is_complex
    kernel.shape.assert_has_rank(3)
    _validate_dtype(decay)
    sk = decay.shape[0]
    assert sk is not None

    assert kernel.shape[0] == sk
    _validate_dtype(kernel)
    assert kernel.shape[0] == decay.shape[0]
    assert kernel.shape[1] == decay.shape[1]
    dt_ = _split_spatial_dt(dt, sk)

    # implementation start
    kernel = tf.unstack(kernel, axis=0)
    decay = tf.unstack(decay, axis=0)
    terms = [featureless_temporal_event_conv(*args) for args in zip(dt_, kernel, decay)]
    return tf.add_n(terms)


def binary_spatio_temporal_event_conv(
    features: BoolTensor,
    dt: Union[tf.SparseTensor, Sequence[tf.SparseTensor]],
    kernel: FloatTensor,
    decay: FloatTensor,
) -> FloatTensor:
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
        assert kernel.dtype.is_complex
    kernel.shape.assert_has_rank(3)
    _validate_dtype(decay)
    sk = decay.shape[0]
    assert sk is not None

    assert kernel.shape[0] == sk
    _validate_dtype(kernel)
    assert kernel.shape[0] == decay.shape[0]
    assert kernel.shape[1] == 2 * decay.shape[1]
    features.shape.assert_has_rank(1)
    assert features.dtype.is_bool

    dt_ = _split_spatial_dt(dt, sk)

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
    kernel: FloatTensor,
    decay: FloatTensor,
    combine: str = "unstack",
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
        dt: rank-3 `SpareTensor` with `dense_shape` [sk, n_out, n_in] and values
            of non-negative time differences.
        kernel: [sk, tk, f_in, f_out] kernel weights.
        decay: [sk, tk] decay weights in units per-time.

    Returns:
        [n_out, f_out] output features.
    """
    if decay.dtype.is_complex:
        assert kernel.dtype.is_complex
        features = as_complex(features)
    decay.shape.assert_has_rank(2)
    kernel.shape.assert_has_rank(4)
    _validate_dtype(decay)
    sk = decay.shape[0]
    assert sk is not None

    assert kernel.shape[0] == sk
    _validate_dtype(kernel)

    assert kernel.shape[:2] == decay.shape
    assert kernel.shape[-2] == features.shape[-1]
    features.shape.assert_has_rank(2)

    dt_ = _split_spatial_dt(dt, sk)

    # implementation start
    kernel = tf.unstack(kernel, axis=0)
    decay = tf.unstack(decay, axis=0)
    terms = [
        temporal_event_conv(
            features, *args, validate=False, combine=combine, use_csr=use_csr
        )
        for args in zip(dt_, kernel, decay)
    ]
    return tf.add_n(terms)


def _split_spatial_dt(dt: Union[tf.SparseTensor, Sequence[tf.SparseTensor]], sk: int):
    # arg checking
    if isinstance(dt, tf.SparseTensor):
        _validate_dtype(dt)
        shape = getattr(dt, "shape")
        if shape.ndims == 4:
            # remove batch dim
            dt = sparse_ops.remove_leading_dim(dt)
        else:
            shape.assert_has_rank(3)
        dt_ = sparse_ops.unstack(dt, num_partitions=sk)
    elif isinstance(dt, tf.Tensor):
        raise ValueError("dense dt not supported")
    elif hasattr(dt, "__iter__"):
        dt_ = []
        for d in dt:
            assert isinstance(d, tf.SparseTensor)
            shape = d.shape
            if shape.ndims == 3:
                # remove batch dim
                d = sparse_ops.remove_leading_dim(d)
            else:
                shape.assert_has_rank(2)
            _validate_dtype(d)
            dt_.append(d)
    else:
        raise ValueError("Unrecognized dt type {}".format(dt))
    return dt_
