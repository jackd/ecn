import tensorflow as tf

BoolTensor = tf.Tensor
IntTensor = tf.Tensor
FloatTensor = tf.Tensor


def global_event_conv(in_features: FloatTensor, in_times: FloatTensor,
                      out_times: FloatTensor, sparse_indices: IntTensor,
                      kernel: FloatTensor, decay: FloatTensor) -> FloatTensor:
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
        in_times: [n_in] float tensor of input event times.
        out_times: [n_out] float tensor of output event times.
        sparse_indices: [E, 2] sparse indices for each kernel.
        kernel: [tk, f_in, f_out] kernel weights.
        decay: [tk] decay weights in units per-time.

    Returns:
        [n_out, f_out] output features.
    """
    in_features.shape.assert_has_rank(2)
    in_times.shape.assert_has_rank(1)
    out_times.shape.assert_has_rank(1)
    sparse_indices.assert_has_rank(2)
    kernel.assert_has_rank(3)
    decay.assert_has_rank(1)
    assert (kernel.shape[0] == kernel.shape[0])
    assert (in_features.shape[0] == in_times.shape[0])

    dense_shape = (tf.size(out_times), tf.size(in_times))
    out_indices, in_indices = tf.split(sparse_indices, axis=-1)
    t_out = tf.gather(out_times, out_indices)
    t_in = tf.gather(in_times, in_indices)
    sparse_values = tf.exp(tf.expand_dims(decay, axis=-1) * (t_in - t_out))

    def map_fn(kernel, sparse_values):
        st = tf.SparseTensor(sparse_indices, sparse_values, dense_shape)
        features = tf.sparse.sparse_dense_matmul(st, in_features)
        return tf.matmul(features, kernel)

    kernel = tf.unstack(kernel, axis=0)
    sparse_values = tf.unstack(sparse_values, axis=0)
    features = tf.add_n([map_fn(k, sv) for k, sv in zip(kernel, sparse_values)])
    return features


def event_conv(in_features: FloatTensor, in_times: FloatTensor,
               out_times: FloatTensor, sparse_indices: IntTensor,
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
        E: number of edges

    Args:
        in_features: [n_in, f_in] float tensor of input event features.
        in_times: [n_in] float tensor of input event times.
        out_times: [n_out] float tensor of output event times.
        sparse_indices: [sk, E?, 2] sparse indices for each kernel.
        kernel: [sk, tk, f_in, f_out] kernel weights.
        decay: [tk] decay weights in units per-time.

    Returns:
        [n_out, f_out] output features.
    """

    def map_fn(sparse_indices, kernel):
        """
        Args:
            sparse_indices: [E, 2] int
            kernel: [f_in, f_out] float

        Returns:
            features: n_out, f_out
        """
        return global_event_conv(in_features, in_times, out_times,
                                 sparse_indices, kernel, decay)

    if isinstance(sparse_indices, tf.Tensor):
        sparse_indices = tf.unstack(sparse_indices, axis=0)
    elif isinstance(sparse_indices, tf.RaggedTensor):
        sparse_indices = tf.split(sparse_indices.values,
                                  sparse_indices.row_lengths())
    else:
        raise ValueError(
            'sparse_indices must be a Tensor or RaggedTensor, got {}'.format(
                sparse_indices))
    kernel = tf.unstack(kernel, axis=0)
    terms = [map_fn(si, k) for si, k in zip(sparse_indices, kernel)]
    return tf.add_n(terms)
