import tensorflow as tf
import ecn.np_utils.conv as _np_conv

BoolTensor = tf.Tensor
IntTensor = tf.Tensor
FloatTensor = tf.Tensor


def global_event_conv(in_features: FloatTensor, in_times: FloatTensor,
                      out_times: FloatTensor, sparse_indices: IntTensor,
                      valid_length: IntTensor, kernel: FloatTensor,
                      decay_time: float) -> FloatTensor:
    """
    Global event convolution.

    Documentation uses the following:
        n_in: number of input events
        n_out: number of output events
        f_in: number of input features per input event
        f_out: number of output features per output event
        E: number of edges

    Args:
        in_features: [n_in, f_in] float tensor of input event features.
        in_times: [n_in] float tensor of input event times.
        out_times: [n_out] float tensor of output event times.
        sparse_indices: [E, 2] sparse indices for each kernel.
        valid_length: [] lengths of valid indices in sparse_indices. All
            entries must be in [0, E).
        kernel: [f_in, f_out] kernel weights.
        decay_time: float indicating temporal decay, in units of time.

    Returns:
        [n_out, f_out] output features.
    """
    E = tf.shape(sparse_indices, out_type=tf.int64)[-2]
    dense_shape = (tf.size(out_times), tf.size(in_times))
    mask = tf.sequence_mask(valid_length, maxlen=E)
    out_indices, in_indices = tf.split(sparse_indices, axis=-1)
    t_out = tf.gather(out_times, out_indices)
    t_in = tf.gather(in_times, in_indices)
    sparse_values = tf.exp((t_in - t_out) / decay_time)
    sparse_values = tf.where(mask, sparse_values, tf.zeros_like(sparse_values))
    st = tf.SparseTensor(sparse_indices, sparse_values, dense_shape)
    features = tf.sparse.sparse_dense_matmul(st, in_features)
    features = tf.matmul(features, kernel)
    return features


def event_conv(in_features: FloatTensor, in_times: FloatTensor,
               out_times: FloatTensor, sparse_indices: IntTensor,
               valid_lengths: IntTensor, kernel: FloatTensor,
               decay_time: float) -> FloatTensor:
    """
    Event convolution.

    Documentation uses the following:
        n_in: number of input events
        n_out: number of output events
        f_in: number of input features per input event
        f_out: number of output features per output event
        k: number of elements of the kernel. E.g. a 2x2 conv has k=4
        E: number of edges

    Args:
        in_features: [n_in, f_in] float tensor of input event features.
        in_times: [n_in] float tensor of input event times.
        out_times: [n_out] float tensor of output event times.
        sparse_indices: [k, E, 2] sparse indices for each kernel.
        valid_lengths: [k] lengths of valid indices in sparse_indices. All
            entries must be in [0, E).
        kernel: [k, f_in, f_out] kernel weights.
        decay_time: float indicating temporal decay, in units of time.

    Returns:
        [n_out, f_out] output features.
    """

    def map_fn(sparse_indices, valid_length, kernel):
        """
        Args:
            sparse_indices: [E, 2] int
            valid_length: [] int
            kernel: [f_in, f_out] float

        Returns:
            features: n_out, f_out
        """
        return global_event_conv(in_features, in_times, out_times,
                                 sparse_indices, valid_length, kernel,
                                 decay_time)

    terms = []
    args = (tf.unstack(arg) for arg in (sparse_indices, valid_lengths, kernel))
    terms = [map_fn(*a) for a in zip(*args)]
    return tf.add_n(terms)


def unlearned_polarity_event_conv(polarity: BoolTensor, in_times: IntTensor,
                                  in_coords: IntTensor, out_times: IntTensor,
                                  out_coords: IntTensor, in_length: IntTensor,
                                  out_length: IntTensor, decay_time: int,
                                  stride: int) -> FloatTensor:
    """
    Args:
        polarity: input polarities.
        in_times: input event times.
        in_coords: input event coordinates.
        out_times: output event times.
        decay_time: time for polarity to decay by a factor of 1.
        stride: convolution stride.

    Returns:
        event_polarities: [n_out, stride, stride, 2] float tensor of decayed
            polarity values.

    """
    padded_length = out_times.size

    def fn(polarity, in_times, in_coords, in_length, out_times, out_coords,
           out_length):
        out_length = out_length.numpy()
        in_length = in_length.numpy()
        return _np_conv.unlearned_polarity_event_conv(
            polarity.numpy()[:in_length],
            in_times.numpy()[:in_length],
            in_coords.numpy()[:in_length],
            out_times.numpy()[:out_length],
            out_coords.numpy()[:out_length],
            decay_time=decay_time,
            stride=stride,
            out_length=padded_length)[0]

    event_polarities = tf.py_function(fn, [
        polarity, in_times, in_coords, in_length, out_times, out_coords,
        out_length
    ], tf.float32)
    event_polarities.set_shape((out_length, stride, stride, 2))
    return event_polarities
