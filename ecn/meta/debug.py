from typing import Optional, Callable
import tensorflow as tf
from ecn.meta import core


class DebugBuilderContext(core.MetaBuilderContext):

    def __init__(self, batch_size: int = 2):
        self._batch_size = batch_size
        self._model_inputs = []

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _batch(self, tensor: tf.Tensor, flat=False):
        if not flat:
            tensor = tf.expand_dims(tensor, axis=0)
        return tf.tile(
            tensor,
            (self.batch_size, *(1 for _ in range(tensor.shape.ndims - 1))))

    # def learning_phase(self) -> tf.Tensor:
    #     return tf.keras.backend.learning_phase()

    def batch(self, tensor: core.TensorLike):
        if isinstance(tensor, tf.Tensor):
            return self._batch(tensor)
        elif isinstance(tensor, tf.SparseTensor):
            values = self._batch(tensor.values, flat=True)
            indices = self._batch(tensor.indices, flat=True)
            b = tf.expand_dims(tf.range(self.batch_size), axis=-1)
            b = tf.tile(b, (1, tf.shape(values)[0]))
            indices = tf.concat((tf.expand_dims(b, 0), indices), axis=-1)
            dense_shape = tf.concat([(self.batch_size,), tensor.dense_shape],
                                    axis=0)
            return tf.SparseTensor(indices, values, dense_shape)
        elif isinstance(tensor, tf.RaggedTensor):
            values = self._batch(tensor.values, flat=True)
            row_lengths = self._batch(tensor.row_lengths(), flat=True)
            rl2 = tf.tile(tf.expand_dims(tensor.nrows(), 0), self.batch_size)
            return tf.RaggedTensor.from_row_lengths(
                tf.RaggedTensor.from_row_lengths(values, row_lengths), rl2)
        else:
            raise TypeError(
                'Invalid type `tensor`: must be TensorLike, got {}'.format(
                    tensor))

    def model_input(self, tensor: core.TensorLike, name: Optional[str] = None):
        assert (tensor.shape[0] == self.batch_size)
        out = tf.identity(tensor, name=name)
        self._model_inputs.append(out)
        return out


def debug_builder(build_fn: Callable, inputs, batch_size: int = 2):
    with DebugBuilderContext(batch_size=batch_size):
        outputs = build_fn(*inputs)
    return outputs
