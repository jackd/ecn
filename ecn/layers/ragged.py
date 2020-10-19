import functools
from typing import Optional

import tensorflow as tf

from composite_layers.utils import wrap
from ecn.ops import ragged as ragged_ops

IntTensor = tf.Tensor
BoolTensor = tf.Tensor


@functools.wraps(ragged_ops.row_sorted)
def row_sorted(row_indices: IntTensor, col_indices: IntTensor, values: IntTensor):
    return wrap(ragged_ops.row_sorted, row_indices, col_indices, values)


@functools.wraps(ragged_ops.transpose_csr)
def transpose_csr(
    indices: IntTensor,
    splits: IntTensor,
    values: tf.Tensor,
    nrows_out: Optional[IntTensor] = None,
    validate=True,
):
    return wrap(
        ragged_ops.transpose_csr, indices, splits, values, nrows_out, validate=validate
    )


@functools.wraps(ragged_ops.gather_rows)
def gather_rows(values: tf.Tensor, row_splits: IntTensor, indices: IntTensor):
    return wrap(ragged_ops.gather_rows, values, row_splits, indices)
