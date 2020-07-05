from typing import Optional, Tuple

import tensorflow as tf

IntTensor = tf.Tensor
BoolTensor = tf.Tensor


def row_sorted(row_indices: IntTensor, col_indices: IntTensor, values: IntTensor):
    order = tf.argsort(row_indices)
    return (
        tf.gather(row_indices, order),
        tf.gather(col_indices, order),
        tf.gather(values, order),
    )


def transpose_csr(
    indices: IntTensor,
    splits: IntTensor,
    values: tf.Tensor,
    nrows_out: Optional[IntTensor] = None,
    validate=True,
) -> Tuple[IntTensor, IntTensor, tf.Tensor]:
    indices = tf.convert_to_tensor(indices, dtype_hint=tf.int64)
    splits = tf.convert_to_tensor(splits, dtype_hint=tf.int64)
    values = tf.convert_to_tensor(values)
    col_indices = tf.ragged.row_splits_to_segment_ids(splits)  # initial row indices
    row_indices, col_indices, values = row_sorted(indices, col_indices, values)

    rt = tf.RaggedTensor.from_value_rowids(
        col_indices, row_indices, nrows=nrows_out, validate=validate
    )
    indices = rt.values
    splits = rt.row_splits
    return indices, splits, values


def mask_rows(
    values: tf.Tensor, row_splits: IntTensor, mask: BoolTensor
) -> Tuple[tf.Tensor, IntTensor]:
    rt = tf.RaggedTensor.from_row_splits(values, row_splits, validate=False)
    mask = tf.RaggedTensor.from_row_splits(mask, row_splits, validate=False)
    rt = tf.ragged.boolean_mask(rt, mask)
    return rt.values, rt.row_splits


def gather_rows(values: tf.Tensor, row_splits: IntTensor, indices: IntTensor):
    rt = tf.RaggedTensor.from_row_splits(values, row_splits, validate=False)
    rt = tf.gather(rt, indices)
    return rt.values, rt.row_splits
