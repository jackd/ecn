from typing import Tuple, Optional
import tensorflow as tf
from kblocks.ops.ragged import splits_to_ids
from kblocks.ops.ragged import ids_to_splits
from kblocks.ops.ragged import splits_to_lengths
from kblocks.ops.ragged import lengths_to_splits
IntTensor = tf.Tensor
BoolTensor = tf.Tensor


def row_sorted(row_indices: IntTensor, col_indices: IntTensor,
               values: IntTensor):
    order = tf.argsort(row_indices)
    return (
        tf.gather(row_indices, order),
        tf.gather(col_indices, order),
        tf.gather(values, order),
    )


def transpose_csr(indices: IntTensor,
                  splits: IntTensor,
                  values: tf.Tensor,
                  nrows_out: Optional[IntTensor] = None,
                  validate=True) -> Tuple[IntTensor, IntTensor, tf.Tensor]:
    indices = tf.convert_to_tensor(indices, dtype_hint=tf.int64)
    splits = tf.convert_to_tensor(splits, dtype_hint=tf.int64)
    values = tf.convert_to_tensor(values)
    col_indices = tf.ragged.row_splits_to_segment_ids(
        splits)  # initial row indices
    row_indices, col_indices, values = row_sorted(indices, col_indices, values)

    rt = tf.RaggedTensor.from_value_rowids(col_indices,
                                           row_indices,
                                           nrows=nrows_out,
                                           validate=validate)
    return rt.values, rt.row_splits, values
    # row_splits = tf.ragged.segment_ids_to_row_splits(row_indices,
    #                                                  out_type=indices.dtype)
    # return col_indices, row_splits, values


def mask_rows(values: tf.Tensor, row_splits: IntTensor,
              mask: BoolTensor) -> Tuple[tf.Tensor, IntTensor]:
    rt = tf.RaggedTensor.from_row_splits(values, row_splits, validate=False)
    mask = tf.RaggedTensor.from_row_splits(mask, row_splits, validate=False)
    rt = tf.ragged.boolean_mask(rt, mask)
    return rt.values, rt.row_splits


def gather_rows(values: tf.Tensor, row_splits: IntTensor, indices: IntTensor):
    rt = tf.RaggedTensor.from_row_splits(values, row_splits, validate=False)
    rt = tf.gather(rt, indices)
    return rt.values, rt.row_splits
