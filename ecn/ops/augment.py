from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf

IntArray = np.ndarray

IntTensor = tf.Tensor
BoolTensor = tf.Tensor
FloatTensor = tf.Tensor

GridShape = Union[IntTensor, IntArray, Tuple[int, int]]
MaybeBool = Union[bool, float, tf.Tensor]


def to_bool(maybe: MaybeBool) -> Union[bool, BoolTensor, FloatTensor]:
    if isinstance(maybe, (tf.Tensor, np.ndarray)):
        # dtype = maybe.dtype
        dtype = getattr(maybe, "dtype")
        if dtype.is_bool:
            return maybe
        elif dtype.is_floating:
            return tf.random.uniform(()) < maybe
        else:
            raise ValueError(
                f"maybe must be a bool or float if a tensor, " "got dtype {maybe.dtype}"
            )
    elif isinstance(maybe, bool):
        return maybe
    else:
        return tf.random.uniform(()) < maybe


def rotation2d_matrix(radians):
    c = tf.cos(radians)
    s = tf.sin(radians)
    return tf.reshape(tf.stack([c, s, -s, c]), (2, 2))


def rotate2d(coords, radians, center=None):
    coords.shape.assert_has_rank(2)
    assert coords.shape[1] == 2
    radians.shape.assert_has_rank(0)
    rot = rotation2d_matrix(radians)
    if center is None:
        return tf.matmul(coords, rot)
    else:
        center = tf.convert_to_tensor(center, dtype=coords.dtype)
        return tf.matmul(coords - center, rot) + center


def reverse_time(
    time: IntTensor, coords: IntTensor, polarity: IntTensor
) -> Tuple[IntTensor, IntTensor, BoolTensor]:
    time = time[-1] - time
    polarity = tf.logical_not(polarity)
    time, coords, polarity = (tf.reverse(t, axis=[0]) for t in (time, coords, polarity))
    return time, coords, polarity


def flip_dim(
    coords: tf.Tensor, grid_shape: GridShape, dim: Union[int, Sequence[int]]
) -> tf.Tensor:
    coords = tf.unstack(coords, axis=-1)
    if isinstance(dim, int):
        coords[dim] = grid_shape[dim] - 1 - coords[dim]
    else:
        for d in dim:
            coords[d] = grid_shape[d] - 1 - coords[d]
    return tf.stack(coords, axis=-1)


def smart_cond(
    condition: Union[bool, BoolTensor], if_true: Callable, if_false: Callable
):
    """
    Similar to tf.cond but just calls the relevant functio if condition is bool.

    This is useful when we want to potentially avoid using `tf.cond`, e.g. when
    we want to use `tf.graph_util.import_meta_graph`.
    """
    if isinstance(condition, bool):
        return if_true() if condition else if_false()
    else:
        return tf.cond(condition, if_true, if_false)


def augment(
    time: IntTensor,
    coords: IntTensor,
    polarity: BoolTensor,
    grid_shape: GridShape,
    flip_lr: MaybeBool = False,
    flip_ud: MaybeBool = False,
    flip_time: MaybeBool = False,
    rotate_limits: Optional[Tuple[float, float]] = None,
) -> Tuple[IntTensor, IntTensor, BoolTensor, Optional[BoolTensor]]:
    mask = None
    if all(b is False for b in (flip_lr, flip_ud, flip_time)) and rotate_limits is None:
        return time, coords, polarity, mask
    if flip_lr is not False or flip_ud is not False:
        # autograph complains about lambdas
        def flipx():
            return grid_shape[0] - 1 - x

        def no_flipx():
            return x

        def flipy():
            return grid_shape[1] - 1 - y

        def no_flipy():
            return y

        x, y = tf.unstack(coords, axis=-1)
        x = smart_cond(to_bool(flip_lr), flipx, no_flipx)
        y = smart_cond(to_bool(flip_ud), flipy, no_flipy)
        coords = tf.stack((x, y), axis=-1)

    def flipt():
        return reverse_time(time, coords, polarity)

    def no_flipt():
        return (time, coords, polarity)

    time, coords, polarity = smart_cond(to_bool(flip_time), flipt, no_flipt)

    if rotate_limits is not None:
        min_rot, max_rot = rotate_limits
        radians = tf.random.uniform((), minval=min_rot, maxval=max_rot)
        dtype = coords.dtype
        coords = tf.cast(coords, tf.float32)
        coords = rotate2d(coords, radians, center=tf.cast(grid_shape, tf.float32) / 2)
        coords = tf.cast(tf.round(coords), dtype)
        mask = tf.reduce_all(tf.logical_and(coords >= 0, coords < grid_shape), axis=-1)
        time = tf.boolean_mask(time, mask)
        coords = tf.boolean_mask(coords, mask)
        polarity = tf.boolean_mask(polarity, mask)

    return time, coords, polarity, mask


if __name__ == "__main__":
    from ecn.problems import sources

    # source = sources.nmnist_source()
    source = sources.cifar10_dvs_source()
    grid_shape = source.meta["grid_shape"]
    for features, label in source.get_dataset("validation"):
        sources.vis_example((features, label))
        time, coords, polarity, _ = augment(
            features["time"],
            features["coords"],
            features["polarity"],
            grid_shape=grid_shape,
            flip_ud=True,
        )
        features = dict(time=time, coords=coords, polarity=polarity)
        sources.vis_example((features, label), reverse_xy=True)
