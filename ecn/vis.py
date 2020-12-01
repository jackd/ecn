from typing import Callable, Optional, Sequence

import gin
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import logging

import ecn.components as comp
import events_tfds.vis.anim as anim
import meta_model.pipeline as pl
from events_tfds.vis.image import as_frame, as_frames
from meta_model.utils import ModelMap


def _get_cached_stream_dataset(
    meta_model_func: Callable, dataset: tf.data.Dataset, num_dims: int
):
    batcher = tf.data.experimental.dense_to_ragged_batch(batch_size=1)
    builder = pl.PipelinedModelBuilder(dataset.element_spec, batcher=batcher)
    with builder:
        with comp.stream_accumulator() as streams:
            inputs = builder.pre_cache_inputs
            logging.info("Building meta model...")
            meta_model_func(*inputs)
            logging.info("Successfully built!")
        times = []
        coords = []
        static_shapes = []
        for stream in streams:
            times.append(stream.times)
            if isinstance(stream, comp.SpatialStream):
                coords.append(stream.shaped_coords)
                static_shapes.append(stream.grid.static_shape)
            else:
                coords.append(tf.zeros((tf.shape(times)[0], num_dims), dtype=tf.int64))
                static_shapes.append((1,) * num_dims)

        has_polarity = "polarity" in inputs[0]
        outputs = (tuple(times), tuple(coords))
        if has_polarity:
            polarity = inputs[0]["polarity"]
            outputs = (*outputs, polarity)

    dataset = dataset.map(ModelMap(builder.build_pre_cache_model(outputs)))

    del builder
    return dataset, static_shapes


def _get_cached_adjacency_dataset(
    meta_model_func: Callable,
    dataset: tf.data.Dataset,
    augment_func: Optional[Callable] = None,
):
    if augment_func is not None:
        dataset = dataset.map(augment_func)
    batcher = tf.data.experimental.dense_to_ragged_batch(batch_size=1)
    builder = pl.PipelinedModelBuilder(dataset.element_spec, batcher=batcher)
    with builder:
        with comp.stream_accumulator() as streams:
            with comp.convolver_accumulator() as convolvers:
                inputs = builder.pre_cache_inputs
                logging.info("Building multi graph...")
                meta_model_func(*inputs)
                logging.info("Successfully built!")
        stream_indices = {s: i for i, s in enumerate(streams)}
        outputs = []
        decay_times = []
        stream_indices = tuple(
            (stream_indices[c.in_stream], stream_indices[c.out_stream])
            for c in convolvers
        )
        for c in convolvers:
            outputs.append((c.indices, c.splits, c.in_stream.times, c.out_stream.times))
            decay_times.append(c.decay_time)

    dataset = dataset.map(ModelMap(builder.build_pre_cache_model(outputs)))
    return dataset, decay_times, stream_indices, len(convolvers)


@gin.configurable(module="ecn.vis")
def vis_streams2d(
    meta_model_func: Callable,
    dataset: tf.data.Dataset,
    augment_func: Optional[Callable] = None,
    num_frames: int = 20,
    fps: int = 4,
    group_size: int = 1,
    skip_vis: int = False,
    flip_up_down: bool = False,
):
    """
    Visualize 2D event streams.

    Args:
        meta_model_func: meta-model building function
        dataset: base dataset to visualize.
        augment_func: optional map function to apply to dataset.
        num_frames: number of frames in the final animation.
        fps: frame-rate
        group_size: if greater than 1, mean / std stats are printed for the size of each
            stream over this number of examples.
        skip_vis: if True, the visualization will be skipped and only example stats will
            be printed to screen.
        flip_up_down: if True, flips images in the final animation up/down.
    """
    if augment_func is not None:
        dataset = dataset.map(augment_func)
    dataset, static_shapes = _get_cached_stream_dataset(
        meta_model_func, dataset, num_dims=2
    )

    sizes = np.zeros((group_size, len(static_shapes)), dtype=np.int64)
    for i, example in enumerate(dataset):
        example = tf.nest.map_structure(lambda x: x.numpy(), example)

        if len(example) == 3:
            times, coords, polarity = example
        else:
            times, coords = example
            polarity = None

        assert len(times) == len(coords) == len(static_shapes)

        img_data = []
        ii = i % group_size
        sizes[ii] = [t.shape[0] for t in times]
        if ii == group_size - 1:
            print(f"{i}: sizes: {np.mean(sizes, axis=0).astype(np.int64)}")
            if group_size > 1:
                print(f"{i}: stds : {np.std(sizes, axis=0).astype(np.int64)}")
        if skip_vis:
            continue

        for i, (t, c, shape) in enumerate(zip(times, coords, static_shapes)):
            if t.shape[0] > 0:
                imgs = as_frames(
                    c,
                    t,
                    num_frames=num_frames,
                    polarity=polarity[: c.shape[0]] if i == 0 else None,
                    shape=shape,
                    flip_up_down=flip_up_down,
                )
                img_data.append(imgs)
        anim.animate_frames_multi(
            *img_data, fps=fps, figsize=(10, 2), tight_layout=True
        )


@gin.configurable(module="ecn.vis")
def vis_streams1d(
    meta_model_func: Callable,
    dataset: tf.data.Dataset,
    augment_func: Optional[Callable] = None,
):
    """Visualize 1D event streams."""
    if augment_func is not None:
        dataset = dataset.map(augment_func)
    dataset, static_shapes = _get_cached_stream_dataset(
        meta_model_func, dataset, num_dims=1
    )
    print("num_channels: {}".format([shape[0] for shape in static_shapes]))
    for example in dataset:
        example = tf.nest.map_structure(lambda x: x.numpy(), example)
        if len(example) == 3:
            times, coords, polarity = example
        else:
            times, coords = example
            polarity = None
        del polarity

        print("Sizes: {}".format([t.shape[0] for t in times]))
        xys = []
        for (t, c, shape) in zip(times, coords, static_shapes):
            (size,) = shape
            c = np.squeeze(c, axis=-1)
            assert np.max(c) < size
            xys.append(((c + 0.5) / size, t))
        plt.figure()
        for i, xy in enumerate(xys):
            plt.scatter(*xy, s=(i + 1))
        plt.show()


@gin.configurable(module="ecn.vis")
def vis_adjacency(
    meta_model_func: Callable,
    dataset: tf.data.Dataset,
    augment_func: Optional[Callable] = None,
) -> None:
    """
    Visualize sparse adjacency matrices.

    Each visualization is a row of sparse matrices along with a histogram of values,
    where values are given by `exp(-dt / decay_time)`.

    Args:
        meta_model_func: meta-model build function, mapping dataset inputs to
            outputs. The actual outputs are ignored. Streams are recorded as they are
            created and the relevant adjacency matrices extracted.
        dataset: tf.data.Dataset used ot create meta-models.
        augment_func: optional map function to apply to dataset.
    """
    if augment_func is not None:
        dataset = dataset.map(augment_func)

    def _vis_single_adjacency(
        indices, splits, in_times, out_times, decay_time, ax0, ax1
    ):
        from scipy.sparse import coo_matrix  # pylint: disable=import-outside-toplevel

        if indices.size == 0:
            return
        print(f"{in_times.size} -> {out_times.size} events")
        print(f"{indices.size} edges")
        row_lengths = splits[1:] - splits[:-1]
        i = np.repeat(np.arange(row_lengths.size), row_lengths, axis=0)
        j = indices
        values = np.exp(-(out_times[i] - in_times[j]) / decay_time)
        sp = coo_matrix((values, (i, j)))
        print(f"Sparsity: {sp.nnz / np.prod(sp.shape) * 100}%")
        ax0.spy(sp, markersize=0.5)
        ax0.set_xticks([])

        ax1.hist(values)

    dataset, decay_times, stream_indices, nc = _get_cached_adjacency_dataset(
        meta_model_func, dataset
    )
    for convolvers in dataset:
        convolvers = tf.nest.map_structure(lambda c: c.numpy(), convolvers)
        _, axes = plt.subplots(2, nc, squeeze=False)
        for i, (conv_data, decay_time, indices) in enumerate(
            zip(convolvers, decay_times, stream_indices)
        ):
            s0, s1 = indices
            title = f"Conv {s0} -> {s1}"
            ax0, ax1 = axes[:, i]
            _vis_single_adjacency(*conv_data, decay_time=decay_time, ax0=ax0, ax1=ax1)
            ax0.title.set_text(title)

        plt.show()


@gin.configurable(module="ecn.vis")
def vis_example2d(
    example,
    num_frames: int = 20,
    fps: int = 4,
    reverse_xy: bool = False,
    flip_up_down: bool = False,
    class_names: Optional[Sequence[str]] = None,
):
    """
    Visualize 2d example as an animated sequence of frames.

    Args:
        example: element from unbatched image stream, tuple of (features, label),
            where features is a dict with "coords", "time" and "polarity" keys.
        num_frames: total number of frames in resulting animation.
        fps: frame-rate
        reverse_xy: if True, transposes coordinates.
        flip_up_down: if true, flips up/down (after possible transposition from
            `reverse_xy`).
        class_names: if provided, prints `class_names[label]` rather than `label`.
    """

    features, label = example
    coords = features["coords"]
    time = features["time"]
    time = (time - tf.reduce_min(time)).numpy()
    polarity = features["polarity"].numpy()
    if class_names is not None:
        print(class_names[label.numpy()])
    else:
        print(label.numpy())

    coords = (coords - tf.reduce_min(coords, axis=0)).numpy()
    print(f"{time.shape[0]} events over {time[-1] - time[0]} dt")
    if reverse_xy:
        coords = coords[:, -1::-1]

    if num_frames == 1:
        frame = as_frame(coords, polarity)
        plt.imshow(frame)
        plt.show()
    else:
        frames = as_frames(
            coords, time, polarity, num_frames=num_frames, flip_up_down=flip_up_down
        )
        anim.animate_frames(frames, fps=fps, tight_layout=True)
