import tensorflow as tf
from ecn.pipelines.core import CustomPipeline
import ecn.ops.spike as spike_ops
import ecn.ops.neighbors as neigh_ops
import ecn.ops.conv as conv_ops


def scnn_pipeline(batch_size: int, filters=(8, 16, 32), **kwargs):
    num_layers = len(filters)

    def pre_cache_map(events, labels, weights):
        times = events['time']
        coords = events['coords']
        coords = coords - tf.reduce_min(coords, axis=0)
        polarity = events['polarity']
        out = dict(coords=coords, times=times)
        in_size = events['num_events'] if 'num_events' in events else tf.size(
            times, out_type=tf.int64)
        del events

        decay_time = 10000
        stride = 2
        event_duration = 10000
        spatial_buffer_size = 8
        max_out_events = 2048
        threshold = 2.
        reset_potential = 1.
        max_neighbors = 2048

        all_layer_kwargs = []
        # first layer - use unlearned conv
        out_times, out_coords, out_size = spike_ops.spike_threshold(
            times, coords // stride, in_size, decay_time, max_out_events,
            threshold, reset_potential)
        event_polarities = conv_ops.unlearned_polarity_event_conv(
            polarity,
            times,
            coords,
            in_size,
            out_times,
            out_coords,
            out_size,
            decay_time=decay_time,
            stride=stride)

        all_layer_kwargs.append(
            dict(out_times=out_times,
                 out_coords=out_coords,
                 out_size=out_size,
                 event_polarities=event_polarities))

        times = out_times
        coords = out_coords
        in_size = out_size
        decay_time *= 2
        event_duration *= 2
        max_out_events //= 4
        max_neighbors //= 2
        spatial_buffer_size *= 2

        for _ in range(1, num_layers):
            out_times, out_coords, out_size = spike_ops.spike_threshold(
                times, coords // stride, in_size, decay_time, max_out_events,
                threshold, reset_potential)
            indices, splits = neigh_ops.compute_neighbors(
                times, coords, in_size, out_times, out_coords, out_size, stride,
                event_duration, spatial_buffer_size, max_neighbors)
            all_layer_kwargs.append(
                dict(out_times=out_times,
                     out_coords=out_coords,
                     out_size=out_size,
                     neigh_indices=indices,
                     neigh_splits=splits))

            times = out_times
            coords = out_coords
            in_size = out_size
            decay_time *= 2
            event_duration *= 2
            max_out_events //= 4
            max_neighbors //= 2
            spatial_buffer_size *= 2
        out['layer_kwargs'] = tuple(all_layer_kwargs)

        # global
        global_times, global_length = spike_ops.global_spike_threshold(
            times,
            in_size,
            max_out_events,
            decay_time,
            threshold=threshold,
            reset_potential=reset_potential)

        global_indices, global_splits = neigh_ops.compute_global_neighbors(
            times, in_size, global_times, global_length, event_duration,
            max_neighbors)
        out['global_kwargs'] = dict(times=global_times,
                                    length=global_length,
                                    indices=global_indices,
                                    splits=global_splits)

        return out, labels

    def post_batch_map(inputs, labels):
        # TODO: index fixes
        global_kwargs = inputs['global_kwargs']
        maxlen = global_kwargs['global_times'].shape[1]
        labels = tf.tile(tf.expand_dims(labels, axis=-1), (1, maxlen))
        weights = tf.sequence_mask(global_kwargs['length'], maxlen=maxlen)
        labels = tf.reshape(labels, (-1,))
        weights = tf.cast(tf.reshape(weights, (-1,)), tf.float32)
        raise NotImplementedError()
        # return out, labels, weights

    pre_batch_map = None
    return CustomPipeline(batch_size, pre_cache_map, pre_batch_map,
                          post_batch_map, **kwargs)


# @nb.njit()
# def pre_batch_map_np(in_times,
#                      in_coords,
#                      num_layers=3,
#                      decay_time=10000,
#                      stride=2,
#                      event_duration=10000,
#                      spatial_buffer_size=32,
#                      max_neighbors_factor=16,
#                      max_out_events=512,
#                      threshold=2.,
#                      reset_potential=-1.):
#     all_times = []
#     all_coords = []
#     all_event_lengths = []

#     all_index_values = []
#     all_index_splits = []

#     for _ in range(3):
#         max_out_events //= 4
#         out_times, out_coords, out_events = spike_ops.spike_threshold(
#             in_times,
#             in_coords // stride,
#             decay_time,
#             max_out_events,
#             threshold=threshold,
#             reset_potential=reset_potential)

#         index_values, index_splits, _ = neigh.compute_neighbors(
#             in_times,
#             in_coords,
#             out_times,
#             out_coords,
#             stride=stride,
#             event_duration=event_duration,
#             spatial_buffer_size=spatial_buffer_size,
#             max_neighbors=max_neighbors_factor * max_out_events,
#         )

#         all_times.append(out_times)
#         all_coords.append(out_coords)
#         all_event_lengths.append(out_events)
#         all_index_values.append(index_values)
#         all_index_splits.append(index_splits)

#         max_out_events //= 4
#         event_duration *= 2
#         decay_time *= 2

#         in_times = out_times[:out_events]
#         in_coords = out_coords[:out_events]

#     # global spike / neighbors
#     global_times, global_length = spike_ops.global_spike_threshold(
#         in_times,
#         max_out_events,
#         decay_time,
#         threshold=threshold,
#         reset_potential=reset_potential)

#     indices, splits, _ = neigh.compute_global_neighbors(in_times, global_times,
#                                                         100, event_duration,
#                                                         1024)
#     global_vals = (global_times, global_length, indices, splits)

#     return tuple(all_times + all_coords + all_event_lengths + all_index_values +
#                  all_index_splits) + global_vals
