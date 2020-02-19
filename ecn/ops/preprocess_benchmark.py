raise NotImplementedError('deprecated')
# import functools
# import tensorflow as tf
# from ecn.benchmark_utils import benchmark
# from ecn.benchmark_utils import run_benchmarks
# import ecn.ops.preprocess as pp
# import ecn.ops.preprocess_tf as pp_tf
# from ecn.pipelines import scnn

# from events_tfds.events.nmnist import NMNIST

# kwargs = dict(stride=2,
#               decay_time=10000,
#               event_duration=10000 * 8,
#               spatial_buffer_size=32,
#               num_layers=3,
#               reset_potential=-2.)

# def map_fn(events, labels):
#     return pp.preprocess_network_trimmed(events['time'], events['coords'],
#                                          events['polarity'], **kwargs), labels

# def map_fn_tf(events, labels):
#     return pp.preprocess_network_trimmed(events['time'], events['coords'],
#                                          events['polarity'], **kwargs), labels

# with tf.Graph().as_default():
#     ds = NMNIST().as_dataset(split='train', as_supervised=True)
#     mapped = ds.map(map_fn)
#     mapped_tf = ds.map(map_fn_tf)
#     scnn_mapped = ds.map(functools.partial(scnn.pre_map, **kwargs))

#     out0 = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
#     out1 = tf.compat.v1.data.make_one_shot_iterator(mapped).get_next()
#     out2 = tf.compat.v1.data.make_one_shot_iterator(scnn_mapped).get_next()
#     out3 = tf.compat.v1.data.make_one_shot_iterator(mapped_tf).get_next()

#     with tf.compat.v1.Session() as sess:

#         def f0():
#             sess.run(out0)

#         def f1():
#             sess.run(out1)

#         def f2():
#             sess.run(out2)

#         def f3():
#             sess.run(out3)

#         benchmark(name='base')(f0)
#         benchmark(name='mapped')(f1)
#         benchmark(name='scnn')(f2)
#         benchmark(name='mapped_tf')(f3)
#         run_benchmarks(10, 100)
