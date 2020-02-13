import functools
import timeit
import tensorflow as tf
import numpy as np
import numba


@numba.njit()
def wrapped_sum(x, z):
    return np.sum(x) + z


with tf.Graph().as_default() as graph:
    x = tf.range(10, dtype=tf.int64)
    y = 2 * x
    out = tf.numpy_function(functools.partial(wrapped_sum, z=4), (x,), tf.int64)
    out.set_shape(())

    tf.keras.layers.Dense(2)(tf.reshape(out, (1, 1)))

graph_def = graph.as_graph_def()


def subgraph(graph_def, input_op_names, output_names):

    @tf.function
    def f(*inputs):
        return tf.graph_util.import_graph_def(graph_def,
                                              input_map=dict(
                                                  zip(input_op_names, inputs)),
                                              return_elements=output_names)

    return f


f = subgraph(graph_def, [x.op.name], [y.name])
print(f(tf.range(3, dtype=tf.int64)))
# with tf.Graph().as_default() as imported_graph:
#     input_map = {x.op.name: tf.constant([3, 4], dtype=tf.int64)}
#     return_elements = [y.name, out.name]
#     rebuilt = tf.graph_util.import_graph_def(graph_def, input_map,
#                                              return_elements)
#     print(rebuilt)

#     with tf.compat.v1.Session() as sess:

#         def single_run():
#             return sess.run(rebuilt)

#         print('First: ', timeit.timeit(single_run, number=1))
#         print('Second: ', timeit.timeit(single_run, number=1))
#         print(single_run())
