import matplotlib.pyplot as plt
import tensorflow as tf

import multi_graph as mg
from ecn import components as comp
from ecn.problems import sources

Lambda = tf.keras.layers.Lambda

source = sources.nmnist_source()
grid_shape = source.meta["grid_shape"]
num_frames = 4
batch_size = 8


def build_fn(features, labels, weights=None):
    del labels, weights
    times = features["time"]
    coords = features["coords"]
    polarity = features["polarity"]

    grid = comp.Grid(grid_shape)
    stream = comp.SpatialStream(grid, times, coords, bucket_sizes=False)
    with mg.pre_batch_context():
        t_start = stream.cached_times[0]
        t_end = stream.cached_times[-1] + 1
    t_start = mg.batch(t_start)
    t_end = mg.batch(t_end)

    polarity = stream.prepare_model_inputs(polarity)
    batch_size = Lambda(lambda x: x.nrows())(polarity)
    polarity = Lambda(lambda x: tf.one_hot(tf.cast(x.values, tf.uint8), 2))(polarity)

    out = stream.mean_voxelize(
        polarity, t_start, t_end, num_frames, batch_size=batch_size
    )
    return out, (), ()


built = mg.build_multi_graph(build_fn, source.element_spec, batch_size=batch_size)

dataset = source.get_dataset("train")
dataset = (
    dataset.map(built.pre_cache_map)
    .map(built.pre_batch_map)
    .batch(batch_size)
    .map(built.post_batch_map)
)

model = built.trained_model
C = 2
J = 4
for features, labels, weights in dataset:
    out = model(features).numpy()

    for b in range(batch_size):
        fig, ax = plt.subplots(C, J)
        for c in range(C):
            for j in range(J):
                ax[c, j].imshow(out[b, j, :, :, c].T)
        plt.show()
