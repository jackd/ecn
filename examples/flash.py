import matplotlib.pyplot as plt
import tensorflow as tf
from ecn.problems import sources
from ecn import components as comp
from ecn import multi_graph as mg

Lambda = tf.keras.layers.Lambda

source = sources.nmnist_source()
grid_shape = source.meta['grid_shape']
num_frames = 4
batch_size = 8


def build_fn(features, labels, weights=None):
    times = features['time']
    coords = features['coords']
    polarity = features['polarity']

    grid = comp.Grid(grid_shape)
    stream = comp.SpatialStream(grid, times, coords, bucket_sizes=False)
    with mg.pre_batch_context():
        t_start = stream.cached_times[0]
        t_end = stream.cached_times[-1] + 1
    t_start = mg.batch(t_start)
    t_end = mg.batch(t_end)

    polarity = stream.prepare_model_inputs(polarity)
    batch_size = Lambda(lambda x: x.nrows())(polarity)
    polarity = Lambda(lambda x: tf.one_hot(tf.cast(x.values, tf.uint8), 2))(
        polarity)

    out = stream.flash(polarity,
                       t_start,
                       t_end,
                       num_frames,
                       batch_size=batch_size)
    return out, (), ()


built = mg.build_multi_graph(build_fn,
                             source.example_spec,
                             batch_size=batch_size)

dataset = source.get_dataset('train')
dataset = dataset.map(built.pre_cache_map).map(
    built.pre_batch_map).batch(batch_size).map(built.post_batch_map)

model = built.trained_model
for features, labels, weights in dataset:
    out = model(features).numpy()

    for b in range(batch_size):
        fig, ax = plt.subplots(2, 4)
        for c in range(2):
            for j in range(4):
                ax[c, j].imshow(out[b, j, :, :, c].T)
        plt.show()
