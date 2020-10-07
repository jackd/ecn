import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ecn.problems import sources

source = sources.asl_dvs_source()

total = source.epoch_length("train")


def map_fn(events, labels):
    del labels
    return tf.shape(events["time"])[0]


batch_size = 128

dataset = source.get_dataset("train").map(map_fn, -1).batch(batch_size)
out = np.zeros((total,), dtype=np.int64)
for i, example in enumerate(tqdm(dataset, total=total // batch_size + 1)):
    out[i * batch_size : (i + 1) * batch_size] = example.numpy()

print("median", np.median(out))
print("mean", np.mean(out))
print("max", np.max(out))
