import events_tfds.events.asl_dvs  # pylint: disable=unused-import
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm


def map_fn(events, labels):
    del labels
    return tf.shape(events["time"])[0]


batch_size = 128

dataset = tfds.load("asl_dvs", split="train", as_supervised=True).map(map_fn, -1)
total = len(dataset)
dataset = dataset.batch(batch_size)
out = np.zeros((total,), dtype=np.int64)
for i, example in enumerate(tqdm(dataset, total=total // batch_size + 1)):
    out[i * batch_size : (i + 1) * batch_size] = example.numpy()

print("median", np.median(out))
print("mean", np.mean(out))
print("max", np.max(out))
