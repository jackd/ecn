import numpy as np
import tensorflow as tf
from tqdm import tqdm

max_size = 10 ** 7


def gen():
    while True:
        example_size = np.random.randint(max_size)
        yield np.random.uniform(size=example_size)


dataset = tf.data.Dataset.from_generator(gen, tf.float64, (None,))

dataset = dataset.cache("/home/guest/dom-ws/data/tmp")

for example in tqdm(dataset):
    pass
