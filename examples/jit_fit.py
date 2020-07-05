import tensorflow as tf
import tensorflow_datasets as tfds

layers = tf.keras.layers


@tf.function(experimental_compile=True)
def f(inp):
    x = tf.cast(inp, tf.float32) / 255
    x = layers.Conv2D(16, 2, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 2, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return layers.Dense(10, activation=None)(x)


inp = tf.keras.Input(shape=(28, 28, 1), dtype=tf.uint8)
logits = f(inp)
model = tf.keras.Model(inp, logits)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

builder = tfds.builder("mnist")
model.fit(
    builder.as_dataset(split="train", as_supervised=True),
    train_steps=builder.info.splits["train"].num_examples,
)
