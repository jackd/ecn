import tensorflow as tf
import tensorflow_datasets as tfds

Lambda = tf.keras.layers.Lambda


def _complex(args):
    return tf.complex(*args)


def swish(x: tf.Tensor):
    return tf.where(tf.math.real(x) < -5, tf.zeros_like(x), x / (1 + tf.math.exp(-x)))


def modified_swish(x: tf.Tensor):
    # x * sigmoid(real(x))
    if x.dtype.is_complex:
        # return x / tf.cast(1 + tf.math.exp(-tf.math.real(x)), tf.complex64)
        real = tf.math.real(x)
        imag = tf.math.imag(x)
        denom = 1 + tf.math.exp(-real)
        return tf.complex(real / denom, imag / denom)
    else:
        return swish(x)


def unpack_complex(x):
    return tf.math.real(x), tf.math.imag(x)


def dropout(x, dropout_rate: float):
    if dropout_rate == 0:
        return x
    layer = tf.keras.layers.Dropout(dropout_rate)

    if x.dtype.is_complex:
        real = layer(Lambda(tf.math.real)(x))
        imag = layer(Lambda(tf.math.imag)(x))
        return Lambda(_complex)([real, imag])
    else:
        return layer(x)


class ComplexDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, input_shape):
        if self.built:
            return
        self.kernel_real = self.add_weight(
            "kernel", shape=(input_shape[-1], self.units), initializer="glorot_uniform"
        )
        self.kernel_imag = self.add_weight(
            "kernel", shape=(input_shape[-1], self.units), initializer="glorot_uniform"
        )
        # self.kernel = tf.complex(self.kernel_real, self.kernel_imag)
        # self.kernel = self.add_weight('kernel',
        #                               shape=(input_shape[-1], self.units),
        #                               initializer='ones',
        #                               dtype=tf.complex64)
        super().build(input_shape)

    @property
    def kernel(self):
        return tf.complex(self.kernel_real, self.kernel_imag)

    def call(self, x):
        # real = tf.math.real(x)
        # imag = tf.math.imag(x)
        # out_real = tf.matmul(real, self.kernel_real) - tf.matmul(
        #     imag, self.kernel_imag)
        # out_imag = tf.matmul(real, self.kernel_imag) + tf.matmul(
        #     imag, self.kernel_real)
        # return tf.cast(out_real, tf.complex64) + tf.cast(
        #     out_imag, tf.complex64) * tf.complex(0., 1.)
        # kernel = tf.complex(self.kernel_real, self.kernel_imag)
        return self.activation(tf.matmul(x, self.kernel))
        # return self.activation(tf.complex(out_real, out_imag))


inp = tf.keras.Input(shape=(28, 28, 1), dtype=tf.float32)
x = tf.keras.layers.Flatten()(inp)
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = dropout(x, 0.5)
real = tf.keras.layers.Dense(256)(x)
imag = tf.keras.layers.Dense(256)(x)

comp = Lambda(_complex)([real, imag])
# comp = Lambda(swish)(comp)
comp = ComplexDense(128, activation=None)(comp)
comp = Lambda(swish)(comp)
comp = ComplexDense(128, activation=swish)(comp)
# comp = Lambda(swish)(comp)
x, y = Lambda(unpack_complex)(comp)

features = Lambda(tf.concat, arguments=dict(axis=-1))([x, y])
features = tf.keras.layers.BatchNormalization()(features)
features = dropout(x, 0.5)

logits = tf.keras.layers.Dense(10)(features)

model = tf.keras.Model(inp, logits)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

#################################


def map_fn(x, labels):
    return tf.cast(x, tf.float32) / 255, labels


builder = tfds.image.MNIST()
builder.download_and_prepare()
splits = ("train", "test")
batch_size = 32
datasets = (builder.as_dataset(split=split, as_supervised=True) for split in splits)
train_ds, val_ds = (
    ds.shuffle(128).map(map_fn).repeat().batch(batch_size) for ds in datasets
)
train_steps, val_steps = (
    builder.info.splits[split].num_examples // batch_size for split in splits
)

model.fit(
    train_ds,
    epochs=10,
    steps_per_epoch=train_steps,
    validation_data=val_ds,
    validation_steps=val_steps,
)
