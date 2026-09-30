from __future__ import annotations

import tensorflow as tf

from .contract import CHANNELS, CLASS_NAMES, IMAGE_SIZE


def build_model() -> tf.keras.Model:
    """Build an MCU-friendly CNN using only TFLite Micro-supported operators."""
    inputs = tf.keras.Input((*IMAGE_SIZE, CHANNELS), name="rgb", dtype=tf.float32)
    # Keep raw 0..255 pixels at the model boundary. This avoids a separate Mul
    # preprocessing op and leaves a five-op TFLite Micro graph.
    x = tf.keras.layers.Conv2D(
        8, 3, strides=2, padding="same", use_bias=False, name="stem_conv"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="stem_relu6")(x)

    for index, filters in enumerate((12, 20, 28), start=1):
        x = tf.keras.layers.DepthwiseConv2D(
            3,
            strides=2,
            padding="same",
            use_bias=False,
            name=f"block{index}_dw",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"block{index}_dw_bn")(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name=f"block{index}_dw_relu6")(x)
        x = tf.keras.layers.Conv2D(
            filters, 1, padding="same", use_bias=False, name=f"block{index}_pw"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"block{index}_pw_bn")(x)
        x = tf.keras.layers.ReLU(max_value=6.0, name=f"block{index}_pw_relu6")(x)

    x = tf.keras.layers.GlobalAveragePooling2D(name="global_mean")(x)
    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES), activation="softmax", name="commands"
    )(x)
    model = tf.keras.Model(inputs, outputs, name="drone_nav_tiny")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
