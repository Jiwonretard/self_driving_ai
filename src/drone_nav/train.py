from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf

from .contract import (
    CLASS_NAMES,
    IMAGE_SIZE,
    MAX_MODEL_BYTES,
    MODEL_FILENAME,
    validate_dataset_layout,
)
from .model import build_model

AUTOTUNE = tf.data.AUTOTUNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and quantize drone navigation CNN")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-model-kib", type=int, default=MAX_MODEL_BYTES // 1024)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.05)
    parser.add_argument(
        "--firmware-model",
        type=Path,
        default=Path("firmware/main/model") / MODEL_FILENAME,
    )
    return parser.parse_args()


def load_split(path: Path, batch_size: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="int",
        class_names=list(CLASS_NAMES),
        color_mode="rgb",
        batch_size=batch_size,
        image_size=IMAGE_SIZE,
        shuffle=shuffle,
        seed=seed,
    )
    return dataset.prefetch(AUTOTUNE)


def balanced_class_weights(train_dir: Path) -> dict[int, float]:
    counts = Counter(
        {
            index: sum(1 for path in (train_dir / label).iterdir() if path.is_file())
            for index, label in enumerate(CLASS_NAMES)
        }
    )
    if any(counts[index] == 0 for index in range(len(CLASS_NAMES))):
        raise ValueError(f"every training class must contain images: {dict(counts)}")
    total = sum(counts.values())
    class_count = len(CLASS_NAMES)
    return {index: total / (class_count * counts[index]) for index in range(class_count)}


def augment(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    # Horizontal flipping would swap left/right labels, so it is intentionally omitted.
    images = tf.image.random_brightness(images, max_delta=22.0)
    images = tf.image.random_contrast(images, lower=0.80, upper=1.20)
    images = tf.clip_by_value(images, 0.0, 255.0)
    return images, labels


def representative_data(dataset: tf.data.Dataset, max_batches: int = 100):
    count = 0
    for images, _ in dataset.unbatch().batch(1):
        yield [tf.cast(images, tf.float32)]
        count += 1
        if count >= max_batches:
            break


def convert_uint8(model: tf.keras.Model, calibration: tf.data.Dataset) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_data(calibration)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    return converter.convert()


def evaluate_tflite(model_content: bytes, dataset: tf.data.Dataset) -> float:
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]
    in_scale, in_zero = input_info["quantization"]
    correct = 0
    total = 0

    for images, labels in dataset.unbatch().batch(1):
        values = images.numpy()
        if input_info["dtype"] == np.uint8:
            values = np.clip(np.rint(values / in_scale + in_zero), 0, 255).astype(np.uint8)
        interpreter.set_tensor(input_info["index"], values)
        interpreter.invoke()
        prediction = int(np.argmax(interpreter.get_tensor(output_info["index"])[0]))
        correct += prediction == int(labels.numpy()[0])
        total += 1
    if total == 0:
        raise ValueError("test dataset is empty")
    return correct / total


def main() -> None:
    args = parse_args()
    errors = validate_dataset_layout(args.data)
    if errors:
        raise SystemExit("Invalid dataset:\n- " + "\n- ".join(errors))

    tf.keras.utils.set_random_seed(args.seed)
    train_raw = load_split(args.data / "train", args.batch_size, True, args.seed)
    val_ds = load_split(args.data / "val", args.batch_size, False, args.seed)
    test_ds = load_split(args.data / "test", args.batch_size, False, args.seed)
    train_ds = train_raw.map(augment, num_parallel_calls=AUTOTUNE)
    class_weights = balanced_class_weights(args.data / "train")
    print("Class weights:", class_weights)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "best.weights.h5"
    model = build_model()
    model.fit(
        train_ds,
        validation_data=val_ds,
        class_weight=class_weights,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=7, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint, monitor="val_loss", save_best_only=True, save_weights_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
            ),
        ],
    )

    _, float_accuracy = model.evaluate(test_ds, verbose=0)
    tflite_bytes = convert_uint8(model, train_raw)
    tflite_accuracy = evaluate_tflite(tflite_bytes, test_ds)
    model_bytes = len(tflite_bytes)

    if model_bytes > args.max_model_kib * 1024:
        raise SystemExit(
            f"Quantized model is too large: {model_bytes / 1024:.1f} KiB "
            f"> {args.max_model_kib} KiB"
        )
    accuracy_drop = float(float_accuracy) - tflite_accuracy
    if accuracy_drop > args.max_accuracy_drop:
        raise SystemExit(
            f"Quantization accuracy drop is too high: {accuracy_drop:.3f} "
            f"> {args.max_accuracy_drop:.3f}"
        )

    model_path = args.output / MODEL_FILENAME
    model_path.write_bytes(tflite_bytes)
    labels_path = args.output / "labels.txt"
    labels_path.write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    metrics = {
        "float_test_accuracy": float(float_accuracy),
        "uint8_test_accuracy": tflite_accuracy,
        "accuracy_drop": accuracy_drop,
        "model_bytes": model_bytes,
        "input_shape": [1, *IMAGE_SIZE, 3],
        "classes": list(CLASS_NAMES),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    args.firmware_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, args.firmware_model)
    print(json.dumps(metrics, indent=2))
    print(f"Firmware model: {args.firmware_model}")


if __name__ == "__main__":
    main()
