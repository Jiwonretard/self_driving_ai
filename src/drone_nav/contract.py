from __future__ import annotations

from pathlib import Path

CLASS_NAMES = ("forward", "left", "right", "stop")
IMAGE_SIZE = (96, 96)
CHANNELS = 3
MODEL_FILENAME = "drone_nav_int8.tflite"
MAX_MODEL_BYTES = 180 * 1024


def validate_dataset_layout(root: Path) -> list[str]:
    """Return human-readable dataset layout errors without importing TensorFlow."""
    errors: list[str] = []
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            errors.append(f"missing split directory: {split_dir}")
            continue
        actual = {p.name for p in split_dir.iterdir() if p.is_dir()}
        expected = set(CLASS_NAMES)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            errors.append(f"{split}: missing classes: {', '.join(missing)}")
        if extra:
            errors.append(f"{split}: unexpected classes: {', '.join(extra)}")
    return errors

