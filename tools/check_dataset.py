from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image

from drone_nav.contract import CLASS_NAMES, validate_dataset_layout

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    errors = validate_dataset_layout(args.data)
    if errors:
        raise SystemExit("Invalid dataset:\n- " + "\n- ".join(errors))

    broken: list[Path] = []
    counts: Counter[tuple[str, str]] = Counter()
    for split in ("train", "val", "test"):
        for label in CLASS_NAMES:
            for path in (args.data / split / label).iterdir():
                if path.suffix.lower() not in EXTENSIONS:
                    continue
                try:
                    with Image.open(path) as image:
                        image.verify()
                    counts[(split, label)] += 1
                except Exception:
                    broken.append(path)

    for split in ("train", "val", "test"):
        print(split + ": " + ", ".join(f"{c}={counts[(split, c)]}" for c in CLASS_NAMES))
    if broken:
        raise SystemExit("Unreadable images:\n- " + "\n- ".join(map(str, broken)))
    if any(counts[(split, label)] == 0 for split in ("train", "val", "test") for label in CLASS_NAMES):
        raise SystemExit("Every split/class directory must contain at least one image")


if __name__ == "__main__":
    main()

