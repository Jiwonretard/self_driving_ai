from __future__ import annotations

import argparse
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TFLite image classifier via MediaPipe Tasks")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()

    labels = [line.strip() for line in args.labels.read_text().splitlines() if line.strip()]
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    options = mp.tasks.vision.ImageClassifierOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        max_results=len(labels),
        score_threshold=0.0,
    )
    with mp.tasks.vision.ImageClassifier.create_from_options(options) as classifier:
        result = classifier.classify(mp_image)

    categories = result.classifications[0].categories
    for category in categories:
        name = category.category_name or labels[category.index]
        print(f"{name:>8}: {category.score:.4f}")


if __name__ == "__main__":
    main()

