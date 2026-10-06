from __future__ import annotations

import argparse
import csv
import errno
import json
import os
import shutil
from collections import Counter
from pathlib import Path

CLASS_NAMES = ("forward", "left", "right", "stop")
PARTITION_NAMES = {"train": "train", "valid": "val", "validation": "val", "val": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PULP-Dronet v3 labels into image-classification folders"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dataset"))
    parser.add_argument("--yaw-threshold", type=float, default=0.2)
    parser.add_argument("--collision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy image bytes instead of creating space-saving hard links",
    )
    return parser.parse_args()


def command_for(yaw_rate: float, collision: float, yaw_threshold: float, collision_threshold: float) -> str:
    if collision >= collision_threshold:
        return "stop"
    if yaw_rate > yaw_threshold:
        return "left"
    if yaw_rate < -yaw_threshold:
        return "right"
    return "forward"


def safe_name(source_root: Path, acquisition_dir: Path, filename: str) -> str:
    relative = acquisition_dir.relative_to(source_root)
    prefix = "__".join(relative.parts)
    return f"{prefix}__{Path(filename).name}"


def place_file(source: Path, destination: Path, copy: bool) -> None:
    if destination.exists():
        if source.samefile(destination):
            return
        raise FileExistsError(f"destination already exists with different content: {destination}")
    if copy:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve()
    output_root = args.output.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"source directory not found: {source_root}")
    if args.yaw_threshold < 0:
        raise SystemExit("--yaw-threshold must be non-negative")
    if not 0.0 <= args.collision_threshold <= 1.0:
        raise SystemExit("--collision-threshold must be between 0 and 1")

    label_files = sorted(source_root.rglob("labels_partitioned.csv"))
    if not label_files:
        raise SystemExit(f"no labels_partitioned.csv files found under {source_root}")

    for partition in ("train", "val", "test"):
        for label in CLASS_NAMES:
            (output_root / partition / label).mkdir(parents=True, exist_ok=True)

    counts: Counter[tuple[str, str]] = Counter()
    missing_images: list[str] = []
    manifest_rows: list[dict[str, str]] = []
    skipped_partitions: Counter[str] = Counter()

    for label_file in label_files:
        acquisition_dir = label_file.parent
        with label_file.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                raw_partition = row["partition"].strip().lower()
                partition = PARTITION_NAMES.get(raw_partition)
                if partition is None:
                    if raw_partition in {"", "none"}:
                        skipped_partitions[raw_partition or "empty"] += 1
                        continue
                    raise ValueError(f"unknown partition {raw_partition!r} in {label_file}")

                filename = Path(row["filename"]).name
                image_path = acquisition_dir / "images" / filename
                if not image_path.is_file():
                    missing_images.append(str(image_path))
                    continue

                yaw_rate = float(row["label_yaw_rate"])
                collision = float(row["label_collision"])
                label = command_for(
                    yaw_rate,
                    collision,
                    args.yaw_threshold,
                    args.collision_threshold,
                )
                destination = output_root / partition / label / safe_name(
                    source_root, acquisition_dir, filename
                )
                place_file(image_path, destination, args.copy)
                counts[(partition, label)] += 1
                manifest_rows.append(
                    {
                        "partition": partition,
                        "label": label,
                        "yaw_rate": str(yaw_rate),
                        "collision": str(collision),
                        "source": str(image_path.relative_to(source_root)),
                        "destination": str(destination.relative_to(output_root)),
                    }
                )

    manifest_path = output_root / "pulp_dronet_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("partition", "label", "yaw_rate", "collision", "source", "destination"),
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "source": str(source_root),
        "output": str(output_root),
        "link_mode": "copy" if args.copy else "hardlink",
        "yaw_threshold": args.yaw_threshold,
        "collision_threshold": args.collision_threshold,
        "label_files": len(label_files),
        "missing_images": len(missing_images),
        "skipped_partitions": dict(skipped_partitions),
        "counts": {
            partition: {label: counts[(partition, label)] for label in CLASS_NAMES}
            for partition in ("train", "val", "test")
        },
    }
    (output_root / "pulp_dronet_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if missing_images:
        preview = "\n- ".join(missing_images[:10])
        raise SystemExit(
            f"conversion completed with {len(missing_images)} missing images; first entries:\n- {preview}"
        )


if __name__ == "__main__":
    main()
