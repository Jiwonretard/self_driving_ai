from pathlib import Path

from drone_nav.contract import CLASS_NAMES, validate_dataset_layout


def test_class_order_is_firmware_contract() -> None:
    assert CLASS_NAMES == ("forward", "left", "right", "stop")


def test_valid_layout(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        for label in CLASS_NAMES:
            (tmp_path / split / label).mkdir(parents=True)
    assert validate_dataset_layout(tmp_path) == []


def test_layout_rejects_missing_and_extra_classes(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        for label in CLASS_NAMES:
            if not (split == "val" and label == "stop"):
                (tmp_path / split / label).mkdir(parents=True)
    (tmp_path / "train" / "up").mkdir()
    errors = validate_dataset_layout(tmp_path)
    assert any("val: missing classes: stop" in error for error in errors)
    assert any("train: unexpected classes: up" in error for error in errors)

