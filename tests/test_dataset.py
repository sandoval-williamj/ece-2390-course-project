from pathlib import Path

import cv2
import numpy as np
import pytest

from roadsign.dataset import (
    PROHIBITORY_CLASS_IDS,
    DatasetError,
    dataset_statistics,
    load_gtsdb,
    original_split,
)


def write_image(path: Path) -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_loads_valid_annotations_and_filters_targets(tmp_path: Path) -> None:
    write_image(tmp_path / "00000.ppm")
    write_image(tmp_path / "00001.ppm")
    (tmp_path / "gt.txt").write_text(
        "00000.ppm;10;5;19;14;1\n00001.ppm;0;0;9;9;17\n",
        encoding="utf-8",
    )

    dataset = load_gtsdb(tmp_path, split="train")

    assert dataset.records[0].annotations[0].bbox == (10, 5, 20, 15)
    assert len(dataset.target_annotations) == 1
    assert dataset.target_annotations[0].class_id in PROHIBITORY_CLASS_IDS
    assert dataset_statistics(dataset)["class_frequencies"] == {1: 1, 17: 1}


def test_rejects_out_of_bounds_box(tmp_path: Path) -> None:
    write_image(tmp_path / "00000.ppm")
    (tmp_path / "gt.txt").write_text(
        "00000.ppm;0;0;60;39;1\n", encoding="utf-8"
    )

    with pytest.raises(DatasetError, match="invalid box"):
        load_gtsdb(tmp_path)


def test_eval_requires_evaluation_annotations(tmp_path: Path) -> None:
    write_image(tmp_path / "00000.ppm")
    write_image(tmp_path / "00600.ppm")
    (tmp_path / "gt.txt").write_text(
        "00000.ppm;0;0;9;9;1\n", encoding="utf-8"
    )

    with pytest.raises(DatasetError, match="use --split dev"):
        load_gtsdb(tmp_path, split="eval")


def test_dev_is_last_twenty_percent_of_available_training_images(tmp_path: Path) -> None:
    for index in range(5):
        write_image(tmp_path / f"{index:05d}.ppm")
    (tmp_path / "gt.txt").write_text("", encoding="utf-8")

    dataset = load_gtsdb(tmp_path, split="dev")

    assert [record.filename for record in dataset.records] == ["00004.ppm"]


def test_official_archive_layout_ignores_class_crops(tmp_path: Path) -> None:
    (tmp_path / "gt.txt").write_text("", encoding="utf-8")
    full_dir = tmp_path / "FullIJCNN2013"
    crop_dir = full_dir / "01"
    crop_dir.mkdir(parents=True)
    write_image(full_dir / "00000.ppm")
    write_image(crop_dir / "00000.ppm")
    (full_dir / "gt.txt").write_text(
        "00000.ppm;10;5;19;14;1\n", encoding="utf-8"
    )

    dataset = load_gtsdb(tmp_path, split="train")

    assert dataset.annotation_path == full_dir / "gt.txt"
    assert [record.path for record in dataset.records] == [full_dir / "00000.ppm"]


@pytest.mark.parametrize(
    ("filename", "expected"),
    [("00000.ppm", "train"), ("00599.ppm", "train"), ("00600.ppm", "eval")],
)
def test_original_split(filename: str, expected: str) -> None:
    assert original_split(filename) == expected

