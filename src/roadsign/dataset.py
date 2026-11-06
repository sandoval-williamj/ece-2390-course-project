from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2

BBox = tuple[int, int, int, int]

PROHIBITORY_CLASS_IDS = frozenset({0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16})
PROHIBITORY_CLASS_SOURCE = (
    "https://github.com/houbensebastian/GermanTrafficSignBenchmarks"
    "#sign-categories"
)


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Annotation:
    filename: str
    bbox: BBox
    class_id: int


@dataclass(frozen=True)
class ImageRecord:
    filename: str
    path: Path
    original_split: str
    annotations: tuple[Annotation, ...]

    @property
    def target_annotations(self) -> tuple[Annotation, ...]:
        return tuple(
            item for item in self.annotations if item.class_id in PROHIBITORY_CLASS_IDS
        )


@dataclass(frozen=True)
class GTSDBDataset:
    data_dir: Path
    annotation_path: Path
    split: str
    records: tuple[ImageRecord, ...]

    @property
    def target_annotations(self) -> tuple[Annotation, ...]:
        return tuple(
            annotation
            for record in self.records
            for annotation in record.target_annotations
        )


def original_split(filename: str) -> str:
    stem = Path(filename).stem
    if not stem.isdigit():
        raise DatasetError(f"GTSDB image filename is not numeric: {filename}")
    index = int(stem)
    if 0 <= index < 600:
        return "train"
    if 600 <= index < 900:
        return "eval"
    raise DatasetError(f"GTSDB image index is outside 00000-00899: {filename}")


def discover_images(data_dir: Path | str) -> dict[str, Path]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise DatasetError(f"Dataset directory does not exist: {data_dir}")

    paths = sorted({*data_dir.glob("*.ppm"), *data_dir.glob("*.PPM")})
    if not paths:
        raise DatasetError(
            f"No GTSDB .ppm images found in {data_dir}. "
            "For a scorable development run, place the original 600 training "
            "images and their gt.txt there."
        )

    images: dict[str, Path] = {}
    for path in paths:
        if path.name in images:
            raise DatasetError(f"Duplicate image filename found: {path.name}")
        original_split(path.name)
        images[path.name] = path
    return images


def find_annotation_file(data_dir: Path | str) -> Path:
    data_dir = Path(data_dir)
    candidates = sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file() and path.name.lower() == "gt.txt"
    )
    if not candidates:
        raise DatasetError(f"No gt.txt annotation file found in {data_dir}")
    if len(candidates) == 1:
        return candidates[0]

    adjacent_counts = {
        path: len(list(path.parent.glob("*.ppm")))
        + len(list(path.parent.glob("*.PPM")))
        for path in candidates
    }
    largest_count = max(adjacent_counts.values())
    best = [path for path, count in adjacent_counts.items() if count == largest_count]
    if largest_count > 0 and len(best) == 1:
        return best[0]

    joined = ", ".join(str(path) for path in candidates)
    raise DatasetError(f"Could not choose between multiple gt.txt files: {joined}")


def parse_annotations(
    annotation_path: Path | str, images: dict[str, Path]
) -> tuple[Annotation, ...]:
    annotation_path = Path(annotation_path)
    annotations: list[Annotation] = []
    image_sizes: dict[str, tuple[int, int]] = {}

    with annotation_path.open(encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = [field.strip() for field in line.split(";")]
            if fields[0].lower() == "filename":
                continue
            if len(fields) != 6:
                raise DatasetError(
                    f"{annotation_path}:{line_number}: expected 6 fields, got {len(fields)}"
                )

            filename = Path(fields[0]).name
            if filename not in images:
                raise DatasetError(
                    f"{annotation_path}:{line_number}: image not found: {filename}"
                )
            try:
                x1, y1, x2_inclusive, y2_inclusive, class_id = map(int, fields[1:])
            except ValueError as error:
                raise DatasetError(
                    f"{annotation_path}:{line_number}: coordinates and class must be integers"
                ) from error

            if filename not in image_sizes:
                image = cv2.imread(str(images[filename]), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise DatasetError(f"OpenCV could not read image: {images[filename]}")
                height, width = image.shape[:2]
                image_sizes[filename] = (width, height)
            width, height = image_sizes[filename]

            valid = (
                0 <= x1 <= x2_inclusive < width
                and 0 <= y1 <= y2_inclusive < height
            )
            if not valid:
                raise DatasetError(
                    f"{annotation_path}:{line_number}: invalid box "
                    f"({x1}, {y1}, {x2_inclusive}, {y2_inclusive}) for "
                    f"{width}x{height} image"
                )

            annotations.append(
                Annotation(
                    filename=filename,
                    bbox=(x1, y1, x2_inclusive + 1, y2_inclusive + 1),
                    class_id=class_id,
                )
            )
    return tuple(annotations)


def _select_images(images: dict[str, Path], split: str) -> list[Path]:
    if split == "all":
        return sorted(images.values())

    train_images = sorted(
        path for name, path in images.items() if original_split(name) == "train"
    )
    if split == "train":
        return train_images
    if split == "eval":
        return sorted(
            path for name, path in images.items() if original_split(name) == "eval"
        )
    if split == "dev":
        if not train_images:
            return []
        dev_count = max(1, (len(train_images) + 4) // 5)
        return train_images[-dev_count:]
    raise DatasetError(f"Unknown split: {split}")


def load_gtsdb(data_dir: Path | str, split: str = "all") -> GTSDBDataset:
    data_dir = Path(data_dir)
    annotation_path = find_annotation_file(data_dir)
    images = discover_images(annotation_path.parent)
    annotations = parse_annotations(annotation_path, images)

    if split == "eval" and not any(
        original_split(annotation.filename) == "eval" for annotation in annotations
    ):
        raise DatasetError(
            "The annotation file contains no evaluation-image annotations. "
            "Official evaluation cannot be calculated; use --split dev instead."
        )

    selected_paths = _select_images(images, split)
    if not selected_paths:
        raise DatasetError(f"No images are available for split '{split}'")

    by_filename: dict[str, list[Annotation]] = {name: [] for name in images}
    for annotation in annotations:
        by_filename[annotation.filename].append(annotation)

    records = tuple(
        ImageRecord(
            filename=path.name,
            path=path,
            original_split=original_split(path.name),
            annotations=tuple(by_filename[path.name]),
        )
        for path in selected_paths
    )
    return GTSDBDataset(
        data_dir=data_dir,
        annotation_path=annotation_path,
        split=split,
        records=records,
    )


def dataset_statistics(dataset: GTSDBDataset) -> dict[str, Any]:
    annotations = [
        annotation for record in dataset.records for annotation in record.annotations
    ]
    frequencies = Counter(annotation.class_id for annotation in annotations)
    return {
        "split": dataset.split,
        "image_count": len(dataset.records),
        "annotated_image_count": sum(bool(record.annotations) for record in dataset.records),
        "all_annotation_count": len(annotations),
        "target_annotation_count": len(dataset.target_annotations),
        "images_with_target_count": sum(
            bool(record.target_annotations) for record in dataset.records
        ),
        "class_frequencies": dict(sorted(frequencies.items())),
    }


def group_annotations(
    annotations: Iterable[Annotation],
) -> dict[str, list[Annotation]]:
    grouped: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        grouped.setdefault(annotation.filename, []).append(annotation)
    return grouped

