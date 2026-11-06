from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

from roadsign.dataset import DatasetError, GTSDBDataset, ImageRecord, load_gtsdb
from roadsign.detector import Detection, detect_candidates
from roadsign.evaluate import EvaluationResult, evaluate_detections, metrics_at_confidence
from roadsign.visualize import (
    save_milestone_figure,
    save_precision_recall,
    save_sample_detection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GTSDB HSV baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/gtsdb"))
    parser.add_argument(
        "--split", choices=("train", "dev", "eval"), default="dev"
    )
    parser.add_argument(
        "--iou",
        type=float,
        nargs="+",
        default=[0.5, 0.6],
        help="One or more IoU thresholds",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--save-all-detections",
        action="store_true",
        help="Save one annotated PNG for every image in the evaluated split",
    )
    return parser.parse_args()


def run_detector(dataset: GTSDBDataset) -> list[Detection]:
    detections: list[Detection] = []
    for record in dataset.records:
        image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if image is None:
            raise DatasetError(f"OpenCV could not read image: {record.path}")
        detections.extend(detect_candidates(image, record.filename))
    return detections


def write_predictions(path: Path, detections: list[Detection]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "x1", "y1", "x2", "y2", "confidence"])
        for detection in detections:
            writer.writerow(
                [
                    detection.filename,
                    *detection.bbox,
                    f"{detection.confidence:.8f}",
                ]
            )


def split_description(dataset: GTSDBDataset) -> str:
    if dataset.split == "dev":
        return (
            "Development subset: final 20% of available original training images "
            "in filename order; not an official GTSDB test result."
        )
    if dataset.split == "eval":
        return (
            "Original GTSDB evaluation images with locally supplied annotations; "
            "scored with preliminary conventional AP, not official benchmark scoring."
        )
    return "Original GTSDB training images; not an official test result."


def build_metrics(
    dataset: GTSDBDataset,
    detections: list[Detection],
    results: list[EvaluationResult],
    confidence_threshold: float,
) -> dict[str, object]:
    records = dataset.records
    per_iou: dict[str, object] = {}
    for result in results:
        operating = metrics_at_confidence(result, confidence_threshold)
        per_iou[f"{result.iou_threshold:.2f}"] = {
            "average_precision": result.average_precision,
            "operating_point": operating,
        }

    return {
        "evaluation_label": "preliminary conventional AP; not official GTSDB scoring",
        "evaluated_split": dataset.split,
        "split_description": split_description(dataset),
        "first_image": records[0].filename,
        "last_image": records[-1].filename,
        "image_count": len(records),
        "ground_truth_count": len(dataset.target_annotations),
        "detection_count": len(detections),
        "iou_thresholds": [result.iou_threshold for result in results],
        "confidence_score_definition": (
            "Fraction of red-mask pixels inside a candidate box; heuristic ranking "
            "score, not a calibrated probability."
        ),
        "operating_confidence_threshold": confidence_threshold,
        "ap_convention": (
            "All-points interpolated area under the precision-recall envelope; "
            "detections with tied confidence are accumulated as one PR point."
        ),
        "box_convention": "(x1, y1, x2, y2), with x2 and y2 exclusive",
        "official_gtsdb_scoring": False,
        "results": per_iou,
    }


def choose_sample(
    dataset: GTSDBDataset,
    detections: list[Detection],
) -> tuple[ImageRecord, list[Detection]]:
    by_image: dict[str, list[Detection]] = {}
    for detection in detections:
        by_image.setdefault(detection.filename, []).append(detection)

    for record in dataset.records:
        if record.target_annotations and by_image.get(record.filename):
            return record, by_image[record.filename]
    for record in dataset.records:
        if record.target_annotations:
            return record, by_image.get(record.filename, [])
    for record in dataset.records:
        if by_image.get(record.filename):
            return record, by_image[record.filename]
    return dataset.records[0], []


def save_all_detection_images(
    dataset: GTSDBDataset,
    detections: list[Detection],
    confidence_threshold: float,
    output_dir: Path,
) -> None:
    by_image: dict[str, list[Detection]] = {}
    for detection in detections:
        if detection.confidence >= confidence_threshold:
            by_image.setdefault(detection.filename, []).append(detection)

    output_dir.mkdir(parents=True, exist_ok=True)
    for record in dataset.records:
        image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if image is None:
            raise DatasetError(f"OpenCV could not read image: {record.path}")
        output_path = output_dir / f"{Path(record.filename).stem}.png"
        save_sample_detection(
            image,
            record.target_annotations,
            by_image.get(record.filename, []),
            record.filename,
            output_path,
        )


def main() -> int:
    args = parse_args()
    if any(not 0.0 <= threshold <= 1.0 for threshold in args.iou):
        print("IoU thresholds must be between 0 and 1", file=sys.stderr)
        return 2
    if not 0.0 <= args.confidence_threshold <= 1.0:
        print("Confidence threshold must be between 0 and 1", file=sys.stderr)
        return 2

    try:
        dataset = load_gtsdb(args.data_dir, args.split)
        if not dataset.target_annotations:
            raise DatasetError(
                f"Split '{args.split}' has no prohibitory ground truth; refusing to "
                "report AP."
            )
        detections = run_detector(dataset)
    except DatasetError as error:
        print(f"Dataset error: {error}", file=sys.stderr)
        return 2

    results = [
        evaluate_detections(detections, dataset.target_annotations, threshold)
        for threshold in dict.fromkeys(args.iou)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(args.output_dir / "predictions.csv", detections)
    metrics = build_metrics(
        dataset, detections, results, args.confidence_threshold
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    sample, sample_detections = choose_sample(dataset, detections)
    image = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Could not read {sample.path}", file=sys.stderr)
        return 2
    visible_detections = [
        item
        for item in sample_detections
        if item.confidence >= args.confidence_threshold
    ]
    save_sample_detection(
        image,
        sample.target_annotations,
        visible_detections,
        sample.filename,
        args.output_dir / "sample_detection.png",
    )
    if args.save_all_detections:
        save_all_detection_images(
            dataset,
            detections,
            args.confidence_threshold,
            args.output_dir / "sample_detections",
        )
    save_precision_recall(results, args.output_dir / "precision_recall.png")
    save_milestone_figure(
        image,
        sample.target_annotations,
        visible_detections,
        sample.filename,
        results,
        args.output_dir / "milestone_figure.png",
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

