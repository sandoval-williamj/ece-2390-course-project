from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from roadsign.dataset import Annotation, BBox, group_annotations
from roadsign.detector import Detection


@dataclass(frozen=True)
class MatchResult:
    detections: tuple[Detection, ...]
    confidence: NDArray[np.float64]
    true_positive: NDArray[np.int64]
    false_positive: NDArray[np.int64]


@dataclass(frozen=True)
class EvaluationResult:
    matches: MatchResult
    thresholds: NDArray[np.float64]
    precision: NDArray[np.float64]
    recall: NDArray[np.float64]
    average_precision: float
    ground_truth_count: int
    iou_threshold: float


def intersection_over_union(box_a: BBox, box_b: BBox) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])

    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_detections(
    detections: Iterable[Detection],
    ground_truth: Iterable[Annotation],
    iou_threshold: float,
) -> MatchResult:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be between 0 and 1")

    ranked = tuple(
        sorted(
            detections,
            key=lambda item: (-item.confidence, item.filename, item.bbox),
        )
    )
    ground_truth_by_image = group_annotations(ground_truth)
    matched: dict[str, set[int]] = {
        filename: set() for filename in ground_truth_by_image
    }
    true_positive = np.zeros(len(ranked), dtype=np.int64)
    false_positive = np.zeros(len(ranked), dtype=np.int64)

    for detection_index, detection in enumerate(ranked):
        candidates = ground_truth_by_image.get(detection.filename, [])
        available = [
            (index, annotation)
            for index, annotation in enumerate(candidates)
            if index not in matched.get(detection.filename, set())
        ]
        if not available:
            false_positive[detection_index] = 1
            continue

        best_index, best_annotation = max(
            available,
            key=lambda item: intersection_over_union(
                detection.bbox, item[1].bbox
            ),
        )
        overlap = intersection_over_union(detection.bbox, best_annotation.bbox)
        if overlap >= iou_threshold:
            true_positive[detection_index] = 1
            matched.setdefault(detection.filename, set()).add(best_index)
        else:
            false_positive[detection_index] = 1

    return MatchResult(
        detections=ranked,
        confidence=np.asarray(
            [detection.confidence for detection in ranked], dtype=np.float64
        ),
        true_positive=true_positive,
        false_positive=false_positive,
    )


def precision_recall_curve(
    matches: MatchResult, ground_truth_count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if not matches.detections:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty

    cumulative_tp = np.cumsum(matches.true_positive)
    cumulative_fp = np.cumsum(matches.false_positive)
    group_ends = np.flatnonzero(
        np.r_[matches.confidence[:-1] != matches.confidence[1:], True]
    )
    tp = cumulative_tp[group_ends]
    fp = cumulative_fp[group_ends]
    precision = tp / np.maximum(tp + fp, 1)
    if ground_truth_count:
        recall = tp / ground_truth_count
    else:
        recall = np.zeros_like(precision, dtype=np.float64)
    return precision.astype(float), recall.astype(float), matches.confidence[group_ends]


def average_precision(
    precision: NDArray[np.float64],
    recall: NDArray[np.float64],
    ground_truth_count: int,
) -> float:
    if ground_truth_count == 0 or precision.size == 0:
        return 0.0

    recall_points = np.concatenate(([0.0], recall, [1.0]))
    precision_points = np.concatenate(([0.0], precision, [0.0]))
    precision_points = np.maximum.accumulate(precision_points[::-1])[::-1]
    changes = np.flatnonzero(recall_points[1:] != recall_points[:-1])
    area = np.sum(
        (recall_points[changes + 1] - recall_points[changes])
        * precision_points[changes + 1]
    )
    return float(area)


def evaluate_detections(
    detections: Iterable[Detection],
    ground_truth: Iterable[Annotation],
    iou_threshold: float = 0.5,
) -> EvaluationResult:
    ground_truth = tuple(ground_truth)
    matches = match_detections(detections, ground_truth, iou_threshold)
    precision, recall, thresholds = precision_recall_curve(matches, len(ground_truth))
    ap = average_precision(precision, recall, len(ground_truth))
    return EvaluationResult(
        matches=matches,
        thresholds=thresholds,
        precision=precision,
        recall=recall,
        average_precision=ap,
        ground_truth_count=len(ground_truth),
        iou_threshold=iou_threshold,
    )


def metrics_at_confidence(
    result: EvaluationResult, confidence_threshold: float
) -> dict[str, float | int]:
    selected = result.matches.confidence >= confidence_threshold
    true_positive = int(result.matches.true_positive[selected].sum())
    false_positive = int(result.matches.false_positive[selected].sum())
    detection_count = true_positive + false_positive
    precision = true_positive / detection_count if detection_count else 0.0
    recall = (
        true_positive / result.ground_truth_count
        if result.ground_truth_count
        else 0.0
    )
    return {
        "confidence_threshold": confidence_threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "precision": precision,
        "recall": recall,
    }

