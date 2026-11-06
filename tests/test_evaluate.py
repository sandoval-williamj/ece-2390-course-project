import numpy as np
import pytest

from roadsign.dataset import Annotation
from roadsign.detector import Detection
from roadsign.evaluate import (
    evaluate_detections,
    intersection_over_union,
    metrics_at_confidence,
)


def annotation(filename: str, bbox: tuple[int, int, int, int]) -> Annotation:
    return Annotation(filename=filename, bbox=bbox, class_id=1)


def detection(
    filename: str,
    bbox: tuple[int, int, int, int],
    confidence: float,
) -> Detection:
    return Detection(filename=filename, bbox=bbox, confidence=confidence)


def test_iou_for_identical_partial_and_disjoint_boxes() -> None:
    assert intersection_over_union((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert intersection_over_union((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(
        1 / 3
    )
    assert intersection_over_union((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0


def test_perfect_predictions_have_ap_one() -> None:
    ground_truth = [annotation("a.ppm", (0, 0, 10, 10))]
    predictions = [detection("a.ppm", (0, 0, 10, 10), 0.8)]

    result = evaluate_detections(predictions, ground_truth)

    assert result.average_precision == 1.0
    assert result.precision.tolist() == [1.0]
    assert result.recall.tolist() == [1.0]


def test_no_predictions_is_safe_and_has_zero_ap() -> None:
    result = evaluate_detections([], [annotation("a.ppm", (0, 0, 10, 10))])

    assert result.average_precision == 0.0
    assert result.precision.size == 0
    assert metrics_at_confidence(result, 0.1)["recall"] == 0.0


def test_incorrect_box_is_false_positive() -> None:
    result = evaluate_detections(
        [detection("a.ppm", (20, 20, 30, 30), 0.9)],
        [annotation("a.ppm", (0, 0, 10, 10))],
    )

    assert result.matches.true_positive.tolist() == [0]
    assert result.matches.false_positive.tolist() == [1]
    assert result.average_precision == 0.0


def test_duplicate_detection_matches_ground_truth_once() -> None:
    box = (0, 0, 10, 10)
    result = evaluate_detections(
        [detection("a.ppm", box, 0.9), detection("a.ppm", box, 0.8)],
        [annotation("a.ppm", box)],
    )

    assert result.matches.true_positive.tolist() == [1, 0]
    assert result.matches.false_positive.tolist() == [0, 1]


def test_detection_in_image_without_target_is_false_positive() -> None:
    result = evaluate_detections(
        [detection("empty.ppm", (0, 0, 10, 10), 0.7)],
        [annotation("sign.ppm", (0, 0, 10, 10))],
    )

    assert result.matches.false_positive.tolist() == [1]
    assert result.recall.tolist() == [0.0]


def test_multiple_signs_are_matched_one_to_one() -> None:
    ground_truth = [
        annotation("a.ppm", (0, 0, 10, 10)),
        annotation("a.ppm", (20, 20, 30, 30)),
    ]
    predictions = [
        detection("a.ppm", (20, 20, 30, 30), 0.9),
        detection("a.ppm", (0, 0, 10, 10), 0.8),
    ]

    result = evaluate_detections(predictions, ground_truth)

    assert result.matches.true_positive.tolist() == [1, 1]
    assert result.average_precision == 1.0


def test_tied_confidences_form_one_pr_point() -> None:
    ground_truth = [annotation("a.ppm", (0, 0, 10, 10))]
    predictions = [
        detection("a.ppm", (0, 0, 10, 10), 0.5),
        detection("empty.ppm", (0, 0, 10, 10), 0.5),
    ]

    result = evaluate_detections(predictions, ground_truth)

    assert result.thresholds.tolist() == [0.5]
    assert result.precision.tolist() == [0.5]
    assert result.recall.tolist() == [1.0]
    assert result.average_precision == 0.5


def test_confidence_ranking_affects_ap() -> None:
    ground_truth = [annotation("a.ppm", (0, 0, 10, 10))]
    predictions = [
        detection("a.ppm", (0, 0, 10, 10), 0.4),
        detection("empty.ppm", (0, 0, 10, 10), 0.9),
    ]

    result = evaluate_detections(predictions, ground_truth)

    assert result.average_precision == 0.5


def test_iou_threshold_is_configurable() -> None:
    ground_truth = [annotation("a.ppm", (0, 0, 10, 10))]
    predictions = [detection("a.ppm", (0, 0, 10, 5), 0.8)]

    at_50 = evaluate_detections(predictions, ground_truth, iou_threshold=0.5)
    at_60 = evaluate_detections(predictions, ground_truth, iou_threshold=0.6)

    assert at_50.matches.true_positive.tolist() == [1]
    assert at_60.matches.false_positive.tolist() == [1]


def test_empty_ground_truth_with_prediction_is_safe() -> None:
    result = evaluate_detections(
        [detection("empty.ppm", (0, 0, 10, 10), 0.9)], []
    )

    assert result.average_precision == 0.0
    assert np.all(result.recall == 0.0)

