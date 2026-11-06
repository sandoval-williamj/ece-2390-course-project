import cv2
import numpy as np
import pytest

from roadsign.detector import DetectorConfig, detect_candidates, red_mask


def test_detects_red_square_and_scores_red_fraction() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (39, 39), (0, 0, 255), thickness=-1)

    detections = detect_candidates(image, "square.ppm")

    assert len(detections) == 1
    assert detections[0].filename == "square.ppm"
    assert detections[0].bbox == (20, 20, 40, 40)
    assert detections[0].confidence == pytest.approx(1.0)


def test_uses_both_red_hue_intervals() -> None:
    hsv = np.zeros((30, 60, 3), dtype=np.uint8)
    hsv[:, :30] = (2, 255, 255)
    hsv[:, 30:] = (178, 255, 255)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    mask = red_mask(image)

    assert np.all(mask == 255)


def test_rejects_non_square_candidate_by_aspect_ratio() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 20), (69, 29), (0, 0, 255), thickness=-1)

    assert detect_candidates(image, "wide.ppm") == []


def test_confidence_is_red_proportion_not_probability() -> None:
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (39, 39), (0, 0, 255), thickness=-1)
    cv2.rectangle(image, (70, 10), (99, 39), (0, 0, 255), thickness=4)

    detections = detect_candidates(
        image,
        "two.ppm",
        DetectorConfig(max_area_fraction=0.5),
    )

    assert len(detections) == 2
    assert detections[0].confidence > detections[1].confidence
    assert all(0.0 <= item.confidence <= 1.0 for item in detections)


def test_rejects_non_bgr_input() -> None:
    with pytest.raises(ValueError, match="BGR"):
        red_mask(np.zeros((20, 20), dtype=np.uint8))

