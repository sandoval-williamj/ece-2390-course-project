from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from roadsign.dataset import BBox


@dataclass(frozen=True)
class DetectorConfig:
    hue_low_max: int = 10
    hue_high_min: int = 170
    saturation_min: int = 70
    value_min: int = 50
    kernel_size: int = 3
    min_size: int = 10
    min_area: int = 100
    max_area_fraction: float = 0.20
    min_aspect_ratio: float = 0.65
    max_aspect_ratio: float = 1.35


@dataclass(frozen=True)
class Detection:
    filename: str
    bbox: BBox
    confidence: float


def red_mask(
    image_bgr: NDArray[np.uint8], config: DetectorConfig = DetectorConfig()
) -> NDArray[np.uint8]:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with shape (height, width, 3)")

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    low_red = cv2.inRange(
        hsv,
        (0, config.saturation_min, config.value_min),
        (config.hue_low_max, 255, 255),
    )
    high_red = cv2.inRange(
        hsv,
        (config.hue_high_min, config.saturation_min, config.value_min),
        (179, 255, 255),
    )
    mask = cv2.bitwise_or(low_red, high_red)

    kernel = np.ones((config.kernel_size, config.kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _valid_candidate(
    width: int,
    height: int,
    image_area: int,
    config: DetectorConfig,
) -> bool:
    area = width * height
    aspect_ratio = width / height
    return (
        width >= config.min_size
        and height >= config.min_size
        and area >= config.min_area
        and area <= image_area * config.max_area_fraction
        and config.min_aspect_ratio <= aspect_ratio <= config.max_aspect_ratio
    )


def detect_candidates(
    image_bgr: NDArray[np.uint8],
    filename: str,
    config: DetectorConfig = DetectorConfig(),
) -> list[Detection]:
    mask = red_mask(image_bgr, config)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    detections: list[Detection] = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if not _valid_candidate(width, height, image_area, config):
            continue
        red_fraction = float(np.count_nonzero(mask[y : y + height, x : x + width]))
        red_fraction /= width * height
        detections.append(
            Detection(
                filename=filename,
                bbox=(x, y, x + width, y + height),
                confidence=min(1.0, max(0.0, red_fraction)),
            )
        )

    return sorted(
        detections,
        key=lambda item: (-item.confidence, item.bbox, item.filename),
    )

