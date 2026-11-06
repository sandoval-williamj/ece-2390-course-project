import json
import sys
from pathlib import Path

import cv2
import numpy as np

from scripts.run_baseline import main


def test_baseline_writes_all_outputs_for_synthetic_data(
    tmp_path: Path, monkeypatch
) -> None:
    for index in range(5):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        if index == 4:
            cv2.rectangle(image, (10, 10), (29, 29), (0, 0, 255), thickness=-1)
        assert cv2.imwrite(str(tmp_path / f"{index:05d}.ppm"), image)
    (tmp_path / "gt.txt").write_text(
        "00004.ppm;10;10;29;29;1\n", encoding="utf-8"
    )
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_baseline.py",
            "--data-dir",
            str(tmp_path),
            "--split",
            "dev",
            "--iou",
            "0.5",
            "0.6",
            "--output-dir",
            str(output_dir),
            "--save-all-detections",
        ],
    )

    assert main() == 0

    expected = {
        "predictions.csv",
        "metrics.json",
        "sample_detection.png",
        "precision_recall.png",
        "milestone_figure.png",
        "sample_detections",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    assert (output_dir / "sample_detections" / "00004.png").is_file()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["image_count"] == 1
    assert metrics["ground_truth_count"] == 1
    assert metrics["results"]["0.50"]["average_precision"] == 1.0
    assert metrics["results"]["0.60"]["average_precision"] == 1.0

