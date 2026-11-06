from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from roadsign.dataset import Annotation
from roadsign.detector import Detection
from roadsign.evaluate import EvaluationResult


def _show_image(
    axes: Axes,
    image_bgr: NDArray[np.uint8],
    ground_truth: Iterable[Annotation],
    detections: Iterable[Detection],
    title: str,
) -> None:
    axes.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    axes.set_title(title)
    axes.axis("off")

    for annotation in ground_truth:
        x1, y1, x2, y2 = annotation.bbox
        axes.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="limegreen",
                linewidth=2,
            )
        )
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        axes.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="darkorange",
                linewidth=2,
            )
        )
        axes.text(
            x1,
            max(0, y1 - 3),
            f"{detection.confidence:.2f}",
            color="darkorange",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 1},
        )

    axes.legend(
        handles=[
            Patch(facecolor="none", edgecolor="limegreen", label="Ground truth"),
            Patch(facecolor="none", edgecolor="darkorange", label="Prediction"),
        ],
        loc="lower right",
        framealpha=0.9,
    )


def _show_precision_recall(
    axes: Axes, results: Sequence[EvaluationResult]
) -> None:
    for result in results:
        label = f"IoU {result.iou_threshold:.2f}, AP {result.average_precision:.3f}"
        if result.recall.size:
            recall = np.r_[0.0, result.recall]
            precision = np.r_[1.0, result.precision]
            axes.step(recall, precision, where="post", linewidth=2, label=label)
        else:
            axes.plot([], [], label=label)

    axes.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.05),
        xlabel="Recall",
        ylabel="Precision",
        title="Prohibitory-sign precision–recall",
    )
    axes.grid(alpha=0.25)
    axes.legend(loc="lower left")


def save_ground_truth_image(
    image_bgr: NDArray[np.uint8],
    annotations: Iterable[Annotation],
    filename: str,
    output_path: Path | str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(10, 6))
    _show_image(axes, image_bgr, annotations, [], filename)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_sample_detection(
    image_bgr: NDArray[np.uint8],
    ground_truth: Iterable[Annotation],
    detections: Iterable[Detection],
    filename: str,
    output_path: Path | str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(10, 6))
    _show_image(axes, image_bgr, ground_truth, detections, filename)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_precision_recall(
    results: Sequence[EvaluationResult], output_path: Path | str
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(6, 5))
    _show_precision_recall(axes, results)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_milestone_figure(
    image_bgr: NDArray[np.uint8],
    ground_truth: Iterable[Annotation],
    detections: Iterable[Detection],
    filename: str,
    results: Sequence[EvaluationResult],
    output_path: Path | str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    _show_image(axes[0], image_bgr, ground_truth, detections, filename)
    _show_precision_recall(axes[1], results)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

