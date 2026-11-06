# ECE 2390 Road Sign Detection

Classical computer-vision baselines for the University of Pittsburgh ECE 2390
course project by William Sandoval Casas and John Ferguson.

The November 13, 2026 milestone implements an HSV red-sign candidate detector
and preliminary evaluation infrastructure for the German Traffic Sign Detection
Benchmark (GTSDB). It does **not** implement HOG+LDA and does **not** claim to
reproduce the published 70.33% result.

## Current data status

The repository intentionally does not contain GTSDB data. The local file
`/Users/sandovalwilliamj/Downloads/TestIJCNN2013.zip` was inspected on November
5, 2026. It contains the official 300 test images (`00000.ppm` through
`00299.ppm`) and `ReadMe.txt`, but no ground-truth file. The test archive alone
can support inference, but it cannot support precision, recall, or AP.

For the milestone, obtain the original training package with its `gt.txt` and
place or extract its contents under:

```text
data/gtsdb/
├── 00000.ppm
├── ...
├── 00599.ppm
└── gt.txt
```

Nested extraction directories are also accepted. Do not alter the original
images or annotations. Do not mix the test ZIP into this directory: its names
restart at `00000.ppm` and collide with the training names. The loader refuses
duplicate filenames.

If released evaluation annotations later become available, add an explicit
layout adapter before treating the separate test archive as the `eval` split.
Until then, use the deterministic `dev` subset and label its results as
development results.

## Setup

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run pytest
```

The lockfile fixes the resolved environment. Runtime dependencies are NumPy,
OpenCV (headless), and Matplotlib; pytest is the only development dependency.

## Dataset inspection

```bash
uv run python scripts/inspect_dataset.py \
    --data-dir data/gtsdb \
    --split train \
    --save-example outputs/ground_truth.png
```

The loader validates image references and boxes, reports class frequencies, and
converts the annotation file's inclusive right/bottom coordinates to internal
half-open boxes `(x1, y1, x2, y2)`.

GTSDB's corrected prohibitory grouping is:

```text
0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16
```

The mapping comes from the [maintained German Traffic Sign Benchmarks repository
published by GTSDB author Sebastian Houben](https://github.com/houbensebastian/GermanTrafficSignBenchmarks#sign-categories).

## Run the baseline

When only the annotated training package is available, evaluate the final 20%
of its images in filename order. On the complete training set this is images
`00480.ppm` through `00599.ppm`:

```bash
uv run python scripts/run_baseline.py \
    --data-dir data/gtsdb \
    --split dev \
    --iou 0.5 0.6 \
    --save-all-detections \
    --output-dir outputs
```

The command creates:

```text
outputs/
├── predictions.csv
├── metrics.json
├── sample_detection.png
├── sample_detections/       # present with --save-all-detections
├── precision_recall.png
└── milestone_figure.png
```

The detector converts BGR to HSV, thresholds two red hue ranges, performs a
3-by-3 morphological opening and closing, extracts external contours, and
filters boxes by size and aspect ratio. A detection's confidence is the
fraction of red-mask pixels inside its box. This is a heuristic ranking score,
not a calibrated probability. Ground truth is never passed to the detector.

The default operating threshold reported in `metrics.json` is confidence 0.10.
It affects the documented precision/recall operating point and displayed sample
boxes, not the AP ranking.

## Evaluation convention

Predictions are sorted by decreasing confidence and greedily matched one-to-one
to unmatched target boxes in the same image. IoU uses half-open coordinates.
Equal-confidence predictions are accumulated into one precision-recall point.

AP is the all-points interpolated area under the precision-recall envelope,
equivalent to the post-2010 PASCAL VOC convention. `--iou` accepts any threshold
from zero to one; the milestone reports 0.50 and 0.60.

These are conventional AP calculations. They are not automatically equivalent
to the official GTSDB evaluation procedure described by [Houben et al.
(2013)](https://doi.org/10.1109/IJCNN.2013.6706807). Every metrics file is
labeled accordingly.

## Tests

```bash
uv run pytest
```

Tests cover annotation validation, target filtering, split selection, both red
hue bands, candidate filtering, confidence ranking, IoU, empty predictions,
incorrect boxes, duplicate detections, negative images, multiple signs, tied
scores, AP, configurable IoU thresholds, and the five-file end-to-end output.
The integration test uses a synthetic image with a known result; it is not a
reported project experiment.

## Milestone report

`reports/milestone.tex` contains clearly marked result placeholders. Run the
real development experiment first, transfer the values from
`outputs/metrics.json`, and then compile from the `reports` directory:

```bash
cd reports
pdflatex milestone.tex
```

## Limitations and next steps

- Acquire the 600-image training archive and verified `gt.txt`, then run and
  inspect the development experiment.
- Tune HSV/morphology parameters only on training/development data and record
  all changes.
- Acquire released evaluation annotations or implement the official benchmark
  submission/scoring format before claiming evaluation-set results.
- Document common HSV failure modes: red background clutter, faded signs,
  shadows, occlusion, and merged contours.
- After the milestone, reproduce HOG+LDA and the official GTSDB scoring protocol
  before comparing with the paper's 70.33% prohibitory-sign baseline.


