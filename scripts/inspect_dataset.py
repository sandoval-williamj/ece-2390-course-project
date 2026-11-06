from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from roadsign.dataset import DatasetError, dataset_statistics, load_gtsdb
from roadsign.visualize import save_ground_truth_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a local GTSDB dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data/gtsdb"))
    parser.add_argument(
        "--split", choices=("all", "train", "dev", "eval"), default="all"
    )
    parser.add_argument(
        "--save-example",
        type=Path,
        help="Optional path for an image with prohibitory ground-truth boxes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        dataset = load_gtsdb(args.data_dir, args.split)
    except DatasetError as error:
        print(f"Dataset error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(dataset_statistics(dataset), indent=2))
    if args.save_example:
        record = next(
            (item for item in dataset.records if item.target_annotations),
            dataset.records[0],
        )
        image = cv2.imread(str(record.path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Could not read {record.path}", file=sys.stderr)
            return 2
        save_ground_truth_image(
            image,
            record.target_annotations,
            record.filename,
            args.save_example,
        )
        print(f"Saved {args.save_example}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

