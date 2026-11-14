"""Lazy command-line entry point for CT prediction and label discovery."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from totalsegmentator_mlx.labels import LABELS, MODEL_PARTS, QUALITY_MODELS

if TYPE_CHECKING:
    from totalsegmentator_mlx.models import Mode


class Arguments(argparse.Namespace):
    """Typed namespace for the ``predict`` and ``labels`` subcommands.

    Only the fields of the chosen subcommand are set by argparse.
    """

    command: str
    image_path: Path
    output_dir: Path
    weights_path: Path | None
    mode: str
    model_part: str | None
    classes: list[str] | None
    roi_subset: list[str] | None
    robust_crop: bool
    body_seg: bool
    force_split: bool
    overwrite: bool
    probabilities: bool


def _parser() -> argparse.ArgumentParser:
    """Build the prediction and label-discovery command parser."""
    parser = argparse.ArgumentParser(description="Native TotalSegmentator v2 CT inference on Metal")
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser("predict", help="Segment a NIfTI CT on its original grid")
    predict.add_argument("-i", "--image", dest="image_path", type=Path, required=True)
    predict.add_argument("-o", "--output-dir", type=Path, required=True)
    predict.add_argument(
        "--weights", dest="weights_path", type=Path, help="Weights folder; default downloads"
    )
    predict.add_argument(
        "--mode",
        choices=["standard", *QUALITY_MODELS],
        default="standard",
        help="standard: five 1.5 mm parts; fast: 3 mm model; coarse: 6 mm model (upstream fastest)",
    )
    predict.add_argument("--model-part", choices=list(MODEL_PARTS))
    predict.add_argument(
        "--classes", nargs="+", help="Keep these canonical structure names after inference"
    )
    predict.add_argument(
        "--roi-subset",
        nargs="+",
        help="Localize these structures at 6 mm and crop; standard mode runs only their parts",
    )
    predict.add_argument(
        "--robust-crop", action="store_true", help="Locate --roi-subset with the 3 mm model"
    )
    predict.add_argument(
        "--body-seg", action="store_true", help="Crop to the body with the Dataset300 model"
    )
    predict.add_argument("--force-split", action="store_true")
    predict.add_argument("--overwrite", action="store_true")
    predict.add_argument(
        "--probabilities", action="store_true", help="Also write probabilities.npz and .pkl"
    )
    subparsers.add_parser("labels", help="Print released global label IDs as JSON")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run a subcommand; ``labels`` prints the global ID-to-name table as JSON.

    Exceptions that are ``OSError``, ``ValueError``, ``RuntimeError`` or ``ImportError``
    (which includes the ``medmlx_core`` errors) are reported on stderr with exit status 2.
    ``predict`` reports stages on stderr and prints the output paths as JSON on success.
    """
    parser = _parser()
    args = parser.parse_args(argv, namespace=Arguments())
    if args.command == "labels":
        print(json.dumps(LABELS, indent=2))
        return
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    try:
        from totalsegmentator_mlx.medmlx import segment

        result = segment(
            image_path=args.image_path,
            output_dir=args.output_dir,
            weights_path=args.weights_path,
            classes=args.classes,
            mode=cast("Mode", args.mode),
            model_part=args.model_part,
            roi_subset=args.roi_subset,
            robust_crop=args.robust_crop,
            body_seg=args.body_seg,
            force_split=args.force_split,
            probabilities=args.probabilities,
            overwrite=args.overwrite,
        )
        print(json.dumps(result["outputs"], indent=2))
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(2, f"{parser.prog}: {error}\n")
