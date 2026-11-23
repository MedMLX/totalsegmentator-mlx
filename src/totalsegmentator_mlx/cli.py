"""Command-line entry point: ``predict`` is generated from ``run``, ``labels`` is hand-written."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from importlib.metadata import version

from medmlx_core.cli import add_options, call, execute

from totalsegmentator_mlx.labels import LABELS
from totalsegmentator_mlx.medmlx import run

_PROG = "totalsegmentator-mlx"


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``predict`` or ``labels`` and return the exit status.

    ``predict`` has one option per ``run`` argument, logs stages on stderr and prints the
    run result as JSON; ``labels`` prints the global label ID-to-name table as JSON.
    """
    parser = argparse.ArgumentParser(
        prog=_PROG, description="Native TotalSegmentator v2 CT inference on Metal"
    )
    parser.add_argument("--version", action="version", version=version(_PROG))
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser("predict", help="Segment a NIfTI CT on its original grid")
    add_options(predict, run)
    subparsers.add_parser("labels", help="Print released global label IDs as JSON")
    arguments = parser.parse_args(argv)
    if arguments.command == "labels":
        return execute(_PROG, lambda: LABELS)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    return execute(_PROG, lambda: call(run, arguments))
