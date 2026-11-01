"""Admit the released CT model identities before loading model weights."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from medmlx_core.errors import AssetNotReadyError, InvalidInputError

from totalsegmentator_mlx.labels import (
    BODY_LABELS,
    BODY_MODEL,
    LABELS,
    MODEL_PARTS,
    PART_LABELS,
    QUALITY_MODELS,
)

# nnunet-mlx compatibility profile for the released TotalSegmentator v2 CT models, and the
# arithmetic mode that runs MLX kernels in the pinned PyTorch CPU float32 operation order.
SOURCE_PROFILE = "totalsegmentator-v2-ct"
ARITHMETIC = "torch-order-fp32"

type Mode = Literal["standard", "fast", "coarse"]
MODES: tuple[Mode, ...] = ("standard", "fast", "coarse")
# Isotropic model grid in mm, as upstream resamples before each model.
TASK_SPACING: dict[int, float] = {
    **dict.fromkeys(PART_LABELS, 1.5),
    QUALITY_MODELS["fast"]: 3.0,
    QUALITY_MODELS["coarse"]: 6.0,
    BODY_MODEL: 6.0,
}


def model_labels(task_id: int) -> dict[int, str]:
    """Return a model's local label map; only the five parts need global remapping."""
    if task_id in PART_LABELS:
        return PART_LABELS[task_id]
    return BODY_LABELS if task_id == BODY_MODEL else LABELS


def select_parts(model_part: str | None) -> tuple[int, ...]:
    """Return the dataset IDs of the standard parts, in merge order, or just ``model_part``'s."""
    if model_part is None:
        return tuple(MODEL_PARTS.values())
    if model_part not in MODEL_PARTS:
        raise InvalidInputError(f"model_part must be one of {', '.join(MODEL_PARTS)}")
    return (MODEL_PARTS[model_part],)


def select_models(
    mode: str, model_part: str | None, roi_subset: Mapping[int, str] | None
) -> tuple[int, ...]:
    """Choose the dataset IDs to run for a mode, in merge order.

    ``standard`` gives the five 1.5 mm parts (291-295), or the one named ``model_part``;
    ``fast`` and ``coarse`` give the single 3 mm (297) or 6 mm (298) model. An ``roi_subset``
    (global ID to name) keeps only the standard parts that contain its labels.

    Raises:
        InvalidInputError: Unknown mode or part, or ``model_part`` with a non-standard mode.
    """
    if mode not in MODES:
        raise InvalidInputError(f"mode must be one of {', '.join(MODES)}")
    if mode != "standard":
        if model_part is not None:
            raise InvalidInputError("model_part requires mode='standard'")
        return (QUALITY_MODELS[mode],)
    parts = select_parts(model_part)
    if roi_subset is None:
        return parts
    names = set(roi_subset.values())
    return tuple(task for task in parts if names & set(PART_LABELS[task].values()))


def select_labels(prompt: list[str] | tuple[str, ...] | None) -> dict[int, str]:
    """Return global ID -> name for the canonical names in ``prompt`` (all 117 if ``None``).

    Raises ``InvalidInputError`` for an empty prompt or a name outside the released labels.
    """
    if prompt is None:
        return dict(LABELS)
    if not prompt or set(prompt) - set(LABELS.values()):
        raise InvalidInputError("Choose canonical TotalSegmentator organ names")
    return {number: name for number, name in LABELS.items() if name in prompt}


def model_folder(root: Path, task_id: int) -> Path:
    """Return ``root`` itself if it holds ``plans.json``, else the ``<root>/<task_id>`` folder."""
    return root if (root / "plans.json").is_file() else root / str(task_id)


def read_object(path: Path) -> dict[str, object]:
    """Read a JSON object; reject a missing file or non-object with ``AssetNotReadyError``.

    Malformed JSON and filesystem read errors propagate from the underlying reader.
    """
    if not path.is_file():
        raise AssetNotReadyError(f"Required model metadata is missing: {path}")
    payload: object = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise AssetNotReadyError(f"Expected a JSON object: {path}")
    return cast(dict[str, object], payload)


def validate_metadata(folder: Path, task_id: int) -> None:
    """Check Dataset ``task_id`` name, one CT channel and exact labels in metadata.

    This metadata check does not authenticate checkpoint identity or origin.

    Raises:
        AssetNotReadyError: ``plans.json`` or ``dataset.json`` is missing or does not match.
        ValueError: Metadata is not valid JSON.
    """
    plans = read_object(folder / "plans.json")
    dataset = read_object(folder / "dataset.json")
    labels = model_labels(task_id)
    expected = {"background": 0, **{name: label for label, name in labels.items()}}
    if (
        str(plans.get("dataset_name", "")).split("_")[0] != f"Dataset{task_id}"
        or dataset.get("labels") != expected
        or dataset.get("channel_names") != {"0": "CT"}
    ):
        raise AssetNotReadyError(f"Model must match Dataset{task_id}, its CT channel and labels")


def validate_models(root: Path, task_ids: tuple[int, ...]) -> None:
    """Check metadata and the converted fold-0 checkpoint for every model before any input is read.

    Raises:
        AssetNotReadyError: A model folder is missing or mismatched, or its checkpoint lacks
            ``state.json`` or ``weights.safetensors``.
    """
    for task_id in task_ids:
        folder = model_folder(root, task_id)
        validate_metadata(folder, task_id)
        checkpoint = folder / "fold_0" / "checkpoint_final"
        for name in ("state.json", "weights.safetensors"):
            path = checkpoint / name
            if not path.is_file():
                raise AssetNotReadyError(f"Converted fold-0 checkpoint file is missing: {path}")
