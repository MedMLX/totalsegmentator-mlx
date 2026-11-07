"""MedMLX discovery and Jobs runner owned by the native TotalSegmentator package."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import cast

from medmlx_core.errors import InvalidInputError

from totalsegmentator_mlx.labels import LABELS
from totalsegmentator_mlx.models import ARITHMETIC, SOURCE_PROFILE, Mode

MODEL = {
    "id": "totalsegmentator",
    "label": "TotalSegmentator",
    "task": "segmentation",
    "run": "totalsegmentator_mlx.medmlx:run",
    # Measured MLX peak for standard mode on a 768x768x101 abdominal CT; larger inputs need more.
    "peak_memory_gb": 5.0,
    "summary": "CT segmentation (standard, fast or coarse; optional ROI/body crop) on Metal.",
    "labels": {str(number): name for number, name in LABELS.items()},
}


def _folder(name: str, value: object) -> Path:
    """Return a path argument as an absolute path with ``~`` expanded, without resolving links."""
    if not isinstance(value, (str, PathLike)):
        raise InvalidInputError(f"{name} must be a folder path")
    return Path(cast(str | PathLike[str], value)).expanduser().absolute()


def run(
    *,
    weights_path: str | None = None,
    image_path: str,
    output_dir: str,
    prompt: Sequence[str] | None = None,
    mode: str = "standard",
    roi_subset: Sequence[str] | None = None,
    robust_crop: bool = False,
    body_seg: bool = False,
    overwrite: bool = False,
) -> dict[str, object]:
    """Write mask.nii.gz with global CT labels and original input geometry.

    The mask is uint8 with global TotalSegmentator v2 ``total`` label IDs 1-117
    (``details["labels"]`` lists them) and background 0. See ``segment`` for the CT workflow.

    Args:
        weights_path: Optional root of native dataset folders; omitted weights
            download automatically from MedMLX/totalsegmentator-mlx. Folders ``291``-``295`` for
            ``standard``, ``297`` for ``fast``, ``298`` for ``coarse``; crops additionally
            need ``298`` (ROI), ``297`` (robust ROI) or ``300`` (body). Each folder contains
            ``plans.json``, ``dataset.json`` and ``fold_0/checkpoint_final/`` with both
            ``state.json`` and ``weights.safetensors``. A model folder can be passed
            directly when only that dataset is required.
        image_path: Finite 3D NIfTI-1 or NIfTI-2 CT in Hounsfield units, with millimeter
            or unspecified spatial units.
        output_dir: Destination for ``mask.nii.gz``; created if missing.
        prompt: Canonical structure names to retain; other labels become background.
            Cannot be combined with ``roi_subset``.
        mode: ``standard`` (default), ``fast`` or ``coarse``.
        roi_subset: Canonical structure names to localize and crop to; cannot be combined
            with ``prompt`` or ``body_seg``.
        robust_crop: Locate ``roi_subset`` with the 3 mm model instead of the 6 mm one;
            requires ``roi_subset``.
        body_seg: Crop to the Dataset300 body mask plus 3 mm.
        overwrite: Replace an existing mask when true; an input alias is still refused.

    Returns:
        ``{"outputs": {"mask": absolute POSIX path}, "details": {...}}``, where ``details``
        records the backend, mode, source profile, arithmetic, folds and label table.
        Replacing an existing mask requires ``overwrite``. Every argument is checked before
        weights download. Errors are those of ``segment`` (``InvalidInputError``,
        ``AssetNotReadyError``, ``ModelExecutionError``).
    """
    from medmlx_core.weights import weights_dir, weights_revision

    from totalsegmentator_mlx.pipeline import check_request, segment_request

    mask = _folder("output_dir", output_dir) / "mask.nii.gz"
    local = None if weights_path is None else _folder("weights_path", weights_path)
    request = check_request(
        image_path,
        output_path=mask,
        prompt=prompt,
        mode=cast(Mode, mode),
        roi_subset=roi_subset,
        robust_crop=robust_crop,
        body_seg=body_seg,
        overwrite=overwrite,
    )
    folder = weights_dir("totalsegmentator") if local is None else local
    segment_request(request, model_dir=folder)
    return {
        "outputs": {"mask": mask.as_posix()},
        "details": {
            "weights_revision": weights_revision(folder),
            "backend": "metal",
            "mode": mode,
            "source_profile": SOURCE_PROFILE,
            "arithmetic": ARITHMETIC,
            "folds": [0],
            "labels": [{"name": name, "value": number} for number, name in LABELS.items()],
        },
    }
