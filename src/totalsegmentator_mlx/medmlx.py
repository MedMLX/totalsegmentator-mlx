"""TotalSegmentator's public operation, its reusable model and the MedMLX declaration.

Importing this module does not import MLX or the nnunet-mlx engine.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from medmlx_core.files import PathLike, input_file, output_targets, publish
from medmlx_core.outputs import Artifact, artifact
from medmlx_core.weights import model_folder, weights_revision

from totalsegmentator_mlx.labels import LABELS
from totalsegmentator_mlx.models import ARITHMETIC, SOURCE_PROFILE, Mode

if TYPE_CHECKING:
    from nnunet_mlx import NiftiPrediction

    from totalsegmentator_mlx.engine import Networks
    from totalsegmentator_mlx.pipeline import Request

MODEL = {
    "id": "totalsegmentator",
    "label": "TotalSegmentator",
    "task": "segmentation",
    "run": "totalsegmentator_mlx.medmlx:segment",
    "cohort": "totalsegmentator_mlx.medmlx:load_model",
    # Measured MLX peak for standard mode on a 768x768x101 abdominal CT; larger inputs need more.
    "peak_memory_gb": 5.0,
    "summary": "CT segmentation (standard, fast or coarse; optional ROI/body crop) on Metal.",
    "labels": {str(number): name for number, name in LABELS.items()},
}

# Every role is reserved on each run, so a run never leaves another run's files beside its own.
OUTPUT_FILES = {
    "mask": "mask.nii.gz",
    "probabilities": "probabilities.npz",
    "probability_properties": "probabilities.pkl",
}

type RunResult = dict[str, object]


def segment(
    *,
    image_path: PathLike,
    output_dir: PathLike,
    weights_path: PathLike | None = None,
    classes: Sequence[str] | None = None,
    mode: Mode = "standard",
    model_part: str | None = None,
    roi_subset: Sequence[str] | None = None,
    robust_crop: bool = False,
    body_seg: bool = False,
    force_split: bool = False,
    probabilities: bool = False,
    overwrite: bool = False,
) -> RunResult:
    """Segment a CT and write ``mask.nii.gz`` with global labels on the input grid.

    The mask is uint8 with global TotalSegmentator v2 ``total`` label IDs 1-117
    (``details["labels"]`` lists them) and background 0, with the input's shape,
    affine and header plus the upstream label table. Each network is released after
    its part; use :func:`load_model` to keep them loaded across CTs.

    Args:
        image_path: Finite 3D NIfTI-1 or NIfTI-2 CT (``.nii`` or ``.nii.gz``) in
            Hounsfield units, with millimeter or unspecified spatial units.
        output_dir: Folder for the outputs; created if missing.
        weights_path: Root of the native dataset folders as published; omitted weights
            download from MedMLX/totalsegmentator-mlx. Folders ``291``-``295`` serve
            ``standard``, ``297`` ``fast`` and ``298`` ``coarse``; crops also need ``298``
            (ROI), ``297`` (robust ROI) or ``300`` (body). Each holds ``plans.json``,
            ``dataset.json`` and ``fold_0/checkpoint_final/`` with ``state.json`` and
            ``weights.safetensors``. A single dataset folder serves a run that needs only it.
        classes: Canonical structure names to keep; other labels become background
            after inference. Cannot be combined with ``roi_subset``.
        mode: ``standard`` (five 1.5 mm parts), ``fast`` (3 mm, upstream ``--fast``) or
            ``coarse`` (6 mm, upstream ``--fastest``).
        model_part: One standard part (``organs``, ``vertebrae``, ``cardiac``,
            ``muscles``, ``ribs``); standard mode only.
        roi_subset: Canonical structure names to localize at 6 mm and crop to, plus
            20 mm; standard mode then runs only their parts. Excludes ``classes``,
            ``model_part`` and ``body_seg``.
        robust_crop: Locate ``roi_subset`` with the 3 mm model; requires ``roi_subset``.
        body_seg: Crop to the Dataset300 body mask plus 3 mm.
        force_split: Split into three overlapping z slabs even for smaller volumes.
        probabilities: Also write ``probabilities.npz`` (float32 local-label
            probabilities on the model reader's grid) and ``probabilities.pkl`` (its
            nnU-Net geometry and crop properties). Needs one model and no crop.
        overwrite: Replace existing outputs; an output may never be the input or a
            model file.

    Returns:
        ``{"outputs": {role: path}, "artifacts": {...}, "details": {...}}`` with roles
        ``mask`` and, with ``probabilities``, ``probabilities`` and
        ``probability_properties``. The mask artifact is a ``label_map`` on
        ``image_path``'s grid whose labels are the classes the run can write (all
        117, or the ``classes`` or ``roi_subset`` structures).

    Raises:
        InvalidInputError: Invalid options, CT or output paths; checked before weights
            download.
        AssetNotReadyError: Weights are missing, incomplete or mismatched.
        ModelExecutionError: Metal is unavailable or a restored grid or label map is invalid.
    """
    from totalsegmentator_mlx.engine import Networks

    request, targets = _case(
        image_path, output_dir, classes=classes, mode=mode, model_part=model_part,
        roi_subset=roi_subset, robust_crop=robust_crop, body_seg=body_seg,
        force_split=force_split, probabilities=probabilities, overwrite=overwrite,
    )
    folder = model_folder("totalsegmentator", weights_path)
    networks = Networks(retain=False)
    try:
        return _segment(request, targets, folder, weights_revision(folder), networks)
    finally:
        networks.close()


class TotalSegmentatorModel:
    """TotalSegmentator with its weights folder fixed and networks kept loaded until closed.

    Each network loads on first use and stays resident, so later CTs skip loading;
    standard mode with crops keeps up to eight networks. :meth:`segment` has the
    :func:`segment` contract without ``weights_path``. CTs run one at a time.
    """

    def __init__(self, *, weights_path: PathLike | None = None) -> None:
        """Resolve (downloading if omitted) the weights folder; networks load lazily."""
        from totalsegmentator_mlx.engine import Networks

        self._folder = model_folder("totalsegmentator", weights_path)
        self._revision = weights_revision(self._folder)
        self._networks: Networks | None = Networks(retain=True)

    def segment(
        self,
        *,
        image_path: PathLike,
        output_dir: PathLike,
        classes: Sequence[str] | None = None,
        mode: Mode = "standard",
        model_part: str | None = None,
        roi_subset: Sequence[str] | None = None,
        robust_crop: bool = False,
        body_seg: bool = False,
        force_split: bool = False,
        probabilities: bool = False,
        overwrite: bool = False,
    ) -> RunResult:
        """Segment one CT with the loaded networks; see :func:`segment`.

        Raises:
            RuntimeError: The model is closed.
        """
        networks = self._networks
        if networks is None:
            raise RuntimeError("TotalSegmentator model is closed")
        request, targets = _case(
            image_path, output_dir, classes=classes, mode=mode, model_part=model_part,
            roi_subset=roi_subset, robust_crop=robust_crop, body_seg=body_seg,
            force_split=force_split, probabilities=probabilities, overwrite=overwrite,
        )
        return _segment(request, targets, self._folder, self._revision, networks)

    def close(self) -> None:
        """Release the networks; later :meth:`segment` calls raise ``RuntimeError``."""
        if self._networks is not None:
            self._networks.close()
            self._networks = None

    def __enter__(self) -> TotalSegmentatorModel:
        """Return this model for a ``with`` block that closes it on exit."""
        return self

    def __exit__(self, *args: object) -> None:
        """Close the model."""
        self.close()


def load_model(*, weights_path: PathLike | None = None) -> TotalSegmentatorModel:
    """Load TotalSegmentator once to segment several CTs, or for a MedMLX cohort.

    ``weights_path`` has the meaning it has in :func:`segment`. Close the returned
    model, or use it as a context manager, to release its networks.
    """
    return TotalSegmentatorModel(weights_path=weights_path)


def _case(
    image_path: PathLike,
    output_dir: PathLike,
    *,
    classes: Sequence[str] | None,
    mode: Mode,
    model_part: str | None,
    roi_subset: Sequence[str] | None,
    robust_crop: bool,
    body_seg: bool,
    force_split: bool,
    probabilities: bool,
    overwrite: bool,
) -> tuple[Request, dict[str, Path]]:
    """Check one CT's arguments and output targets without weights or MLX."""
    from totalsegmentator_mlx.pipeline import check_request

    image = input_file("image_path", image_path)
    request = check_request(
        image, classes=classes, mode=mode, model_part=model_part, roi_subset=roi_subset,
        robust_crop=robust_crop, body_seg=body_seg, force_split=force_split,
        probabilities=probabilities,
    )
    targets = output_targets(output_dir, OUTPUT_FILES, overwrite=overwrite, inputs=[image])
    return request, targets


def _segment(
    request: Request,
    targets: dict[str, Path],
    folder: Path,
    revision: str | None,
    networks: Networks,
) -> RunResult:
    """Admit the models, segment, then publish this run's files together."""
    from totalsegmentator_mlx.imageio import save_nifti
    from totalsegmentator_mlx.pipeline import check_models, run_request

    check_models(request, folder, list(targets.values()))
    labels, prediction = run_request(request, folder, networks)
    writes: dict[Path, Callable[[Path], None]] = {
        targets["mask"]: lambda path: save_nifti(labels, path)
    }
    if prediction is not None:
        writes[targets["probabilities"]] = lambda path: _save_probabilities(prediction, path)
        writes[targets["probability_properties"]] = lambda path: _save_properties(
            prediction, path
        )
    publish(writes, remove=[path for path in targets.values() if path not in writes])
    roles = [role for role, path in targets.items() if path in writes]
    artifacts: dict[str, Artifact] = {
        "mask": artifact(
            kind="label_map",
            coordinate_space="image_physical",
            grid_of="image_path",
            labels=[{"value": number, "name": name} for number, name in request.selected.items()],
        )
    }
    if prediction is not None:
        # nnU-Net's reader grid of the resampled RAS CT, which is not written.
        artifacts["probabilities"] = artifact(kind="scalar_field", coordinate_space="unresolved")
    return {
        "outputs": {role: targets[role].as_posix() for role in roles},
        "artifacts": artifacts,
        "details": {
            "weights_revision": revision,
            "backend": "metal",
            "mode": request.mode,
            "source_profile": SOURCE_PROFILE,
            "precision": ARITHMETIC,
            "folds": [0],
            "labels": [{"name": name, "value": number} for number, name in LABELS.items()],
        },
    }


def _save_probabilities(prediction: NiftiPrediction, path: Path) -> None:
    # A handle keeps NumPy from appending ".npz" to the staging name.
    with path.open("wb") as handle:
        np.savez_compressed(handle, probabilities=prediction.probabilities)


def _save_properties(prediction: NiftiPrediction, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(prediction.properties, handle)
