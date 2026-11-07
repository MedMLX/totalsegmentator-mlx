"""Delegate each CT network to the public nnunet-mlx Metal inference boundary."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
from medmlx_core.errors import ModelExecutionError

from totalsegmentator_mlx.imageio import load_nifti, spatial_affine
from totalsegmentator_mlx.labels import BODY_MODEL
from totalsegmentator_mlx.models import ARITHMETIC, SOURCE_PROFILE, TASK_SPACING, model_folder

if TYPE_CHECKING:
    from nnunet_mlx import ModelFolderPredictor

_runtime_lock = RLock()


def _model_identity(folder: Path) -> tuple[tuple[int, int, int, int, int] | None, ...]:
    """Invalidate cached weights/metadata on replacement or an in-place update."""
    paths = (
        folder / "plans.json",
        folder / "dataset.json",
        folder / "fold_0/checkpoint_final/state.json",
        folder / "fold_0/checkpoint_final/weights.safetensors",
        folder / "conversion.json",
    )
    identities: list[tuple[int, int, int, int, int] | None] = []
    for path in paths:
        if path.name == "conversion.json" and not path.exists():
            identities.append(None)
            continue
        info = path.stat()
        identities.append(
            (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        )
    return tuple(identities)


def _step_fraction(task_id: int) -> float:
    """Return upstream's tile step: 0.8 below a 3 mm grid and 0.5 otherwise."""
    return 0.8 if TASK_SPACING[task_id] < 3.0 else 0.5


# maxsize 8 keeps all five parts, both single-model tiers and the body model loaded.
@lru_cache(maxsize=8)
def _load_predictor(
    folder: Path,
    task_id: int,
    identity: tuple[tuple[int, int, int, int, int] | None, ...],
) -> ModelFolderPredictor:
    """Load one fold-0 model onto Metal; ``identity`` is only the cache key (file stats).

    Raises:
        ModelExecutionError: Metal is unavailable, the model does not take one CT channel, or a
            non-body model carries test-time mirroring axes.
    """
    from nnunet_mlx import ModelFolderPredictor, load_model_bundle
    from nnunet_mlx.runtime import runtime_info

    if runtime_info().backend != "metal":
        raise ModelExecutionError("TotalSegmentator requires an Apple Metal GPU")
    bundle = load_model_bundle(
        folder,
        configuration="3d_fullres",
        folds=[0],
        source_profile=SOURCE_PROFILE,
        arithmetic=ARITHMETIC,
    )
    if bundle.image_channels != 1 or bundle.input_channels != 1:
        raise ModelExecutionError("CT models require exactly one CT input channel")
    if task_id == BODY_MODEL:
        # The body model was trained with mirroring; upstream predicts it without TTA.
        bundle = replace(bundle, mirroring_axes=())
    elif bundle.mirroring_axes:
        raise ModelExecutionError("TotalSegmentator CT requires a no-mirroring model bundle")
    # Upstream nnU-Net blends every model's tiles in float16; float32 blending
    # changed 549 voxels of AMOS22 case amos_0008 against the upstream standard pipeline.
    return ModelFolderPredictor(
        bundle,
        step_fraction=_step_fraction(task_id),
        compile_models=False,
        accumulation="nnunet-fp16",
    )


def prepare_models(root: Path, task_ids: tuple[int, ...]) -> None:
    """Populate the existing Metal predictor cache while host CT preparation runs."""
    with _runtime_lock:
        for task_id in task_ids:
            folder = model_folder(root, task_id).resolve()
            _load_predictor(folder, task_id, _model_identity(folder))


def predict_image(image: nib.Nifti1Image, folder: Path, task_id: int) -> nib.Nifti1Image:
    """Return local labels on the supplied XYZ grid, retaining loaded networks.

    ``image`` is float32 CT on the model grid (1.5, 3 or 6 mm); the returned labels are
    this model's own IDs (``model_labels(task_id)``), not yet merged into global IDs.

    The public engine NIfTI API transfers final labels from Metal. The lock owns
    the cached predictors' single-GPU lifecycle across concurrent API callers.
    """
    with _runtime_lock:
        folder = folder.resolve()
        predictor = _load_predictor(folder, task_id, _model_identity(folder))
        return predictor.predict_nifti(image)


def predict_part(
    inputs: Path,
    output: Path,
    folder: Path,
    task_id: int,
    *,
    save_probabilities: bool,
) -> None:
    """Export file-backed predictions and optional local-label probabilities.

    Reads ``*_0000.nii.gz`` from ``inputs`` and writes ``<case>.nii.gz`` (plus ``.npz`` and
    ``.pkl`` when ``save_probabilities``) to ``output``, then checks that each result keeps
    its input's shape and affine.
    """
    with _runtime_lock:
        folder = folder.resolve()
        predictor = _load_predictor(folder, task_id, _model_identity(folder))
        predictor.predict_folder(
            inputs,
            output,
            save_probabilities=save_probabilities,
            overwrite=False,
        )
    for path in sorted(inputs.glob("*_0000.nii.gz")):
        source = load_nifti(path)
        result = load_nifti(output / path.name.replace("_0000", ""))
        if result.shape != source.shape or not np.allclose(
            spatial_affine(result),
            spatial_affine(source),
            rtol=0,
            atol=1e-5,
        ):
            raise ModelExecutionError("Engine segmentation geometry differs from its CT input")
