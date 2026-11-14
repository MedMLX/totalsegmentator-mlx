"""Delegate each CT network to the public nnunet-mlx Metal inference boundary."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Protocol, cast

import nibabel as nib
from medmlx_core.errors import ModelExecutionError

from totalsegmentator_mlx.labels import BODY_MODEL
from totalsegmentator_mlx.models import ARITHMETIC, SOURCE_PROFILE, TASK_SPACING

if TYPE_CHECKING:
    from nnunet_mlx import ModelFolderPredictor, NiftiPrediction

_runtime_lock = RLock()

type ModelIdentity = tuple[tuple[int, int, int, int, int] | None, ...]


class _Memory(Protocol):
    """MLX's free-buffer cache control."""

    def clear_cache(self) -> None: ...


class Networks:
    """Fold-0 networks on Metal, owned by one ``segment`` call or one loaded model.

    With ``retain=False`` (one-shot ``segment``) a network is loaded for one model's
    slabs and released, with MLX's free-buffer cache, before the next model loads, so
    one network is resident at a time. With ``retain=True`` (``load_model``) every
    network used stays loaded until :meth:`close`. Prediction holds a process-wide
    lock because all networks share the one Metal device.
    """

    def __init__(self, *, retain: bool) -> None:
        """Start with no network loaded; each loads on first use."""
        self._retain = retain
        self._loaded: dict[tuple[Path, int], ModelFolderPredictor] = {}

    def predict_images(
        self, images: Sequence[nib.Nifti1Image], folder: Path, task_id: int
    ) -> list[nib.Nifti1Image]:
        """Return each image's local labels on its own grid, in order.

        ``images`` are float32 CT on the model grid (1.5, 3 or 6 mm); labels are the
        model's own IDs (``model_labels(task_id)``), not yet merged into global IDs.
        """
        with _runtime_lock:
            predictor = self._network(folder, task_id)
            try:
                return [predictor.predict_nifti(image) for image in images]
            finally:
                del predictor
                self._release()

    def predict_probabilities(
        self, image: nib.Nifti1Image, folder: Path, task_id: int
    ) -> NiftiPrediction:
        """Return labels plus reader-grid probabilities and properties for one image."""
        with _runtime_lock:
            predictor = self._network(folder, task_id)
            try:
                return predictor.predict_nifti_probabilities(image)
            finally:
                del predictor
                self._release()

    def close(self) -> None:
        """Release every retained network and MLX's free-buffer cache."""
        with _runtime_lock:
            self._loaded.clear()
            _clear_mlx_cache()

    def _network(self, folder: Path, task_id: int) -> ModelFolderPredictor:
        key = folder.resolve(), task_id
        predictor = self._loaded.get(key)
        if predictor is None:
            predictor = load_predictor(*key)
            if self._retain:
                self._loaded[key] = predictor
        return predictor

    def _release(self) -> None:
        if not self._retain:
            _clear_mlx_cache()


def _clear_mlx_cache() -> None:
    """Return MLX's free buffers to the system; a no-op before MLX is imported."""
    module = sys.modules.get("mlx.core")
    if module is not None:
        cast(_Memory, module).clear_cache()


def _model_identity(folder: Path) -> ModelIdentity:
    """Identify the model files by inode, size and times, to detect edits during loading."""
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


def load_predictor(folder: Path, task_id: int) -> ModelFolderPredictor:
    """Load one fold-0 model onto Metal with TotalSegmentator's inference settings.

    Raises:
        ModelExecutionError: Metal is unavailable, the model does not take one CT channel,
            a non-body model carries test-time mirroring axes, or the model files changed
            while loading.
    """
    identity = _model_identity(folder)
    predictor = _load_predictor(folder, task_id)
    if _model_identity(folder) != identity:
        raise ModelExecutionError("Model files changed while loading; retry with stable assets")
    return predictor


def _load_predictor(folder: Path, task_id: int) -> ModelFolderPredictor:
    """Build the predictor; ``load_predictor`` checks the files around this call."""
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
