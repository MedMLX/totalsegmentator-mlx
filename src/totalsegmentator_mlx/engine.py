"""Delegate each CT network to the public nnunet-mlx Metal inference boundary."""

from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import nibabel as nib
import numpy as np
from medmlx_core.errors import ModelExecutionError

from totalsegmentator_mlx.imageio import load_nifti, spatial_affine
from totalsegmentator_mlx.labels import BODY_MODEL
from totalsegmentator_mlx.models import ARITHMETIC, SOURCE_PROFILE, TASK_SPACING

if TYPE_CHECKING:
    from nnunet_mlx import ModelFolderPredictor, NiftiPrediction

_runtime_lock = RLock()

type ModelIdentity = tuple[tuple[int, int, int, int, int] | None, ...]
type CacheKey = tuple[Path, int, ModelIdentity]


class _Memory(Protocol):
    """Native allocator counters, separate from predictor ownership accounting."""

    def get_active_memory(self) -> int: ...
    def get_cache_memory(self) -> int: ...
    def clear_cache(self) -> None: ...


class PredictorCacheInfo(TypedDict):
    """Predictor budget and overlapping native MLX allocation/cache counters."""

    entries: int
    retained_bytes: int
    max_bytes: int | None
    max_entries: int
    mlx_active_bytes: int | None
    mlx_cache_bytes: int | None


_cache: OrderedDict[CacheKey, tuple[ModelFolderPredictor, int]] = OrderedDict()
_max_bytes: int | None = None
_max_entries = 8


def configure_predictor_cache(*, max_bytes: int | None = None, max_entries: int = 8) -> None:
    """Set a process-local retained-predictor budget and release the previous cache.

    The default keeps all eight standard/tier/body predictors for warm reuse.
    ``max_entries=1`` retains one active network; zero disables reuse. The byte
    budget covers retained model state, not inference scratch or MLX's free
    allocator cache. A model larger than the budget runs uncached. Calls and
    policy changes share the runtime lock, so active inference cannot be evicted.
    """
    global _max_bytes, _max_entries
    if (type(max_entries) is not int or max_entries < 0
            or (max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0))):
        raise ValueError("Predictor cache limits must be nonnegative integers or max_bytes=None")
    with _runtime_lock:
        if (_max_bytes, _max_entries) != (max_bytes, max_entries):
            clear_predictor_cache()
            _max_bytes, _max_entries = max_bytes, max_entries


def clear_predictor_cache(*, release_memory: bool = True) -> None:
    """Release retained predictors and, by default, free unused MLX allocator buffers."""
    with _runtime_lock:
        _cache.clear()
        module = sys.modules.get("mlx.core")
        if release_memory and module is not None:
            cast(_Memory, module).clear_cache()


def predictor_cache_info() -> PredictorCacheInfo:
    """Return retained-state accounting and separate native counters without loading MLX."""
    with _runtime_lock:
        module = sys.modules.get("mlx.core")
        memory = None if module is None else cast(_Memory, module)
        return {"entries": len(_cache), "retained_bytes": sum(size for _, size in _cache.values()),
                "max_bytes": _max_bytes, "max_entries": _max_entries,
                "mlx_active_bytes": None if memory is None else memory.get_active_memory(),
                "mlx_cache_bytes": None if memory is None else memory.get_cache_memory()}


def _predictor(folder: Path, task_id: int) -> ModelFolderPredictor:
    """Get one verified predictor under the caller-held runtime lock."""
    identity = _model_identity(folder)
    key = folder, task_id, identity
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key][0]
    evicted = False
    # A changed checkpoint replaces its previous identity immediately, rather
    # than consuming another slot while stale model state remains resident.
    for previous in tuple(_cache):
        if previous[:2] == key[:2]:
            del _cache[previous]
            evicted = True
    weights = identity[3]
    assert weights is not None
    estimate = weights[2]
    while _cache and (len(_cache) >= _max_entries or (_max_bytes is not None
            and sum(size for _, size in _cache.values()) + estimate > _max_bytes)):
        _cache.popitem(last=False)
        evicted = True
    memory = cast(_Memory, import_module("mlx.core"))
    if evicted:
        memory.clear_cache()
    before = memory.get_active_memory()
    predictor = _load_predictor(folder, task_id)
    if _model_identity(folder) != identity:
        raise ModelExecutionError("Model files changed while loading; retry with stable assets")
    # Materialized model allocations are measured serially; the safetensors
    # file size is a conservative floor that includes its metadata header.
    resident_bytes = max(estimate, memory.get_active_memory() - before)
    while _cache and _max_bytes is not None and (
            sum(size for _, size in _cache.values()) + resident_bytes > _max_bytes):
        _cache.popitem(last=False)
    if _max_entries and (_max_bytes is None or resident_bytes <= _max_bytes):
        _cache[key] = predictor, resident_bytes
    return predictor


def _model_identity(folder: Path) -> ModelIdentity:
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


def _load_predictor(
    folder: Path,
    task_id: int,
) -> ModelFolderPredictor:
    """Load one fold-0 model onto Metal.

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


def predict_image(image: nib.Nifti1Image, folder: Path, task_id: int) -> nib.Nifti1Image:
    """Return local labels on the supplied XYZ grid, retaining loaded networks.

    ``image`` is float32 CT on the model grid (1.5, 3 or 6 mm); the returned labels are
    this model's own IDs (``model_labels(task_id)``), not yet merged into global IDs.

    The public engine NIfTI API transfers final labels from Metal. The lock owns
    the cached predictors' single-GPU lifecycle across concurrent API callers.
    """
    with _runtime_lock:
        folder = folder.resolve()
        predictor = _predictor(folder, task_id)
        return predictor.predict_nifti(image)



def predict_probabilities(
    image: nib.Nifti1Image, folder: Path, task_id: int,
) -> NiftiPrediction:
    """Return labels plus reader-grid probabilities/properties without staging the CT."""
    with _runtime_lock:
        folder = folder.resolve()
        predictor = _predictor(folder, task_id)
        return predictor.predict_nifti_probabilities(image)

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
        predictor = _predictor(folder, task_id)
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
