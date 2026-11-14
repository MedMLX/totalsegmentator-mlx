"""CT workflow tiers, auxiliary crops and restoration to the original NIfTI grid."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import nibabel as nib
import numpy as np
from medmlx_core.errors import InvalidInputError

from totalsegmentator_mlx.geometry import (
    Labels,
    canonical_ct,
    crop_ct,
    join_slabs,
    load_ct,
    open_ct,
    resample_ct,
    restore_labels,
    source_labels,
    split_ct,
)
from totalsegmentator_mlx.labels import BODY_MODEL, QUALITY_MODELS
from totalsegmentator_mlx.localization import (
    BODY_MARGIN_MM,
    ROI_MARGIN_MM,
    clean_body,
    localization_mask,
)
from totalsegmentator_mlx.models import (
    TASK_SPACING,
    Mode,
    model_folder,
    select_labels,
    select_models,
    validate_models,
)
from totalsegmentator_mlx.output import add_label_table, merge_part

if TYPE_CHECKING:
    from nnunet_mlx import NiftiPrediction

    from totalsegmentator_mlx.engine import Networks

# Stage messages ("Predicting Dataset291", ...) at INFO; the CLI prints them.
_log = logging.getLogger("totalsegmentator_mlx")


def _infer_parts(
    slabs: Sequence[nib.Nifti1Image],
    root: Path,
    task_ids: tuple[int, ...],
    networks: Networks,
    *,
    probabilities: bool,
) -> tuple[list[Labels], NiftiPrediction | None]:
    """Predict prepared slabs directly and merge local labels in the original task order.

    Probability runs admit one model and one unsplit slab (``check_request``). Their
    arrays and reader/crop properties come from nnU-Net's in-memory boundary.
    """
    combined = [np.zeros(slab.shape, dtype=np.uint8, order="F") for slab in slabs]
    # XYZ Fortran storage is dense CZYX to the engine, shared by all five parts.
    model_images = [
        nib.Nifti1Image(np.asfortranarray(slab.dataobj, dtype=np.float32), slab.affine)
        for slab in slabs
    ]
    prediction: NiftiPrediction | None = None
    for task_id in task_ids:
        _log.info("Predicting Dataset%d", task_id)
        folder = model_folder(root, task_id)
        if probabilities:
            prediction = networks.predict_probabilities(model_images[0], folder, task_id)
            images = [prediction.segmentation]
        else:
            images = networks.predict_images(model_images, folder, task_id)
        for labels, image in zip(combined, images, strict=True):
            merge_part(labels, cast(Labels, np.asanyarray(image.dataobj)), task_id)
    return combined, prediction


def _segment_grid(
    source: nib.Nifti1Image,
    root: Path,
    task_ids: tuple[int, ...],
    networks: Networks,
    *,
    force_split: bool,
    probabilities: bool,
) -> tuple[nib.Nifti1Image, NiftiPrediction | None]:
    """Run models sharing one spacing and restore uint8 labels to the source grid.

    The CT is reoriented to RAS, resampled to the models' isotropic grid,
    split into slabs when required, merged in model order and restored with
    nearest-neighbor labels to the source shape, affine and header.
    """
    _log.info("Preparing CT")
    canonical = canonical_ct(source)
    prepared = resample_ct(canonical, TASK_SPACING[task_ids[0]])
    slabs = split_ct(prepared, force_split=force_split, multimodel=len(task_ids) > 1)
    if probabilities and len(slabs) != 1:
        raise InvalidInputError("probabilities require an unsplit input")
    combined, prediction = _infer_parts(
        slabs, root, task_ids, networks, probabilities=probabilities
    )
    labels = join_slabs(combined, prepared.shape)
    if task_ids == (BODY_MODEL,):
        # Upstream cleans body components on the 6 mm grid before restoration.
        zooms = nib.Nifti1Image(labels, prepared.affine).header.get_zooms()
        labels = clean_body(labels, zooms)
    _log.info("Restoring source grid")
    return restore_labels(labels, canonical, source), prediction


def _same_path(first: Path, second: Path) -> bool:
    """Compare names and existing inodes, including symlink and hard-link aliases."""
    return first.resolve() == second.resolve() or (
        first.is_file() and second.is_file() and first.samefile(second)
    )


@dataclass(frozen=True)
class Request:
    """Segmentation arguments checked by ``check_request``; model folders are not checked yet."""

    image: nib.Nifti1Image
    mode: Mode
    task_ids: tuple[int, ...]
    crop_task: int | None
    roi: dict[int, str] | None
    selected: dict[int, str]
    filter_labels: bool
    force_split: bool
    probabilities: bool


def _names(name: str, value: Sequence[str] | None) -> list[str] | None:
    """Return structure names as a list; a bare string or a non-string item is rejected."""
    if value is None:
        return None
    items = cast(object, value)
    if isinstance(items, str) or not isinstance(items, (list, tuple)):
        raise InvalidInputError(f"{name} must be a list of canonical structure names")
    names = cast(list[object] | tuple[object, ...], items)
    if not all(isinstance(item, str) for item in names):
        raise InvalidInputError(f"{name} must be a list of canonical structure names")
    return [str(item) for item in names]


def check_request(
    image_path: Path,
    *,
    mode: Mode = "standard",
    model_part: str | None = None,
    classes: Sequence[str] | None = None,
    roi_subset: Sequence[str] | None = None,
    robust_crop: bool = False,
    body_seg: bool = False,
    force_split: bool = False,
    probabilities: bool = False,
) -> Request:
    """Check every ``segment`` option that needs no model folder, reading only the CT header.

    Call this before downloading weights so a bad argument fails without network or
    model work. Output paths are checked by the caller.

    Raises:
        InvalidInputError: Invalid options or booleans, or an unreadable or non-NIfTI CT.
    """
    flags = {
        "robust_crop": robust_crop,
        "body_seg": body_seg,
        "force_split": force_split,
        "probabilities": probabilities,
    }
    for name, flag in flags.items():
        if not isinstance(cast(object, flag), bool):
            raise InvalidInputError(f"{name} must be true or false")
    if not image_path.name.lower().endswith((".nii", ".nii.gz")):
        raise InvalidInputError(f"image_path must be a .nii or .nii.gz file: {image_path}")
    image = open_ct(image_path)
    roi_names = _names("roi_subset", roi_subset)
    roi = None if roi_names is None else select_labels(roi_names)
    if roi is not None and (classes is not None or model_part is not None or body_seg):
        raise InvalidInputError(
            "roi_subset cannot be combined with classes, model_part or body_seg"
        )
    if robust_crop and roi is None:
        raise InvalidInputError("robust_crop requires roi_subset")
    task_ids = select_models(mode, model_part, roi)
    crop_task = None
    if roi is not None:
        crop_task = QUALITY_MODELS["fast" if robust_crop else "coarse"]
    elif body_seg:
        crop_task = BODY_MODEL
    if probabilities and (len(task_ids) != 1 or crop_task is not None):
        raise InvalidInputError(
            "probabilities require one model (fast, coarse or one model_part) and no crop"
        )
    selected = roi if roi is not None else select_labels(_names("classes", classes))
    return Request(
        image=image,
        mode=mode,
        task_ids=task_ids,
        crop_task=crop_task,
        roi=roi,
        selected=selected,
        filter_labels=roi is not None or classes is not None,
        force_split=force_split,
        probabilities=probabilities,
    )


def check_models(request: Request, root: Path, outputs: Sequence[Path]) -> None:
    """Admit the request's model folders and keep ``outputs`` off their files.

    Raises:
        AssetNotReadyError: A required converted model is missing or mismatched.
        InvalidInputError: An output is one of the model files.
    """
    task_ids = request.task_ids
    validate_models(root, task_ids if request.crop_task is None else (request.crop_task, *task_ids))
    filenames = (
        "plans.json",
        "dataset.json",
        "conversion.json",
        "fold_0/checkpoint_final/state.json",
        "fold_0/checkpoint_final/weights.safetensors",
        "fold_0/checkpoint_final.pth",
    )
    folders = {model_folder(root, task) for task in TASK_SPACING}
    folders.update(root / str(task) for task in TASK_SPACING)
    assets = {folder / name for folder in folders for name in filenames}
    if any(_same_path(output, asset) for output in outputs for asset in assets):
        raise InvalidInputError("Output must not overwrite a model asset")


def run_request(
    request: Request, root: Path, networks: Networks
) -> tuple[nib.Nifti1Image, NiftiPrediction | None]:
    """Segment a checked request whose models ``check_models`` admitted.

    The CT (finite, 3D, Hounsfield units, millimeter or unspecified spatial units) is
    reoriented to RAS, resampled with a cubic spline to the model grid (1.5, 3 or 6 mm)
    and staged as int32. No intensity clipping or normalization happens in this package;
    the staged values go to the nnunet-mlx engine as they are. Labels are restored by
    nearest neighbor to the input shape, affine and orientation. The label image is
    NIfTI-1 with a copy of the input header (data type uint8) and the upstream Caret label
    table as a header extension. Label IDs are the ``total`` task's global IDs 1-117
    (0 is background; see ``labels.LABELS``).

    ``mode`` selects the upstream ``total`` task tier: ``standard`` runs the five 1.5 mm
    parts (291-295), ``fast`` the 3 mm model (297, upstream --fast) and ``coarse`` the
    6 mm model (298, upstream --fastest). ``classes`` filters the full result after
    inference.

    ``roi_subset`` follows upstream: a rough 6 mm total segmentation (3 mm 297 with
    ``robust_crop``) locates the named structures, the CT is cropped to them plus 20 mm,
    and other labels are removed. Standard mode runs only the parts containing those
    structures; fast and coarse each run their one model. ``body_seg`` instead crops to
    the Dataset300 body mask plus 3 mm. Crops are restored by zero filling. An empty
    localization returns an all-zero image with the source affine and the label table
    but not the source header, without running the requested models.

    Returns:
        The label image, and the probability prediction when the request asked for it.

    Raises:
        ModelExecutionError: Metal is unavailable or a restored grid or label map is invalid.
    """
    task_ids, crop_task, roi = request.task_ids, request.crop_task, request.roi
    source = load_ct(request.image)
    prediction: NiftiPrediction | None = None
    if crop_task is None:
        result, prediction = _segment_grid(
            source,
            root,
            task_ids,
            networks,
            force_split=request.force_split,
            probabilities=request.probabilities,
        )
    else:
        _log.info("Localizing body" if roi is None else "Localizing ROI subset")
        rough, _ = _segment_grid(
            source, root, (crop_task,), networks, force_split=False, probabilities=False
        )
        mask = localization_mask(np.asanyarray(rough.dataobj), None if roi is None else set(roi))
        if not mask.any():
            # Upstream returns a header-free empty image for an empty crop.
            result = nib.Nifti1Image(np.zeros(source.shape, dtype=np.uint8), source.affine)
        else:
            margin = BODY_MARGIN_MM if roi is None else ROI_MARGIN_MM
            cropped, box = crop_ct(source, mask, margin)
            part, _ = _segment_grid(
                cropped,
                root,
                task_ids,
                networks,
                force_split=request.force_split,
                probabilities=False,
            )
            labels = np.zeros(source.shape, dtype=np.uint8)
            labels[box] = np.asanyarray(part.dataobj)
            result = source_labels(labels, source)
    if request.filter_labels:
        values = np.asanyarray(result.dataobj)
        filtered = np.where(np.isin(values, list(request.selected)), values, 0).astype(np.uint8)
        result = nib.Nifti1Image(filtered, result.affine, result.header)
    add_label_table(result, request.selected)
    return result, prediction
