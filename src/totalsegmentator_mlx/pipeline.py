"""CT workflow tiers, auxiliary crops and restoration to the original NIfTI grid."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import nibabel as nib
import numpy as np
from medmlx_core.errors import InvalidInputError

from totalsegmentator_mlx.engine import predict_image, predict_part
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
from totalsegmentator_mlx.imageio import load_nifti, save_nifti
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
from totalsegmentator_mlx.output import (
    add_label_table,
    merge_part,
    require_folder,
    validate_destination,
    write_segmentation,
)


def _infer_parts(
    staging: Path,
    slabs: Sequence[nib.Nifti1Image],
    root: Path,
    task_ids: tuple[int, ...],
    probabilities: Path | None,
    progress: Callable[[str], None],
) -> list[Labels]:
    """Run each model on every slab and merge its local labels into global IDs in task order.

    Without ``probabilities`` the slabs are predicted in memory. With it (one unsplit slab
    and one model only) predictions go through the engine's file-based path so its
    ``.npz`` probabilities and ``.pkl`` properties can be copied to ``probabilities``.
    """
    # Engine XYZ labels are views of dense ZYX arrays; match their storage during merges.
    combined = [np.zeros(slab.shape, dtype=np.uint8, order="F") for slab in slabs]
    if probabilities is None:
        # XYZ Fortran storage is dense CZYX to the engine, shared by all five parts.
        model_images = [
            nib.Nifti1Image(np.asfortranarray(slab.dataobj, dtype=np.float32), slab.affine)
            for slab in slabs
        ]
        for task_id in task_ids:
            progress(f"Predicting Dataset{task_id}")
            folder = model_folder(root, task_id)
            for slab, labels in zip(model_images, combined, strict=True):
                image = predict_image(slab, folder, task_id)
                merge_part(labels, cast(Labels, np.asanyarray(image.dataobj)), task_id)
        return combined
    incoming = staging / "input"
    incoming.mkdir()
    for index, slab in enumerate(slabs, 1):
        save_nifti(slab, incoming / f"s{index:02d}_0000.nii.gz")
    for task_id in task_ids:
        progress(f"Predicting Dataset{task_id}")
        output = staging / str(task_id)
        predict_part(
            incoming,
            output,
            model_folder(root, task_id),
            task_id,
            save_probabilities=True,
        )
        for index, labels in enumerate(combined, 1):
            image = load_nifti(output / f"s{index:02d}.nii.gz")
            merge_part(labels, np.asanyarray(image.dataobj), task_id)
        probabilities.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output / "s01.npz", probabilities)
        shutil.copyfile(output / "s01.pkl", probabilities.with_suffix(".pkl"))
    return combined


def _segment_grid(
    source: nib.Nifti1Image,
    root: Path,
    task_ids: tuple[int, ...],
    *,
    force_split: bool,
    probabilities: Path | None,
    progress: Callable[[str], None],
) -> nib.Nifti1Image:
    """Run models sharing one spacing and restore uint8 labels to the source grid.

    The CT is reoriented to RAS, resampled to the models' isotropic grid,
    split into slabs when required, merged in model order and restored with
    nearest-neighbor labels to the source shape, affine and header.
    """
    progress("Preparing CT")
    canonical = canonical_ct(source)
    prepared = resample_ct(canonical, TASK_SPACING[task_ids[0]])
    slabs = split_ct(prepared, force_split=force_split, multimodel=len(task_ids) > 1)
    if probabilities is not None and len(slabs) != 1:
        raise InvalidInputError("Probabilities require an unsplit input")
    with TemporaryDirectory(prefix="totalsegmentator_mlx_") as temporary:
        combined = _infer_parts(Path(temporary), slabs, root, task_ids, probabilities, progress)
    labels = join_slabs(combined, prepared.shape)
    if task_ids == (BODY_MODEL,):
        # Upstream cleans body components on the 6 mm grid before restoration.
        zooms = nib.Nifti1Image(labels, prepared.affine).header.get_zooms()
        labels = clean_body(labels, zooms)
    progress("Restoring source grid")
    return restore_labels(labels, canonical, source)


def _protect_input(destination: Path, source: Path | None) -> None:
    """Reject output paths sharing the input's name, symlink target or inode."""
    if source is not None and (
        destination.resolve() == source.resolve()
        or (destination.is_file() and source.is_file() and destination.samefile(source))
    ):
        raise InvalidInputError("Output must not overwrite the input CT")


def _input_paths(image: str | Path | nib.Nifti1Image) -> set[Path]:
    """Find named image files and named proxy backing files when available."""
    if not isinstance(image, nib.Nifti1Image):
        return {Path(image)}
    paths: set[Path] = set()
    filename = image.get_filename()
    if filename is not None:
        paths.add(Path(filename))
    backing: object = (
        image.dataobj.filename
        if isinstance(image.dataobj, np.memmap)
        else getattr(image.dataobj, "file_like", None)
    )
    if not isinstance(backing, (str, PathLike)):
        backing = getattr(backing, "name", None)
    if isinstance(backing, str):
        paths.add(Path(backing))
    elif isinstance(backing, PathLike):
        pathlike = cast(PathLike[str], backing)
        paths.add(Path(pathlike))
    return paths


def _same_path(first: Path, second: Path) -> bool:
    """Compare names and existing inodes, including symlink and hard-link aliases."""
    return first.resolve() == second.resolve() or (
        first.is_file() and second.is_file() and first.samefile(second)
    )


def _protect_outputs(outputs: Sequence[Path], root: Path) -> None:
    """Reject collisions among artifacts and the known files in model folders."""
    if any(_same_path(first, second) for first, second in combinations(outputs, 2)):
        raise InvalidInputError("Outputs must not refer to the same file")
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


@dataclass(frozen=True)
class Request:
    """Segmentation arguments checked by ``check_request``; model folders are not checked yet."""

    image: nib.Nifti1Image
    task_ids: tuple[int, ...]
    crop_task: int | None
    roi: dict[int, str] | None
    selected: dict[int, str]
    filter_labels: bool
    destination: Path | None
    probabilities: Path | None
    force_split: bool
    overwrite: bool

    @property
    def outputs(self) -> list[Path]:
        """Files this request writes: the mask, then probabilities and their properties."""
        probabilities = self.probabilities
        return ([] if self.destination is None else [self.destination]) + (
            [] if probabilities is None else [probabilities, probabilities.with_suffix(".pkl")]
        )


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


def _open_input(image_path: str | Path | nib.Nifti1Image) -> nib.Nifti1Image:
    """Return an in-memory image unchanged, or open an existing ``.nii``/``.nii.gz`` header."""
    image = cast(object, image_path)
    if isinstance(image, nib.Nifti1Image):
        return image
    if not isinstance(image, (str, PathLike)):
        raise InvalidInputError("image_path must be a NIfTI file path or image")
    path = Path(cast(str | PathLike[str], image)).expanduser()
    if not path.name.lower().endswith((".nii", ".nii.gz")) or not path.is_file():
        raise InvalidInputError(f"image_path must be an existing .nii or .nii.gz file: {path}")
    return open_ct(path)


def check_request(
    image_path: str | Path | nib.Nifti1Image,
    *,
    output_path: str | Path | None = None,
    mode: Mode = "standard",
    model_part: str | None = None,
    prompt: Sequence[str] | None = None,
    roi_subset: Sequence[str] | None = None,
    robust_crop: bool = False,
    body_seg: bool = False,
    force_split: bool = False,
    probabilities_path: str | Path | None = None,
    overwrite: bool = False,
) -> Request:
    """Check every ``segment`` argument that needs no model folder, reading only the CT header.

    Arguments are those of ``segment``. Call this before downloading weights so a bad
    argument fails without network or model work.

    Raises:
        InvalidInputError: Invalid options, booleans, CT file or output path.
    """
    flags = {
        "robust_crop": robust_crop,
        "body_seg": body_seg,
        "force_split": force_split,
        "overwrite": overwrite,
    }
    for name, flag in flags.items():
        if not isinstance(cast(object, flag), bool):
            raise InvalidInputError(f"{name} must be true or false")
    image = _open_input(image_path)
    roi_names = _names("roi_subset", roi_subset)
    roi = None if roi_names is None else select_labels(roi_names)
    if roi is not None and (prompt is not None or model_part is not None or body_seg):
        raise InvalidInputError("roi_subset cannot be combined with prompt, model_part or body_seg")
    if robust_crop and roi is None:
        raise InvalidInputError("robust_crop requires roi_subset")
    task_ids = select_models(mode, model_part, roi)
    crop_task = None
    if roi is not None:
        crop_task = QUALITY_MODELS["fast" if robust_crop else "coarse"]
    elif body_seg:
        crop_task = BODY_MODEL
    selected = roi if roi is not None else select_labels(_names("prompt", prompt))
    source_paths = _input_paths(image)
    destination = None if output_path is None else Path(output_path)
    if destination is not None:
        validate_destination(destination, overwrite=overwrite)
        for source_path in source_paths:
            _protect_input(destination, source_path)
    probabilities = None if probabilities_path is None else Path(probabilities_path)
    if probabilities is not None:
        if len(task_ids) != 1 or crop_task is not None or probabilities.suffix != ".npz":
            raise InvalidInputError(
                "Probabilities require one model_part or mode, no crop and an .npz destination"
            )
        require_folder(probabilities)
        for path in (probabilities, probabilities.with_suffix(".pkl")):
            if path.exists() and not overwrite:
                raise InvalidInputError(f"Output already exists: {path}")
            for source_path in source_paths:
                _protect_input(path, source_path)
    return Request(
        image=image,
        task_ids=task_ids,
        crop_task=crop_task,
        roi=roi,
        selected=selected,
        filter_labels=roi is not None or prompt is not None,
        destination=destination,
        probabilities=probabilities,
        force_split=force_split,
        overwrite=overwrite,
    )


def segment(
    image_path: str | Path | nib.Nifti1Image,
    *,
    model_dir: str | Path,
    output_path: str | Path | None = None,
    mode: Mode = "standard",
    model_part: str | None = None,
    prompt: Sequence[str] | None = None,
    roi_subset: Sequence[str] | None = None,
    robust_crop: bool = False,
    body_seg: bool = False,
    force_split: bool = False,
    probabilities_path: str | Path | None = None,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
) -> nib.Nifti1Image:
    """Segment CT on Metal and return uint8 global labels on its original voxel grid.

    The CT (finite, 3D, Hounsfield units, millimeter or unspecified spatial units) is
    reoriented to RAS, resampled with a cubic spline to the model grid (1.5, 3 or 6 mm)
    and staged as int32. No intensity clipping or normalization happens in this package;
    the staged values go to the nnunet-mlx engine as they are. Labels are restored by nearest
    neighbor to the input shape, affine and orientation. The returned image is NIfTI-1 with
    a copy of the input header (data type uint8) and the upstream Caret label table added as
    a header extension. Label IDs are the ``total`` task's global IDs 1-117 (0 is background;
    see ``labels.LABELS``).

    ``mode`` selects the upstream ``total`` task tier: ``standard`` runs the
    five 1.5 mm parts (291-295), ``fast`` the 3 mm model (297, upstream
    --fast) and ``coarse`` the 6 mm model (298, upstream --fastest).
    ``prompt`` filters the full result after inference.

    ``roi_subset`` follows upstream: a rough 6 mm total segmentation (3 mm 297
    with ``robust_crop``) locates the named structures, the CT is cropped to
    them plus 20 mm, and other labels are removed. Standard mode runs only the
    parts containing those structures; fast and coarse each run their one model.
    ``body_seg`` instead crops to the Dataset300 body mask plus 3 mm.
    Crops are restored by zero filling. An empty localization returns an empty
    segmentation without running the requested models.

    Probabilities retain the engine's local labels and require one unsplit,
    uncropped model. No weights are downloaded. Existing outputs require
    explicit overwrite. The CT input, known model metadata/checkpoint files,
    and outputs are checked for aliases before inference; concurrent filesystem
    changes after the check are not locked out. File-backed image objects are
    protected when their filename, proxy backing file, or NumPy memmap has a
    path; anonymous or opaque backing objects may not expose an identifiable file.

    Args:
        image_path: CT NIfTI path, or an in-memory NIfTI-1 or NIfTI-2 image; never modified.
        model_dir: Root of converted model folders ``<id>/``; a folder holding
            ``plans.json`` is used directly, which only suits a single-model selection.
            Each model folder contains ``plans.json``, ``dataset.json`` and
            ``fold_0/checkpoint_final/`` with ``state.json`` and ``weights.safetensors``.
        output_path: Optional ``.nii``/``.nii.gz`` destination; it must not be the input
            file, and an existing file needs ``overwrite``.
        mode: ``standard``, ``fast`` or ``coarse``.
        model_part: One standard part (``organs``, ``vertebrae``, ``cardiac``, ``muscles``,
            ``ribs``); standard mode only.
        prompt: Canonical structure names to keep; other labels are set to 0 after inference.
        roi_subset: Canonical structure names to localize and crop to; excludes ``prompt``,
            ``model_part`` and ``body_seg``.
        robust_crop: Locate ``roi_subset`` with the 3 mm model instead of the 6 mm one.
        body_seg: Crop to the Dataset300 body mask plus 3 mm.
        force_split: Split into three overlapping z slabs even for smaller volumes.
        probabilities_path: ``.npz`` destination for local-label probabilities (the engine's
            ``.pkl`` properties go beside it).
        overwrite: Allow replacing existing outputs.
        progress: Called with a short message at each stage.

    Returns:
        The label image. When an ROI or body localization is empty it is an all-zero
        image with the source affine and the label table but not the source header.

    Raises:
        InvalidInputError: Invalid options or non-bool flags, a missing or unreadable CT,
            or an invalid or existing output path; checked before the model folders.
        AssetNotReadyError: A required converted model is missing or mismatched.
        ModelExecutionError: Metal is unavailable or a restored grid or label map is invalid.
    """
    request = check_request(
        image_path,
        output_path=output_path,
        mode=mode,
        model_part=model_part,
        prompt=prompt,
        roi_subset=roi_subset,
        robust_crop=robust_crop,
        body_seg=body_seg,
        force_split=force_split,
        probabilities_path=probabilities_path,
        overwrite=overwrite,
    )
    return segment_request(request, model_dir=model_dir, progress=progress)


def segment_request(
    request: Request,
    *,
    model_dir: str | Path,
    progress: Callable[[str], None] | None = None,
) -> nib.Nifti1Image:
    """Check the model folders for a checked request, then segment and write its outputs.

    See ``segment`` for the workflow, outputs and errors.
    """
    task_ids, crop_task, roi = request.task_ids, request.crop_task, request.roi
    force_split, probabilities = request.force_split, request.probabilities
    root = Path(model_dir)
    validate_models(root, task_ids if crop_task is None else (crop_task, *task_ids))
    _protect_outputs(request.outputs, root)
    report: Callable[[str], None] = progress if progress is not None else lambda _message: None
    source = load_ct(request.image)
    if crop_task is None:
        result = _segment_grid(
            source,
            root,
            task_ids,
            force_split=force_split,
            probabilities=probabilities,
            progress=report,
        )
    else:
        report("Localizing body" if roi is None else "Localizing ROI subset")
        rough = _segment_grid(
            source, root, (crop_task,), force_split=False, probabilities=None, progress=report
        )
        mask = localization_mask(np.asanyarray(rough.dataobj), None if roi is None else set(roi))
        if not mask.any():
            # Upstream returns a header-free empty image for an empty crop.
            result = nib.Nifti1Image(np.zeros(source.shape, dtype=np.uint8), source.affine)
        else:
            margin = BODY_MARGIN_MM if roi is None else ROI_MARGIN_MM
            cropped, box = crop_ct(source, mask, margin)
            part = _segment_grid(
                cropped,
                root,
                task_ids,
                force_split=force_split,
                probabilities=None,
                progress=report,
            )
            labels = np.zeros(source.shape, dtype=np.uint8)
            labels[box] = np.asanyarray(part.dataobj)
            result = source_labels(labels, source)
    if request.filter_labels:
        values = np.asanyarray(result.dataobj)
        filtered = np.where(np.isin(values, list(request.selected)), values, 0).astype(np.uint8)
        result = nib.Nifti1Image(filtered, result.affine, result.header)
    add_label_table(result, request.selected)
    if request.destination is not None:
        report("Saving segmentation")
        write_segmentation(result, request.destination, overwrite=request.overwrite)
    return result
