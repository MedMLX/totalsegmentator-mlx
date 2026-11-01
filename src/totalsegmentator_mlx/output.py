"""Global label merging and self-describing NIfTI segmentation output."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
from medmlx_core.errors import InvalidInputError, ModelExecutionError
from nibabel.nifti1 import Nifti1Extension

from totalsegmentator_mlx.geometry import Labels
from totalsegmentator_mlx.imageio import save_nifti
from totalsegmentator_mlx.labels import LABELS, PART_LABELS
from totalsegmentator_mlx.models import model_labels


def merge_part(combined: Labels, local: Labels, task_id: int) -> None:
    """Apply nonzero global labels in part order; later parts overwrite earlier parts.

    ``local`` holds the model's own label IDs and is mapped by name to global IDs for the
    five standard parts (for example Dataset295 local 1, ``rib_left_1``, becomes 92).
    Single models (fast, coarse and body) already predict their final label IDs.
    ``combined`` is updated in place; background (0) never overwrites an earlier label.

    Raises:
        ModelExecutionError: ``local`` has the wrong shape, is not integer, or contains IDs
            outside the model's released label map.
    """
    part = model_labels(task_id)
    if local.shape != combined.shape or local.dtype.kind not in "iu":
        raise ModelExecutionError("Part segmentation shape or integer dtype is invalid")
    if local.min(initial=0) < 0 or local.max(initial=0) > max(part):
        raise ModelExecutionError(
            "Part segmentation contains labels outside its released label map"
        )
    global_ids = {name: number for number, name in LABELS.items()}
    lookup = np.zeros(max(part) + 1, dtype=np.uint8)
    for local_id, name in part.items():
        lookup[local_id] = global_ids[name] if task_id in PART_LABELS else local_id
    mapped = lookup[local]
    np.copyto(combined, mapped, where=mapped != 0)


def add_label_table(image: nib.Nifti1Image, labels: Mapping[int, str]) -> None:
    """Append the upstream Caret NIfTI label table, including its label color cycle.

    The table is a header extension (not voxel data) listing ``labels`` (global ID to name);
    colors cycle through twelve fixed RGB values by ``ID % 12``, as in upstream.
    """
    colors = (
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (255, 0, 128),
        (128, 255, 128),
        (0, 128, 255),
        (128, 128, 128),
        (185, 170, 155),
    )
    body: list[str] = []
    for number, name in labels.items():
        red, green, blue = colors[number % len(colors)]
        body.append(
            f'<Label Key="{number}" Red="{red / 255}" Green="{green / 255}" '
            f'Blue="{blue / 255}" Alpha="1"><![CDATA[{name}]]></Label>\n'
        )
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?> <CaretExtension>  '
        '<Date><![CDATA[2013-07-14T05:45:09]]></Date>   <VolumeInformation Index="0">   '
        "<LabelTable>\n" + "".join(body) + "\n  </LabelTable>  <StudyMetaDataLinkSet>  "
        "</StudyMetaDataLinkSet>  <VolumeType><![CDATA[Label]]></VolumeType>   "
        "</VolumeInformation></CaretExtension>\n              "
    )
    cast(list[object], image.header.extensions).append(Nifti1Extension(0, xml.encode()))


def require_folder(path: Path) -> None:
    """Reject an output path whose nearest existing parent is a file, not a folder."""
    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir():
        raise InvalidInputError(f"Output folder is a file: {parent}")


def validate_destination(path: Path, *, overwrite: bool) -> None:
    """Require a ``.nii``/``.nii.gz`` path that is new, or an existing file with ``overwrite``."""
    if not str(path).endswith((".nii", ".nii.gz")):
        raise InvalidInputError("Output must be a .nii or .nii.gz multilabel image")
    require_folder(path)
    if path.exists() and (not overwrite or not path.is_file()):
        raise InvalidInputError(f"Output already exists: {path}")


def write_segmentation(image: nib.Nifti1Image, path: Path, *, overwrite: bool) -> None:
    """Write the mask, creating parent directories; the format follows the file extension.

    The mask is written to ``.<pid>.<name>`` beside ``path`` and renamed into place, so a
    failed write never replaces an earlier mask.
    """
    validate_destination(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{os.getpid()}.{path.name}")
    try:
        save_nifti(image, partial)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)
