"""Global label merging and the NIfTI label table of the segmentation output."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import nibabel as nib
import numpy as np
from medmlx_core.errors import ModelExecutionError
from nibabel.nifti1 import Nifti1Extension

from totalsegmentator_mlx.geometry import Labels
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
