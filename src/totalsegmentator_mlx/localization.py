"""Upstream auxiliary body and ROI localization masks on the source CT grid."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import cast

import numpy as np
from numpy.typing import NDArray

from totalsegmentator_mlx.geometry import Labels

# Upstream crop margins in mm: 20 around an ROI subset, 3 around the body.
ROI_MARGIN_MM = 20.0
BODY_MARGIN_MM = 3.0
# Upstream removes body_extremities components up to 50,000 mm^3.
_EXTREMITY_MIN_MM3 = np.float32(50000)


def _components(mask: NDArray[np.bool_]) -> tuple[NDArray[np.int32], int]:
    """Label 6-connected components with SciPy's default structure."""
    label = cast(
        Callable[[NDArray[np.bool_]], tuple[NDArray[np.int32], int]],
        import_module("scipy.ndimage").label,
    )
    return label(mask)


def clean_body(labels: Labels, zooms: tuple[float, ...]) -> Labels:
    """Apply upstream body postprocessing on the 6 mm model grid, in place.

    Keep only the largest 6-connected body_trunc (1) component, the first on
    ties, and remove body_extremities (2) components at or below 50,000 mm^3
    or larger than 10 billion voxels.
    ``zooms`` are the model grid's float32 NIfTI header zooms.
    """
    trunc, extremities = 1, 2
    components, count = _components(labels == trunc)
    if count:
        sizes = np.bincount(np.ravel(components))[1:]
        labels[(labels == trunc) & (components != np.argmax(sizes) + 1)] = 0
    components, count = _components(labels == extremities)
    if count:
        sizes = np.bincount(np.ravel(components))
        lower = _EXTREMITY_MIN_MM3 / np.prod(np.asarray(zooms, dtype=np.float32))
        # Sizes are voxel counts; 1e10 is the upper bound of upstream's blob-size interval.
        removed = (sizes <= lower) | (sizes > 1e10)
        labels[(labels == extremities) & removed[components]] = 0
    return labels


def localization_mask(labels: Labels, roi_ids: set[int] | None) -> NDArray[np.bool_]:
    """Select ROI labels from a rough total segmentation, or any body label.

    ``labels`` is a uint8 label volume on the source grid. ``roi_ids`` are global label IDs
    from the rough ``total`` segmentation; ``None`` selects every nonzero body label.
    Returns a boolean mask on the same grid.
    """
    return labels > 0 if roi_ids is None else np.isin(labels, list(roi_ids))
