"""Outer CT resampling and reversible NIfTI orientation in voxel (x, y, z) order.

Volumes are 3D arrays in NIfTI storage order ``(x, y, z)``. ``canonical_ct`` reorients to
the closest RAS+ axes, ``resample_ct`` moves to an isotropic model grid (mm) and
``restore_labels`` returns labels to the source shape, orientation and affine.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Sequence
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import nibabel as nib
import numpy as np
from medmlx_core.errors import InvalidInputError, ModelExecutionError
from nibabel.filebasedimages import ImageFileError
from nibabel.spatialimages import HeaderDataError
from numpy.typing import NDArray

from totalsegmentator_mlx.imageio import load_nifti, spatial_affine
from totalsegmentator_mlx.resampling import cubic_zoom, oriented_nearest_labels

type Labels = NDArray[np.uint8]

# NiBabel and gzip failures for unreadable, truncated or non-NIfTI files.
_READ_ERRORS = (ImageFileError, HeaderDataError, OSError, EOFError, zlib.error)


class DtypeSetter(Protocol):
    """Declare the NiBabel header method that records a NIfTI data dtype."""

    def set_data_dtype(self, datatype: object) -> None:
        """Set the data dtype recorded by a NIfTI header."""
        ...


def open_ct(path: str | Path) -> nib.Nifti1Image:
    """Open a NIfTI CT lazily, reading only its header.

    Raises:
        InvalidInputError: The file is missing, unreadable or not a NIfTI image.
    """
    try:
        return load_nifti(path)
    except _READ_ERRORS as error:
        raise InvalidInputError(f"image_path is not a readable NIfTI image: {error}") from error


def load_ct(source: str | Path | nib.Nifti1Image) -> nib.Nifti1Image:
    """Load a finite, nonempty 3D CT with an invertible affine; voxel spacing is in mm.

    ``source`` is a path or an in-memory NIfTI-1 or NIfTI-2 image. Intensities are expected in
    Hounsfield units (not checked) and must be integer or float and finite. Spatial units
    must be millimeters or unspecified. Retain validated proxy values so canonicalization
    does not decode them again. Array-backed inputs are returned unchanged; caller data
    and headers are not modified.

    Raises:
        InvalidInputError: The file cannot be read as NIfTI, or the volume is not 3D, is
            empty or non-numeric, has non-finite values, has a non-invertible or non-finite
            affine, or has non-mm spatial units.
    """
    image = source if isinstance(source, nib.Nifti1Image) else open_ct(source)
    try:
        values = np.asanyarray(image.dataobj)
    except _READ_ERRORS as error:
        raise InvalidInputError(f"CT voxel data cannot be read: {error}") from error
    if values.ndim != 3 or not values.size or values.dtype.kind not in "iuf":
        raise InvalidInputError("Input must be a nonempty three-dimensional CT")
    if not np.isfinite(values).all():
        raise InvalidInputError("CT intensities must be finite")
    affine = image.affine
    if (
        affine is None
        or not np.isfinite(affine).all()
        or abs(float(np.linalg.det(affine[:3, :3]))) < 1e-12
        or not np.allclose(affine[3], [0, 0, 0, 1])
    ):
        raise InvalidInputError("CT requires a finite, invertible spatial affine")
    if image.header.get_xyzt_units()[0] not in {"unknown", "mm"}:
        raise InvalidInputError("CT spatial units must be millimeters")
    return (
        image
        if cast(object, image.dataobj) is values
        else nib.Nifti1Image(values, affine, image.header)
    )


def canonical_ct(source: nib.Nifti1Image) -> nib.Nifti1Image:
    """Reorient to the closest canonical (RAS+) axes, dropping the header.

    Only the voxel array and affine are kept (axes are flipped or permuted, nothing is
    interpolated). Restore the source orientation with ``restore_labels``.
    """
    # Match upstream's header-free staging before canonical reorientation.
    clean = nib.Nifti1Image(np.asanyarray(source.dataobj), source.affine)
    canonicalize = cast(
        Callable[[nib.Nifti1Image], object],
        import_module("nibabel").as_closest_canonical,
    )
    canonical = canonicalize(clean)
    if not isinstance(canonical, nib.Nifti1Image):
        raise InvalidInputError("Canonical CT must remain a NIfTI-1 image")
    return canonical


def resample_ct(image: nib.Nifti1Image, target: float) -> nib.Nifti1Image:
    """Stage cubic CT values on an isotropic ``target`` mm grid as int32, as upstream does.

    ``image`` is canonical-orientation CT whose header zooms are the voxel size in mm.
    Resampling is a cubic spline (SciPy ``zoom`` semantics, nearest boundary) in float64;
    results are then cast to int32, which truncates toward zero. When the spacing already
    equals ``target`` the values are only cast. Raises ``InvalidInputError`` if the field
    of view is smaller than one model voxel. The input validator does not check that
    intensities or spline overshoots fit in int32 before conversion.
    """
    spacing = np.asarray(image.header.get_zooms(), dtype=np.float64)
    factors = spacing / target
    shape = np.rint(np.asarray(image.shape) * factors)
    if np.any(shape < 1):
        raise InvalidInputError(f"CT field of view is too small for the {target} mm model grid")
    if np.array_equal(spacing, [target] * 3):
        values = np.asanyarray(image.dataobj).astype(np.int32)
    else:
        values = cubic_zoom(np.asarray(image.get_fdata(), dtype=np.float64), factors).astype(
            np.int32
        )
    affine = np.array(image.affine, copy=True)
    affine[:3, :3] /= factors
    return nib.Nifti1Image(values, affine)


def split_ct(
    image: nib.Nifti1Image, *, force_split: bool, multimodel: bool
) -> tuple[nib.Nifti1Image, ...]:
    """Use upstream's three overlapping z slabs for large multi-model volumes or on request.

    Splitting happens for multi-model runs whose grid exceeds 512*512*900 voxels with more
    than 200 slices, or when ``force_split``. The slabs cover the thirds of the z axis, each
    extended by a margin of about 20 slices into its neighbors (float64 values);
    ``join_slabs`` drops the margins again. Raises ``InvalidInputError`` if a split is
    requested for fewer than 60 slices.
    """
    depth = image.shape[2]
    split = force_split or (multimodel and np.prod(image.shape) > 512 * 512 * 900 and depth > 200)
    if not split:
        return (image,)
    if depth < 60:
        raise InvalidInputError("Splitting requires at least 60 slices on the model grid")
    third, margin = depth // 3, 20
    values = image.get_fdata()
    # Each slab retains the upstream staging affine; merge uses voxel indices.
    slices = (
        slice(None, third + margin),
        slice(third + 1 - margin, 2 * third + margin),
        slice(2 * third + 1 - margin, None),
    )
    return tuple(nib.Nifti1Image(values[:, :, item], image.affine) for item in slices)


def join_slabs(slabs: Sequence[Labels], shape: tuple[int, ...]) -> Labels:
    """Join one volume, or three ``split_ct`` slab predictions, into labels of ``shape``.

    Each adjacent pair overlaps by 39 slices. The merge discards 20 slices from
    the end of the earlier slab and 19 from the start of the later slab, so each
    z slice comes from exactly one prediction.
    """
    if len(slabs) == 1:
        return slabs[0]
    if len(slabs) != 3:
        raise ModelExecutionError("Expected one volume or three prediction slabs")
    third, margin = shape[2] // 3, 20
    joined = np.empty(shape, dtype=np.uint8)
    joined[:, :, :third] = slabs[0][:, :, :-margin]
    joined[:, :, third : 2 * third] = slabs[1][:, :, margin - 1 : -margin]
    joined[:, :, 2 * third :] = slabs[2][:, :, margin - 1 :]
    return joined


def restore_labels(
    labels: Labels,
    canonical: nib.Nifti1Image,
    source: nib.Nifti1Image,
) -> nib.Nifti1Image:
    """Restore nearest-neighbor labels to the exact source shape, affine and orientation.

    ``labels`` is on the resampled grid of ``canonical`` (the ``canonical_ct`` image of
    ``source``). Labels are scaled up to the canonical shape with nearest-neighbor
    sampling, then reoriented to match ``source``. Raises ``ModelExecutionError`` if the
    result does not have the source's shape and affine.
    """
    factors = np.asarray(canonical.shape) / np.asarray(labels.shape)
    orientations = import_module("nibabel.orientations")
    orientation = cast(Callable[[object], NDArray[np.float64]], orientations.io_orientation)
    ras = cast(Callable[[str], NDArray[np.float64]], orientations.axcodes2ornt)
    transform = cast(
        Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]],
        orientations.ornt_transform,
    )(ras("RAS"), orientation(source.affine))
    restored = oriented_nearest_labels(labels, factors, transform)
    # The affine ``as_reoriented`` gives the restored canonical-grid image.
    canonical_shape = tuple(
        round(size * factor) for size, factor in zip(labels.shape, factors, strict=True)
    )
    inverse = cast(
        Callable[[NDArray[np.float64], tuple[int, ...]], NDArray[np.float64]],
        orientations.inv_ornt_aff,
    )
    affine = spatial_affine(canonical).dot(inverse(transform, canonical_shape))
    if restored.shape != source.shape or not np.allclose(
        affine, spatial_affine(source), rtol=0, atol=1e-5
    ):
        raise ModelExecutionError("Restored segmentation geometry differs from its CT input")
    return source_labels(restored, source)


def source_labels(labels: Labels, source: nib.Nifti1Image) -> nib.Nifti1Image:
    """Wrap uint8 labels with the source affine and a copy of its header."""
    header = source.header.copy()
    cast(DtypeSetter, header).set_data_dtype(np.uint8)
    return nib.Nifti1Image(labels, source.affine, header)


def crop_ct(
    source: nib.Nifti1Image, mask: NDArray[np.bool_], margin_mm: float
) -> tuple[nib.Nifti1Image, tuple[slice, slice, slice]]:
    """Crop CT to a nonempty source-grid mask's bounding box plus a margin, staged as int32.

    The margin converts to whole voxels per source axis by truncation, using
    affine column norms as upstream's header-free copy does. Cropped float64
    values truncate to int32 before reorientation and resampling, and the
    affine translation moves to the first retained voxel.
    """
    probe = nib.Nifti1Image(np.zeros((1, 1, 1), dtype=np.uint8), source.affine)
    zooms = np.asarray(probe.header.get_zooms())
    margin = (np.full(3, margin_mm) / zooms).astype(int)
    occupied = [np.flatnonzero(mask.any(axis=other)) for other in ((1, 2), (0, 2), (0, 1))]
    lower = [max(0, int(axis[0]) - pad) for axis, pad in zip(occupied, margin, strict=True)]
    upper = [
        min(size, int(axis[-1]) + 1 + pad)
        for axis, pad, size in zip(occupied, margin, mask.shape, strict=True)
    ]
    box = (slice(lower[0], upper[0]), slice(lower[1], upper[1]), slice(lower[2], upper[2]))
    values = np.asarray(np.asanyarray(source.dataobj)[box], dtype=np.float64).astype(np.int32)
    affine = np.array(source.affine, dtype=np.float64, copy=True)
    affine[:3, 3] = (affine @ np.array([*lower, 1], dtype=np.float64))[:3]
    return nib.Nifti1Image(values, affine), box
