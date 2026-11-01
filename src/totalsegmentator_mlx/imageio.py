"""Typed NIfTI boundaries for NiBabel's partially annotated load/save API."""

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import nibabel as nib
import numpy as np
from medmlx_core.errors import InvalidInputError
from numpy.typing import NDArray


class ImageWriter(Protocol):
    """Declare the NiBabel image method that writes an image to a filename."""

    def to_filename(self, filename: Path) -> None:
        """Write the image to a filename using its format."""
        ...


def load_nifti(path: str | Path) -> nib.Nifti1Image:
    """Load a NIfTI file lazily (voxels stay an array proxy).

    Accepts both NIfTI-1 and NIfTI-2 files: NiBabel's ``Nifti2Image`` subclasses
    ``Nifti1Image`` and exposes the same affine/header/dataobj interface, and
    upstream TotalSegmentator's own loader (``nifti_io.require_nifti1``) applies the
    identical ``isinstance(image, Nifti1Image)`` check, so this matches its behavior.

    Raises ``InvalidInputError`` if NiBabel returns a non-NIfTI image. File and
    decoding errors from NiBabel propagate.
    """
    loader = cast(Callable[[str], object], import_module("nibabel").load)
    image = loader(str(path))
    if not isinstance(image, nib.Nifti1Image):
        raise InvalidInputError("Input must be a NIfTI image")
    return image


def save_nifti(image: nib.Nifti1Image, path: Path) -> None:
    """Write ``image`` to ``path``; compression follows the file extension (``.nii.gz``)."""
    cast(ImageWriter, image).to_filename(path)


def spatial_affine(image: nib.Nifti1Image) -> NDArray[np.float64]:
    """Return the 4x4 voxel-to-world affine as float64, without converting units.

    The caller must interpret or validate the header's spatial units.
    """
    affine = image.affine
    if affine is None:
        raise InvalidInputError("NIfTI image requires a spatial affine")
    return np.asarray(affine, dtype=np.float64)
