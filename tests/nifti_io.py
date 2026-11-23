"""Typed NiBabel loader for tests that inspect written NIfTI files and headers."""

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import cast

import nibabel as nib


def load_nifti(path: str | Path) -> nib.Nifti1Image:
    """Load a NIfTI file lazily with its header and affine."""
    loader = cast(Callable[[str], object], import_module("nibabel").load)
    image = loader(str(path))
    assert isinstance(image, nib.Nifti1Image)
    return image
