"""Security behavior at the segmentation output boundary."""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from medmlx_core.errors import InvalidInputError

from totalsegmentator_mlx.imageio import load_nifti
from totalsegmentator_mlx.output import write_segmentation


def test_writer_does_not_follow_a_staging_symlink(tmp_path: Path) -> None:
    """A writable output folder cannot redirect the mask write to another file."""
    victim = tmp_path / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    (tmp_path / f".{os.getpid()}.mask.nii.gz").symlink_to(victim)
    path = tmp_path / "mask.nii.gz"
    image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4))

    write_segmentation(image, path, overwrite=False)

    assert victim.read_bytes() == b"keep"
    assert load_nifti(path).shape == image.shape


def test_writer_rejects_a_dangling_destination_symlink(tmp_path: Path) -> None:
    """An existing link counts as output even when its target is missing."""
    path = tmp_path / "mask.nii.gz"
    path.symlink_to(tmp_path / "missing.nii.gz")
    image = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), np.eye(4))

    with pytest.raises(InvalidInputError, match="already exists"):
        write_segmentation(image, path, overwrite=False)

    assert path.is_symlink()
