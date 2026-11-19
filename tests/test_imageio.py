"""NIfTI serialization through the single-member gzip writer."""

import gzip
import zlib
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

from totalsegmentator_mlx.imageio import ImageWriter, load_nifti, save_nifti


def test_parallel_gzip_matches_serial_nifti(tmp_path: Path, ct_data: NDArray[np.float32]) -> None:
    """An output crossing several deflate chunks must decode fully in its first member."""
    voxels = ct_data[:256, :256, :53]
    bins = [-1000, -600, -100, 0, 50, 200]
    labels = np.asfortranarray(np.digitize(voxels, bins).astype(np.uint8))
    image = nib.Nifti1Image(labels, np.diag([0.8, 0.8, 2.5, 1.0]))
    serial, parallel = tmp_path / "serial.nii.gz", tmp_path / "parallel.nii.gz"
    cast(ImageWriter, image).to_filename(serial)
    save_nifti(image, parallel)

    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    assert decoder.decompress(parallel.read_bytes()) == gzip.decompress(serial.read_bytes())
    assert decoder.eof and not decoder.unused_data
    np.testing.assert_array_equal(load_nifti(parallel).dataobj, labels)
