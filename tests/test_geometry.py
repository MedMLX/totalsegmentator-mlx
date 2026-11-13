"""Resampling parity with the SciPy zoom calls upstream TotalSegmentator makes."""

from importlib import import_module
from typing import Literal, Protocol, cast

import nibabel as nib
import numpy as np
import pytest
from numpy.typing import NDArray

from totalsegmentator_mlx.geometry import resample_ct, split_ct
from totalsegmentator_mlx.resampling import cubic_zoom, nearest_labels


class ScipyZoom(Protocol):
    """The float64 SciPy zoom calls used as independent resampling references."""

    def __call__(
        self,
        input: NDArray[np.float64],
        zoom: NDArray[np.float64],
        *,
        order: Literal[0, 3],
        mode: Literal["nearest"],
    ) -> NDArray[np.float64]:
        """Run SciPy zoom as the independent resampling reference."""
        ...


class SerializedImage(Protocol):
    """NIfTI serialization without optional writer keyword arguments."""

    def to_bytes(self) -> bytes:
        """Serialize a NIfTI image to bytes."""
        ...


scipy_zoom = cast(ScipyZoom, import_module("scipy.ndimage").zoom)


@pytest.mark.parametrize("order", [0, 3])
@pytest.mark.parametrize("factor", [None, 1.0, 0.99])
@pytest.mark.parametrize(
    "shape,target",
    [((1, 3, 7), (5, 1, 9)), ((17, 21, 6), (27, 18, 17)), ((16, 15, 14), (7, 11, 3))],
)
def test_resampling_matches_scipy_coordinates_and_cubic_coefficients(
    order: Literal[0, 3],
    factor: float | None,
    shape: tuple[int, int, int],
    target: tuple[int, int, int],
    ct_data: NDArray[np.float32],
) -> None:
    """Compare nearest and cubic resampling exactly with SciPy across shapes."""
    image = ct_data[200:200 + shape[0], 200:200 + shape[1], 40:40 + shape[2]]
    source = np.asarray(image > 50, dtype=np.uint8) if order == 0 else image.astype(">f8")
    source = source.transpose(2, 1, 0)[::-1]
    factors = np.asarray(target) / source.shape if factor is None else np.full(3, factor)
    if order == 0:
        # The original outer workflow sampled a float64 copy of the uint8 labels.
        expected = scipy_zoom(source.astype(np.float64), factors, order=0, mode="nearest")
        expected = expected.astype(np.uint8)
        actual = nearest_labels(source, factors)
    else:
        float_source = cast(NDArray[np.float64], source)
        expected = scipy_zoom(float_source, factors, order=3, mode="nearest")
        actual = cubic_zoom(float_source, factors)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.view(np.uint8), expected.view(np.uint8))


def test_ct_resampling_preserves_scaled_int32_values_and_affine(
    ct_data: NDArray[np.float32],
) -> None:
    """Compare scaled CT resampling and output affine with SciPy."""
    values = ct_data[200:207, 200:209, 40:50].astype(np.int16)
    affine = np.array([[0, -2.3, 0, 11], [1.7, 0, 0, -8], [0, 0, 3.1, 4], [0, 0, 0, 1]])
    source = nib.Nifti1Image(values, affine)
    source.header["scl_slope"], source.header["scl_inter"] = 1.5, -1024
    source = nib.Nifti1Image.from_bytes(cast(SerializedImage, source).to_bytes())
    factors = np.asarray(source.header.get_zooms(), dtype=np.float64) / 1.5
    decoded = cast(NDArray[np.float64], source.get_fdata())
    expected = scipy_zoom(decoded, factors, order=3, mode="nearest").astype(np.int32)
    expected_affine = np.array(source.affine, copy=True)
    expected_affine[:3, :3] /= factors

    actual = resample_ct(source, 1.5)

    assert actual.get_data_dtype() == np.dtype(np.int32)
    np.testing.assert_array_equal(actual.dataobj, expected)
    np.testing.assert_array_equal(actual.affine, expected_affine)


@pytest.mark.parametrize("order", ["C", "F"])
def test_split_ct_keeps_native_integer_slab_views(
    order: Literal["C", "F"], ct_data: NDArray[np.float32],
) -> None:
    """Split an already resampled CT without allocating another full-volume dtype."""
    values = np.array(ct_data[200:203, 200:205, :90], dtype=np.int32, order=order)
    image = nib.Nifti1Image(values, np.diag([1.5, 1.5, 1.5, 1.0]))
    slabs = split_ct(image, force_split=True, multimodel=True)
    for slab, interval in zip(slabs, (slice(0, 50), slice(11, 80), slice(41, 90)), strict=True):
        actual = np.asanyarray(slab.dataobj)
        assert actual.dtype == np.int32
        assert np.shares_memory(actual, values)
        np.testing.assert_array_equal(actual, values[:, :, interval])
        np.testing.assert_array_equal(slab.affine, image.affine)
