"""CPU resampling in voxel (x, y, z) order with SciPy nearest boundaries."""

from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from itertools import pairwise
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray


class SplineFilter(Protocol):
    """Declare the SciPy line filter that produces cubic spline coefficients."""

    def __call__(
        self,
        input: NDArray[np.float64],
        order: Literal[3],
        *,
        axis: int,
        output: NDArray[np.float64],
        mode: Literal["nearest"],
    ) -> NDArray[np.float64]:
        """Filter float64 input along one axis into the supplied output."""
        ...


class CoordinateSampler(Protocol):
    """Declare the SciPy sampler that evaluates cubic spline coefficients."""

    def __call__(
        self,
        input: NDArray[np.float64],
        coordinates: NDArray[np.float64],
        *,
        output: NDArray[np.float64],
        order: Literal[3],
        mode: Literal["nearest"],
        prefilter: Literal[False],
    ) -> NDArray[np.float64]:
        """Sample float64 spline coefficients at the supplied coordinates."""
        ...


def cubic_zoom(values: NDArray[np.float64], factors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Match SciPy's 3D cubic zoom with at most eight CPU workers.

    Filter full independent lines in SciPy's axis order using float64 spline
    coefficients. Output slabs sample global coordinates; splitting the input
    would change the cubic boundary conditions. ``values`` and ``factors`` are in voxel
    (x, y, z) order; each output extent is ``round(size * factor)``.
    """
    shape = tuple(round(size * factor) for size, factor in zip(values.shape, factors, strict=True))
    output = np.empty(shape, dtype=np.float64)
    if np.all(factors == 1):
        output[...] = values
        return output
    if not output.size:
        return output

    scale = np.divide(
        np.asarray(values.shape) - 1,
        np.asarray(shape) - 1,
        out=np.ones(3, dtype=np.float64),
        where=np.asarray(shape) != 1,
    )
    scipy = import_module("scipy.ndimage")
    filter_line = cast(SplineFilter, scipy.spline_filter1d)
    sample = cast(CoordinateSampler, scipy.map_coordinates)
    # SciPy nearest-boundary cubic zoom extends every input axis by twelve voxels.
    padding, workers = 12, 8
    coefficients = np.pad(values, padding, mode="edge").astype(np.float64, copy=False)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for axis in range(3):
            split_axis = 1 if axis == 0 else 0
            edges = [
                int(value)
                for value in np.linspace(0, coefficients.shape[split_axis], workers + 1, dtype=int)
            ]

            def filter_slab(
                bounds: tuple[int, int], *, axis: int = axis, split_axis: int = split_axis
            ) -> None:
                """Filter a disjoint coefficient slab along the current axis."""
                lower, upper = bounds
                if lower == upper:
                    return
                selection = [slice(None)] * 3
                selection[split_axis] = slice(lower, upper)
                view = coefficients[tuple(selection)]
                filter_line(view, 3, axis=axis, output=view, mode="nearest")

            list(pool.map(filter_slab, pairwise(edges)))

        coordinates_y = np.arange(shape[1], dtype=np.float64) * scale[1] + padding
        coordinates_z = np.arange(shape[2], dtype=np.float64) * scale[2] + padding

        def sample_slab(lower: int) -> None:
            """Sample global cubic coordinates for one output slab."""
            upper = min(lower + 8, shape[0])
            coordinates = np.empty((3, upper - lower, shape[1], shape[2]), dtype=np.float64)
            coordinates[0] = (np.arange(lower, upper, dtype=np.float64) * scale[0] + padding)[
                :, None, None
            ]
            coordinates[1] = coordinates_y[None, :, None]
            coordinates[2] = coordinates_z[None, None, :]
            sample(
                coefficients,
                coordinates,
                output=output[lower:upper],
                order=3,
                mode="nearest",
                prefilter=False,
            )

        list(pool.map(sample_slab, range(0, shape[0], 8)))
    return output


def nearest_labels(labels: NDArray[np.uint8], factors: NDArray[np.float64]) -> NDArray[np.uint8]:
    """Restore uint8 labels with SciPy's endpoint-aligned nearest coordinates.

    Order zero selects floor(coordinate + 0.5), including half-coordinate ties.
    Only three index vectors and disjoint uint8 output slabs are allocated.
    """
    shape = tuple(round(size * factor) for size, factor in zip(labels.shape, factors, strict=True))
    if shape == labels.shape:
        return labels.copy()
    indices: list[NDArray[np.intp]] = []
    for source, target in zip(labels.shape, shape, strict=True):
        scale = (source - 1) / (target - 1) if target != 1 else 1.0
        indices.append(np.floor(np.arange(target, dtype=np.float64) * scale + 0.5).astype(np.intp))
    output = np.empty(shape, dtype=np.uint8)

    def sample_slab(lower: int) -> None:
        """Copy nearest-neighbor label indices into one output slab."""
        upper = min(lower + 8, shape[0])
        output[lower:upper] = labels[np.ix_(indices[0][lower:upper], indices[1], indices[2])]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(sample_slab, range(0, shape[0], 8)))
    return output
