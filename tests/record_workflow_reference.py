"""Record TotalSegmentator 2.12.0 outer workflows on an HCC-TACE CT crop."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from importlib import import_module, metadata
from pathlib import Path
from typing import Protocol, TypedDict, cast

import nibabel as nib
import numpy as np


class Options(TypedDict, total=False):
    fast: bool
    fastest: bool
    roi_subset: list[str]
    body_seg: bool


class UpstreamApi(Protocol):
    download_pretrained_weights: Callable[[int], None]

    def totalsegmentator(
        self, image: str, output: None, *, ml: bool, task: str, device: str,
        nr_thr_resamp: int, nr_thr_saving: int, quiet: bool, **options: object,
    ) -> object: ...


class NnunetModule(Protocol):
    nnUNetv2_predict: object


class LabelMaps(Protocol):
    class_map: Mapping[str, Mapping[int, str]]
    class_map_5_parts: Mapping[str, Mapping[int, str]]
    map_taskid_to_partname_ct: Mapping[int, str]


def load_image(path: Path) -> nib.Nifti1Image:
    """Read a NIfTI-1 CT through the installed upstream NiBabel."""
    loader = cast(Callable[[str], object], import_module("nibabel").load)
    image = loader(str(path))
    if not isinstance(image, nib.Nifti1Image):
        raise ValueError(f"Expected a NIfTI CT at {path}")
    return image


def save_image(image: nib.Nifti1Image, path: Path) -> None:
    """Write a NIfTI image without changing its header."""
    writer = cast(Callable[[nib.Nifti1Image, str], None], import_module("nibabel").save)
    writer(image, str(path))


def record(ct_path: Path, output: Path) -> None:
    """Run pinned upstream resampling and merging with controlled network labels."""
    if metadata.version("TotalSegmentator") != "2.12.0":
        raise ValueError("The workflow reference requires TotalSegmentator 2.12.0")
    ct = load_image(ct_path)
    if ct.affine is None:
        raise ValueError("The HCC-TACE CT must have a spatial affine")
    start = np.array([150, 250, 30])
    data = np.asarray(ct.dataobj)[150:190, 250:290, 30:90]
    if data.shape != (40, 40, 60):
        raise ValueError(f"Unexpected CT crop shape {data.shape}")
    affine = ct.affine.copy()
    affine[:3, 3] += affine[:3, :3] @ start
    images = {
        "anisotropic": nib.Nifti1Image(data, affine, header=ct.header),
        "permuted": nib.Nifti1Image(
            data.transpose(1, 0, 2), affine[:, [1, 0, 2, 3]], header=ct.header,
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        save_image(image, output / f"{name}_ct.nii.gz")

    with tempfile.TemporaryDirectory(prefix="totalseg-reference-") as temporary:
        config = Path(temporary)
        os.environ["TOTALSEG_HOME_DIR"] = temporary
        os.environ["TOTALSEG_WEIGHTS_PATH"] = temporary
        (config / "config.json").write_text(json.dumps({
            "totalseg_id": "reference", "send_usage_stats": False,
            "prediction_counter": 0, "statistics_disclaimer_shown": True,
        }))
        api = cast(UpstreamApi, import_module("totalsegmentator.python_api"))
        nnunet = cast(NnunetModule, import_module("totalsegmentator.nnunet"))
        maps = cast(LabelMaps, import_module("totalsegmentator.map_to_binary"))

        def no_download(task: int) -> None:
            """Network inference is controlled at the upstream folder boundary."""

        def predict(
            directory: Path, destination: Path, task: int, *args: object, **kwargs: object,
        ) -> None:
            labels = (
                maps.class_map_5_parts[maps.map_taskid_to_partname_ct[task]]
                if task in range(291, 296)
                else maps.class_map["body" if task == 300 else "total"]
            )
            for source in Path(directory).glob("*_0000.nii.gz"):
                image = load_image(source)
                values = np.asarray(image.dataobj, dtype=np.int64)
                prediction = ((values + task) % (max(labels) + 1)).astype(np.uint8)
                path = Path(destination) / source.name.replace("_0000.nii.gz", ".nii.gz")
                save_image(nib.Nifti1Image(prediction, image.affine), path)

        api.download_pretrained_weights = no_download
        nnunet.nnUNetv2_predict = predict
        cases: tuple[tuple[str, str, Options], ...] = (
            ("anisotropic", "anisotropic", {}),
            ("permuted", "permuted", {}),
            ("anisotropic", "anisotropic_coarse", {"fastest": True}),
            ("permuted", "permuted_fast", {"fast": True}),
            ("anisotropic", "anisotropic_roi", {
                "roi_subset": ["kidney_right", "liver", "vertebrae_T12"],
            }),
            ("permuted", "permuted_body", {"body_seg": True}),
        )
        for source, name, options in cases:
            print(f"Recording {name}", flush=True)
            result = api.totalsegmentator(
                str(output / f"{source}_ct.nii.gz"), None, ml=True, task="total",
                device="cpu", nr_thr_resamp=1, nr_thr_saving=1, quiet=True, **options,
            )
            if not isinstance(result, nib.Nifti1Image):
                raise TypeError("TotalSegmentator did not return a NIfTI image")
            save_image(result, output / f"{name}_reference.nii.gz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    record(arguments.ct, arguments.output_dir)


if __name__ == "__main__":
    main()
