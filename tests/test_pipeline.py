"""Runner argument rejection and model admission, before any inference."""

from pathlib import Path
from typing import NoReturn, cast

import nibabel as nib
import numpy as np
import pytest
from medmlx_core.errors import AssetNotReadyError, InvalidInputError
from numpy.typing import NDArray

from totalsegmentator_mlx import segment
from totalsegmentator_mlx.imageio import save_nifti
from totalsegmentator_mlx.models import Mode


@pytest.mark.parametrize("fault", ["identity", "labels", "weights"])
def test_model_admission_fails_before_input(
    tmp_path: Path, model_root: Path, fault: str, ct_data: NDArray[np.float32],
) -> None:
    """Reject incomplete or incompatible model assets before inference."""
    folder = model_root / "295"
    if fault == "identity":
        (folder / "plans.json").write_text('{"dataset_name": "DatasetOther"}')
    elif fault == "weights":
        (folder / "fold_0" / "checkpoint_final" / "weights.safetensors").unlink()
    else:
        (folder / "dataset.json").write_text('{"labels": {}, "channel_names": {"0": "MR"}}')
    image = tmp_path / "ct.nii.gz"
    save_nifti(nib.Nifti1Image(ct_data[200:208, 200:208, 40:48].astype(np.int16), np.eye(4)), image)
    with pytest.raises(AssetNotReadyError):
        segment(image_path=image, output_dir=tmp_path / "out", weights_path=model_root)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_image",
        "text_image",
        "unknown_mode",
        "string_boolean",
        "existing_output",
        "before_download",
    ],
)
def test_segment_rejects_bad_arguments(
    tmp_path: Path, model_root: Path, monkeypatch: pytest.MonkeyPatch, problem: str,
    ct_data: NDArray[np.float32],
) -> None:
    """Each rejection category fails before inference and before any weights download."""

    def download(model_id: str, **options: object) -> NoReturn:
        raise AssertionError(f"downloaded {model_id} before checking arguments")

    monkeypatch.setattr("medmlx_core.weights.weights_dir", download)
    image = tmp_path / "ct.nii.gz"
    data = ct_data[200:208, 200:208, 40:48].astype(np.int16)
    save_nifti(nib.Nifti1Image(data, np.diag([1.5] * 3 + [1])), image)
    output = tmp_path / "job"
    weights: str | None = str(model_root)
    mode = "standard"
    overwrite = False
    if problem == "missing_image":
        image = tmp_path / "absent.nii.gz"
    elif problem == "text_image":
        image = tmp_path / "ct.txt"
        image.write_text("not a CT")
    elif problem == "unknown_mode":
        mode = "fastest"
    elif problem == "string_boolean":
        overwrite = cast(bool, "false")
    elif problem == "existing_output":
        output.mkdir()
        (output / "mask.nii.gz").write_bytes(b"previous")
    else:
        weights, mode = None, "fastest"
    with pytest.raises(InvalidInputError):
        segment(
            weights_path=weights,
            image_path=str(image),
            output_dir=str(output),
            mode=cast(Mode, mode),
            overwrite=overwrite,
        )
