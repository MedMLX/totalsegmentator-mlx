"""Runner argument rejection and model admission, before any inference."""

from pathlib import Path
from typing import NoReturn, cast

import nibabel as nib
import numpy as np
import pytest
from medmlx_core.errors import AssetNotReadyError, InvalidInputError

from totalsegmentator_mlx import pipeline
from totalsegmentator_mlx.imageio import save_nifti
from totalsegmentator_mlx.medmlx import run


@pytest.mark.parametrize("fault", ["identity", "labels", "weights"])
def test_model_admission_fails_before_input(model_root: Path, fault: str) -> None:
    """Reject incomplete or incompatible model assets before inference."""
    folder = model_root / "295"
    if fault == "identity":
        (folder / "plans.json").write_text('{"dataset_name": "DatasetOther"}')
    elif fault == "weights":
        (folder / "fold_0" / "checkpoint_final" / "weights.safetensors").unlink()
    else:
        (folder / "dataset.json").write_text('{"labels": {}, "channel_names": {"0": "MR"}}')
    image = nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.int16), np.eye(4))
    with pytest.raises(AssetNotReadyError):
        pipeline.segment(image, model_dir=model_root)


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
def test_run_rejects_bad_arguments(
    tmp_path: Path, model_root: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    """Each rejection category fails before inference and before any weights download."""

    def download(model_id: str) -> NoReturn:
        raise AssertionError(f"downloaded {model_id} before checking arguments")

    monkeypatch.setattr("medmlx_core.weights.weights_dir", download)
    image = tmp_path / "ct.nii.gz"
    save_nifti(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.int16), np.diag([1.5] * 3 + [1])), image)
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
        run(
            weights_path=weights,
            image_path=str(image),
            output_dir=str(output),
            mode=mode,
            overwrite=overwrite,
        )
