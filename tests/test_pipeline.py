"""Runner argument rejection and model admission, before any inference."""

import os
from pathlib import Path
from typing import NoReturn, cast

import nibabel as nib
import numpy as np
import pytest
from medmlx_core.errors import AssetNotReadyError, InvalidInputError
from nnunet_mlx import NiftiPrediction

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


def _probability_result(image: nib.Nifti1Image, folder: Path, task_id: int) -> NiftiPrediction:
    """Supply owned output arrays at the engine boundary for publication-only tests."""
    from totalsegmentator_mlx.models import model_labels

    labels = np.ones(image.shape, dtype=np.uint8)
    probabilities = np.zeros((len(model_labels(task_id)) + 1, *image.shape[::-1]), dtype=np.float32)
    probabilities[1] = 1
    return NiftiPrediction(nib.Nifti1Image(labels, image.affine), probabilities,
                           {"spacing": (3.0, 3.0, 3.0), "inputs": []})


@pytest.mark.parametrize("failed_artifact", ["probabilities", "properties"])
def test_probability_serialization_failure_preserves_every_destination(
    tmp_path: Path, model_root: Path, monkeypatch: pytest.MonkeyPatch, failed_artifact: str,
) -> None:
    """A failed write leaves previous mask/NPZ/PKL bytes and removes the private staging area."""
    monkeypatch.setattr(pipeline, "predict_probabilities", _probability_result)
    source = nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.diag([3] * 3 + [1]))
    mask, probabilities = tmp_path / "mask.nii.gz", tmp_path / "probabilities.npz"
    targets = (mask, probabilities, probabilities.with_suffix(".pkl"))
    for target in targets:
        target.write_bytes(target.name.encode())

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("injected serialization failure")

    if failed_artifact == "probabilities":
        monkeypatch.setattr(pipeline.np, "savez_compressed", fail)
    else:
        monkeypatch.setattr(pipeline.pickle, "dump", fail)
    with pytest.raises(OSError, match="serialization"):
        pipeline.segment(source, model_dir=model_root, mode="fast", output_path=mask,
                         probabilities_path=probabilities, overwrite=True)
    expected = [target.name.encode() for target in targets]
    assert [target.read_bytes() for target in targets] == expected
    assert not list(tmp_path.glob(".totalsegmentator-*"))


@pytest.mark.parametrize("failed_output", ["properties", "mask"])
def test_probability_publication_failure_restores_prior_files(
    tmp_path: Path, model_root: Path, monkeypatch: pytest.MonkeyPatch, failed_output: str
) -> None:
    monkeypatch.setattr(pipeline, "predict_probabilities", _probability_result)
    source = nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.diag([3] * 3 + [1]))
    mask, probabilities = tmp_path / "mask.nii.gz", tmp_path / "probabilities.npz"
    properties = probabilities.with_suffix(".pkl")
    old = {path: path.name.encode() for path in (mask, probabilities, properties)}
    for path, contents in old.items():
        path.write_bytes(contents)
    replace = os.replace
    failed = False

    failing_path = properties if failed_output == "properties" else mask

    def fail_publication(staged: Path | str, destination: Path | str) -> None:
        nonlocal failed
        if destination == failing_path and not failed:
            failed = True
            raise OSError("injected publication failure")
        replace(staged, destination)

    monkeypatch.setattr(pipeline.os, "replace", fail_publication)
    with pytest.raises(OSError, match="injected publication failure"):
        pipeline.segment(
            source, model_dir=model_root, mode="fast", output_path=mask,
            probabilities_path=probabilities, overwrite=True,
        )
    for path, contents in old.items():
        assert path.read_bytes() == contents
    assert not list(tmp_path.glob(".totalsegmentator-*"))


def test_probability_publication_replaces_links_and_rejects_input_output_collisions(
    tmp_path: Path, model_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication replaces destination links while preflight protects aliased user data."""
    monkeypatch.setattr(pipeline, "predict_probabilities", _probability_result)
    source = tmp_path / "ct.nii.gz"
    save_nifti(nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.float32), np.diag([3] * 3 + [1])),
               source)
    probabilities = tmp_path / "probabilities.npz"
    properties = probabilities.with_suffix(".pkl")
    old = tmp_path / "user-data"
    old.write_bytes(b"preserve")
    for target in (probabilities, properties):
        target.symlink_to(old)
    # Both outputs initially alias one another; reject before replacing either.
    with pytest.raises(InvalidInputError):
        pipeline.segment(source, model_dir=model_root, mode="fast",
                         probabilities_path=probabilities, overwrite=True)
    properties.unlink()
    other = tmp_path / "other-user-data"
    other.write_bytes(b"also preserve")
    properties.symlink_to(other)
    pipeline.segment(source, model_dir=model_root, mode="fast",
                     probabilities_path=probabilities, overwrite=True)
    assert not probabilities.is_symlink() and not properties.is_symlink()
    assert old.read_bytes() == b"preserve" and other.read_bytes() == b"also preserve"
    with np.load(probabilities) as arrays:
        assert arrays["probabilities"].dtype == np.float32
    probabilities.unlink()
    probabilities.hardlink_to(source)
    before = source.read_bytes()
    with pytest.raises(InvalidInputError):
        pipeline.segment(source, model_dir=model_root, mode="fast",
                         probabilities_path=probabilities, overwrite=True)
    assert source.read_bytes() == before
