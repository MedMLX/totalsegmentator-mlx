"""External CT and model-metadata fixtures."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from totalsegmentator_mlx.imageio import load_nifti
from totalsegmentator_mlx.models import TASK_SPACING, model_labels


@pytest.fixture(scope="session")
def ct_data() -> NDArray[np.float32]:
    """Read the external HCC-TACE-Seg HCC_063 contrast image."""
    default = Path.home() / ".cache" / "medmlx" / "test-data" / "hcc_063_ct.nii.gz"
    path = os.environ.get("MEDMLX_TEST_CT", str(default))
    return np.asarray(load_nifti(path).dataobj, dtype=np.float32)


@pytest.fixture
def model_root(tmp_path: Path) -> Path:
    """Model root with empty checkpoint files and matching metadata for every model used.

    The folders pass admission (``validate_models``) but hold no real network, so tests
    must stub the engine.
    """
    root = tmp_path / "models"
    for task_id in TASK_SPACING:
        labels = model_labels(task_id)
        folder = root / str(task_id)
        weights = folder / "fold_0" / "checkpoint_final" / "weights.safetensors"
        weights.parent.mkdir(parents=True)
        weights.touch()
        (weights.parent / "state.json").write_text("{}")
        (folder / "plans.json").write_text(json.dumps({"dataset_name": f"Dataset{task_id}_CT"}))
        (folder / "dataset.json").write_text(
            json.dumps(
                {
                    "channel_names": {"0": "CT"},
                    "labels": {"background": 0, **{name: value for value, name in labels.items()}},
                }
            )
        )
    return root
