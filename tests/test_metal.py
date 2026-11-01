"""Run the registered runner end to end on Metal with tiny synthetic, untrained CT networks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import nibabel as nib
import numpy as np
from nnunet_mlx.model.network import PlainConvUNet
from nnunet_mlx.model.topology import PlainConvUNetSpec
from nnunet_mlx.runtime import import_mlx, runtime_info
from nnunet_mlx.storage.checkpoint import (
    CheckpointIdentity,
    InferenceCheckpointState,
    save_inference_checkpoint,
)

from totalsegmentator_mlx.imageio import load_nifti, save_nifti
from totalsegmentator_mlx.medmlx import run
from totalsegmentator_mlx.models import model_labels

if TYPE_CHECKING:
    from mlx.core import array


class SegmentationHead(Protocol):
    """Synthetic decoder output bias on Metal."""

    bias: array


class SyntheticNetwork(Protocol):
    """First head attached by the nnU-Net decoder builder."""

    seg0: SegmentationHead


PARTS = (
    "Dataset291_TotalSegmentator_part1_organs_1559subj",
    "Dataset292_TotalSegmentator_part2_vertebrae_1532subj",
    "Dataset293_TotalSegmentator_part3_cardiac_1559subj",
    "Dataset294_TotalSegmentator_part4_muscles_1559subj",
    "Dataset295_TotalSegmentator_part5_ribs_1559subj",
)


def _synthetic_plans() -> dict[str, object]:
    """Minimal nnU-Net plans for a one-channel 3d_fullres model with no mirroring."""
    resampling = {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None}
    return {
        "plans_name": "nnUNetPlans",
        "image_reader_writer": "SimpleITKIO",
        "label_manager": "LabelManager",
        "transpose_forward": [2, 0, 1],
        "transpose_backward": [1, 2, 0],
        "foreground_intensity_properties_per_channel": {"0": {}},
        "configurations": {
            "3d_fullres": {
                "data_identifier": "nnUNetPlans_3d_fullres",
                "batch_size": 2,
                "batch_dice": False,
                "patch_size": [8, 8, 8],
                "spacing": [1.5, 1.5, 1.5],
                "normalization_schemes": ["NoNormalization"],
                "use_mask_for_norm": [False],
                "resampling_fn_data": "resample_data_or_seg_to_shape",
                "resampling_fn_data_kwargs": resampling,
                "resampling_fn_seg": "resample_data_or_seg_to_shape",
                "resampling_fn_seg_kwargs": {**resampling, "is_seg": True, "order": 1},
                "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
                "resampling_fn_probabilities_kwargs": {**resampling, "order": 1},
                "architecture": {
                    "network_class_name": (
                        "dynamic_network_architectures.architectures.unet.PlainConvUNet"
                    ),
                    "arch_kwargs": {
                        "features_per_stage": [4, 8],
                        "kernel_sizes": [[3, 3, 3]] * 2,
                        "strides": [[1, 1, 1], [2, 2, 2]],
                        "n_conv_per_stage": [2, 2],
                        "n_conv_per_stage_decoder": [2],
                        "conv_bias": True,
                        "conv_op": "torch.nn.modules.conv.Conv3d",
                        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                        "nonlin": "torch.nn.modules.activation.LeakyReLU",
                        "nonlin_kwargs": {"negative_slope": 0.01},
                    },
                },
            }
        },
    }


def _write_synthetic_model(
    root: Path,
    task: int,
    dataset_name: str,
    *,
    trainer: str = "nnUNetTrainerNoMirroring",
    mirroring: tuple[int, ...] = (),
) -> None:
    """Write an untrained model folder (plans, dataset, checkpoint) for task ``task``.

    ``mirroring`` lists the test-time mirroring axes recorded in the checkpoint state.
    """
    plans = _synthetic_plans()
    plans["dataset_name"] = dataset_name
    dataset = {
        "channel_names": {"0": "CT"},
        "file_ending": ".nii.gz",
        "labels": {"background": 0, **{name: n for n, name in model_labels(task).items()}},
    }
    folder = root / str(task)
    folder.mkdir(parents=True)
    (folder / "plans.json").write_text(json.dumps(plans))
    (folder / "dataset.json").write_text(json.dumps(dataset))
    spec = PlainConvUNetSpec.from_plans(
        plans,
        configuration="3d_fullres",
        input_channels=1,
        num_classes=len(model_labels(task)) + 1,
    )
    model = PlainConvUNet(spec)
    mx, _ = import_mlx()
    # Favor label 1 so a synthetic crop model always localizes a nonempty region.
    head = cast(SyntheticNetwork, model).seg0
    head.bias = mx.array(
        np.asarray([0.0, 1e3] + [0.0] * (len(model_labels(task)) - 1), dtype=np.float32)
    )
    state = InferenceCheckpointState(
        identity=CheckpointIdentity(trainer, "3d_fullres", 0),
        current_epoch=0,
        spatial_dims=3,
        inference_allowed_mirroring_axes=mirroring,
        dropped_source_keys=(),
    )
    save_inference_checkpoint(folder / "fold_0" / "checkpoint_final", model=model, state=state)


def test_runner_body_crop_executes_five_parts_without_mirroring_on_metal(tmp_path: Path) -> None:
    """Run the body crop and five-part merge through the runner on Metal with untrained models."""
    assert runtime_info().backend == "metal"
    root = tmp_path / "synthetic_models"
    for task, name in zip(range(291, 296), PARTS, strict=True):
        _write_synthetic_model(root, task, name)
    # Dataset300 is released with mirroring; the native body crop must disable it.
    _write_synthetic_model(
        root, 300, "Dataset300_body_6mm_1559subj", trainer="nnUNetTrainer", mirroring=(0, 1, 2)
    )
    source = nib.Nifti1Image(np.full((16, 16, 16), 40, dtype=np.int16), np.diag([1.5, 1.5, 1.5, 1]))
    image = tmp_path / "ct.nii.gz"
    save_nifti(source, image)
    result = run(
        weights_path=str(root), image_path=str(image), output_dir=str(tmp_path), body_seg=True
    )
    mask = load_nifti(cast(dict[str, str], result["outputs"])["mask"])
    assert mask.shape == source.shape
    np.testing.assert_array_equal(mask.affine, source.affine)
    labels = np.asanyarray(mask.dataobj)
    # Every part favors its first local label; ribs (rib_left_1, global 92) merge last.
    assert labels.dtype == np.uint8 and np.all(labels == 92)
