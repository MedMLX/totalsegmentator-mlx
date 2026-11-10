<div align="center">

# totalsegmentator-mlx

Segment 117 CT structures with TotalSegmentator v2 in native MLX. Includes standard, fast and coarse modes.

[![PyPI](https://img.shields.io/pypi/v/totalsegmentator-mlx)](https://pypi.org/project/totalsegmentator-mlx/)
[![Python](https://img.shields.io/pypi/pyversions/totalsegmentator-mlx)](https://pypi.org/project/totalsegmentator-mlx/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://github.com/MedMLX/totalsegmentator-mlx/blob/v0.1.9/LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MLX-black?logo=apple)](https://github.com/ml-explore/mlx)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-weights-yellow)](https://huggingface.co/MedMLX/totalsegmentator-mlx)
[![MedMLX](https://img.shields.io/badge/MedMLX-runtime-A51C30)](https://github.com/MedMLX/medmlx)

[Model weights](https://huggingface.co/MedMLX/totalsegmentator-mlx) · [MedMLX](https://github.com/MedMLX/medmlx)

<img src="docs/example.png" alt="TotalSegmentator standard-mode (1.5 mm) 3D surfaces and axial CT segmentation" width="100%">

Standard mode (1.5 mm): 3D surfaces from [autoPET-III](https://huggingface.co/datasets/YongchengYAO/autoPET-III-Lite) and axial CT from [AMOS22](https://zenodo.org/records/7262581). Surfaces and contours are smoothed for display.

</div>

| Input | Output |
| :--- | :--- |
| 3D CT · NIfTI | Multilabel mask on the input grid |

| Model | Weights download | Peak MLX memory | Measured on |
| :--- | ---: | ---: | :--- |
| `totalsegmentator` | 824 MB | 5.0 GB | Standard mode, 768×768×101 abdominal CT |

## Install

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "totalsegmentator-mlx>=0.1.9"
```

## Run

```python
from totalsegmentator_mlx.medmlx import run

result = run(image_path="ct.nii.gz", output_dir="out")
print(result["outputs"]["mask"])
```

Weights download from `MedMLX/totalsegmentator-mlx` on first use. Use `medmlx setup totalsegmentator` to download ahead of time. For local weights, pass the downloaded bundle folder as `weights_path`.

### Inputs and outputs

| Argument | Meaning |
| --- | --- |
| `image_path` | One finite 3D CT in NIfTI-1 or NIfTI-2, in Hounsfield units |
| `output_dir` | Directory for `mask.nii.gz` |
| `weights_path` | Optional root of the model folders below |
| `mode` | `standard` (default, five 1.5 mm parts), `fast` (3 mm model, upstream `--fast`) or `coarse` (6 mm model, upstream `--fastest`) |
| `prompt` | Structure names to keep after inference; label IDs stay global |
| `roi_subset` | Locate these structures, crop to them plus 20 mm and drop other labels |
| `robust_crop` | Locate `roi_subset` with the 3 mm model instead of the 6 mm one |
| `body_seg` | Crop to the Dataset300 body model plus 3 mm |
| `overwrite` | `False` by default; an existing mask is never replaced |

- **Cropping:** with `roi_subset`, standard mode runs only the parts that contain those
  structures. `roi_subset` cannot be combined with `prompt` or `body_seg`, and
  `robust_crop` requires `roi_subset`. Cropped regions are restored by zero filling; an
  empty localization returns an empty mask.
- **Units:** spatial units must be millimetres or unspecified.
- **Mask:** a uint8 NIfTI-1 with global labels 1–117 (0 background) on the input grid,
  with the upstream label table as a header extension.
- **Return value:** `{"outputs": {"mask": path}, "details": {...}}`, with run metadata in
  `details`.
- **Errors:** invalid inputs raise `InvalidInputError`; missing or mismatched weights
  raise `AssetNotReadyError`.

### Model folders

A local `weights_path` holds one folder per dataset, each with `plans.json`,
`dataset.json` and `fold_0/checkpoint_final/` containing `state.json` and
`weights.safetensors`. Only the folders the chosen options use are required, and a
single model folder can be passed directly when only that dataset is needed.

| Folder | Model | Used by |
| --- | --- | --- |
| `291/`–`295/` | five 1.5 mm parts | `standard` |
| `297/` | 3 mm total model | `fast`; `roi_subset` with `robust_crop` |
| `298/` | 6 mm total model | `coarse`; `roi_subset` |
| `300/` | 6 mm body model | `body_seg` |

### Command line

Download the [safetensors model bundle](https://huggingface.co/MedMLX/totalsegmentator-mlx),
then segment:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "MedMLX/totalsegmentator-mlx",
    revision="6c03d6f90ccf53f224c3985cbdecf1eaa96555af",
    local_dir="models",
)
PY
totalsegmentator-mlx predict -i ct.nii.gz -o mask.nii.gz --model-dir models
```

`predict` takes the runner options as flags (`--mode`, `--organ` for `prompt`,
`--roi-subset`, `--robust-crop`, `--body-seg`, `--overwrite`) and adds three more:

| Flag | Effect |
| :--- | :--- |
| `--model-part` | Runs one standard part: `organs`, `vertebrae`, `cardiac`, `muscles` or `ribs` |
| `--force-split` | Uses upstream's three overlapping slabs |
| `--probabilities-path probabilities.npz` | Also exports local-label probabilities for one unsplit, uncropped model |

`totalsegmentator-mlx labels` prints the label IDs.

### Python API

```python
from totalsegmentator_mlx import segment

labels = segment("ct.nii.gz", model_dir="models", mode="fast")  # nibabel image, uint8
```

`segment` takes the command-line options as keyword arguments, plus an optional
`output_path`. It never downloads weights.

### Limits

- Supports the `total` CT task: Dataset291–295, 297 and 298 (117 labels), plus
  Dataset300 for body cropping.
- Models use `3d_fullres`, fold 0, without mirroring.
- Input and output are NIfTI. Other TotalSegmentator tasks and its extra export,
  statistics and training options are unsupported.
- Inference requires Apple Silicon with Metal.

## Use with MedMLX

Use the [MedMLX MCP server](https://github.com/MedMLX/medmlx/blob/main/docs/mcp.md) to run `totalsegmentator` from an MCP client. Weights download on first use.

```sh
uv tool install --python 3.13 "medmlx[segment,mcp]"
MEDMLX="$(uv tool dir --bin --color never)/medmlx"
mkdir -p "$HOME/medmlx-work"
claude mcp add --scope user medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"
```

For Codex, use `codex mcp add medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"`. Other MCP clients take the same command and arguments. The same install provides the `medmlx` CLI, for example `medmlx describe totalsegmentator`.

> **For AI agents:** prefer the MedMLX MCP tools (`list_models`, `describe_model`, `model_readiness`, `run_model`, `get_job`) over writing Python against this package.

## License

[Apache-2.0](https://github.com/MedMLX/totalsegmentator-mlx/blob/v0.1.9/LICENSE). See [third-party notices](https://github.com/MedMLX/totalsegmentator-mlx/blob/v0.1.9/THIRD_PARTY_NOTICES.md). Model weights retain their publisher terms.
