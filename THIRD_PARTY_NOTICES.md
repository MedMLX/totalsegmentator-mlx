# Third-party notices

MedMLX-authored additions are licensed under [Apache-2.0](LICENSE).
The upstream attribution and license notices are retained below.

totalsegmentator-mlx extracts the existing MedMLX integration and the
TotalSegmentator v2 CT outer pipeline into a native model package.

TotalSegmentator source and label tables:
  https://github.com/wasserth/TotalSegmentator
  Apache License, Version 2.0 (included as LICENSE).

The extracted CT path preserves canonical orientation, cubic 1.5 mm CT
resampling with int32 staging, ordered five-part global label merging,
nearest-neighbor label restoration and the NIfTI label table. Other upstream
tasks, backends and application services are not included.

The MedMLX adapter is by Joseph Sandoval and Alfredo Sandoval Jr., under the
Apache License, Version 2.0.

nnunet-mlx is a separate dependency and retains its own Apache-2.0 and MIT
notices. Model checkpoint terms are separate from these source-code licenses.

Dependencies declared in [pyproject.toml](pyproject.toml) are installed separately
and retain their respective licenses.

## Example image

`docs/example.png` shows CT and model-derived surfaces from
[autoPET-III-Lite](https://huggingface.co/datasets/YongchengYAO/autoPET-III-Lite),
licensed [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). It retains
those terms; the repository's Apache-2.0 license does not relicense it.
