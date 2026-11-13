# Image test input

The geometry, pipeline, and Metal tests use the converted HCC-TACE-Seg HCC_063
contrast CT at `~/.cache/medmlx/test-data/hcc_063_ct.nii.gz` by default.
`MEDMLX_TEST_CT` overrides the path. They use CT crops for SciPy comparisons
and assign controlled spatial geometry where needed.

For the upstream workflow tests, use an environment with TotalSegmentator
2.12.0 to generate references outside Git:

```sh
python tests/record_workflow_reference.py \
  --ct "${MEDMLX_TEST_CT:-$HOME/.cache/medmlx/test-data/hcc_063_ct.nii.gz}" \
  --output-dir "${MEDMLX_TEST_REFERENCES:-work/real-reference}"
```

The workflow test reads `work/real-reference` by default;
`MEDMLX_TEST_REFERENCES` overrides that path. The directory contains CT crops
as well as masks.
