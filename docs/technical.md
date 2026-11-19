# Technical notes

## M1 Max latency and memory

Six paired perfkit rounds compared the public `run()` with 0.3.0
(`40a2e07`) on an Apple M1 Max (64 GiB, macOS 27.2.1), with the same
lockfile and released weights. Both revisions used Python 3.13.18, MLX 0.32.4
and nnunet-mlx 0.9.0. Each fresh process ran one cold call followed by two
warm calls. The calls include CT reads, per-part model loading, inference
and mask publication.

| HCC-TACE-Seg case | Cold call, s | Warm call, s | Peak process, MiB | Peak MLX, MiB |
| --- | ---: | ---: | ---: | ---: |
| HCC_074 fast (512 × 512 × 185) | 2.160 → 1.746 | 1.693 → 1.448 | 7,884 → 7,890 | 5,345 → 5,345 |
| HCC_039 standard (512 × 512 × 77) | 4.707 → 4.398 | 4.191 → 4.104 | 5,905 → 5,960 | 3,818 → 3,818 |

The table reports medians. The standard warm interval reached into perfkit's
2% minimum-effect band. The largest fast-mode process peaks were 8.07 GiB
for 0.3.0 and 8.08 GiB for the port.

## Cubic resampling footprint

The M1 Max resampled HCC_074 to 240 × 240 × 308 voxels at 1.5 mm.
These are median peak process footprints from six perfkit rounds. The
16-worker runs forced that count on the 10-core M1 Max.

| Cubic workers | Planes per sampling slab | Peak footprint | Change |
| ---: | ---: | ---: | ---: |
| 8 (0.3.0) | 8 | 1,481 MiB | |
| 16 | 8 | 1,593 MiB | +7.6% |
| 16 | 4 | 1,493 MiB | +0.8% |
| 10 (M1 Max default) | 6 | 1,475 MiB | −0.4% |

The sampler can use up to 16 workers. It sizes each slab so the coordinate
buffers cover at most 64 planes in total, matching the eight-worker budget.
The float64 CT array is temporary for the spline pass. The resampled int32
values and affine matched 0.3.0 with both 10 and 16 bounded workers.

## Output equivalence

The timed fast and standard runs produced the same mask voxels, affine and
header, including extensions, as 0.3.0. Each mask opened in NiBabel, and its
single gzip member decoded to the serial writer's NIfTI bytes. With
probabilities enabled on HCC_039, fast and standard organs runs also produced
byte-identical `.npz` and `.pkl` sidecars.
