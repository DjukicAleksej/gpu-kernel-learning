# Optimization log

All times below came from GPU Mode's remote evaluator. Remote compilation time
and local wall-clock time are not kernel runtime.

## Grayscale target

- Problem: PMPP_v2 `grayscale_v2`
- GPU: NVIDIA A100
- Input/output: FP32 `[size, size, 3]` -> `[size, size]`
- Reported benchmark: `size=16384`, seed `54352`
- Method: Popcorn `test` before non-ranked Popcorn `benchmark`

| Version | Main change | Mean runtime | Speedup vs v0 | Correct? | Test / benchmark IDs |
| --- | --- | ---: | ---: | :---: | --- |
| v0 | Official PyTorch starter | 10.6 +/- 0.00 ms | 1.00x | yes | 936123 / 936124 |
| v1 | Fused scalar CUDA kernel | 2.55 +/- 0.002 ms | 4.16x | yes | 936126 / 936128 |
| v2 | Four pixels/thread, `float4` I/O | **2.37 +/- 0.001 ms** | **4.47x** | yes | 936129 / 936130 |

Speedups use the displayed evaluator means, so they should not be quoted with
more than two decimal places.

## v0: official PyTorch baseline

Hypothesis: the eager expression is convenient but launches multiple operations
and materializes an image-sized multiplication result.

Result: all three tests passed. Benchmark reported `10.6 +/- 0.00 ms` for the
final 16384-square case.

## v1: fuse the operation

Change: one thread computes one output pixel in one native CUDA kernel. The three
products stay in registers; the input is read once and the output is written
once.

Expected effect: fewer launches and no intermediate tensor/global-memory
roundtrip.

Result: all tests passed. `2.55 +/- 0.002 ms`, **4.16x faster than v0**.

Minimum traffic for the reported shape:

```text
16384^2 pixels * (3 input floats + 1 output float) * 4 bytes = 4 GiB
4 GiB / 2.55 ms ~= 1.68 TB/s
```

This suggests that further gains will be bandwidth/instruction-efficiency
limited rather than compute-throughput limited.

## v2: vectorize four pixels

Change: one thread processes four adjacent pixels using three `float4` loads and
one `float4` store. Four RGB pixels are twelve contiguous floats:

```text
[R0 G0 B0 R1] [G1 B1 R2 G2] [B2 R3 G3 B3]
```

Expected effect: the same HBM bytes but fewer address calculations and load/store
instructions, with aligned 16-byte accesses.

Result: all tests passed. `2.37 +/- 0.001 ms`, a **1.076x speedup over v1**
(**7.1% lower runtime**) and **4.47x faster than v0**. Minimum-traffic
effective bandwidth is about
`1.81 TB/s`.

Decision: promote v2. Stop blind performance tuning because gains are
diminishing and the minimum-traffic estimate suggests a bandwidth limit;
profiling is needed to confirm the bottleneck before another speed-focused
version.

## Failed matmul experiment (kept deliberately)

- Problem: PMPP_v2 `matmul_v2`, FP16, A100
- Version: one CUDA thread per output, FP32 sequential accumulator
- Test submission: `936122`
- Outcome: failed correctness; **not benchmarked**

Mismatch counts reported by the official tests:

| Shape `(M,N,K)` | Mismatched output elements |
| --- | ---: |
| `(64,64,64)` | 4 |
| `(128,128,128)` | 23 |
| `(256,256,256)` | 217 |
| `(32,512,32)` | 4 |
| `(64,1024,64)` | 43 |

Likely cause: a different floating-point reduction order or implementation from
the cuBLAS-backed reference, combined with a tolerance tighter than a typical
FP16 ULP at these magnitudes. The evidence is consistent with this explanation
but does not isolate it conclusively. The code is retained as an educational
attempt, but no performance claim is attached to it.
