# GPU Kernel Learning: first GPU Mode result

I started this project after being encouraged to explore GPU Mode kernel
optimization. I knew C++ but had almost no practical CUDA or GPU-architecture
experience. The goal was to become productive quickly through a tight loop:
build, test, measure, understand, and make one justified change at a time.

## Result at a glance

On GPU Mode's open PMPP_v2 `grayscale_v2` problem, a fused and then vectorized
native CUDA implementation reduced the reported runtime from **10.6 ms to
2.37 ms** on the same remote NVIDIA A100: **4.47x faster than the official
PyTorch starter**.

| Version | Main change | A100 runtime | Speedup vs v0 | Correct? |
| --- | --- | ---: | ---: | :---: |
| v0 | Official PyTorch starter | 10.6 +/- 0.00 ms | 1.00x | yes |
| v1 | One fused CUDA kernel | 2.55 +/- 0.002 ms | 4.16x | yes |
| v2 | Four pixels/thread with `float4` I/O | **2.37 +/- 0.001 ms** | **4.47x** | yes |

These are real GPU Mode `benchmark` results for the reported `16384 x 16384`
case, not estimates. Each custom version first passed all official correctness
tests. No ranked/leaderboard-mode submission was made.

## Problem

Convert a contiguous FP32 RGB image of shape `[H, W, 3]` to grayscale
`[H, W]`:

```text
Y = 0.2989 R + 0.5870 G + 0.1140 B
```

The operation does little arithmetic relative to the data it moves, so launch
overhead and global-memory traffic matter more than raw floating-point compute.

## Iterations

### v0: official baseline

The starter creates a weights tensor and expresses multiplication, reduction,
and output assignment as eager PyTorch operations. It is correct and convenient,
but it needs multiple operations and an intermediate tensor.

### v1: kernel fusion

One CUDA thread owns one output pixel. It reads the pixel's R/G/B values,
computes the weighted sum in registers, and writes the result once. This removes
the intermediate image-sized tensor and collapses the work into one launch.

Measured result: **2.55 ms**, a **4.16x speedup** over v0.

### v2: vectorized memory access

One thread handles four pixels. Three aligned `float4` loads read the twelve
input floats and one `float4` store writes four outputs. This does not reduce the
required HBM bytes; the hypothesis was that it would reduce indexing and
memory-instruction overhead.

Measured result: **2.37 ms**, a **1.076x speedup over v1** (**7.1% lower
runtime**) and **4.47x faster than v0**. The minimum input/output traffic is
4 GiB, corresponding to roughly 1.81 TB/s of effective bandwidth at the
measured time. The smaller second gain is consistent with approaching a
memory-bandwidth limit, but profiling is needed to confirm that bottleneck.

## Benchmark scope

- Platform: GPU Mode remote **NVIDIA A100** (the runner did not expose the
  specific A100 form factor).
- Mode: Popcorn `test`, followed by non-ranked `benchmark`.
- The current task config uses `ranking_by: last`; for these runs KernelBot
  passed only the final `size=16384` benchmark to the evaluator, even though the
  metadata lists smaller shapes. This README reports only the row actually
  returned.
- Raw benchmark outputs:
  [v0](pmpp_v2/grayscale_py/results/v0_pytorch_benchmark_a100.txt),
  [v1](pmpp_v2/grayscale_py/results/v1_fused_cuda_benchmark_a100.txt), and
  [v2](pmpp_v2/grayscale_py/results/v2_vectorized_cuda_benchmark_a100.txt).
- Sanitized CLI evidence for the three passing tests and failed matmul attempt:
  [submission summaries](pmpp_v2/grayscale_py/results/submission_test_summaries.txt).
- This competition submission relies on the official evaluator's fresh,
  separately allocated, aligned contiguous tensors and required default CUDA
  execution path. It is not presented as a drop-in PyTorch operator for
  arbitrary offset/overlapping views.
- Full experiment history and submission IDs are in
  [the optimization log](notes/optimization-log.md).

## Matmul attempt

I first tried the PMPP_v2 FP16 matmul problem with a one-thread-per-output CUDA
kernel. It compiled and ran on A100, but failed the official correctness check.
The mismatches are consistent with its sequential FP32 reduction rounding some
FP16 results differently from the cuBLAS-backed reference, though that cause was
not isolated conclusively. I did not benchmark it or call it correct. Keeping
this attempt documents an important lesson: mathematically equivalent
floating-point algorithms can violate a strict numerical contract when their
reduction orders differ.

## What I learned

- host code versus device kernels and the meaning of `__global__`;
- how grids, blocks, threads, and `blockIdx * blockDim + threadIdx` map work;
- why NVIDIA groups 32 threads into a warp and why nearby addresses matter;
- registers, shared memory, caches, and global VRAM;
- coalesced and vectorized memory access;
- kernel fusion and arithmetic intensity;
- correctness-before-performance and honest failed-experiment logging;
- why shared-memory tiling is useful for data-reuse problems such as matmul,
  although the PMPP matmul numerical contract needs more work before that path
  can be benchmarked correctly.

The concise concept notes are in [CUDA basics](notes/cuda-basics.md).

## Repository layout

```text
gpu-kernel-learning/
|-- README.md
|-- notes/
|   |-- cuda-basics.md
|   `-- optimization-log.md
`-- pmpp_v2/
    |-- grayscale_py/
    |   |-- submission.py          # current best, self-contained v2
    |   |-- experiments/           # versioned experiment snapshots
    |   `-- results/               # raw outputs and test evidence
    `-- matmul_py/
        |-- submission.py
        `-- experiments/v0_naive.py
```

## Reproduce

After registering the current Popcorn CLI, run from `pmpp_v2/grayscale_py`:

```powershell
popcorn-cli submit --no-tui --mode test experiments/v2_vectorized_cuda.py
popcorn-cli submit --no-tui --mode benchmark --output results/v2_vectorized_cuda_benchmark_a100.txt experiments/v2_vectorized_cuda.py
```

Every experiment is a single Python submission containing embedded C++/CUDA via
PyTorch `load_inline`. POPCORN directives pin it to `grayscale_v2` on A100.

This run used Popcorn CLI `1.3.30`. Follow the official Popcorn README for the
current Windows release/install steps, then authenticate once with
`popcorn-cli register github` (or Discord). The generated `.popcorn`, `.codex`,
and `.claude` files are intentionally retained because `popcorn setup` uses them
to record the problem configuration and single-file submission workflow.

## References

- [GPU Mode Popcorn CLI](https://github.com/gpu-mode/popcorn-cli)
- [GPU Mode PMPP_v2 reference kernels](https://github.com/gpu-mode/reference-kernels/tree/main/problems/pmpp_v2)
- [Sankalp's kernel auto-research write-up](https://sankalp.bearblog.dev/autoresearch/)
- [Simon Boehm's CUDA matmul worklog](https://siboehm.com/articles/22/CUDA-MMM)

## AI usage

I used AI as a pair programmer and learning assistant to explain CUDA concepts,
review kernels, brainstorm optimizations, and debug implementations. I tested,
benchmarked, and worked through the reasoning behind each optimization. Failed
ideas and AI-assisted code are documented rather than presented as work derived
entirely from scratch.
