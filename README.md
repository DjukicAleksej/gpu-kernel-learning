# GPU Kernel Optimization — A100 #7

CUDA optimization experiments for GPU Mode's `grayscale_v2` challenge.

## Result

**#7 on the NVIDIA A100 leaderboard**

- Best ranked runtime: **2375.680 us** (v10, submission 940445)
- Problem: `grayscale_v2`
- GPU: NVIDIA A100
- Input: `16384 x 16384` FP32 RGB image
- All official correctness tests passed

Leaderboard:
https://www.gpumode.com/leaderboard/538

The project starts from the official PyTorch implementation and progressively
reduces runtime through kernel fusion, vectorized memory access, and low-level
CUDA tuning.

## Performance progression

| Version | Strategy | Representative A100 runtime |
| --- | --- | ---: |
| v0 | Official PyTorch baseline | 10.6 ms |
| v1 | Fused CUDA kernel | 2.55 ms |
| v2 | `float4` vectorized memory access | 2.37 ms |
| Ranked best | v10: 128-thread exact-grid CUDA | **2375.680 us — #7** |

The largest improvement came from kernel fusion.

The original implementation expressed grayscale conversion through several
PyTorch operations. Moving the entire computation into a single CUDA kernel
removed intermediate work and reduced the operation to one streaming pass over
the image.

Vectorizing the memory path then reduced indexing and memory-instruction
overhead.

## Problem

For each RGB pixel:

    Y = 0.2989 R + 0.5870 G + 0.1140 B

Input:

    [H, W, 3] float32

Output:

    [H, W] float32

For a `16384 x 16384` image, the minimum input/output traffic is approximately
4 GiB.

At runtimes around 2.38 ms, this corresponds to roughly 1.8 TB/s of effective
memory bandwidth.

Because the arithmetic per pixel is tiny compared with the amount of memory
traffic, later optimization becomes primarily a memory-throughput problem.

## Optimization strategy

### v0 — PyTorch baseline

The official implementation establishes correctness and the starting runtime.

Measured runtime:

    10.6 ms

### v1 — fused CUDA kernel

One CUDA thread processes one output pixel.

Each thread:

1. loads R, G and B,
2. computes the weighted sum in registers,
3. writes one grayscale value.

This removes the intermediate image-sized work performed by the original
PyTorch expression.

Measured runtime:

    2.55 ms

Speedup versus the baseline:

    4.16x

### v2 — vectorized memory access

The next implementation processes four pixels per thread.

Twelve input floats are read using three aligned `float4` loads:

    [R0 G0 B0 R1]
    [G1 B1 R2 G2]
    [B2 R3 G3 B3]

and four grayscale outputs are written using one `float4` store:

    [Y0 Y1 Y2 Y3]

Representative benchmark:

    2.37 ms

This reduced indexing and memory-instruction overhead while keeping the kernel
as a simple streaming operation.

### v3 — block-size tuning

Tested 128 threads per block instead of 256.

The goal was to determine whether occupancy or scheduling changes could improve
memory throughput.

No meaningful improvement was observed.

### v4 — thread coarsening

Tested assigning multiple vectorized pixel groups to each thread.

The experiment also explored read-only loads and fused multiply-add operations.

The goal was to reduce per-thread indexing overhead and expose additional
independent memory operations.

### v5 — cache and store behavior

Tested read-only input loads together with a write-through output strategy.

The goal was to investigate whether cache behavior could improve a nearly pure
streaming workload.

### v6 — exact-grid specialization

For the large benchmark shape, the number of pixel groups divides the launch
geometry exactly.

A specialized hot path removes the bounds check for that exact configuration.

The implementation passed correctness tests but did not consistently beat the
best ranked result.

### v7 — explicit PTX memory operations

Tested explicit vectorized PTX global-memory loads and stores:

    ld.global.v4.f32
    st.global.v4.f32

The goal was to control the generated memory operations directly instead of
depending entirely on compiler lowering of C++ `float4`.

The implementation passed correctness testing but did not produce a consistent
ranked improvement.

## Why the last few microseconds are difficult

Grayscale conversion has extremely low arithmetic intensity.

For each pixel the kernel must read:

    3 x float32 = 12 bytes

and write:

    1 x float32 = 4 bytes

for only a handful of floating-point operations.

Once framework overhead and intermediate tensors are removed, performance is
therefore dominated by global-memory throughput.

This explains the optimization pattern:

    PyTorch baseline
           |
           | large gain from fusion
           v
    fused CUDA
           |
           | smaller gain from vectorization
           v
    vectorized CUDA
           |
           | tiny gains/losses from micro-tuning
           v
    memory-bandwidth region

## Ranked result

GPU Mode `grayscale_v2`

    GPU: NVIDIA A100
    Best ranked runtime: 2375.680 us
    Rank at time recorded: #7
    Correctness: PASSED

The exact leaderboard position can change as new submissions are added.

## September 3 follow-up

The top-three goal is still in progress. Fresh official runs and recovered
historical results are recorded with full precision in
[the optimization log](notes/optimization-log.md#september-3-2026-follow-up)
and [public API evidence](pmpp_v2/grayscale_py/results/2026-09-03-public-evidence.json).
Benchmark times are not ranked scores. `submission.py` is an exact copy of v10,
the kernel that earned the best verified ranked result (940445). The original
v2 result was 2378.069 us; the new score improves it by 2.389 us. The gap to
third place at verification was 0.768 us, so the top-three goal remains open.

New isolated experiments include v8's cached-streaming output stores, v9's
512-thread blocks, and v10's 128-thread exact grid. v8 was rejected before GPU
testing by the service's source-admission filter and is paused pending organizer
review; it must not be ranked. The rejection and its source-level explanation
are preserved in [the rejection record](pmpp_v2/grayscale_py/results/v8_admission_rejection.txt).

The later v11 experiment tested 64-thread blocks. It passed correctness but
ranked slower, so the canonical implementation remains v10. Full benchmark
comparisons and submission IDs are recorded in the optimization log.

## Repository structure

    gpu-kernel-learning/
    |
    |-- README.md
    |-- notes/
    |   |-- cuda-basics.md
    |   `-- optimization-log.md
    |
    `-- pmpp_v2/
        `-- grayscale_py/
            |-- submission.py
            |
            |-- experiments/
            |   |-- v0_pytorch.py
            |   |-- v1_fused_cuda.py
            |   |-- v2_vectorized_cuda.py
            |   |-- v3_block128.py
            |   |-- v4_coarsened4.py
            |   |-- v5_readonly_wt.py
            |   |-- v6_exact_grid.py
            |   |-- v7_ptx_exact_grid.py
            |   |-- v8_streaming_store.py
            |   |-- v9_block512.py
            |   |-- v10_block128.py
            |   `-- v11_block64.py
            |
            `-- results/
                |-- ranked_result_a100.txt
                |-- v0_pytorch_benchmark_a100.txt
                |-- v1_fused_cuda_benchmark_a100.txt
                `-- v2_vectorized_cuda_benchmark_a100.txt

## Running an experiment

From:

    pmpp_v2/grayscale_py

Run correctness tests:

    popcorn-cli submit --no-tui --mode test submission.py

Run a benchmark:

    popcorn-cli submit --no-tui --mode benchmark submission.py

## Key takeaways

- Kernel fusion produced the largest performance improvement.
- Coalesced global-memory access is critical for streaming CUDA workloads.
- Vectorization reduced indexing and memory-instruction overhead.
- Performance changes must be measured rather than inferred from code alone.
- Remote GPU benchmark variance matters when comparing differences of only a
  few microseconds.
- Once a kernel approaches the memory-bandwidth ceiling, increasingly
  low-level changes can have very small effects.

