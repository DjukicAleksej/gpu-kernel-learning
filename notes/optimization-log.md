# Grayscale v2 Optimization Log

## Target

GPU Mode `grayscale_v2` on NVIDIA A100.

Best ranked result during this optimization run:

    2378.069 us
    #7 on A100

All ranked code passed the official correctness tests.

---

## v0 — PyTorch baseline

Runtime:

    10.6 ms

Purpose:

Establish the official reference implementation and baseline.

---

## v1 — fused CUDA

Runtime:

    2.55 ms

Changes:

- replaced multiple PyTorch operations with one CUDA kernel;
- one thread computes one grayscale output;
- removed intermediate image-sized operations;
- performed RGB weighting directly in registers.

This was the largest optimization step.

Speedup over baseline:

    4.16x

---

## v2 — float4 vectorization

Representative runtime:

    2.37 ms

Changes:

- four pixels per thread;
- three `float4` input loads for twelve RGB floats;
- one `float4` output store;
- reduced indexing and memory-instruction overhead.

This brought the kernel close to the memory-throughput limit.

---

## Ranked result

Best ranked runtime:

    2378.069 us

A100 leaderboard position at the time:

    #7

Correctness:

    PASSED

---

## v3 — 128-thread blocks

Hypothesis:

A smaller block size could improve occupancy or scheduling.

Tested:

    256 threads/block -> 128 threads/block

Result:

No meaningful performance improvement.

---

## v4 — thread coarsening

Hypothesis:

Assigning more independent work to each thread could reduce indexing overhead
and expose additional memory-level parallelism.

Explored:

- multiple pixel groups per thread;
- read-only loads;
- fused multiply-add operations.

No sufficiently consistent improvement over the best implementation was found.

---

## v5 — read-only loads and write-through stores

Hypothesis:

Because the workload is dominated by streaming memory traffic, changing cache
and store behavior could increase effective bandwidth.

Explored:

- read-only input loads;
- write-through output stores.

No sufficiently consistent improvement was observed.

---

## v6 — exact-grid specialization

Hypothesis:

The main `16384 x 16384` benchmark has an exact launch geometry, allowing the
hot path to avoid the bounds branch.

The implementation passed official correctness tests.

The optimization did not consistently outperform the best ranked kernel.

---

## v7 — explicit PTX vector memory operations

Hypothesis:

Explicit vectorized PTX instructions could produce a more efficient or more
predictable memory path.

Explored:

    ld.global.v4.f32
    st.global.v4.f32

The implementation passed correctness testing.

It did not produce a sufficiently consistent ranked improvement to replace the
best result.

---

## Overall progression

Baseline:

    10.6 ms

Best ranked result:

    2.378069 ms

Overall speedup:

    approximately 4.46x

The optimization process showed a clear transition from software/framework
overhead to hardware memory-bandwidth limitations.

Kernel fusion created the largest gain.

Vectorized memory access produced another measurable improvement.

After that point, changes to launch geometry, branching, cache behavior,
coarsening, and explicit PTX had much smaller effects because the workload was
already dominated by global-memory throughput.

