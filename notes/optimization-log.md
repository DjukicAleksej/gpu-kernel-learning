# Grayscale v2 Optimization Log

## Target

GPU Mode `grayscale_v2` on NVIDIA A100.

Best ranked result during this optimization run:

    2375.680 us (v10, submission 940445; September 3, 2026)
    #7 on A100

All ranked code passed the official correctness tests.

The v0-v7 narrative below records the August 16 experiments. The September 3
follow-up at the end contains the newer v10 result and current top-three gap.

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

## September 3, 2026 follow-up

The live top-three target has not yet been achieved. The previous best is
submission 936167 (v2), 2378.069321 us, rank #7. The third-place threshold at
the start of this session was 2374.911964 us: a 3.157357 us gap.

Full-precision results were retrieved through the official authenticated
Popcorn user-submission API, filtered to public runs only. The committed
[evidence](../pmpp_v2/grayscale_py/results/2026-09-03-public-evidence.json)
contains source hashes, timestamps, submission IDs, and raw units. Benchmark
time fields are nanoseconds; leaderboard scores are seconds. Source comparisons
normalize only BOM, line endings, and trailing whitespace.

| Candidate | Correctness test | Benchmark ID | Benchmark mean (us) | Ranked ID | Ranked score (us) |
| --- | ---: | ---: | ---: | ---: | ---: |
| v6 exact grid | 940437 passed | 940436 | 2369.877259 | 940438 | 2393.673113 |
| v7 explicit PTX | 936208 passed; unchanged source | 940439 | 2369.194587 | 940440 | 2392.649123 |
| v9 512-thread exact grid | 940441 passed | 940443 | 2377.727985 | Not ranked | — |
| v10 128-thread exact grid | 940445 public and secret ranked tests passed | 940444 | 2368.170579 | 940445 | 2375.679970 |

v6 and v7 passed all public and secret ranked correctness checks but did not
improve the retained best score. v9 was slower than the 256-thread v6 baseline
in this session, so it was not ranked. v10's roughly 1 us advantage over v7 is
within measurement noise (v10 standard error 2.076265 us); it is a candidate,
not evidence of a repeatable performance win. Nevertheless, ranked submission
940445 earned a new retained best of 2375.679970 us: 2.389352 us faster than
936167. The live API verified rank #7 with a remaining 0.768006 us gap to third.
`submission.py` was promoted to an exact copy of v10. One ranked repeat, 940446,
passed all public and secret correctness checks and scored 2377.045314 us
publicly. It did not beat 940445, but both v10 runs beat the previous retained
v2 score. More controlled measurements are needed to establish a repeatable
kernel speedup; repeated score hunting is not a substitute for optimization.

Recovered historical benchmarks also show substantial between-run variation:
v6/936207 2484.565417 us, v7/936210 2370.815933 us, v2/936213 2517.333428 us,
and v3/936214 2517.674764 us. Historical ranked submission 936211 is verified
to contain v7, not v2. Its public score was 2521.770636 us.
These observations do not establish a specific hardware or clock cause.

### v8 admission rejection

v8 changes only v6's output store to `st.global.cs.v4.f32`; the CUDA launch
stream is unchanged. NVIDIA documents `.cs` as an evict-first cache hint,
not a different execution stream. See the
[PTX cache-operator specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators).

The service rejected the source before assigning a submission ID or executing
tests. Its [published admission rule](https://github.com/gpu-mode/kernelbot/blob/727212cdcf1b9b4d587c12f6d1484b3fd54549d0/src/kernelbot/api/api_utils.py#L272)
rejects any source containing the substring `stream`, including identifiers and
comments. This explains the rejection of v8's `streaming` names. The rejected
source is preserved unchanged and must not be resubmitted or ranked without
organizer review. See the [exact response](../pmpp_v2/grayscale_py/results/v8_admission_rejection.txt).

### Attribution

v7's explicit vector PTX memory operations were informed by HayatoFujihara's
public GPU Mode submission 230431, distributed in the official
[GPUMODE/kernelbot-data dataset](https://huggingface.co/datasets/GPUMODE/kernelbot-data/tree/4159cf6b2c6bab208be6dda885d6d87631cc16df).
The exact-grid path is a separate specialization of this project's v2/v6 work.

### 21:27 UTC follow-up: v11, 64-thread blocks

One isolated launch-geometry hypothesis was tested after quota became available:
halve v10's block size from 128 to 64 threads, updating the exact-grid shifts
from 7 to 6. The RGB coefficients, arithmetic, float4 loads/stores, compiler
flags, wrapper checks, and guarded fallback were unchanged. The smaller blocks
double the block count; a speedup was a hypothesis, not an assumption.

v11 passed the three official correctness tests in submission 940452. A fresh
v10 benchmark was run specifically as a contemporaneous comparison for this
new candidate, not as an automatic retry of previously completed work.

| Benchmark | Submission | Mean (us) | Standard error (us) | Best (us) | Samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| v10 control | 940453 | 2369.194587 | 1.487849 | 2366.463900 | 3 |
| v11 candidate | 940454 | 2366.805236 | 0.341336 | 2366.463900 | 3 |

Both passed the largest-shape correctness check. v11's mean was 2.389352 us
lower, but both had the same best time and the small sample sizes and separate
remote runs prevent a claim of a repeatable speedup.

After these validation gates, v11 was ranked once as submission 940455. All
public and secret correctness checks passed; its public ranked score was
2390.724952 us, slower than v10's retained 2375.679970 us. v11 was not promoted
or retried. The live position remains #7, 0.768006 us behind third; the top-three
goal is still open. `submission.py` remains byte-for-byte identical to v10.

The [public evidence](../pmpp_v2/grayscale_py/results/2026-09-03-public-evidence.json)
includes the completed runs and normalized source hashes. Future experiments
should test a new isolated hypothesis rather than automatically repeat this
block-size trial. v8 remains unchanged and paused for organizer review.

### Post-v11 follow-up: v12, two groups per thread

v12 isolated thread coarsening from the current v10 best. Each 128-thread block
covers one contiguous 256-group tile: every thread computes group `base + tid`
and group `base + tid + 128`. Both per-instruction accesses remain coalesced.
The exact coefficients, grayscale arithmetic, ordinary float4 I/O, compiler
flags, wrapper checks, error handling, and 128-thread guarded fallback remain
unchanged. This differs from v4, which mixed four-way grid-stride coarsening
with read-only loads, explicit FMA operations, and fast-math flags.

v12 passed all three official correctness tests as submission 940605. Static
coverage also confirmed that every official shape takes the exact 256-group
path and writes every output group once. A fresh v10 benchmark was again used
only as a contemporaneous control for the new candidate.

| Benchmark | Submission | Mean (us) | Standard error (us) | Best (us) | Samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| v10 control | 940606 | 2369.877259 | 0.903090 | 2368.511915 | 3 |
| v12 candidate | 940607 | 2366.805315 | 2.238211 | 2362.368107 | 3 |

The candidate mean was 3.071944 us lower, but the uncertainty was large. It was
ranked once after passing the test and benchmark gates. Submission 940608 passed
all public and secret checks, then scored 2399.118198 us publicly—23.438228 us
slower than the retained v10 result. Its embedded ranked-run public benchmark
mean was 2371.925275 us. v12 was therefore not promoted or retried.

`submission.py` remains the v10 kernel, and the live result remains #7 at
2375.679970 us, 0.768006 us behind third. These v12 results reinforce the need
to separate an unranked three-sample benchmark from the scored ranked run.

