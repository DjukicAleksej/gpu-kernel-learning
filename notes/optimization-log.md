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

    2.375680 ms

Overall speedup:

    approximately 4.46x

The optimization process showed a clear transition from software/framework
overhead to hardware memory-bandwidth limitations.

Kernel fusion created the largest gain.

Vectorized memory access produced another measurable improvement.

After that point, changes to launch geometry, branching, cache behavior,
coarsening, and explicit PTX had much smaller effects because the workload was
already dominated by global-memory throughput.

## September 3, 2026 onward follow-up

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

### Post-v12 follow-up: v13, L1 no-allocation store hint

v13 returned to the current v10 best and changed only its float4 output store.
The ordinary C++ store was replaced with explicit PTX:

    st.global.L1::no_allocate.v4.f32

NVIDIA documents `L1::no_allocate` as a cache-allocation performance hint. It
does not change store correctness or memory-consistency semantics. The PTX store
grammar permits the four-element FP32 vector form, and this L1 hint is supported
on A100's sm_80 target. The exact-grid launch, guarded fallback, input loads,
coefficients, arithmetic, compiler flags, wrapper checks, and launch-error check
remain identical to v10. See NVIDIA's [PTX store instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-st)
and [cache eviction-priority hints](https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-eviction-priority-hints).

The new source is independent of the rejected v8 `.cs` experiment and does not
rename or resubmit it. A full source scan found no alternate CUDA execution path
or the service's prohibited substring. v13 then passed every official public
correctness test as submission 940616.

| Benchmark | Submission | Mean (us) | Standard error (us) | Best (us) | Samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| v10 control | 940617 | 2367.829243 | 0.682672 | 2366.463900 | 3 |
| v13 candidate | 940618 | 2369.535923 | 0.000000 | 2369.535923 | 3 |

The v13 mean was 1.706680 us slower than its fresh v10 control, and its best
sample was also slower. Accordingly, v13 was not a strong candidate for the
ranked workflow: it was not ranked, promoted, or retried. `submission.py`
remains byte-for-byte identical to v10, and the live best remains submission
940445 at 2375.679970 us (#7), 0.768006 us behind third.

### Post-v13 follow-up: v14, L1 input-retention hints

v14 kept v10's launch geometry, output store, coefficients, arithmetic,
wrapper checks, and compiler flags unchanged. It isolated cache-eviction
priority hints on the three 16-byte input loads. The first two loads use
`ld.global.L1::evict_last.v4.f32`, while the final load uses
`ld.global.L1::evict_first.v4.f32`.

The experiment targets the overlap between adjacent warp-wide vector-load
phases. Each phase requests 32 sectors, while the three phases cover only 48
unique sectors in total; retaining the early sectors could therefore preserve
reuse until the last phase. These qualifiers are performance hints and do not
change the values loaded or memory-consistency semantics. NVIDIA documents the
forms in the [PTX load instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-ld)
and supports L1 eviction-priority hints on A100's sm_80 target.

The official test submission 941296 passed all three public correctness cases.
The contemporaneous control and candidate benchmarks were:

| Benchmark | Submission | Mean (us) | Standard error (us) | Best (us) | Samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| v10 control | 941297 | 2369.194587 | 1.365344 | 2366.463900 | 3 |
| v14 candidate | 941298 | 2370.901267 | 0.903090 | 2369.535923 | 3 |

The v14 mean was 1.706680 us slower than its fresh control, and its best sample
was also slower. It therefore did not qualify for a ranked submission and was
not promoted or retried. `submission.py` remains byte-for-byte identical to
v10. Live verification still shows submission 940445 at 2375.679970 us (#7),
0.768006 us behind third.

### Post-v14 follow-up: v15, sequential two-group coarsening

v15 revisited the v12 coarsening hypothesis because v12's unranked benchmark
mean had been 3.071944 us faster than its contemporary v10 control. It changed
the exact hot kernel so each 128-thread block still covers two coalesced
128-group phases, but writes the first result before beginning the second
group. This shortens the source-level live range of the first `float4` result.
The exact RGB coefficients and per-pixel arithmetic order, ordinary `float4`
loads and stores, compiler flags, wrapper checks, guarded fallback, and launch
error check remain unchanged.

Submission 941345 passed all three official public correctness cases. A fresh
v10 control and the v15 candidate then produced:

| Benchmark | Submission | Mean (us) | Standard error (us) | Best (us) | Samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| v10 control | 941346 | 2370.218595 | 1.706680 | 2368.511915 | 3 |
| v15 candidate | 941347 | 2371.839941 | 2.146960 | 2365.439892 | 4 |

Although v15's best sample was 3.072023 us faster, its mean was 1.621346 us
slower than the control and both runs had substantial variation. That is not a
strong benchmark result for promotion into the ranked workflow, so v15 was not
ranked, promoted, or retried. `submission.py` remains byte-for-byte identical
to v10. Live verification still shows submission 940445 at 2375.679970 us
(#7), 0.768006 us behind third.

### Post-v15 follow-up: v16, read-only input loads

v16 returned to v10 and changed only its three ordinary `float4` input loads
to CUDA's `__ldg` read-only loads. This isolates a mechanism that appeared in
the older v4/v5 experiments alongside other changes. The exact RGB
coefficients, arithmetic, 128-thread exact-grid and guarded paths, output
store, compiler flags, validation checks, and launch-error check remain
unchanged.

Submission 941481 passed all three official public correctness cases. The
contemporaneous control and candidate benchmarks were:

| Benchmark | Submission | Mean (us) | Standard error (us) | Best (us) | Samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| v10 control | 941482 | 2366.186698 | 0.877999 | 2364.639997 | 3 |
| v16 candidate | 941483 | 2369.024038 | 1.356602 | 2367.360115 | 3 |

The v16 mean was 2.837340 us slower than the control, and its best sample was
2.720118 us slower. It was therefore not a strong ranked candidate and was not
ranked, promoted, or retried. `submission.py` remains byte-for-byte identical
to v10. Live verification still shows submission 940445 at 2375.679970 us
(#7), 0.768006 us behind third.

### Post-v16 follow-up: v17, write-through output store

v17 returned to v10 and changed only its ordinary `float4` output store to the
PTX write-through form `st.global.wt.v4.f32`. This cleanly isolates a mechanism
that v5 combined with read-only loads, explicit FMA calls, fast-math flags, and
a different block size. It is independent of v8's rejected `.cs` candidate:
v8 remains unchanged and paused, and v17 neither renames nor resubmits it. The
input loads, exact coefficients, arithmetic, 128-thread launch paths, wrapper
checks, compiler flags, and launch-error check remain identical to v10.

Submission 941590 passed all three official public correctness cases. The
contemporaneous control and candidate benchmarks were:

| Benchmark | Submission | Mean (us) | Standard error (us) | Best (us) | Samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| v10 control | 941591 | 2365.781228 | 0.903090 | 2364.415884 | 3 |
| v17 candidate | 941592 | 2367.146571 | 1.806180 | 2364.415884 | 3 |

The candidate matched the control's best sample but its mean was 1.365344 us
slower, with twice the control's standard error. That did not qualify as a
strong ranked result, so v17 was not ranked, promoted, or retried.
`submission.py` remains byte-for-byte identical to v10. Live verification
still shows submission 940445 at 2375.679970 us (#7), 0.768006 us behind third.

### October 10 follow-up: v18, explicit correctly rounded FMA

v18 changes only the hot arithmetic from v10: nested `__fmaf_rn` calls
explicitly form the two multiply-add stages while retaining v10's exact
`0.2989f`, `0.5870f`, and `0.1140f` coefficients. The 128-thread exact-grid
and guarded paths, vector loads/stores, wrapper validation, launch checks, and
compiler flags remain unchanged. The source passed the three official public
correctness tests in submission 950316 before benchmark comparison.

After the previous six-per-hour test-submission limit expired, a fresh
non-ranked v10 control (950356) and v18 candidate (950360) both completed
their official A100 benchmark. The public CLI reported rounded summaries:

| Benchmark | Submission | CLI mean display | CLI uncertainty display |
| --- | ---: | ---: | ---: |
| v10 control | 950356 | 2.37 ms | ±0.000 ms |
| v18 candidate | 950360 | 2.37 ms | ±0.001 ms |

The CLI's public `submissions show --no-code` response does not expose the raw
benchmark fields, so only its rounded summary is retained in the sanitized
evidence file. These readings do not establish a candidate advantage; v18 was
not ranked. Keep `submission.py` on v10 and wait for better evidence or a
concrete isolated optimization. v8 remains unchanged and paused pending
organizer review.

Live leaderboard verification at 2026-10-10T09:51:54Z found v10 submission
940445 still at #8, 2375.679970 us. Third place was Kernel-Zhang, submission
786700, at 2374.399960 us; the gap is 1.280010 us. The sanitized snapshot is
in `pmpp_v2/grayscale_py/results/2026-09-03-leaderboard.json`.

A reversed-order repeat pair was then run: v18 submission 950382 completed at
10:24:20Z and the v10 control 950384 completed at 10:26:08Z. The public CLI
reported 2.38 ±0.001 ms for v18 and 2.48 ±0.000 ms for v10. This apparent
100-us gap conflicts with the preceding pair, where both displayed 2.37 ms,
and with the earlier 2.48-ms control submissions in the evidence archive.
Because the control itself shifted substantially between nearby sessions and
raw measurements are unavailable from the public CLI, service/device variance
dominates the comparison; it is not evidence of a v18 speedup. No ranked
submission was made, `submission.py` remains v10, and v18 should stay
unranked absent a more stable benchmark method or organizer guidance.

The live API was checked again at 2026-10-10T10:26:44Z: submission 940445 is
still #8 at 2375.679970 us, with a 1.280010-us gap to third place (Kernel-Zhang,
submission 786700 at 2374.399960 us). The top-ten snapshot and public benchmark
records were refreshed in the results directory and pushed to GitHub.

A third contemporaneous pair checked whether the earlier 2.48-ms control was
an outlier: v10 control 950404 displayed 2.37 ±0.001 ms and v18 950406 displayed
2.37 ±0.002 ms. Together with the initial 2.37/2.37-ms pair, the repeated
rounding shows no measurable v18 lead. The second pair's 2.38/2.48-ms readings
are not reproducible and should be treated as transient service/device
variation. These benchmark displays are too coarse to support ranking v18, so
no ranked submission was made and canonical v10 was left unchanged.

The live API was rechecked at 2026-10-10T11:01:59Z and remains unchanged:
940445 is #8, 1.280010 us behind third. Sanitized evidence now includes 52
unique public submissions/results.

### October 10 follow-up: v19, bypass L1 for input vectors

v19 changes only how the three aligned `float4` input vectors are fetched:
inline PTX `ld.global.cg.v4.f32` bypasses L1 while retaining the original
coalesced addresses. The exact v10 arithmetic and RGB coefficients, output
store, 128-thread exact-grid/guarded launch paths, checks, and compiler flags
are otherwise unchanged. All three official correctness cases passed in
submission 950450.

A fresh v10 control (950451) displayed 2.37 ±0.001 ms. v19 (950453) displayed
3.62 ±0.002 ms, with displayed best/worst of 3.62/3.63 ms. This is a clear
performance regression for the cache-bypass load policy, so v19 was not
ranked. Keep the experiment as evidence and leave canonical `submission.py`
unchanged at v10. The sanitized submission entries include normalized local
source hashes; raw benchmark fields are not available in the public CLI show
output.

The live API was refreshed at 2026-10-10T12:14:50Z: submission 940445 remains
#8 at 2375.679970 us, 1.280010 us behind third place (Kernel-Zhang, 786700,
2374.399960 us). The leaderboard snapshot and public evidence were updated.

