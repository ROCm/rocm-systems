# Address race checker performance analysis

This document is a design-review snapshot of the core address race checker's
runtime and memory cost as of 2026-10-15. It separates measurements that answer
different questions, identifies the current hot paths, and proposes an
optimization order that preserves findings and retained coverage.

For checker semantics and correctness limits, see
[Address race diagnostics](memory-wait-diagnostics.md#address-race-diagnostics)
and [Address race checking gaps](address-race-checking-gaps.md).

## Conclusions

1. The newer "~41% slower on small memory workloads and ~51% on larger serial
   workloads versus `develop`" result does not establish a regression caused by
   the final checker redesign. Those figures came from synthetic workloads that
   maximize memory accesses per unit of unrelated simulation work. The earlier
   +20.4% to +26.2% figures were geomeans across 20 heterogeneous Gluon kernels.
2. The current race-off path is close to `develop`: -0.03% on the small serial
   set and +1.9% on the large serial set. The measured cost is in the enabled
   checker path, not a general disabled-path regression.
3. A limited same-kernel comparison is directionally favorable to the current
   design. Across FMHA v4 and A16W16 v9, the full-coverage geomean fell from
   +53.4% before the final refactor to +42.6% now. This is not a controlled A/B,
   but it is evidence against the claim that the redesign itself increased
   overhead.
4. The checker is still expensive on memory-heavy workloads. Software profiles
   put 27% to 40% of resolved rocJITsu samples directly in race-check code, led
   by per-dword global shadow traversal, LDS history updates, and lane/range
   preparation.
5. Memory amplification is substantial. One retained global owner for a source
   dword costs about 28.25 shadow bytes before allocator overhead; two retained
   owners cost about 52.25 bytes. The nominal shadow budget therefore does not
   predict process RSS.
6. Optimization should start with counters and exact no-work paths, then reduce
   the number of ranges, dwords, lookups, allocations, and records processed.
   Changing lock topology is not yet justified because lock wait has not been
   measured.

## Measurement definitions

- **Race-off** means the current binary with address race checking disabled.
- **Race-on** means the same binary and workload with address race checking
  enabled.
- **`develop`** means the original branch baseline without the new checker.
- **Default coverage** uses the 512 MiB global shadow budget.
- **Full coverage** in these campaigns uses a 4 GiB budget and emitted no global
  shadow exhaustion warning.
- Unless stated otherwise, percentages compare elapsed time per dispatch with
  race-on against race-off from the same binary.

Timing must be interpreted together with coverage. Once the global shadow
budget is exhausted, later accesses are checked against retained history but
are not retained. A capped run can therefore become faster by checking less
history.

## Performance evidence

### Pre-final-refactor realistic campaign

The broadest realistic campaign used 20
[gfx950 Gluon tutorial kernels](https://github.com/ROCm/gfx950-gluon-tutorials)
with a Release build and `configs/gfx950_mi355x.json` (256 compute units). GEMMs
used 4096 x 4096 matrices, with K = 4096 for fp16 and K = 8192 for fp8 and
MXFP4. FMHA used 16 heads, sequence length 4096, and head dimension 128. Each
configuration used two processes with two timed dispatches each; each kernel
value is the median of four pooled dispatches.

| Configuration | Geomean time per dispatch | Per-kernel range | Added peak memory |
|---|---:|---:|---:|
| Race-off versus contemporaneous `develop` | 0.97x | Within noise | Not measured |
| Race-on, 512 MiB budget | +20.4% | +4% to +52% | 0.7 to 1.3 GiB |
| Race-on, 4 GiB budget | +26.2% | +3% to +73% | 1.4 to 2.1 GiB |

Race-off peak memory was 2.0 to 2.2 GiB. The checker added a roughly constant
0.5 to 1.9 seconds per dispatch at these shapes, so relative cost was highest
for kernels that were otherwise cheap to simulate. Every 512 MiB run exhausted
the global shadow; no 4 GiB run did. No race finding was reported.

This campaign predates the final ownership and lifecycle refactor. The old
checker was owned per command processor, whereas the current checker is shared
by all command processors and compute units in one GPU/SoC. The results are a
useful realistic baseline, not a clean before/after redesign comparison.

### Current synthetic microbenchmarks

Four memory-intensive gfx950 workloads--stream, LDS reduction, GEMM, and
histogram--were run against the current Release binary with checking on and
off. The small set used 262,144 elements and GEMM size 128. The large set used
1,048,576 elements and GEMM size 256. Geomeans below exclude the compute-only
control.

| Execution mode | Small memory set, race-on versus race-off | Large memory set, race-on versus race-off |
|---|---:|---:|
| Default dispatch threading | +37.6% | +54.1% |
| Serial dispatch | +41.1% | +48.4% |

The compute-only control measured +1.2% with default threading and -2.8% with
serial dispatch, both consistent with noise. In the less noisy serial runs,
race-off was -0.03% versus `develop` over the five small workloads and +1.9%
over the four large workloads.

Combining those baselines, race-on measured +40.7% on the small memory set and
+51.2% on the large memory set versus `develop`. These are the source of the
"about 41% / 51%" headline. The +41.1% / +48.4% race-on/race-off values isolate
the current checker's incremental cost.

These synthetics deliberately remove most non-memory work. They expose the
upper-cost regime of the checker and are not directly comparable to the
20-kernel geomean. The realistic campaign already contained individual kernels
at +52% with the default budget and +73% with full coverage.

### Current mutation-harness sweep

A serial gfx1250 sweep used 21 small mutation shaders and five processes per
variant, with ordinary wait checking enabled in both variants. Enabling address
race checking added 3.7% geomean whole-process wall time, with a bootstrap 95%
interval of approximately +1.9% to +5.4%. The global shadow did not exhaust.

These CI-sized kernels show the low-cost end of the range. Their whole-process
timing includes startup and is not comparable to per-dispatch Gluon timing.

### Current representative rerun

FMHA v4 and A16W16 v9 were rerun on the current checker with one process per
variant and three timed dispatches. The table uses the dispatch medians. Raw
stderr confirmed that the 512 MiB runs exhausted the shadow and the 4 GiB runs
did not; the benchmark helper's old `degraded` parser did not recognize the new
warning text.

| Kernel | Race-off median | 512 MiB overhead | 4 GiB overhead | Added peak RSS, 512 MiB / 4 GiB |
|---|---:|---:|---:|---:|
| FMHA v4 | 2.844 s | +42.3% | +50.9% | 1.43 / 1.58 GiB |
| A16W16 v9 | 3.731 s | +29.7% | +34.8% | 1.36 / 2.38 GiB |
| Two-kernel geomean | -- | +35.9% | +42.6% | -- |

For the same two kernels, medians from the pre-final-refactor campaign were
+49.2% with the default budget and +53.4% with 4 GiB. The current result is
directionally faster by 13.3 and 10.8 percentage points, respectively.

This is still not a controlled redesign A/B. The runs happened at different
times, and the current rerun used only one process per variant. It should be
treated as evidence against a large redesign regression, not a precise estimate
of the redesign's benefit.

## Why the headline figures differ

The campaigns weight different work:

- The 20-kernel geomean mixes compute-heavy and memory-heavy kernels. Race
  bookkeeping is diluted when instruction execution dominates.
- The synthetic campaign maximizes routed memory operations and touched
  dwords. It intentionally exposes bookkeeping and shadow-memory cost.
- The mutation sweep measures tiny whole processes, so initialization and
  teardown dilute checker work.
- The default-budget campaigns lose retention after exhaustion. Full-coverage
  runs do more work and can be slower.

The correct conclusion is that enabled checker overhead is workload-sensitive:
approximately 4% for the CI-sized mutation sweep, about 20% to 26% across the
older realistic suite, and roughly 40% to 54% for current memory-intensive
synthetics. These measurements do not show a comparable increase caused by the
final redesign.

## Software profile

Hardware `perf` sampling was unavailable because `perf_event_paranoid=4`.
Userspace SIGPROF sampling was therefore run against the exact current Release
binary. Percentages below are normalized to samples that resolved inside
`librocjitsu.so`.

| Workload | Samples directly in race-check code | Largest resolved costs |
|---|---:|---|
| Streaming global memory | 27% | `global_access` 22% |
| LDS reduction | 27% | Global access 7%, LDS record 6%, `lds_access` 5% |
| GEMM | 40% | `lds_access` 14%, global access 7%, LDS record 7%, range preparation about 6%, clipping about 3% |

These are lower bounds. Allocator and mutex work that resolves to libc is not
attributed to the race-check caller. The profile identifies scalar bookkeeping
cost but does not measure lock wait or prove that contention is dominant.

## Current cost path

### 1. Materialize lane ranges

`ComputeUnitCore::track_address_races` builds per-lane byte ranges, separates
FLAT LDS/global domains, clips LDS ranges to the workgroup allocation, handles
DS2 addresses, and constructs direct-to-LDS effects. Most vector instructions
then enter `for_each_word`, which clears and probes a 64-entry recent-dword
cache and splits each range into dwords.

This work scales with active lanes, elements, and touched dwords even when many
lanes map to a regular contiguous region.

Relevant code:

- [`compute_unit.cpp`](../lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp):
  `clip_to_lds_allocation`, `track_address_races`, and
  `track_tensor_dma_accesses`
- [`memory_race_checker.cpp`](../lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_race_checker.cpp):
  `for_each_word`

### 2. Locate and lock state

Every LDS access hashes its workgroup, locks the map shard, obtains a shared
workgroup state, locks that state, and linearly finds the wave slot. Workgroup
startup registers waves individually and grows the vector-clock matrix from
N x N to (N + 1) x (N + 1) for each newly seen wave.

Global accesses hash pages into 256 shards and switch locks as the lane order
moves between shards. The configured global budget is divided evenly across
the shards, so an active shard can exhaust its allocation while another shard's
allocation is unused.

### 3. Walk pointer-heavy shadows

Every touched dword walks a linked record list:

- LDS uses a dense/sparse dword-head index plus `LdsRecord` links.
- Global memory uses an `unordered_map` of pages, a separate record vector, and
  per-dword linked lists.
- Pending LDS operations retain separate lane-range vectors and can be scanned
  again for overlap and barrier carry.

The layout makes exact byte masks and provenance available, but costs cache
locality, allocations, and pointer chasing.

### 4. Synchronize and retire

Barriers allocate member, merged-clock, and joined vectors, join vector clocks,
prune or clear history, and carry incomplete operations. Dispatch retirement
visits every workgroup and global shard to erase dispatch state. Completion
tokens are classified and attached before it is known whether the resulting
effect will consume LDS pending state.

## Memory amplification

The current global shadow uses 256-dword pages:

- `GlobalPage` contains 256 four-byte heads: 1,024 bytes.
- `kPageOverhead` estimates another 64 bytes for the map node.
- Head metadata is therefore 1,088 / 256 = **4.25 bytes per source dword**.
- `GlobalRecord` is statically asserted to **24 bytes**.

The 4.25-byte figure is amortized across all 256 dwords in an allocated page.
A sparse page still allocates every head, so its effective cost per touched
dword can be much higher. For a densely used page, approximate retained costs
before hash buckets, vector slack, allocator metadata, LDS state, report state,
and other process memory are:

| Retained state for one 4-byte source dword | Approximate shadow bytes | Amplification |
|---|---:|---:|
| Page head plus one owner record | 28.25 B | 7.1x |
| Page head plus two owner records | 52.25 B | 13.1x |

The checker charges page estimates and global-record vector capacity to the
global budget, but the process RSS contains more than that accounting. In the
representative rerun, race-on added about 1.4 GiB RSS at the nominal 512 MiB
limit and 1.6 to 2.4 GiB with full coverage.

## Optimization roadmap

The priority is to reduce work while preserving byte masks, findings, and
retained coverage. Expected payoffs below are hypotheses to validate with
counters and controlled reruns.

### P0: measure fixed work

Add counters for:

- materialized lane ranges and distinct dwords,
- shadow pages and records visited, retained, replaced, and rejected,
- empty versus nonempty record heads,
- allocations and vector growth,
- map/state lookup and lock-wait time,
- pending-operation candidates tested,
- barrier members, clock cells, carried operations, and cleanup work, and
- coverage retained and rejected per dispatch and shard.

This distinguishes algorithmic work, allocation, and contention and prevents a
faster capped run from being mistaken for an optimization.

### P1: remove work from common paths

1. Do not create LDS workgroup state when the allocation is zero.
2. Skip global checking for a dispatch with only one workgroup, because the
   current global checker compares different workgroups only.
3. Attach completion tokens only to effects that can enter LDS pending state.
4. Batch-register the known workgroup wave span and size the vector-clock matrix
   once instead of growing it for each wave.
5. Cache a lifetime-safe workgroup-state handle and wave-slot index on the
   active workgroup/wave.

### P1: reduce range and dword processing

1. Stable-bucket lane ranges by page and dword.
2. Coalesce only compatible contiguous ranges while preserving producer lane
   attribution needed by diagnostics.
3. Add aligned/disjoint and aggregate-in-bounds LDS fast paths.
4. Retain the generic path for gathers, broadcasts, masks, overlap, and
   unaligned accesses.
5. Index pending LDS operations by `[lo, hi)` envelopes or dword buckets so an
   access visits only potentially overlapping incomplete operations.

### P1: localized storage improvements

1. Reuse fixed-size barrier scratch for workgroups with at most 64 waves.
2. Reorder `CompletionToken` fields if layout checks confirm the expected
   reduction from 32 to 24 bytes on LP64.
3. Anchor LDS dword indexing to the known allocation bounds instead of growing
   and probing a generic dense/sparse window.

### P2: improve global-shadow locality and accounting

1. Allocate pages and records from per-dispatch shard arenas or slabs.
2. Replace the pointer-heavy page map with a flatter index where practical.
3. Grow records in balanced chunks rather than reserving one shard's remaining
   budget.
4. Allow busy shards to use otherwise idle budget while retaining an exact
   process-wide cap.
5. Add a correctly invalidated saturated-page miss cache to avoid repeated
   failed retention work after coverage degrades.
6. Intern PCs/sites and pack record provenance toward 16 bytes while preserving
   exact producer PC, workgroup, wave, lane, and byte mask.

### P3: represent regular accesses compactly

Represent affine wave accesses as page runs and expand only irregular gathers.
Compact owner/conflict summaries, indexed in-flight state, and epoch-based LDS
history have the largest potential payoff, but also the greatest correctness
risk. They should follow lower-risk changes and differential tests.

### Conditional: change lock topology

Do not redesign synchronization solely from the current profile. First measure
lock wait and scaling with isolated serial and threaded runs. If contention is
material, evaluate finer page ownership, lock-free read paths, or issuer-local
batching. The current evidence proves bookkeeping cost, not that locks are the
primary bottleneck.

## Planning bounds

If all added elapsed time scaled linearly with race-specific work, reducing
that work would give the following Amdahl-style planning bounds:

| Workload regime | Measured added time | If race-specific work falls 25% | If it falls 50% |
|---|---:|---:|---:|
| Small serial synthetic | +41.1% | +30.8% | +20.6% |
| Large serial synthetic | +48.4% | +36.3% | +24.2% |
| Earlier realistic geomean, default budget | +20.4% | +15.3% | +10.2% |

This is not a forecast. Some costs will not scale linearly, and each
optimization must be measured at fixed coverage.

## Correctness guardrails

Performance changes must preserve:

- byte-precise masks and producer/consumer lane attribution,
- same-wave in-flight LDS wait hazards,
- full, named, and cluster barrier member semantics,
- ended-wave history until it is ordered,
- the atomic-vs-atomic exemption and atomic/non-atomic conflicts,
- sufficient distinct global owners per byte to detect later conflicts,
- exact tensor and direct-to-LDS footprints,
- dispatch cleanup only after all fanout peers retire, and
- explicit coverage degradation when retention fails.

Gate each optimization with the race-checker unit suite, execution-level
simulation tests, mutation shaders, report/config/CLI tests, and representative
performance runs.

## Required sign-off protocol

Re-establish precise percentages before accepting an optimization:

1. Use an isolated or pinned host with a fixed CPU policy.
2. Interleave race-on and race-off processes from the same binary.
3. Use at least five repetitions per variant.
4. Run both the 512 MiB budget and a budget that retains full coverage.
5. Record raw dispatch times, process peak RSS, coverage warnings, findings,
   and checker work counters.
6. Keep realistic Gluon kernels, synthetic controls, and the mutation sweep;
   they expose different cost regimes.
7. Compare optimizations at fixed retained coverage.
8. Add gfx1250/wave32, multi-GPU, and dispatch-concurrency measurements.

The host used for the current campaigns was heavily loaded: approximately
400-560 load average during parts of the synthetic/profile campaign and about
101 during the final rerun on 384 logical CPUs. The current percentages are
directional until repeated under the protocol above.
