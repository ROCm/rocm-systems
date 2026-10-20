# Address race checker performance analysis

This document is a design-review snapshot of the core address race checker's
runtime and memory cost as of 2026-10-20. It separates measurements that answer
different questions, identifies the current hot paths, and proposes an
optimization order that preserves findings and retained coverage.

For checker semantics and correctness limits, see
[Address race diagnostics](memory-wait-diagnostics.md#address-race-diagnostics)
and [Address race checking gaps](address-race-checking-gaps.md).

## Conclusions

### Current family-selection result

1. No independently useful active check meets the requirement that measured
   overhead and its 95% upper bound remain at or below 2%. On the 20-kernel
   full-coverage Gluon suite, `global` was +2.22% (+1.14% to +3.02%),
   `lds-same-wave` was +2.55% (+1.43% to +3.47%), and `lds-cross-wave` was
   +6.13% (+5.01% to +7.15%) versus race-off.
2. Direct-to-LDS is the largest realistic incremental cost. Adding that source
   to all address domains raised the Gluon geomean by 12.67% (+11.55% to
   +13.72%); all checks together were +22.81% (+21.44% to +23.62%). Without
   direct-to-LDS, all address domains were +8.99% (+7.66% to +9.96%).
3. The gfx950 Gluon tensor-DMA comparison, +0.80% (+0.13% to +1.46%), is a
   no-op control because gfx950 executes no tensor-DMA source. On a dedicated
   gfx1250 tensor-DMA workload, enabling the source added 7.46% (+1.76% to
   +9.66%) with cross-wave LDS and 20.89% (+15.08% to +29.39%) with all
   domains. Same-wave-only tensor DMA was unresolved with a +3.97% upper bound.
   It therefore does not qualify as an active check below 2%.
4. Adding same-wave LDS checking after cross-wave checking is cheap on Gluon:
   +0.05% (-0.86% to +1.07%). That does not produce a default-on candidate,
   because the combined LDS selection is still +6.19% (+5.18% to +7.21%).
   It only shows that the two LDS families share most routing, state, and
   history cost.

### Earlier phase results

1. The final P3 build adds 21.2% geomean time versus its race-off mode across
   the 20 heterogeneous Gluon kernels at the default budget and 21.8% with full
   4 GiB coverage. The corresponding comparisons with `develop` are +21.4% and
   +22.0%. P3 race-off is +0.2% versus `develop`; no disabled-path regression
   resolves.
2. Every default-budget P3 Gluon process exhausted retention, while no 4 GiB
   process did. The 4 GiB/default timing difference is +0.5% (-0.9% to +1.4%),
   so full coverage costs no resolved extra time. All 400 processes produced
   matching outputs and no race or wait finding.
3. P3 compactly represents 97% of stream and 99% of LDS-reduction logical lane
   ranges, and removes 122,880 large-LDS range-vector growths. It does not
   reduce shadow dword or record work, and GEMM remains entirely explicit.
   Consequently, the direct P3/P2 synthetic geomean is +0.2% on both sets with
   intervals spanning zero: no elapsed-time gain or regression resolves.
4. P2 and P3 race-on versus `develop` differ by only +0.2% to +0.3% after
   normalizing their separate realistic campaigns. P3's larger on/off ratio
   comes from a lower race-off point estimate, not resolved extra race-on time.
5. A post-P3 lock-scope audit moved callbacks, JSON formatting and file I/O,
   detached-state destruction, and budget release out of shared-state locks.
   Direct P3 comparisons remain statistically flat: -0.4% serial and +0.4%
   threaded without JSON collection, and +0.1% serial and -1.1% threaded with
   collection. The change reduces critical-section responsibilities and
   deadlock exposure, but is not a resolved elapsed-time optimization.
6. The checker is still expensive on memory-heavy workloads. Software profiles
   put 27% to 40% of resolved rocJITsu samples directly in race-check code, led
   by per-dword global shadow traversal, LDS history updates, and lane/range
   preparation. Current P3 samples still lead with exact site lookup and scalar
   shadow callbacks.
7. Memory amplification remains substantial. P2/P3's dense base payload is 20
   modeled bytes per source dword with one record and 16 more for a second
   record, before interned sites, directories, slab slack, and allocator
   overhead. The nominal shadow budget therefore does not predict process RSS.
8. P1 removes the measured common-path fixed work without changing findings or
   degrading retained coverage. Global-only dispatches now create no LDS
   workgroup state; batched LDS setup cuts clock-matrix growth by 75%, cached
   bindings cut large LDS/GEMM workgroup lookups by 90% to 98%, and the aligned
   GEMM fast path avoids expanding 74.9% of its dword candidates.
9. In the controlled P1/P0 matrix, P1 was 2.0% faster on the small set and 0.7%
   faster on the large set without JSON collection; both intervals included
   zero. With collection enabled, P1 was 4.2% and 4.9% faster, with only the
   large interval excluding zero. The exact no-work reductions are real, but
   the aggregate no-report timing gain is modest at these shapes.
10. An audit found that the earlier P0 harness enabled reporting for variants
   named `*-no-report`, because that suffix also ends in `-report`. Its
   report/no-report and no-report/pre-P0 conclusions are invalid. A corrected
   matrix measures report collection at +7.4% for P0 and +2.9% for P1 on the
   large set; the earlier report-to-report P0/pre-P0 comparison remains valid.
11. P2 replaces the global `unordered_map`/relocatable-vector layout with
   per-dispatch flat directories and bounded page, record, and site slabs. The
   budget is now one exact modeled checker-wide cap rather than 256 isolated
   shares, and dispatch cleanup returns its charge immediately.
12. P2 improves modeled storage and budget fairness, but it is not a runtime
    win over P1 on these synthetics. In the final controlled matrix, P2 without
    a report was +16.5% on the noisy small set and +11.1% on the stable large
    set versus P1. Exact site interning is the main new hot path; compact
    8-byte site-directory entries and removal of redundant lookup/tracking work
    recovered seven percentage points from the first P2 large-set result.

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

The historical realistic campaign used 20
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

### Final P2 realistic Gluon campaign

The final P2 Release binary, the same binary with checking off, and original
`develop` were run over the same 20 kernels and shapes. Each variant used five
rotating processes with two timed dispatches; each kernel value is the median
of ten pooled dispatches. The aggregate is the geometric mean of those
per-kernel median ratios. Intervals below are 20,000-sample within-kernel
dispatch bootstraps. JSON statistics collection was disabled.

The P2 binary was built from the staged branch state based on
`ddf6d8a7ac085b473d5e08bb2e7c858b106a6b36` and had SHA-256
`2890bb9d52dea0a387214cfe0ffe200194e204ffcc41c9202f9f4f77a730da19`.
The clean `develop` source was
`d97b52eb129c9bed93fb34b223fcb728e1746334`; its binary SHA-256 was
`0aecb6ad6902fd7cfadd6d17e52d42378921656d51d24f5f2e35d971e40f6598`.
The Gluon tutorials were at `32418e5af4214c8bcd5343eda02574fab5148edf`.

- P2 race-off is +1.4% versus `develop` (-0.8% to +2.5%).
- At the 512 MiB default, P2 race-on is +19.5% versus race-off (+18.2% to
  +21.5%) and +21.1% versus `develop` (+19.3% to +22.2%). Individual kernels
  range from +4.2% to +50.6%.
- At 4 GiB, P2 race-on is +20.0% versus race-off (+18.6% to +22.1%) and +21.6%
  versus `develop` (+19.8% to +22.8%). Individual kernels range from +3.6% to
  +51.0%.
- Full coverage is +0.4% versus the default budget (-0.4% to +1.3%), so no
  timing difference between the two budgets resolves.

All 400 processes exited successfully, each kernel's digest matched across all
variants, and no race or wait finding appeared. All 100 default-budget race-on
processes emitted the retention warning; none of the 100 4 GiB processes did.
The added dispatch-window peak RSS ranged from 0.58 to 1.12 GiB at the default
budget and 0.83 to 1.21 GiB at 4 GiB. Race-off used 1.97 to 2.20 GiB.

Against the pre-final-refactor campaign on the same suite and shapes, the
default-budget point estimate changes from +20.4% to +19.5%, while the
full-coverage estimate changes from +26.2% to +20.0%. Added full-coverage RSS
also falls from 1.4--2.1 GiB to 0.83--1.21 GiB. These runs occurred at different
times and are not a direct revision A/B, but the complete P2 result provides no
evidence of a realistic-workload regression from the redesign.

### Pre-P0/P1 synthetic baseline

Four memory-intensive gfx950 workloads--stream, LDS reduction, GEMM, and
histogram--were run against the post-redesign, pre-P0/P1 Release binary with
checking on and off. The small set used 262,144 elements and GEMM size 128. The
large set used 1,048,576 elements and GEMM size 256. Geomeans below exclude the
compute-only control.

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
the pre-P0/P1 checker's incremental cost.

These synthetics deliberately remove most non-memory work. They expose the
upper-cost regime of the checker and are not directly comparable to the
20-kernel geomean. The realistic campaign already contained individual kernels
at +52% with the default budget and +73% with full coverage.

### P1 versus `develop` and race-off

The final P1 Release binary and the original `develop` binary were then run in
one pinned, rotating nine-repetition matrix over the same eight serial
workloads. The P1 binary ran once with checking disabled and once with checking
enabled. No JSON report was configured. All 216 runs passed and no checked run
degraded coverage.

| Comparison | Small set | Large set |
|---|---:|---:|
| P1 race-off / `develop` | +10.4% (-4.1% to +17.1%) | -0.4% (-5.9% to +3.1%) |
| P1 race-on / P1 race-off | +26.3% (+17.0% to +43.3%) | +41.0% (+34.7% to +43.0%) |
| P1 race-on / `develop` | +39.4% (+26.1% to +48.2%) | +40.4% (+32.3% to +42.2%) |

The small race-off point estimate does not establish a 10% disabled-path
regression: stream and GEMM were bimodal, and the interval includes zero. The
longer large set is stable and shows no resolved race-off cost. Compared with
the older +40.7% / +51.2% race-on-versus-`develop` headline, the latest direct
P1 figures are +39.4% / +40.4%. This direction is favorable, especially on the
large set, but it combines P1 changes with run-to-run host effects; the
P1-versus-P0 matrix below isolates the optimization itself.

### Current mutation-harness sweep

A serial gfx1250 sweep used 21 small mutation shaders and five processes per
variant, with ordinary wait checking enabled in both variants. Enabling address
race checking added 3.7% geomean whole-process wall time, with a bootstrap 95%
interval of approximately +1.9% to +5.4%. The global shadow did not exhaust.

These CI-sized kernels show the low-cost end of the range. Their whole-process
timing includes startup and is not comparable to per-dispatch Gluon timing.

### Earlier two-kernel representative rerun

FMHA v4 and A16W16 v9 were rerun on an earlier checker build with one process per
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
+49.2% with the default budget and +53.4% with 4 GiB. That result is
directionally faster by 13.3 and 10.8 percentage points, respectively.

This is still not a controlled redesign A/B. The runs happened at different
times, and the earlier rerun used only one process per variant. It should be
treated as evidence against a large redesign regression, not a precise estimate
of the redesign's benefit. The complete final P3 campaign below supersedes it
for current overhead.

### Corrected P0/P1 optimization matrix

The first P0 harness used `build.endswith("-report")` to choose report
variants. That expression also matched names ending in `-no-report`, so every
purported no-report variant actually configured a report. The old
P0/pre-P0 no-report row and both old report/no-report conclusions must not be
used. The report-to-report P0/pre-P0 comparison was configured as labeled and
remains valid: +16.7% on the noisy small set and +6.2% on the large set.

The corrected harness uses an explicit set of report variant names. P0 and P1
each ran with and without a report, nine times in rotating order over the same
eight serial synthetics. Processes were pinned to CPUs `2-3,194-195`; each
workload value is the median of nine dispatch times. All 288 runs passed, no run
degraded coverage, and every report had zero rejected bytes.

| Workload | P0 no report | P1 no report | P1/P0 | P0 report | P1 report | P1/P0 |
|---|---:|---:|---:|---:|---:|---:|
| Stream 256K | 97.326 ms | 94.589 ms | -2.8% | 106.162 ms | 98.311 ms | -7.4% |
| LDS reduce 256K | 125.412 ms | 125.757 ms | +0.3% | 165.005 ms | 148.962 ms | -9.7% |
| GEMM 128 | 109.105 ms | 105.788 ms | -3.0% | 113.754 ms | 111.906 ms | -1.6% |
| Histogram 256K | 105.896 ms | 103.350 ms | -2.4% | 122.152 ms | 125.317 ms | +2.6% |
| Stream 1M | 377.892 ms | 361.424 ms | -4.4% | 407.408 ms | 374.462 ms | -8.1% |
| LDS reduce 1M | 473.364 ms | 480.654 ms | +1.5% | 535.197 ms | 517.799 ms | -3.3% |
| GEMM 256 | 743.592 ms | 728.190 ms | -2.1% | 787.509 ms | 776.361 ms | -1.4% |
| Histogram 1M | 296.330 ms | 302.406 ms | +2.1% | 305.300 ms | 284.656 ms | -6.8% |

| Geomean comparison | Small set | Large set |
|---|---:|---:|
| P1 no report / P0 no report | -2.0% (-9.6% to +9.6%) | -0.7% (-6.0% to +1.6%) |
| P1 report / P0 report | -4.2% (-17.5% to +2.7%) | -4.9% (-8.5% to -0.9%) |
| P0 report / P0 no report | +14.6% (+4.4% to +28.8%) | +7.4% (+1.9% to +12.5%) |
| P1 report / P1 no report | +12.1% (-3.0% to +18.1%) | +2.9% (+1.2% to +7.4%) |

The no-report optimization deltas are small relative to the loaded-host noise;
neither interval excludes zero. P1's large-set report result does exclude zero,
and report collection itself has a measurable large-set cost. This corrects the
earlier claim that collection cost was unresolved. The matrix isolates P1 from
P0, while the preceding P1/`develop` matrix answers absolute overhead.

### P2 global-shadow result

The final P2 Release binary and P1 were run in one pinned, rotating
nine-repetition matrix over the same eight serial workloads, each with and
without JSON collection. All 288 cases passed; no run degraded coverage, every
dispatch had zero rejected bytes, and the largest modeled peak was 105.7 MB.

| Geomean comparison | Small set | Large set |
|---|---:|---:|
| P2 no report / P1 no report | +16.5% (+2.5% to +24.4%) | +11.1% (+7.5% to +12.3%) |
| P2 report / P1 report | +1.3% (-3.2% to +12.5%) | +10.6% (+7.0% to +13.3%) |
| P2 report / P2 no report | +0.2% (-6.1% to +9.8%) | +5.4% (+3.5% to +8.1%) |

The large set is the more stable result. P2 was +29.1% over P1 on stream,
+5.8% on LDS reduction, +1.0% on GEMM, and +10.3% on histogram without a
report. Stream exposes the cost of exact site interning: its large dispatch
interned 1.37 million sites for 3.15 million inserted records. GEMM reused sites
much more heavily (24,936 interned for 327,736 total site uses) and was
effectively unchanged.

The first complete P2 build was +18.1% over P1 on the large no-report set and
+15.2% with a report. Sampling identified duplicate site-table probes, repeated
hashing and per-record tracking atomics in the new path. Removing those,
reusing the discovered insertion slot, and packing each site-directory entry
from 16 to 8 bytes reduced the final large-set penalty to +11.1% and +10.6%.
The same packing reduced the large stream's modeled peak from 122.1 MB in the
first P2 build to 105.7 MB.

This is a storage/accounting improvement, not an elapsed-time improvement over
P1. The large stream's median report RSS fell from 330.6 MB on P1 to 315.6 MB
on P2; LDS reduction was unchanged, GEMM rose by 3.0 MB, and histogram was
unchanged. Small workloads can pay slab fragmentation and fixed per-shard
costs, while large, site-heavy stream benefits from the compact directory.

A separate rotating P2 race-off/on/`develop` matrix ran 216 cases. P2 race-off
was -4.3% (-14.2% to +5.8%) on the noisy small set and -1.7% (-4.8% to +2.1%)
on the stable large set, so no disabled-path regression resolved. Race-on was
+43.7% (+32.7% to +59.3%) versus race-off on the small set and +57.9%
(+52.3% to +61.1%) on the large set; versus `develop`, it was +37.5% and
+55.2%. These absolute figures and the direct P2/P1 result answer different
questions and should not be arithmetically combined across matrices.

A rotating five-repetition budget control compared the default 512 MiB cap
with 4 GiB over the same eight workloads. All 80 cases passed with zero
rejected bytes and no coverage degradation. The largest modeled peaks were
105.7 MB at the default and 105.8 MB at 4 GiB, both well below the default cap.
The 4 GiB/default timing geomean was +3.1% on the noisy small set and +1.1% on
the large set. This does not measure a benefit from a larger budget; it confirms
that the P1/P2 and race-off/on matrices compared equal, fully retained coverage.

Seven-repetition standalone probes exercised the new layout directly:

- Dense retention kept 1,048,576 records across 4,096 pages, charged
  25,351,168 bytes, and had median access/cleanup times of 64.82/0.17 ms.
- Sparse retention kept 16,384 records on 16,384 pages and charged 24,457,216
  bytes. Concentrating the same page count in one shard charged 17,567,872
  bytes, exposing directory/slab skew rather than hiding it in a per-shard cap.
- Every cleanup returned current charge to zero. A two-wave cleanup/reuse case
  peaked at 2,383,872 bytes in each wave and released 4,767,744 bytes in total.
- After one failed saturated-page admission, 99,999 repeated attempts hit the
  saturation cache. Retained history still produced a finding, and growth plus
  cleanup invalidated the cache as designed.

The probe timings are implementation diagnostics, not application benchmarks;
their accounting and cleanup invariants are the sign-off result.

### P3 compact-access result

P3 keeps a regular vector access as one affine descriptor containing its base,
stride, size, first lane, and lane count. Contiguous active lanes with
monotonic affine addresses take this path. Sparse or descending lanes,
irregular gathers, per-element masks, combined DS2 accesses, and LDS accesses
requiring per-lane clipping retain the explicit-range path. Unaligned affine
descriptors remain compact at the API and in pending LDS state but use the
generic byte-mask iterator. Aligned descriptors use contiguous or disjoint
dword runs while preserving the first contributing lane for diagnostics.

The final P3 and P2 Release binaries ran in one pinned, rotating,
nine-repetition matrix over the same eight serial synthetics. P3 also ran with
checking off, and the original `develop` binary supplied the disabled-path
control. All 288 processes passed; no checked process rejected bytes or
degraded coverage.

| Geomean comparison | Small set | Large set |
|---|---:|---:|
| P3 race-on / P2 race-on | +0.2% (-7.3% to +5.5%) | +0.2% (-2.3% to +2.9%) |
| P3 race-off / `develop` | +5.1% (-8.3% to +11.4%) | -0.8% (-2.7% to +3.0%) |
| P3 race-on / P3 race-off | +41.6% (+31.6% to +53.1%) | +55.3% (+51.3% to +59.9%) |
| P3 race-on / `develop` | +48.9% (+31.8% to +56.3%) | +54.0% (+51.8% to +60.0%) |

The direct P3/P2 comparison is the phase result: no elapsed-time change
resolves. Individual large-workload medians changed by +1.1% for stream, -2.7%
for LDS reduction, +2.9% for GEMM, and -0.4% for histogram. The separate
on/off ratios moved with noisy race-off baselines and must not be interpreted
as a P3 regression when the direct revision comparison is flat.

The largest dispatch counters explain the result:

| Workload | Affine instructions | Logical ranges compact | Compact pending LDS ops | Fixed work versus P2 |
|---|---:|---:|---:|---|
| Stream 1M | 33.3% | 97.0% | 0 | Unchanged dwords and records |
| LDS reduction 1M | 75.5% | 99.1% | 131,072 | 122,880 range-vector growths removed |
| GEMM 256 | 0% | 0% | 0 | Entirely explicit; unchanged |
| Histogram 1M | 33.3% | 49.6% | 0 | Unchanged dwords and records |

Across every workload, P2 and P3 had exactly equal `dword_candidates`,
`dword_visits`, inserted records, and visited global/LDS records. P3 therefore
removes range materialization and keeps in-flight LDS ranges compact, but does
not yet reduce the dominant per-dword shadow work. The compute path still
checks each active address to prove the affine relation, and each shadow visit
still constructs exact lane provenance. The reused thread-local explicit
vectors also rarely allocate in steady state. GEMM, whose checker path is the
largest of these synthetics, receives no compact descriptor at all.

Current P3 software samples agree with the counters. Stream is led by exact
global-site lookup (8.8% of all CPU-time samples) and the affine per-dword
callback (2.6%). LDS reduction resolves in the affine LDS-record loop, site
lookup, and state update. GEMM remains led by the explicit LDS loop (6.3%),
range preparation (2.5%), global checking (2.4%), and clipping (1.7%).
Sampling percentages are directional single-run evidence, but they show that
P3 moved representation work without displacing shadow traversal and exact
site interning as the main remaining costs.

The final realistic campaign repeated the P2 protocol over the same 20 Gluon
kernels: five rotating processes per variant, two timed dispatches per process,
the default 512 MiB budget, and a full-coverage 4 GiB budget. JSON statistics
collection was disabled. The P3 launcher SHA-256 was
`dc15cf02b309c0fd49f6f84d3b3cc21851ff9574e70951966ee0eb8c23b621fd`;
the source was the complete P3 working-tree state based on
`ddf6d8a7ac085b473d5e08bb2e7c858b106a6b36`. The `develop` binary and Gluon
tutorial revision were unchanged from the P2 campaign.

- P3 race-off was +0.2% versus `develop` (-1.0% to +2.3%).
- At the default budget, P3 race-on was +21.2% versus race-off (+19.4% to
  +22.9%) and +21.4% versus `develop` (+20.2% to +23.7%). Individual on/off
  kernel ratios ranged from +2.8% to +50.7%.
- At 4 GiB, P3 race-on was +21.8% versus race-off (+19.8% to +23.1%) and
  +22.0% versus `develop` (+20.6% to +23.9%). Individual ratios ranged from
  +3.3% to +52.1%.
- Full coverage was +0.5% versus the default budget (-0.9% to +1.4%), so no
  budget timing difference resolves.

All 400 processes exited successfully, all 20 output digests matched across
variants, and no race or wait finding appeared. All 100 default-budget runs
degraded retention; none of the 100 4 GiB runs did. Added dispatch-window peak
RSS ranged from 0.58 to 1.08 GiB at the default and 0.84 to 1.20 GiB at 4 GiB;
race-off used 1.97 to 2.20 GiB.

P2's corresponding on/`develop` results were +21.1% and +21.6%, versus P3's
+21.4% and +22.0%. Normalizing each campaign to its contemporaneous `develop`
changes the P3/P2 point estimates by only +0.2% and +0.3%. The larger P3
on/off values (+21.2%/+21.8% versus +19.5%/+20.0% for P2) come from P3
race-off measuring closer to `develop`, not a resolved increase in race-on
time. The campaigns were not interleaved with each other, while the direct
synthetic P3/P2 matrix above measured +0.2% with intervals spanning zero.
Together, the data show no resolved P3 runtime gain or regression.

### Post-P3 lock-scope audit

The synchronization audit kept locks that establish simulated ordering or
protect shared shadow state, but removed unrelated work from their critical
sections:

- report writers copy a revisioned snapshot under the report-state mutex, then
  serialize JSON and perform file I/O under a separate writer mutex;
- checker report deduplication retains its short mutex, while warning logs,
  report updates, and user reporter callbacks run after it is released;
- workgroup and global-dispatch objects are detached while their map/shard is
  locked, then statistics aggregation, slab destruction, and exact budget
  release run outside that lock;
- command processors mark diagnostics pending while retiring a dispatch and
  flush only after releasing the queue lifecycle mutex, including terminal VM
  faults and failed queue-close paths; and
- parallel workgroup completion no longer performs a report-pending atomic
  exchange for every notification. Parallel retirement flushes at the
  `on_cu_idle()` join point.

Per-instruction CU wave-state locking remains necessary: it orders simulated
issue, scoreboards, barriers, and wave lifetime. The checker also still holds a
workgroup-state lock while updating LDS clocks/history and one global-shard
lock while traversing and updating that shard's records. Local preparation and
all external callbacks are outside those locks where correctness permits.

The final lock-scope Release binary
(`a6a53e868bf508c03a049ab42ca4eae0ff35868b665b52f12a42a7ef0e8614b0`)
was compared directly with the final P3 binary
(`dc15cf02b309c0fd49f6f84d3b3cc21851ff9574e70951966ee0eb8c23b621fd`).
The no-report matrices used 15 rotating repetitions of all eight synthetics;
the serial matrix was pinned to the same four logical CPUs and the threaded
matrix used the normal gfx950 configuration. JSON collection used nine serial
and 15 threaded repetitions. All 864 processes passed, produced the expected
outputs, and retained full coverage.

| Lock-scope/P3 geomean | Small set | Large set | All eight |
|---|---:|---:|---:|
| Serial, no JSON | -1.0% (-8.7% to +5.6%) | +0.2% (-1.9% to +2.4%) | -0.4% (-4.5% to +3.0%) |
| Threaded, no JSON | +0.4% (-4.4% to +4.7%) | +0.3% (-5.0% to +4.4%) | +0.4% (-3.4% to +3.2%) |
| Serial, JSON statistics | -1.5% (-10.7% to +16.5%) | +1.7% (-0.4% to +4.1%) | +0.1% (-4.8% to +8.9%) |
| Threaded, JSON statistics | -1.4% (-4.4% to +4.1%) | -0.8% (-7.8% to +6.0%) | -1.1% (-4.7% to +3.3%) |

Every interval includes zero. The audit therefore introduces neither a
resolved regression nor a resolved speedup. Serial JSON runs averaged about
26--35 ns per measured lock acquisition, including timer cost. In threaded
JSON runs, workgroup-state acquisitions remained about 30--37 ns and
workgroup-map acquisitions about 97--171 ns, while global-shard acquisitions
ranged from 0.30 to 5.81 microseconds depending on workload. P3 and the audited
build alternated which workload had the lower wait, matching the flat elapsed
result. The data show material global-shard contention under threaded
instrumentation, but not a benefit from shortening only the nonessential
regions audited here; the required per-dword shared-shadow traversal remains
inside the shard lock.

### Selectable-family overhead and 2% decision

The current build exposes `race_checking_checks` and
`--race-checking-checks` for five independently selectable families:

- `global`, `lds-same-wave`, and `lds-cross-wave` select address domains;
- `direct-to-lds` and `tensor-dma` add operation sources to the selected
  domains and cannot be selected without a domain; and
- omission or `all` preserves the complete checker.

The final Release launcher had SHA-256
`fd85c7e8a3d8a399fa53f09dfdf74fdb5ea65c8c15c71402dc47895a31d69462`.
Its source was the complete selector working tree based on
`ddf6d8a7ac085b473d5e08bb2e7c858b106a6b36`. JSON statistics collection was
disabled.

The representative matrix used the same 20 gfx950 Gluon kernels, revisions,
and shapes as the P2/P3 campaigns. Every selector used the 4 GiB global-shadow
configuration so comparisons retained full coverage. Each selector ran in
rotating order in five processes per kernel with four timed dispatches per
process. A kernel value is the median of its 20 pooled dispatches; aggregate
values are geometric means of the 20 kernel ratios. Intervals are 20,000-sample
within-kernel dispatch bootstraps.

| Selection | Gluon overhead versus race-off | Per-kernel range | 2% assessment |
|---|---:|---:|---|
| `global` | +2.22% (+1.14% to +3.02%) | -2.32% to +15.49% | Fails: estimate and upper bound exceed 2% |
| `lds-same-wave` | +2.55% (+1.43% to +3.47%) | -2.21% to +8.70% | Fails: estimate and upper bound exceed 2% |
| `lds-cross-wave` | +6.13% (+5.01% to +7.15%) | +0.58% to +25.97% | Fails |
| Both regular LDS families | +6.19% (+5.18% to +7.21%) | +0.48% to +23.66% | Fails |
| All address domains, no direct-to-LDS | +8.99% (+7.66% to +9.96%) | -- | Fails |
| Add direct-to-LDS to all domains | +12.67% (+11.55% to +13.72%) incremental | -1.97% to +33.70% | Fails |
| Add tensor DMA to all domains | +0.80% (+0.13% to +1.46%) incremental | -1.76% to +12.37% | No active gfx950 source; not qualifying evidence |
| `all` | +22.81% (+21.44% to +23.62%) | +3.98% to +52.51% | Fails |

All 800 processes passed. Each kernel had one matching digest across all
selectors, no run emitted a race or wait finding, and no run degraded coverage.
Host load was about 46 on 384 logical CPUs at the start and end. The matrix ran
selectors sequentially; no benchmark process competed with another benchmark
process.

The LDS result has an important dependency detail. Adding same-wave checking
to an already selected cross-wave checker was +0.05% (-0.86% to +1.07%), while
adding cross-wave checking to same-wave was +3.55% (+2.65% to +4.65%). The
cheap first increment reflects shared LDS routing, state, pending-operation,
and lifecycle work. It does not make same-wave checking a viable default-on
bundle: the prerequisite cross-wave configuration is +6.13%, and both
families together are +6.19%.

The source-family Gluon comparisons also need an architecture qualifier.
Direct-to-LDS is active on gfx950 and accounts for the largest incremental
share of the complete realistic result. Tensor DMA is not a gfx950 instruction
source, so `all` and the no-tensor selector should perform the same work there;
their +0.80% difference is a useful measurement-noise control, not the cost of
an executed check.

Dedicated source workloads therefore measured the source switches on paths
that execute them. Fifteen rotating repetitions used 1,024 transfers per
thread/workgroup configuration:

| Active source | Source increment with same-wave LDS | With cross-wave LDS | With all address domains |
|---|---:|---:|---:|
| gfx950 direct-to-LDS | +4.61% (+0.93% to +9.12%) | +15.92% (+12.62% to +18.01%) | +36.21% (+31.39% to +38.45%) |
| gfx1250 tensor DMA | -2.69% (-5.59% to +3.97%) | +7.46% (+1.76% to +9.66%) | +20.89% (+15.08% to +29.39%) |

Tensor DMA with same-wave LDS is unresolved, but its upper bound is +3.97%;
it does not establish a sub-2% active path. Direct-to-LDS fails in every
selected domain combination.

The serial synthetic matrix confirms that the realistic percentages are not
upper bounds for memory-heavy code. Nine rotating repetitions of the eight
small/large workloads produced these large-set results:

| Selection | Large-set overhead versus race-off |
|---|---:|
| `global` | +33.59% (+30.87% to +38.76%) |
| `lds-same-wave` | +11.59% (+9.58% to +18.69%) |
| `lds-cross-wave` | +19.88% (+18.10% to +22.47%) |
| Both regular LDS families | +20.32% (+18.46% to +23.93%) |
| `all` | +56.70% (+54.45% to +64.81%) |

Focused active-path workloads were higher again: global shadow traversal was
+72.06%, same-wave LDS was +17.90%, cross-wave LDS was +38.20%, and both LDS
families were +36.67%. Those probes deliberately maximize checked memory work;
they show why a family near 2% on a heterogeneous suite still cannot be treated
as a universal 2% bound.

The acceptance rule for default-on consideration is that both the point
estimate and the 95% upper bound stay at or below 2% on representative
workloads that actively execute the check, with unchanged findings and
retained coverage. No active family meets that rule. The only sub-2%
incremental results are same-wave LDS added after an already over-budget
cross-wave configuration and a dormant gfx950 tensor-DMA source; neither is an
independent default-on option.

### P0 fixed-work evidence

The largest dispatch from each large workload produced the following counters.
`Record work` is retained records visited / inserted. Lock time is total
measured acquisition time divided by acquisitions across workgroup-map,
workgroup-state, and global-shard locks.

| Workload | Dword candidates / visits | Duplicate candidates skipped | Record work | Workgroups created | Wave-search steps | Vector growths | Mean lock acquisition |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stream 1M | 3.33M / 3.33M | 0% | 0.36M / 3.15M | 4,096 | 0.19M | 17.7K | 26.9 ns |
| LDS reduce 1M | 5.35M / 5.35M | 0% | 1.27M / 5.24M | 4,096 | 1.10M | 205.7K | 26.3 ns |
| GEMM 256 | 37.82M / 9.51M | 74.9% | 10.11M / 7.67M | 256 | 1.03M | 21.9K | 26.8 ns |
| Histogram 1M | 2.13M / 2.13M | 0% | 2.16M / 1.05M | 1,024 | 77.8K | 5.0K | 26.6 ns |

The counters refine the optimization order:

1. Stream and histogram created 4,096 and 1,024 workgroup states despite
   recording zero LDS accesses. Avoiding LDS state for zero-LDS workgroups is
   the clearest exact no-work path.
2. GEMM generated 37.82 million dword candidates and discarded 28.31 million
   (74.9%) in the recent-dword cache. Earlier range coalescing or an affine
   fast path can avoid candidate generation rather than merely rejecting it.
3. Record work is larger than page lookup or lock work. The large dispatches
   inserted 1.05 to 7.67 million records and visited as many as 10.11 million.
4. LDS reduction performed 205,675 vector growths: 122,880 range, 41,835
   record, 24,576 pending, and 16,384 clock-matrix growths. Batch sizing and
   reuse have direct evidence.
5. LDS reduction and GEMM each performed about one million linear wave-search
   steps. Batch registration and a lifetime-safe cached wave index remain
   justified.
6. All three serial lock classes averaged 26 to 28 ns per acquisition. That
   includes the clock calls and uncontended mutex cost, so these data do not
   justify a lock-topology redesign. A threaded scaling run is still required
   to measure contention.

### P1 fixed-work effect

P1 implements the exact common-path changes motivated above. The largest
dispatch counters changed as follows:

| Workload | Dword candidates / visits | Workgroups | Clock growths | Wave-search steps | Workgroup lookups |
|---|---:|---:|---:|---:|---:|
| Stream 1M | 3.33M / 3.33M -> unchanged | 4,096 -> 0 | 16,384 -> 0 | 188,416 -> 0 | 81,920 -> 0 |
| LDS reduce 1M | 5.35M / 5.35M -> unchanged | 4,096 -> 4,096 | 16,384 -> 4,096 | 1,097,728 -> 368,640 | 421,888 -> 40,960 |
| GEMM 256 | 37.82M / 9.51M -> 9.51M / 9.51M | 256 -> 256 | 1,024 -> 256 | 1,030,656 -> 81,920 | 388,096 -> 8,448 |
| Histogram 1M | 2.13M / 2.13M -> unchanged | 1,024 -> 0 | 4,096 -> 0 | 77,824 -> 0 | 32,768 -> 0 |

The zero-LDS path removes all LDS state and teardown work from stream and
histogram. Batch registration sizes each clock matrix once. A cached,
lifetime-safe state handle and wave index avoid map lookup and linear search on
normal LDS accesses while retaining the per-workgroup state mutex. The aligned
range path skips GEMM's overlapping dword prefixes before expansion while
keeping first-lane attribution.

`dword_candidates` now counts only dwords that reach scalar expansion.
`dword_duplicates_skipped` still includes both prefixes skipped before
expansion and duplicates rejected by the recent-dword cache. Therefore the
P0/P1 candidate fields have deliberately different boundaries; `dword_visits`
remains directly comparable.

A one-workgroup `stream:32` probe makes the exact no-work effect explicit. P0
recorded 26 global accesses, one workgroup creation, 25 workgroup lookups,
eight clock growths, 101 wave-search steps and 160 dword visits. P1 recorded
zero for every race-work counter and no completed race-statistics dispatch.

All timed P0 and P1 dispatches retained full coverage (`rejected_bytes` stayed
zero). `retained_bytes` is an allocation-charge counter rather than a coverage
quantity and changed slightly with the P1 allocation pattern, so equal
coverage must not be inferred from equal retained-byte totals.

### P0/P1 validation

The final P1 build passed:

- all 180 tests selected by the complete race-detector unit filter,
- all 136 focused checker, report, configuration, CLI, and execution-plugin
  tests (one unrelated platform-dependent test was skipped),
- the 31-test P1-focused checker and execution selection,
- all 20 mutation-harness unit tests,
- the gfx950 mutation matrix (52 mutants) and gfx1250 matrix (83 mutants), both
  with clean baselines and no inconclusive runs, and
- JSON parsing and schema checks on the emitted benchmark reports, plus all 288
  P0/P1 and 216 P1/`develop` performance cases.

Both mutation matrices retain the same two documented `lds_war_pattern`
equivalent mutants: removing their pre-barrier wait creates no cross-wave byte
conflict, so the address-race checker intentionally emits no finding. They
account for the harness's two reported false negatives on each architecture and
are not a P1 regression.

### P2 validation

The final optimized P2 layout passed:

- all 238 tests selected by the complete race-detector unit filter,
- all 160 focused checker, report, configuration, CLI, and execution tests,
- all 20 mutation-harness unit tests,
- the gfx950 matrix (52 mutants) and gfx1250 matrix (83 mutants), both with
  clean baselines, no inconclusive runs, and only the same two documented
  equivalent misses per architecture,
- a concurrent focused probe under AddressSanitizer and
  UndefinedBehaviorSanitizer with leak detection, and
- all 288 P2/P1, 216 P2/`develop`, and 80 default/4-GiB performance cases with
  no rejected bytes or coverage degradation, plus all 400 final Gluon cases
  with the expected default-budget warnings and full 4 GiB retention.

ThreadSanitizer probes compile, but its runtime cannot initialize on this host.
The monolithic sanitizer test executable also exceeds the x86-64 small code
model at link time, so the focused threaded global-shadow probe is the
sanitizer sign-off.

### P3 validation

The compact representation passed:

- all 242 tests selected by the complete race-detector unit filter,
- all 164 focused checker, report, configuration, CLI, and execution tests,
- all 20 mutation-harness unit tests,
- the gfx950 matrix (52 mutants) and gfx1250 matrix (83 mutants), each with a
  clean baseline, no inconclusive run, and only the same two documented
  equivalent `lds_war_pattern` misses,
- an affine concurrent-shadow probe under AddressSanitizer and
  UndefinedBehaviorSanitizer with leak detection, and
- all 288 controlled P3/P2/`develop` synthetic performance processes with full
  retained coverage and matching fixed-work counts, plus all 400 final Gluon
  processes with matching digests, expected default-budget warnings, and full
  4 GiB retention.

ThreadSanitizer again compiled the focused threaded probe but could not
initialize on this host because of an unexpected memory mapping.

### Post-P3 lock-scope validation

The final lock-scope source passes the same 244-test complete race filter and
166-test focused checker/report/configuration/CLI/execution selection, all 20
mutation-tool unit tests, and the 21-case gfx1250 mutation integration matrix.
The 52-mutant gfx950 matrix again has a clean baseline, no inconclusive run,
and only the two documented equivalent `lds_war_pattern` misses. Thirty-six
focused VM-fault, failed-close, and PM4 queue lifecycle tests pass, including
new checks that direct terminal faults and faulted graceful closes flush only
after queue locks are released. The concurrent cleanup/report tests and those
flush-boundary tests also pass 20 repeated runs.

The threaded checker probe passes AddressSanitizer and
UndefinedBehaviorSanitizer with leak detection. ThreadSanitizer still compiles
and still cannot initialize on this host (`unexpected memory mapping`). All
864 final lock-scope/P3 performance processes passed with matching expected
outputs and no coverage degradation.

### Selectable-family validation

The final selector build passed:

- 174 tests in the complete core race/wait/configuration/checkpoint selection;
- 250 tests in the combined legacy-plugin and core race selection;
- the CLI selector CTest and all 41 mutation-harness Python unit tests;
- the gfx950 mutation matrix (52 mutants) and gfx1250 matrix (83 mutants), both
  with clean baselines, no inconclusive run, no false positive, and only the
  same two documented equivalent `lds_war_pattern` misses;
- execution tests proving that global and cross-wave findings require their
  selected family, and that direct-to-LDS and tensor-DMA sources can be removed
  without disabling the selected address domains;
- configuration, canonical formatting, source-without-domain rejection,
  checkpoint round-trip, omitted-selector compatibility, and CLI/attach/guest
  restriction coverage; and
- an LDS-only retirement regression that verifies dispatch statistics and
  finding deduplication are finalized even when no global family is selected.

All 576 general synthetic, 540 focused-path, 210 active-source, and 800 Gluon
performance processes passed, for 2,126 successful cases. None emitted a
finding or coverage warning. Every Gluon kernel produced one matching digest
across all eight selectors.

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
approximately 4% for the CI-sized mutation sweep and 21% to 22% across the
final P3 realistic suite. The current full-coverage selector campaign measures
+22.81% with all checks and decomposes that result into +2.22% global, +6.19%
regular LDS, and a +12.67% direct-to-LDS source increment; these percentages
interact and must not be added arithmetically. On the synthetic controls, P1
measured roughly 26% on the noisy small set and 41% on the stable large set;
the current all-check large-set result is +56.70%, consistent with P3's earlier
55% point estimate. P2's direct storage-layout cost over P1 was 17% and 11%,
while P3 was flat versus P2. The pre-P1 synthetic campaigns ranged from about
38% to 54%. The implementation history did not cause the earlier apparent jump
from the realistic geomean, but P2's exact site interning remains a measured
synthetic cost that representation-only P3 does not remove.

## Software profile

Hardware `perf` sampling was unavailable because `perf_event_paranoid=4`.
Userspace SIGPROF sampling was therefore run against the exact pre-P1 Release
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

### 1. Classify and iterate lane ranges

`ComputeUnitCore::track_address_races` separates FLAT LDS/global domains,
handles DS2 addresses, and constructs direct-to-LDS effects. P3 recognizes a
contiguous active-lane interval whose addresses form one non-descending affine
sequence. It passes that sequence as one descriptor and keeps it compact in
pending LDS state. The generic path still materializes ranges for sparse,
masked, descending, irregular, combined-DS2, and per-lane-clipped accesses;
tensor DMA also remains explicit.

The checker turns aligned affine descriptors into contiguous or disjoint dword
runs. Unaligned descriptors and explicit ranges use the generic recent-dword
cache, with P1's aligned overlapping-prefix skip. Both forms still visit every
distinct dword/mask sent to the shadow and construct exact first-lane
provenance. Proving an affine sequence also still examines each active lane's
address, so P3 reduces representation traffic rather than the asymptotic
address or shadow work.

Relevant code:

- [`compute_unit.cpp`](../lib/rocjitsu/src/rocjitsu/vm/amdgpu/compute_unit.cpp):
  `clip_to_lds_allocation`, `track_address_races`, affine classification, and
  `track_tensor_dma_accesses`
- [`memory_race_checker.cpp`](../lib/rocjitsu/src/rocjitsu/vm/amdgpu/memory_race_checker.cpp):
  affine and explicit `for_each_word` overloads

### 2. Locate and lock state

P1 workgroup startup obtains LDS state once, registers the known wave span as a
batch, and sizes the vector-clock matrix once. Each wave caches a
generation-checked shared state handle and wave index. Normal LDS accesses
therefore retain the per-workgroup state mutex but skip the map-shard lookup
and linear wave search; uncached callers keep the generic lookup path.
Zero-LDS workgroups allocate no checker state.

Global accesses hash pages into 256 shards and switch locks as the lane order
moves between shards. Each shard keeps per-dispatch flat page and site
directories. All shards reserve from one atomic checker-wide budget, so a busy
shard can use capacity left idle elsewhere.

### 3. Walk pointer-heavy shadows

Every touched dword walks a linked record list:

- LDS uses a dense/sparse dword-head index plus `LdsRecord` links.
- Global memory uses a flat page directory, page/record/site slabs, and
  per-dword linked lists. Records are 16 bytes and name exact interned
  `(pc, wave, lane)` sites through stable slab indices.
- Pending LDS operations keep one affine descriptor when possible; explicit
  fallbacks retain separate lane-range vectors. Both forms can be scanned
  again for overlap and barrier carry.

The layout makes exact byte masks and provenance available, but costs cache
locality, allocations, and pointer chasing.

### 4. Synchronize and retire

Barriers allocate member, merged-clock, and joined vectors, join vector clocks,
prune or clear history, and carry incomplete operations. Dispatch retirement
visits retained workgroups and global shards to erase dispatch state. P1 skips
LDS teardown for workgroups that allocated no checker state and attaches
completion tokens only to effects entering LDS history. A one-workgroup grid
also skips global history entirely because the checker compares global accesses
only between workgroups.

## Memory amplification

P2's modeled global-shadow charge has an exact implementation boundary:

- `GlobalPage` contains 256 four-byte heads (1,024 bytes) and pages are
  allocated in slabs of 16.
- `GlobalRecord` and its exact `GlobalSite` provenance are each 16 bytes;
  record and site slabs hold 1,024 and 256 entries respectively.
- page-directory slots are 16 bytes, site-directory slots are 8 bytes, and
  both deletion-free open-addressed tables grow at 75% occupancy.
- a normal first retention with full initial slabs charges 37,248 bytes. A
  tight budget can admit partial slabs; the minimum exact first-retention
  boundary is 21,888 bytes.

The dense base payload is therefore 20 bytes per source dword with one record
(four bytes of page head plus a 16-byte record), before site and directory
amortization and slab slack. A second owner adds another 16-byte record. Site
cost depends on reuse: one site can serve many records, while a unique site
also needs its 16-byte slab entry and a directory slot.

The cap includes these slab and directory payloads and transactional overlap
during directory growth. It excludes allocator metadata, small owner objects,
slab-descriptor vectors, mutexes, and fixed checker caches, so it remains
different from process RSS. The locality probes below quantify real charged
storage rather than relying on a single bytes-per-record estimate.

## Optimization roadmap

The priority is to reduce work while preserving byte masks, findings, and
retained coverage. Expected payoffs below are hypotheses to validate with
counters and controlled reruns.

### P0 (implemented): measure fixed work

The JSON `memory_diagnostic_report` now records counters for:

- materialized lane ranges and distinct dwords,
- shadow pages and records visited, retained, replaced, and rejected,
- empty versus nonempty record heads,
- allocations and vector growth,
- map/state lookup and lock-wait time,
- pending-operation candidates tested,
- barrier members, clock cells, carried operations, and cleanup work, and
- coverage retained and rejected per dispatch and shard.

The report keeps cumulative totals and details for the eight most recent
completed dispatches with address-race work; `last_dispatch` repeats the newest
entry. Per-shard arrays omit empty shards, so memory remains bounded in
long-lived processes. `retained_bytes` now counts successful modeled
directory/slab charges over time rather than current charge or live RSS;
`rejected_bytes` records failed admissions at the checker-wide budget.
P2-specific counters separately expose reservation attempts/failures/releases,
slab and directory growth, site reuse, saturation-cache behavior, dispatch
release, and checker-wide peak charge. P3 counters distinguish affine versus
explicit accesses and expose logical affine ranges, dword runs/visits, and
compact pending LDS operations. Lock timings include acquisition overhead and
any wait.

Collection is enabled only when `memory_diagnostic_report` is configured.
Console-only checking takes the mode branch but does not increment counters or
call the steady clock. This distinguishes algorithmic work, allocation and
contention, and prevents a faster capped run from being mistaken for an
optimization. The corrected report overhead and fixed-work findings are
reported in the P0/P1 sections above. Report collection is not free: it added
2.9% on the final P1 large set.

### P1 (implemented): remove work from common paths

1. Do not create LDS workgroup state when the allocation is zero.
2. Skip regular aligned overlapping prefixes before dword expansion, including
   the GEMM patterns whose recent-dword cache rejected about three quarters of
   candidates, while preserving first-lane attribution.
3. Skip global checking for a dispatch with only one workgroup, because the
   current global checker compares different workgroups only.
4. Attach completion tokens only to effects that can enter LDS pending state.
5. Batch-register the known workgroup wave span and size the vector-clock matrix
   once instead of growing it for each wave.
6. Cache a lifetime-safe workgroup-state handle and wave-slot index on the
   active workgroup/wave.

The fixed-work and timing effects are reported above. All paths retain the
state mutex, generic irregular-range handling, exact byte masks and producer
lane attribution.

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

### P2 (implemented): improve global-shadow locality and accounting

1. Per-dispatch shard state owns deletion-free flat page/site directories and
   bounded slabs for pages, records, and sites.
2. Stable slab indices replace owning page pointers and relocatable record
   vectors. A record is 16 bytes and names a 16-byte exact provenance site.
3. One atomic budget coordinator serves all 256 shards, uses CAS reservations,
   tracks current/peak charge, and releases a dispatch's modeled bytes only
   after its storage is destroyed.
4. A 64-entry direct-mapped saturation cache per shard skips repeated failed
   growth attempts only. Budget epochs invalidate entries after growth, cleanup,
   or a limit change; retained-record traversal and finding generation always
   run.
5. Report counters expose reservations, failures, releases, slab/directory
   growth, site reuse, and saturation behavior. Recent dispatches include
   released bytes and the checker-wide peak.
6. Hot-path follow-up removed duplicate site probes and per-record tracking
   atomics, reuses the discovered insertion slot, and packs site-directory
   entries to 8 bytes. These changes preserve exact interning and provenance.

### P3 (implemented): represent regular accesses compactly

1. Consecutive active lanes with monotonic affine addresses use one
   `(base, stride, size, first lane, count)` descriptor for global and LDS
   checking.
2. Aligned descriptors use compact contiguous or disjoint dword runs. Unaligned
   descriptors retain generic exact byte-mask iteration.
3. Compact LDS pending operations survive completion pruning and barrier carry
   without a separate lane-range vector.
4. Sparse, descending, irregular, element-masked, DS2, clipped, and tensor-DMA
   accesses retain the explicit fallback.
5. Report counters expose affine accesses, logical ranges, dword runs and
   visits, compact pending operations, and explicit accesses.

The direct P3/P2 synthetic result is flat because this phase changes the input
representation but deliberately preserves the exact count of shadow dword and
record operations. Differential tests cover page crossings, unaligned masks,
overlap/broadcast provenance, pending barrier carry, and explicit fallback.

### P3 follow-up: consume compact runs in bulk

The descriptor is now a safe boundary for larger changes. Batch page/shard
lookup, site/record insertion, and dense LDS-head traversal over an affine run
instead of invoking the scalar shadow callback for every dword. Add fallback
reason counters before extending classification to GEMM's patterns, which were
entirely explicit in the current synthetic. Compact owner/conflict summaries,
indexed in-flight state, and epoch-based LDS history retain the largest
potential payoff, but also the greatest correctness risk.

### Family controls (implemented): target the measured cost

The selector is an attribution and deliberate-coverage control, not an
optimization that makes the complete checker cheaper. The measurements sharpen
the next priorities:

1. Direct-to-LDS is the largest incremental realistic cost. Its global read and
   deferred LDS write currently enter the same scalar shadow paths as ordinary
   accesses; batching both effects and reusing one prepared provenance/site
   context should precede smaller lifecycle changes.
2. Cross-wave LDS carries nearly all of the regular-LDS realistic cost.
   Same-wave comparisons are cheap once that state exists, so optimizing only
   the pending same-wave scan cannot bring the combined family near 2%.
3. Global-only is close to 2% on the heterogeneous Gluon geomean but reaches
   34% on the large synthetic set and 72% on its focused active path. A
   compressed clean-owner/run shadow and bulk affine insertion are required
   before global checking can have a workload-independent 2% claim.
4. Tensor DMA must be judged on gfx1250, not by its dormant gfx950 selector.
   Its active all-domain source increment is 21%, so it needs the same bulk
   global/LDS effect handling as direct-to-LDS.

### Conditional: change lock topology

The post-P3 audit now measures serial and threaded lock behavior. Workgroup
state remains effectively uncontended because workgroups own independent
state, while pages that hash to one global shard show material threaded wait
under JSON instrumentation. Narrowing nonessential critical-section work did
not produce a resolved elapsed-time change because the dominant per-dword
record traversal and mutation must remain serialized today. Before adding
finer page ownership or lock-free read paths, first consume affine runs in
batches or use issuer-local summaries so one acquisition performs more useful
shadow work. Any topology change must retain exact producer provenance,
checker-wide budget accounting, and dispatch cleanup.

## Planning bounds

If all added elapsed time scaled linearly with race-specific work, reducing
that work would give the following Amdahl-style planning bounds:

| Workload regime | Measured added time | If race-specific work falls 25% | If it falls 50% |
|---|---:|---:|---:|
| Current eight-workload serial synthetic | +54.4% | +40.8% | +27.2% |
| Current large serial synthetic | +56.7% | +42.5% | +28.4% |
| Current Gluon geomean, full coverage | +22.8% | +17.1% | +11.4% |

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

The host used for these campaigns was shared and heavily loaded: approximately
400-560 load average during parts of the original synthetic/profile campaign,
about 101 during the representative kernel rerun, 44-47 during the final P1
matrices, roughly 43-70 during the final P2 Gluon matrix, and 42-49 when the
final P3 Gluon matrix started on 384 logical CPUs. The Gluon variants were
rotated and used five processes each, but did not have an exclusive CPU
reservation. The percentages therefore remain directional despite the
dispatch-bootstrap intervals.
