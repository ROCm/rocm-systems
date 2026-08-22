# ConSan gfx1250 status

This is the `gfx1250` workload × instrumentation evidence ledger.  It follows
the acceptance standard of [STATUS_RDNA4.md](STATUS_RDNA4.md), but inherits no
coverage denominator, machine-code identity, fault expectation, timing,
provenance, or green cell from another architecture.

The executable authority is
[`consan_validation.py`](../../tests/dbi/consan/consan_validation.py), with the
experiment contract described by [VALIDATION.md](VALIDATION.md).  This status
matrix is the authoritative progress tracker for gfx1250.

End-to-end evidence is the primary project metric.  Focused builder, decoder,
spill, and resource tests are prerequisites and debugging tools; they cannot
promote a workload cell by themselves.

## Status legend

Every cell uses the same simple maturity scale.  Colors describe how much of
the workload/profile contract currently works; failures, timeouts, coverage
gaps, and missing final evidence move a cell down that scale rather than
introducing separate exceptional states.

- 🩶 **unseen / unassessed:** no useful gfx1250 execution evidence yet;
- 🟥 **does not work:** current execution fails before establishing useful
  workload/profile behavior;
- 🟧 **some things work:** useful execution or instrumentation behavior is
  demonstrated, but substantial correctness, coverage, completion, or
  acceptance gaps remain;
- 🟨 **most things work:** the clean workload and important instrumentation
  path work, with limited coverage or final-acceptance evidence still missing;
- 🟩 **everything works:** every required gate is retained at one frozen
  revision, including clean, coverage, oracle, fault, containment, overhead,
  memory, timeout, health, and provenance evidence.

`N/A` is used only when a fresh gfx1250 inventory proves semantic absence and
records a typed reason.

## Current matrix

The historical final audit `consan-validation-gfx1250-final-audit-149` covered
all 40 cells at commit `9acc4dd9b0`.  Current-tip paired revalidation has
replaced those values, with the latest shared-branch merge audit recorded
below.  Successful process launch or focused unit coverage alone does not
preserve an earlier green claim.

| Workload | SuperCollider | Record/Replay | Sampled | Inline Shadow |
|---|---|---|---|---|
| **P0 Qwen3-0.6B prefill** | 🟧 Fresh current-workspace run transforms 1000/1000 accesses into a 365,536-byte object and reaches execution, but has no oracle or teardown verdict within 90 seconds; prior 1402/1402 evidence likewise had no verdict within 600 seconds | 🟧 Recompiling the unchanged source with the current O3 pipeline removes obsolete giant runtime transpose initializers: the exact full 151,936-logit baseline now passes in 64.49 seconds with the original 2.9-GB parameters. Record/Replay is statically complete at 846/846 accesses plus 80/80 barriers, transforms in 90 ms, and reaches execution, but remains dispatch-active after 28 launches at the bounded 180-second limit and therefore has no final replay/oracle verdict. Preserve this as an explicit emulator-throughput deferral rather than repeatedly extending the wait | 🟧 Current-hook RocJitsu execution retains complete 1000/1000 access plus 90/90 barrier instrumentation and a 3,945,440-byte object, remains dispatch-active through 198 observed launches including the final 151,936-cluster output kernel, but has no oracle or teardown verdict within 600 seconds | 🟧 Current clean-revision instrumentation is complete at 1000/1000 accesses plus 46/46 barriers, emits a 4,748,256-byte object, and reaches execution, but has no oracle or teardown verdict within 30 seconds |
| **P1 Sharktank TP1 prefill** | 🟩 Current clean-revision exact oracle in 12.13 seconds with complete 352/352 access coverage; current paired 1.17x retained | 🟩 Current clean-revision exact oracle in 13.50 seconds with complete 352/352 accesses plus 74/74 barriers, zero diagnostics, and a complete dynamic verdict under the target's bounded validation stride | 🟩 Current clean-revision exact oracle in 20.89 seconds with complete 352/352 accesses plus 64/64 applicable barriers, zero diagnostics or sampled conflicts, and a complete dynamic verdict; current paired 1.51x retained | 🟩 Current clean-revision exact oracle in 30.81 seconds with complete 352/352 accesses plus 37/37 barriers and a complete dynamic verdict; the target-native manifest retains a 60-second bound and current paired 2.11x |
| **P1 Sharktank TP1 decode/combined** | 🟩 Exact decode/combined oracles; 704/704 accesses; current paired 1.09x | 🟩 Exact decode/combined oracles; 704/704 accesses, 74/74 barriers; current paired 1.16x | 🟩 Exact decode/combined oracles; 704/704 accesses, 128/128 applicable barriers; current paired 1.28x | 🟩 Current accepted bundle: exact decode/combined clean and paired oracles, complete 704/704 accesses plus 74/74 barriers, 2.92x maximum paired slowdown, reviewed exact-one detected/pass barrier move, containment, health, cleanup, and clean provenance |
| **P2 Sharktank TP2 family** | 🟧 Current uninstrumented all-mode baseline exceeds 600 seconds; prior frozen bundle retained | 🟧 Current uninstrumented all-mode baseline exceeds 600 seconds; prior frozen bundle retained | 🟧 Current uninstrumented all-mode baseline exceeds 600 seconds; prior frozen bundle retained | 🟧 Current uninstrumented all-mode baseline exceeds 600 seconds; prior frozen bundle retained |
| **P4 hip-moi D128 block attention** | 🟩 Current clean-revision row passes both exact host-reference oracles in 19.85 seconds with complete 18/18 access coverage; prior paired 1.56x retained | 🟩 Current clean-revision row passes both exact oracles in 21.98 seconds with complete 18/18 accesses plus 8/8 barriers, zero diagnostics, and a complete dynamic verdict; prior paired 1.11x retained | 🟩 Current clean-revision row passes both exact oracles in 20.37 seconds with complete 18/18 accesses plus 8/8 applicable barriers and a complete dynamic verdict; prior paired 1.13x retained | 🟩 Current clean-revision row passes both exact oracles in 33.57 seconds with complete 18/18 accesses plus 4/4 barriers and a complete dynamic verdict; prior paired 1.09x retained |
| **P4 hip-moi D128 pressure attention** | 🟩 Current clean-revision row passes all four exact host-reference oracles in 60.15 seconds with complete 24/24 access coverage; prior paired 1.84x retained | 🟩 Current clean-revision row passes all four exact oracles in 73.84 seconds with complete 24/24 accesses plus 8/8 barriers, zero diagnostics, and a complete dynamic verdict; prior paired 1.33x and reviewed-fault bundle retained | 🟩 Current clean-revision row passes all four exact oracles in 61.22 seconds with complete 24/24 accesses plus 8/8 applicable barriers and a complete dynamic verdict; prior paired 1.12x retained | 🟩 Current clean-revision row passes all four exact oracles in 117.48 seconds with complete 24/24 accesses plus 4/4 barriers and a complete dynamic verdict; prior paired 1.29x retained |
| **P4 hip-moi WMMA attention** | 🟩 Current clean-revision row passes both exact host-reference oracles in 15.46 seconds with complete 18/18 access coverage; prior paired 1.78x retained | 🟩 Current clean-revision row passes both exact oracles in 16.91 seconds with complete 18/18 accesses plus 8/8 barriers, zero diagnostics, and a complete dynamic verdict; prior paired 1.17x retained | 🟩 Current clean-revision row passes both exact oracles in 15.22 seconds with complete 18/18 accesses plus 8/8 applicable barriers and a complete dynamic verdict; prior paired 1.15x retained | 🟩 Current clean-revision row passes both exact oracles in 19.91 seconds with complete 18/18 accesses plus 4/4 barriers and a complete dynamic verdict; prior paired 1.17x retained |
| **P4 hip-moi Stream-K arrival** | 🟩 Fresh exact clean run with complete 4/4 access coverage; prior paired 7.38x retained | 🟩 Fresh exact clean run with 4/4 accesses, 8/8 barriers, 10/10 atomics, 16/16 fences, and zero diagnostics; prior paired 2.41x retained | 🟩 Fresh exact clean run with 4/4 accesses, 8/8 applicable barriers, and 10/10 atomics; prior paired 2.72x retained | 🟩 Fresh exact clean run with 4/4 accesses, 4/4 barriers, and 10/10 atomics; prior paired 2.62x retained |
| **P4 hip-moi tree atomic-OR** | 🟩 Fresh exact clean run with complete 4/4 access coverage; prior paired 6.55x retained | 🟩 Fresh exact clean run with 4/4 accesses, 8/8 barriers, 10/10 atomics, 16/16 fences, and zero diagnostics; prior paired 2.04x retained | 🟩 Fresh exact clean run with 4/4 accesses, 8/8 applicable barriers, and 10/10 atomics; prior paired 2.57x retained | 🟩 Fresh exact clean run with 4/4 accesses, 4/4 barriers, and 10/10 atomics; prior paired 2.19x retained |
| **P4 Jakub cooperative matmul** | 🟩 Current clean-revision four-oracle row in 7.50 seconds with complete 70/70 access coverage; prior paired and reviewed-fault evidence retained | 🟩 Current clean-revision four-oracle row in 10.18 seconds with complete 70/70 accesses plus 8/8 barriers, zero diagnostics, and a complete dynamic verdict; prior paired and reviewed-fault evidence retained | 🟩 Final candidate-tree recheck passes all four oracles in 7.06 seconds with complete 70/70 accesses plus 8/8 applicable barriers and complete static and dynamic verdicts; the code-object-wide cluster-tuple gate reservation regression remains fixed, and prior paired and reviewed-fault evidence is retained | 🟩 Current clean-revision four-oracle row in 7.37 seconds with complete 70/70 accesses plus 4/4 barriers and a complete dynamic verdict; prior paired and reviewed-fault evidence retained |

### 2026-08-22 bounded Qwen Record/Replay revalidation

The generated gfx1250 VMFB previously in the workspace was stale relative to
the current IREE O3 pipeline.  It retained a giant
`151936x1024` runtime-transpose initializer and did not complete even an exact
uninstrumented run within 600 seconds.  Recompiling the unchanged checked-in
MLIR with the current compiler and the parameter-overlay options documented in
[VALIDATION.md](VALIDATION.md) removes that obsolete initializer.  The new
219,405-byte VMFB passes the unchanged full-model oracle against the original
2.9-GB parameter archive and 151,936-logit reference in 64.49 seconds through
RocJitsu.  This is a build-artifact readiness repair, not a reduced model or a
weaker oracle.

The paired Record/Replay run discovers and patches all 846 accesses and all 80
barriers in the applicable object, with zero unsupported or resource-failed
sites and a 90-ms transform.  It then remains inside execution after 28
dispatches at the 180-second bound, before final replay, teardown, or the exact
oracle.  The cell therefore remains orange with a concrete return point:
improve or isolate instrumented emulator throughput, or introduce a separately
reviewed reduced E2E denominator, before repeating full qualification.  Do not
turn this into an unbounded wait.

The durable behavioral disposition is already cross-target.  The adjacent
`ReusedLdsGemmPipeline` pair owns Qwen's same-storage reader-retirement edge,
and `HeterogeneousObject` owns its multi-kernel attribution shape; both run
correct/no-diagnostic and incorrect/required-diagnostic members through all
four engines on CDNA3/4/5 and RDNA3/4, plus physical CDNA4.  The compiler
kernarg-preload variant separately covers the generated CDNA3/CDNA4 entry
shape.  The remaining Qwen gap is completion and performance evidence for the
full gfx1250 emulation row, not an uncovered device-observable synchronization
contract.

### 2026-08-21 fixed-stack Inline scalar-route regression

The checked-in reduction of the physical-gfx950 PyTorch `torch.sort` empty-EXEC
failure is intentionally shared with gfx1250.  Expanding each correct/incorrect
member from one to eight LDS access sites selected the scalable Inline route
and exposed a gfx1250 simulator crash in under a second.  Fixed-stack owners
were choosing an available compact indirect router, but its call/PC state was
preserved in per-lane private storage and therefore did not exist for an empty
wave.  The planner now prefers the branch-only scalar-spill route for
fixed-stack RDNA4-family owners; focused gfx1250 and gfx1201 host tests assert
that choice even when the indirect router would otherwise fit.

The final baseline/Inline correct/incorrect matrix passes all 24 rows across
five simulated architectures and physical gfx950 in 1.13 seconds, including
the gfx1250 correct workload's exact scalar oracle and the incorrect workload's
required LDS conflict diagnostic.  This focused evidence does not by itself
promote any E2E matrix cell, but it closes the fast regression-test gap for the
route that the generalized workload uncovered.

### 2026-08-20 bounded Tensile Stream-K revalidation

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx1250-tensile-streamk-baseline-sc-bc6c434`
records accepted baseline and SuperCollider rows through RocJitsu at clean
source revision `bc6c434c03` and hook SHA-256
`1d59d7f12eae10ff42d2d5e9eb0b9d1db5f547a1f333cf72ef2973d5f0de4a22`.
Both produce their one required exact numeric row, finish in 7.22 and 7.73
seconds, and report 7.80 and 9.52 ms of valid device timing.  SuperCollider
has complete 320/320 access coverage and a complete dynamic verdict.  Commit
`bc6c434c03` makes the positive device-timing canary workload-specific so this
one-row functional smoke no longer inherits the 250 ms aggregate required by
repeated empirical measurements.

The preceding all-profile artifact
`rebase-20260820-gfx1250-tensile-streamk-all-b946b9c` first showed that
Record/Replay and Sampled were statically complete but did not reach a numeric
verdict within 55 seconds.  The current repair artifacts
`fix-gfx1250-streamk-rr-prefix1-call-key` and
`fix-gfx1250-streamk-sampled-call-key` supersede those two gaps.  Record/Replay
passes the exact oracle in 14.15 seconds with complete 320/320 access, 22/22
barrier, and 4/4 fence coverage, a complete dynamic verdict, and zero
diagnostics.  Sampled passes in 7.87 seconds with complete 320/320 access and
20/20 barrier coverage and a complete dynamic verdict.  Inline Shadow remains
fully accepted from the preceding artifact: it passes in 27.86 seconds with
complete 320/320 accesses and 11/11 barriers.

Two narrower Record/Replay diagnostics established that the former gap was
not only the ordinary 55-second budget.  Artifact
`diagnostic-gfx1250-streamk-rr-180` remains compute-active after complete
320/320 access, 22/22 barrier, and 4/4 fence instrumentation, but produces no
numeric row within 180 seconds.  More importantly,
`diagnostic-gfx1250-streamk-rr-prefix1-access` disables barrier and atomic
tracking and instruments only the first access candidate, the
`ds_store_b128` at code-object PC `0x40c4`; the client finishes in about seven
seconds but fails its exact numeric oracle.  The selected workgroup emits no
visible record, localizing the defect to the access relocation and its dense
call/probe path rather than report analysis or cumulative runtime cost.  The
repair makes collapsed spill routing compare the caller-PC snapshot retained
in its key SGPR; the old dispatcher instead rebuilt an expected PC in its
shared PC pair and compared that register with itself, making the first route
arm tautologically win and skipping the relocated store.  Focused host
regressions reject that self-comparison for Record/Replay access and barrier
dispatchers and for Sampled access dispatchers; the two full E2E artifacts
above prove the corrected Record/Replay and Sampled behavior.

### 2026-08-20 `007_sk_mxf4gemm_tdm` bounded replay refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx1250-mxf4-tdm-rr-inner600-c55a13a`
reassesses the complete 75-row configuration through RocJitsu at source
revision `c55a13a496` and hook SHA-256
`0fe9aab79ab7a2d8b589ee232a263cde69c8fe6d336d87ba487324e02fc980a7`.
The uninstrumented baseline passes all 75 exact rows in 232.01 seconds.  The
Record/Replay leg runs to its diagnostic 600-second bound, completing 29
exact rows and failing four.  Both PGR1 and PGR2 MT128x128 solutions fail for
problem sizes `(120,120,1,1024)` and `(128,128,1,1024)`, so this current
evidence supersedes the former PGR2-only candidate-145 localization.  Static
instrumentation covers 2448/2448 accesses and 544/544 barrier members, but
only 1/64 fences lowers; termination at the bound leaves no final analysis
verdict.

A focused current-tip diagnostic filters instrumentation to the two failing
MT128x128 kernels and runs only their first exact problem.  Access-only
Record/Replay passes with all 288 filtered access sites.  With barriers
enabled, patch limits 16 and 17 form an exact behavioral boundary: both PGR1
and PGR2 pass with the first 16 access and barrier candidates, while candidate
17 formerly made PGR1 leave every output element at its initial value.  The
newly admitted barrier is candidate 16 in zero-based resource logs, the
loop-end `s_barrier_signal` at `.text+0x143fc` (code-object PC `0x2f4fc`).
Object comparison found that its dense relay host displaced an
`s_set_vgpr_msb 4` transition plus live WMMA instructions.  That made the
appended copy use the new bank while the original continuation was statically
decoded in the old bank.  Dense gfx1250 hosts now require low-bank entry and a
mode-neutral displaced range.  A focused host regression reproduces the
transition-plus-dense-barrier shape, and the same patch-limit-17 E2E command
now passes exact PGR1 and PGR2 oracles with 17/288 accesses, 17/68 barriers,
and 8/8 fences selected.  A subsequent all-supported focused run also passes
both exact rows with 288/288 accesses and 68/68 barriers patched, 35,184
visible events, and zero diagnostics.  Its static verdict remains incomplete
because none of the 8/8 selected fences could be placed; that independent
coverage gap and a fresh full 75-row client verdict remain open.

The preceding 55-second diagnostic
`rebase-20260820-gfx1250-mxf4-tdm-rr-c55a13a` exposed a separate validation
readiness issue: the runner's outer timeout did not replace the Tensile
helper's generic inner default, cutting off baseline and Record/Replay alike.
The target/workload manifest now declares a regression-tested 1800-second
inner bound and 1860-second enclosing bound for the complete configuration.
The longer bound is readiness for a future full run; it does not change the
orange verdict because the current numerical failures occur well before the
600-second diagnostic cutoff.

### 2026-08-20 D128-block clean refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx1250-d128-block-all-8562605`
records a clean-tree baseline and all four strict profiles through RocJitsu at
source revision `856260551a`.  The baseline passes both exact host-reference
oracles in 18.25 seconds.  SuperCollider, Record/Replay, Sampled, and Inline
Shadow pass them in 19.85, 21.98, 20.37, and 33.57 seconds respectively.
Every engine has complete 18/18 access coverage, with 8/8 barriers for
Record/Replay and Sampled and 4/4 for Inline Shadow.  Every instrumented row
has a complete dynamic verdict, and Record/Replay reports zero diagnostics.
The loaded hook SHA-256 is
`59ae90f075525cb84925717b7322d8f0d98b59c7a41b9d3e81e989a5ce0615c4`.

### 2026-08-20 D128-pressure clean refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx1250-d128-pressure-all-a5ef68a`
records a clean-tree baseline and all four strict profiles through RocJitsu at
source revision `a5ef68ad2c`.  The baseline passes all four exact
host-reference oracles in 57.55 seconds.  SuperCollider, Record/Replay,
Sampled, and Inline Shadow pass them in 60.15, 73.84, 61.22, and 117.48
seconds respectively.  Current static coverage is complete at 24/24 accesses
for every engine, plus 8/8 barriers for Record/Replay and Sampled and 4/4 for
Inline Shadow.  Every instrumented row has a complete dynamic verdict;
Record/Replay reports zero diagnostics.  The loaded hook SHA-256 is
`59ae90f075525cb84925717b7322d8f0d98b59c7a41b9d3e81e989a5ce0615c4`.
This clean rebuild supersedes the older 40-access inventory for SuperCollider
and Sampled while retaining the separately accepted paired-overhead and
reviewed-fault evidence.

### 2026-08-20 Aorta-derived PyTorch readiness

The gfx1250 manifest now includes `pytorch-tdm-descriptor-add`,
`pytorch-cluster-load-sync`, `pytorch-torch-mode`, `pytorch-torch-topk`,
`pytorch-torch-sort`, `pytorch-scatter-reduce`, `pytorch-torch-histc`, and
`pytorch-norm-softmax`.  These rows are not promoted into the historical
40-cell matrix above until their shared runtime preflight passes.  The current
workspace wheel, `torch 2.14.0.dev20260722+rocm7.1`, works on physical gfx950
but is not a gfx1250-capable runtime: its bundled ROCr sees the synthetic
gfx1250 agent under RocJitsu, while bundled HIP reports zero supported GPU
agents and the wheel contains gfx950 but no gfx1250 runtime support.

A retained target-specific environment at
`rocm-systems-waitcheck/emulation/rocjitsu/.venv-gfx1250` supplies
`torch 2.11.0+rocm7.15.0a20260719`. When selected explicitly, its
workload-scoped doctor passes a numeric gfx1250 dispatch through RocJitsu and
proves that the exact current ConSan hook is mapped. This is current evidence
for the `histc` row below, not permission to substitute gfx950 code objects or
bypass RocJitsu for any workload.

### 2026-08-20 gfx1250 PyTorch launcher-provenance closure

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx1250-pytorch-histc-rr-launcher-fix-bN76SO`
records baseline and Record/Replay through the RocJitsu `gfx1250_mi455x.json`
launcher with hook SHA-256
`587f327b6c269414c61a79c8915867fdb68fc2a434cc8586ffe41e19e11cfbbc`.
The exact baseline passes in 1.53 seconds. Record/Replay passes in 3.41 seconds
with complete 175/175 access and 168/168 split-barrier-member coverage, zero
diagnostics, and complete analysis, static, and dynamic verdicts.

The first current attempt failed before the workload because provenance's
required PyTorch runtime-identity dispatch omitted the target launcher even
though doctor and the validation row used it. Provenance now applies the exact
launcher to that dispatch as well as to the payload, and records the prefixed
command. The conventional validator suite's 209 tests include a launcher-
propagation regression. The gfx950-only `histc` cadence override remains
target-resolved; this gfx1250 row retains its standard Record/Replay cadence
and 30-second bound.

### 2026-08-20 TP1 prefill post-fast-gate regression

Artifact
`rebase-20260820-gfx1250-tp1-prefill-all-efea098` records a clean-tree
baseline and all four strict profiles through RocJitsu at source revision
`efea0981fd` and hook SHA-256
`663aae26c332ae2eaa8780ebd9020646f27e5c3d20eb44ae77f1dbb7204ecf24`.
The baseline and SuperCollider rows pass the exact prefill oracle in 11.19 and
12.14 seconds; SuperCollider retains complete 352/352 access coverage.

The other three rows expose distinct current gaps.  Record/Replay passes the
numeric oracle and statically patches 352/352 accesses plus 74/74 barriers,
but records no visible evidence across 32 instrumented dispatch packets under
the standard 65,536-stride policy, so its dynamic verdict is incomplete.
Sampled fails closed before execution because its dense fast-gate fallback
emits 2,988 bytes after reserving a 3,100-byte body.  Inline Shadow statically
patches 352/352 accesses plus 37/37 barriers and begins execution, but exceeds
the manifest's 30-second process bound before an oracle or teardown verdict.
The matrix is demoted immediately rather than retaining older green claims;
each cell must be rerun after its issue is fixed.

The Sampled follow-up artifact
`rebase-20260820-gfx1250-tp1-prefill-sampled-body-gate-fix` fixes both the
dense layout rejection and the behavioral failure exposed immediately after
it.  Treating an unavailable early workgroup gate as address-only sampling
admitted hundreds of unrelated workgroups into the finite causal-window
banks, producing 173 false conflicts on this correct workload.  Private
entry-captured workgroup state is now filtered by a vector gate inside the
spill-safe probe body, before its independent LDS-cell selection.  The exact
oracle passes in 19.94 seconds with complete 352/352 access and 64/64 barrier
coverage, 620 visible sampled records, zero diagnostics or conflicts, and a
complete dynamic verdict.  A gfx1250 dense/private-state host regression and
the corresponding gfx950 AccVGPR-boundary regression pin the two lowerings.
The candidate-tree hook SHA-256 is
`18d108f47cf3972487eb96cac0ee0bebe0c04212359cbb1293954bc703823c01`.

Inline Shadow's failure was a stale automation bound rather than an engine
defect.  Artifact `rebase-20260820-gfx1250-tp1-prefill-inline-timeout-fix`
runs without a command-line timeout override and records the target-resolved
60-second manifest contract.  The exact oracle passes in 31.07 seconds with
complete 352/352 access and 37/37 barrier coverage and a complete dynamic
verdict.  The corresponding host regression requires gfx1250 TP1 prefill to
resolve to 60 seconds while gfx950 retains the ordinary 30-second bound.

The clean-revision consolidation artifact
`rebase-20260820-gfx1250-tp1-prefill-all-bcaf8e5` runs the baseline and all
four profiles through RocJitsu at source revision `bcaf8e5b1f` and hook
SHA-256
`59ae90f075525cb84925717b7322d8f0d98b59c7a41b9d3e81e989a5ce0615c4`.
All five rows pass the exact oracle.  Baseline, SuperCollider, Record/Replay,
Sampled, and Inline Shadow take 10.79, 12.13, 13.50, 20.89, and 30.81 seconds,
respectively.  Every instrumented row is analysis-, static-, and
dynamic-complete; their access coverage is 352/352, with 74/74, 64/64, and
37/37 barrier coverage in Record/Replay, Sampled, and Inline Shadow.
Record/Replay emits zero diagnostics.

Record/Replay now uses a checked-in target/workload validation stride of 256:
the production 65,536 stride selects no workgroup in this compact 32-dispatch
schedule.  Executing the selected dense gfx1250 path exposed two real lowering
defects which the no-evidence run could not exercise.  Mixed code objects now
initialize the persistent cluster coordinate to zero for ordinary kernels
which do not receive that launch input.  The dense runtime gate now keeps its
sampling residue out of the live `s_call_i64` return pair; previously it
returned to `0x00007ffe00000000` and crashed RocJitsu's instruction fetch.
Focused emitted-code regressions pin both contracts, and the complete 680-test
`ConSanMoi.*` suite plus all 331 Python validation tests pass.

### 2026-08-20 cluster workgroup identity persistence

The Sampled gate audit also found that the persistent exact-workgroup tuple
retained x/y/z but not gfx1250's cluster-local workgroup ID.  A later gate or
report could therefore read the entry TTMP after arbitrary guest code instead
of using an entry-captured value.  Automatic placement now allocates and
captures a fourth persistent coordinate only for gfx1250 kernels which
actually consume the cluster ID.  Scalar, vector, and private-state prologues,
resource exclusion, emitted gates, reports, and final validation all carry the
same optional coordinate; other targets and gfx1250 kernels without this ABI
input keep the three-coordinate layout.

Two focused host regressions pin both paths.  The dense fast-gate test requires
the entry prologue to capture TTMP6 and the later gate to hash the persistent
copy.  The spill-backed test requires all four coordinates to be captured in
private memory and loaded by the body gate.  The complete 680-test
`ConSanMoi.*` host suite passes with these checks.  This is structural proof of
the repaired lowering; the next clustered E2E refresh remains responsible for
new runtime evidence rather than reinterpreting the earlier accepted bundle.

### 2026-08-20 current Jakub clean refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx1250-jakub-all-47771b9`
runs the baseline and all four strict clean profiles through RocJitsu at source
revision `47771b901b`.  All five rows pass all four exact arithmetic oracles.
Baseline, SuperCollider, Record/Replay, Sampled, and Inline Shadow take 5.95,
7.50, 10.18, 7.15, and 7.37 seconds respectively.  The rebuilt object has
complete 70/70 access coverage in every engine, plus 8/8 barriers in
Record/Replay and Sampled and 4/4 barriers in Inline Shadow.  Record/Replay
emits zero diagnostics, and every instrumented row is analysis-, static-, and
dynamic-complete.

The first current-tip run exposed a Sampled transform rejection after the
cluster-identity persistence change: access-local analysis saw an ordinary
three-coordinate owner, while code-object-wide persistent placement included
a fourth cluster coordinate for a sibling.  Commit `47771b901b` sizes the
dense gate from the persistent tuple it actually consumes and adds a focused
mixed-owner regression.  All 1,229 runnable ConSan-family host tests pass; the
only skip is the opt-in live-object benchmark.  The accepted hook SHA-256 is
`e90e75343df20766f5b5b6eca2c936965df1eddc17087cd96a150bb8f9f3806c`.
This clean evidence preserves the prior accepted paired-overhead and
reviewed-fault qualifications.

Candidate-tree artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx1250-jakub-sampled-owner-recovery-final-FgFPdz`
rechecks Sampled after the CDNA4 mixed-owner placement work. Through RocJitsu's
gfx1250 configuration, all four exact arithmetic oracles pass in 7.06 seconds,
all 70/70 accesses and 8/8 barriers patch, and the static, dynamic, and overall
analysis verdicts are complete. The loaded hook SHA-256 is
`dc7d952995dfbcb0cf8601c10b0b546fefd95f69f055bc08d62e4e2ec5664b2a`.

CLIP BF16 is intentionally omitted from the current acceptance matrix.  Its
uninstrumented execution is not presently practical in the software GPU
environment: the default multi-executor configuration can stall before model
inference, and a single-executor baseline reaches inference but remains too
slow for useful iteration.  Existing static gfx1250 qualification evidence is
not sufficient for promotion; CLIP remains outside the matrix denominator
until baseline execution becomes suitable for end-to-end validation.

### 2026-08-20 TP1 decode/combined Inline bundle

Accepted clean artifact
`rebase-20260820-gfx1250-tp1-decode-inline-bundle-d293c6e` at source
revision `d293c6e7a9` and hook SHA-256
`8928b234058aae810456b3836c83550bca2368c4d8e033cf82ada908f6feef45`
replaces the stale 600-second no-verdict boundary.  Both exact oracles pass:
decode takes 14.769 seconds and combined takes 20.905 seconds for their timed
iterations.  The 75.924-second aggregate process is analysis-, static-, and
dynamic-complete at 704/704 accesses plus 74/74 barriers, with zero diagnostics
or incomplete encounters.

The two modes construct independent IREE models in one Python process.  Their
`SystemContext`/`BoundModule` reference cycles previously retained the first
model's executable and ConSan report allocation until process exit.  The
validation wrapper now explicitly collects each model before constructing the
next independent oracle.  A CPU-only regression disables automatic cyclic GC
and proves that the second model cannot be constructed while the first remains
live.  In the end-to-end run this reduces peak live ConSan report memory from
85,731,696 to 42,865,848 bytes while preserving the exact workload and coverage
contracts.

Paired artifact
`rebase-20260820-gfx1250-tp1-decode-inline-overhead-d293c6e` accepts both
baselines and the complete instrumented row at the same source revision.  The
paired medians are 7.150 seconds decode and 7.115 seconds combined; Inline
Shadow takes 14.773 and 20.783 seconds respectively, for 2.07x and 2.92x
slowdowns.  The workload's checked-in 180-second process bound now covers its
independent model setup as well as both modes, and a host regression pins that
contract instead of relying on an ad hoc command-line timeout.

Reviewed fault artifact
`rebase-20260820-gfx1250-tp1-decode-inline-fault-d293c6e` moves exactly one
selected signal/wait pair later in the decode matmul kernel.  Inline Shadow
emits exactly one attributable diagnostic while both numeric oracles remain
correct, matching the precommitted detected/pass policy.  Mutation accounting
is exactly requested=1, planned=1, applied=1 with complete installation and
reservation evidence; both before/after `rocminfo` and independent target
dispatch probes pass.  The validator now uses the bounded D128-pressure exact
oracle for gfx1250 health instead of selecting the known-unbounded Qwen smoke
merely because its files are present, with host coverage for that target
choice.  Clean, paired, fault, containment, health, cleanup, and provenance
gates are therefore retained at one frozen implementation revision, promoting
the cell to green.

### 2026-08-20 tree atomic-OR clean revalidation

`rebase-20260820-gfx1250-tree-clean-bf5bd13` runs the baseline and all four
strict clean profiles through RocJitsu's native gfx1250 B0 simulator at source
revision `bf5bd139c1`.  All five rows pass the exact acquire/release bitmask and
WMMA-partial oracle.  Every instrumented profile is analysis-, static-, and
dynamic-complete with zero incomplete encounters: 4/4 accesses for every
engine; 8/8 barriers, 10/10 atomics, and 16/16 fences for Record/Replay; 8/8
barriers and 10/10 atomics for Sampled; and 4/4 barriers plus 10/10 atomics for
Inline Shadow.  Record/Replay emits zero diagnostics.  Baseline and profile
wall times are 0.87, 4.11, 4.40, 4.18, and 4.08 seconds, respectively, in
SuperCollider, Record/Replay, Sampled, and Inline Shadow order.  The hook
SHA-256 is
`f2c2da60da4e3e3c60bf778b370de3f4c34f9cab59115ca3eceeb0f9c4141208`.
This clean refresh preserves the previously accepted paired-overhead and fault
evidence rather than replacing it.

### 2026-08-20 WMMA-attention clean revalidation

`rebase-20260820-gfx1250-wmma-all-2ed396c` reruns the baseline and all four
strict clean profiles through RocJitsu's native gfx1250 B0 simulator at source
revision `2ed396cd31`.  All five rows accept both exact host-reference tests.
Every instrumented profile has complete static, analysis, and dynamic coverage
with zero incomplete encounters: 18/18 accesses for all engines, 8/8 barriers
for Record/Replay and Sampled, and 4/4 barriers for Inline Shadow.  The
Record/Replay clean diagnostic contract observes zero diagnostics.  Baseline,
SuperCollider, Record/Replay, Sampled, and Inline Shadow complete in 13.53,
15.46, 16.91, 15.22, and 19.91 seconds respectively.  The loaded hook SHA-256
is `59ae90f075525cb84925717b7322d8f0d98b59c7a41b9d3e81e989a5ce0615c4`.
This clean refresh preserves the previously accepted paired-overhead and fault
evidence rather than replacing it.

### 2026-08-20 Stream-K arrival clean revalidation

`rebase-20260820-gfx1250-streamk-clean-170262c` runs the baseline and all four
strict clean profiles through RocJitsu's native gfx1250 B0 simulator at source
revision `170262caab`.  All five rows pass the exact arrival-counter and WMMA
partial-sum oracle.  Every instrumented profile is analysis-, static-, and
dynamic-complete with zero incomplete encounters: 4/4 accesses for every
engine; 8/8 barriers, 10/10 atomics, and 16/16 fences for Record/Replay; 8/8
barriers and 10/10 atomics for Sampled; and 4/4 barriers plus 10/10 atomics for
Inline Shadow.  Record/Replay emits zero diagnostics.  The hook SHA-256 is
`f2c2da60da4e3e3c60bf778b370de3f4c34f9cab59115ca3eceeb0f9c4141208`.
This clean refresh preserves the previously accepted paired-overhead and fault
evidence rather than replacing it.

### 2026-08-20 D128 block timeout revalidation

`rebase-20260820-gfx1250-d128-all-timeout-fixed` reruns the baseline and all
four strict clean profiles through the native B0 simulator configuration. All
five rows accept both host-reference oracles with hook SHA-256
`5d0f27d1a1fd6968d9665afd98cfdb6b9b38f74b78b230b3625d7d45fa2f27fa`.
The baseline takes 18.41 seconds; the four profile times and current coverage
are recorded in the matrix above.

Record/Replay now retains a materially stronger dynamic-identity contract than
the historical 10.89-second spot run: the current row publishes 5,844 visible
evidence records and all eight admitted barriers without saturating its report,
where the older row retained 18 visible access records and four barriers. The
gfx1250 D128 manifest therefore uses a target-specific 150-second process
bound. A host regression test pins that bound, verifies that the runner uses
it, and separately pins gfx950 to the ordinary 30-second bound.

### 2026-08-20 D128 pressure revalidation

The post-merge rebuild changed the target-native executable to 24 admitted
access ranges and exposed two independent prototype defects. Inline Shadow
first rejected the dense transformed object because every two-word gfx1250
access relay reserved 20 words, while a stranded singleton that must relocate
its host requires 28 words: five for return-PC matching, three for the guest
return adjustment, one conditional branch, six for the long return, plus the
12-word relocated host arm and terminator. The gfx1250 reservation now follows
that worst-case construction. A host regression creates two dense reach
partitions, forces the two-site partition to retain a relocated host, and
requires the transform to succeed. The retained
`rebase-20260820-gfx1250-d128-pressure-inline-reservation-measure` clean row,
with hook SHA-256
`9493c8b4995d47c2578d7707d26966377e4da8b5e72462492797bc9a98f1949e`,
accepts all four exact cases in 117.90 seconds with 24/24 accesses, 4/4
barriers, complete dynamic analysis, and no diagnostics.

Record/Replay then reported typed owner-table saturation (`flags=0x5`) after
retaining only 1,023 address groups at its hottest site, despite occupying
just 5,219 of 1,048,576 table slots. The emitted hash had passed the scalar LDS
address-key register number as the vector-only VOP operand, so dynamic
addresses did not affect the home slot and the site formed a 1,024-probe
collision chain. The operands are now ordered so the scalar address key is the
legal scalar source. The existing emitted-code host regression now pins that
exact instruction contract. The retained
`rebase-20260820-gfx1250-d128-pressure-rr-address-hash-fix` clean row, with hook
SHA-256
`6e8b4a01991d18bfb0d17f17c98510672556fdab74dd490d18e4529be73042c1`,
accepts all four exact cases in 73.87 seconds with 24/24 accesses, 8/8
barriers, 31,256 committed identities, 2,048 address groups at the formerly
failing site, complete dynamic analysis, and no saturation or diagnostics.

The gfx1250 D128-pressure manifest now uses a target-specific 180-second
process bound, providing the same approximate 1.5x simulator margin as the
D128-block row. Runner tests pin both that effective bound and the unchanged
ordinary 30-second gfx950 bound.

The current Record/Replay bundle is completed at source revision
`4bd97f150f`.  Artifact
`rebase-20260820-gfx1250-d128-pressure-rr-overhead-4bd97f1` records a
15.470-second paired baseline median, 20.623-second profile median, and 1.333x
slowdown.  Fresh inventory
`rebase-20260820-gfx1250-d128-pressure-inventory-4bd97f1` reviews the first
compiler signal/wait sequence.  In accepted fault artifact
`rebase-20260820-gfx1250-d128-pressure-rr-fault-4bd97f1-002`, exactly one
requested and planned mutation is installed, the exact GoogleTest oracle
fails, and Record/Replay emits 6,156 attributable diagnostics.  The surviving
object remains analysis-complete at 24/24 accesses and 6/6 barriers; before
and after RocJitsu gfx1250 dispatch probes pass.  The fault campaign uses the
fast D128 oracle for health because the default Qwen health smoke currently
exceeds its 30-second simulator deadline.  The hook SHA-256 for inventory,
paired, and fault evidence is
`7b7b5fc01a9874b05d058492bd79231bbddba597196a258802cb7433ff3eb60f`.

Current-tip artifact
`rebase-20260820-gfx1250-d128-pressure-rr-646711b` rechecks the heavy clean
Record/Replay path after sparse-table replay scheduling changed.  All four
exact oracles pass in 72.63 seconds with complete 24/24 access and 8/8 barrier
coverage, zero diagnostics, and complete static and dynamic verdicts.  Its
baseline passes the same four oracles in 56.80 seconds.  This retains the
accepted paired and reviewed-fault evidence above while proving that the
current replay reader does not regress the 31,256-identity workload.

### Target-native Jakub matmul

hip-moi commits `6a25f44`, `f06db81`, and `0fcd57d` add and strengthen the
`hip_moi_reference_gfx1250_jakub_matmul` executable. Its three parameterized
cases use the gfx1250 wave32 16x16x32 FP16 WMMA layout, a cooperative cross-wave
LDS producer/consumer handoff, and a real two-buffer K=128 ping-pong schedule.
The logical-coordinate oracle independently checks arithmetic, contraction
length, and accumulator placement; as documented in the source, it is invariant
to a bijective shared A/B K-slot permutation. The mixed-architecture build gate
pins this binary to gfx1250 rather than relabeling the RDNA4 executable. The
separate producer-skew case applies bounded volatile private-memory traffic
before every B-fragment handoff, so the scheduling opportunity survives both
unoptimized and optimized compilation. Safe barriers absorb the skew, while
removing a handoff makes the exact oracle fail. This coupling is a workload
schedule property: the kernel does not inspect sanitizer mode or runtime
configuration. Ordinary K32/K128 cases remain separate for timing.

`gfx1250-jakub-skew-final-clean-20260801` is exact and
static/dynamic-complete for all four profiles across all three cases.
`gfx1250-jakub-skew-volatile16k-inventory-20260801` retains the current
`fnv1a64:71c39178e2c3786a` barrier inventory and records hip-moi commit
`0fcd57def36188500a19abaf3b6bca3a6e773034` in the reviewed spec.
`gfx1250-jakub-skew-final-accepted-fault-20260801` accepts the same exact-one
drop in all four profiles: every exact oracle fails, Record/Replay, Sampled,
and Inline Shadow also diagnose the conflict, and SuperCollider retains the
qualified detector miss tracked by `bd-2sjm.1`. Each row records one requested,
planned, applied, and installed mutation plus healthy pre/post simulator
probes; the slowest complete row took 21.10 seconds. The optional
`ConSanGfx1250HipMoiSim.JakubProducerSkewBarrierDrop` CTest locks this execution
contract to the supported simulator launcher.

`gfx1250-jakub-skew-final-overhead-20260801` excludes the producer-skew case,
uses an 800 ms paired baseline, and retains the per-profile slowdowns shown in
the matrix. This keeps the schedule discriminator from diluting timing ratios.

This row is the target-native cooperative LDS/WMMA qualification role, not an
inherited instruction-site denominator. gfx1250 evidence comes only from the
gfx1250 binary and its own oracle, inventory, fault campaign, and timings; the
larger RDNA4 schedule retains its separate target ledger.

### 2026-08-22 Qwen baseline/Record Replay B0 latency check

Artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx1250-qwen-rr-rocjitsu-b0-zLDCUC`
runs an equal 30-second baseline/Record Replay comparison through the checked-in
`gfx1250_mi455x.json` B0 RocJitsu configuration. Its recorded `ROCM_PATH`,
`PATH`, and `LD_LIBRARY_PATH` all select
`/home/ossci/xx/TheRock/build/dist/rocm`; the loaded hook SHA-256 is
`3eaac597ecd8ad73bd2de95b102005651048ab6250cdaf0e5d86c3326e0cecf6`.
The uninstrumented baseline remains active until its 30.20-second bound and
produces neither an error nor an oracle. Record/Replay inventories and patches
all 1,000/1,000 accesses plus 92/92 barriers, spends 106.23 ms in the patcher,
and emits a valid 4,064,224-byte replacement before it too remains active until
the 30.19-second bound. The timeout prevents teardown, so no dynamic or
diagnostic verdict is claimed.

This A/B rules out ConSan transformation cost as the first current-tip
boundary: the baseline itself exceeds the ordinary deadline, and the
instrumented row reaches a statically complete replacement in a small fraction
of that deadline. It does not promote the orange cell because neither row
reaches the independent numerical oracle. Per the tactical-latency rule, a
second long full-model run is deferred; the productive next step is a reduced
Qwen execution denominator or a RocJitsu throughput investigation, not another
identical timeout. The existing heterogeneous-framework checked-in
correct/incorrect pair already owns Qwen's observable multi-kernel publication
and diagnostic-attribution behavior, so this latency finding does not justify
a prototype-layout device fixture.

### 2026-08-20 Qwen current-workspace refresh

Fresh runs at source revision `4d209ed968` and hook SHA-256
`7b7b5fc01a9874b05d058492bd79231bbddba597196a258802cb7433ff3eb60f`
use the checked-out Qwen VMFB and RocJITsu's native gfx1250 B0 simulator.
SuperCollider transforms all 1000 discovered accesses, installs a 365,536-byte
replacement, and reaches execution, but produces no oracle or unload verdict
within 90 seconds. Record/Replay transforms all 1000 accesses and all 92
barriers, installs a 4,039,648-byte replacement, and likewise reaches execution
without a verdict inside 60 seconds. A retained copy of that exact
Record/Replay replacement remains CPU-bound in the standalone B0-to-A0
translator without emitting a diagnostic after five minutes. The run was
bounded at that point, so this is neither a successful translation verdict nor
a reproduction of the older 17-target failure. These results leave both cells
orange and identify translation plus simulated execution latency as the current
first boundary in this workspace.

Clean-revision follow-ups at source revision `1ea9aa986f` and hook SHA-256
`ae1c5b63c70dafdf433586bfb9ae61c67158b1b2f99e7e2124847e4a941e9008`
refresh the two stale cells.  Sampled patches all 1000 accesses plus 90/90
barriers and installs a 3,945,440-byte object.  Inline Shadow patches all 1000
accesses plus 46/46 barriers and installs a 4,748,256-byte object.  Both
replacements are accepted by the current runtime translator and issue repeated
simulated dispatches, but neither produces an oracle or teardown verdict within
the checked-in 30-second bound.  The retained artifacts are
`/home/ossci/xx/consan-validation/rebase-20260820-gfx1250-qwen-sampled-current-1ea9aa9`
and
`/home/ossci/xx/consan-validation/rebase-20260820-gfx1250-qwen-inline-current-1ea9aa9`.
This supersedes the old current-cell claim that Sampled fails loader admission;
all four current Qwen profiles now reach execution, while bounded simulated
execution remains the shared first acceptance gap.

A committed-tip Sampled follow-up at revision `813f0613a8` extends the exact
same standard-profile run to 120 seconds through the explicit RocJITsu
`gfx1250_mi455x.json` launcher.  It again patches all 1000/1000 accesses and 90/90
barriers, installs the 3,945,440-byte replacement, and reaches simulated
execution without an early rejection or diagnostic, but still produces no
oracle or teardown verdict before timeout.  The retained artifact is
`/home/ossci/xx/consan-validation/rebase-20260820-gfx1250-qwen-sampled-rocjitsu-120s-813f061`;
the loaded hook SHA-256 is
`83b9ce72571bcb20e498f2bfae1af489ccf69eba84044c0552d7b683ec4fdaef`.
The cell remains orange: increasing the ordinary validation bound by four is
not enough to turn this large emulated model into an actionable iteration
loop.

A current-hook follow-up at revision `158b25fd5c` extends that same standard
Sampled profile to 600 seconds through RocJITsu's checked-in `gfx1250_mi455x.json`
configuration. Artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx1250-qwen-sampled-600s-independent-abi-v2`
uses hook SHA-256
`dcb50737106a15c068da43ebbc439b83cfddaf08032b280792ddb804b7638712`.
Instrumentation remains complete at 1000/1000 accesses plus 90/90 barriers,
with no resource or lowering failure. The run remains compute-active through
198 observed simulated dispatches, reaching the final 151,936-cluster output
kernel, but still has no numerical oracle or unload verdict at the bound. This
rules out a short post-rebase rejection and strengthens the conclusion that
full-model simulated latency, rather than current ConSan translation or
placement, is the first boundary. The cell remains orange because a bounded
execution without the independent oracle cannot qualify.

### Current large Qwen translation boundary

The current selected-object evidence at `0bf1d17c7d` replaces the earlier
claim that every replacement reached workload execution.  SuperCollider's
394,104-byte expanded object passes the newer standalone B0-to-A0 translator,
but its end-to-end artifact
`consan-validation-large-objects-gfx1250-qwen-sim-sc-v6-20260731` has no oracle
within 600 seconds.  Record/Replay's 4,043,640-byte replacement is statically
complete in
`consan-validation-large-objects-gfx1250-qwen-sim-rr-0bf1d17-20260731`, but
the end-to-end run stops at its 600-second bound after replacement rather than
proving workload execution.

The current Sampled artifact
`consan-validation-large-objects-gfx1250-qwen-sim-sampled-clean-v2-0bf1d17-20260731`
fails loader admission before execution. It loaded hook SHA-256
`2113c773bdcac837e58cbd0cc0e784d7d99d971de3a8e08c377743cd245f8e6e`,
which contains the guest-SCC return fix; the `0bf1d17` suffix identifies the
checked-out source ref rather than the loaded binary. Its 3,396,472-byte object
and the Record/Replay object independently reproduce 17 unrecovered generated
long-return targets in the newer standalone revision translator. Inline
Shadow's 4,666,232-byte object in
`consan-validation-large-objects-gfx1250-qwen-sim-inline-diagnostic-0bf1d17-20260731`
is accepted by the installed runtime translator and begins execution, but has
no oracle within its 60-second diagnostic bound; the newer standalone
translator reports the same 17 return-target gaps.  This version-sensitive
post-instrumentation translation gap is tracked by `bd-1w9.6.26`; no workload
PC, kernel name, register exception, or result exception is an acceptable
solution.

## RocJITsu test-corpus expansion

This is the staging ledger for broadening end-to-end validation with the
gfx1250 Tensile corpus at `$WORKSPACE_ROOT/rocjitsu-test-corpus`, surveyed at
corpus revision `aa54cc8`.  An orange row records target-specific assessment
and an executable proof plan, not acceptance.  Candidates enter the current
matrix only after they have an independent numeric oracle and a
standard-profile clean run; they then follow the same maturity scale as the
existing workloads.

Each instrumentation flavor/engine has its own status cell.  This makes both
horizontal workload maturity and vertical engine maturity visible directly;
no aggregate color may conceal a weaker profile.

The packaged corpus contains 140 gfx1250 code objects from 48 runnable Tensile
configurations.  Forty-four configurations contain workgroup barriers, for
25,809 static signal/wait pairs across the packaged libraries.  These are
static library totals, not executed dynamic counts: a library may contain many
solution kernels while a numeric run selects only a subset.

| Priority | Tracking unit | SuperCollider | Record/Replay | Sampled | Inline Shadow | Notes |
|---|---|---|---|---|---|---|
| P0 | `002_sk_mxf8gemm_explicit` | 🟩 70/70 accesses; current paired 1.04x | 🟩 70/70 accesses; 32/32 barriers; 4/4 fences; current paired 1.08x | 🟩 70/70 accesses; 28/28 barriers; current paired 1.06x | 🟩 70/70 accesses; 32/32 barriers; current paired 1.42x | Exact numeric oracle; all four profile bundles accepted. |
| P0 | `003_sk_mxf4gemm_explicit` | 🟩 42/42 accesses; current paired 1.04x | 🟩 42/42 accesses; 32/32 barriers; 4/4 fences; current paired 1.00x | 🟩 42/42 accesses; 28/28 barriers; current paired 1.08x | 🟩 42/42 accesses; 32/32 barriers; current paired 1.24x | Exact numeric oracle; all four profile bundles accepted. |
| P1 | `037_spmm_tdm_f16_transposes` | 🟩 672/672 accesses; current paired 1.10x | 🟩 672/672 accesses; 176/176 barriers; current paired 1.19x | 🟩 672/672 accesses; 160/160 barriers; current paired 1.19x | 🟩 672/672 accesses; 176/176 barriers; current paired 1.79x | Four numeric clients cover tensor waits and transpose LDS reads; all profiles accepted. |
| P1 | `016_spmm_tdm_all` | 🟩 1610/1610 accesses; current paired 1.15x | 🟩 1610/1610 accesses; 512/512 barriers; current paired 1.24x | 🟩 1610/1610 accesses; 494/494 barriers; current paired 1.19x | 🟩 1610/1610 accesses; 256/256 barriers; strict-capacity current paired 1.61x | Multi-type transpose matrix; all profiles accepted, including strict-capacity Inline Shadow. |
| P1 | `001_sk_mxf8f4gemm_tdm` | 🟩 768/768 accesses; current paired 1.12x | 🟩 Current exact clean run: 768/768 accesses, 102/102 barriers, 24/24 fences | 🟩 768/768 accesses; 180/180 barriers; current paired 1.22x | 🟩 768/768 accesses; 102/102 barriers; current paired 13.38x; reviewed exact-one fault and health accepted | Exact numeric oracle; all profiles accepted, including reviewed Inline Shadow fault evidence. |
| P1 | `004_sk_mxf8gemm_tdm` | 🟩 992/992 accesses; current paired 1.20x | 🟩 Current exact clean run: 992/992 accesses, 102/102 barriers, 24/24 fences | 🟩 992/992 accesses; 180/180 barriers; current paired 1.23x | 🟧 Compute-active through 600, 1200, and 1800 seconds; no verdict | Only Inline Shadow remains: execution has no verdict at the stated bound. |
| P1 | `007_sk_mxf4gemm_tdm` | 🟩 2448/2448 accesses; current paired 1.35x | 🟩 All 96/96 exact numerical rows pass with complete 2448/2448 access, 544/544 barrier, and 64/64 fence coverage and lossless replay. The current paired slowdown is 19.47x. A prospectively reviewed exact-one redundant-barrier mutation is reached, preserves the exact oracle, produces the expected qualified miss, and passes containment, health, and cleanup. Focused host regressions plus the adjacent high-bank/double-barrier LDS device pair protect the distilled behavior | 🟩 2448/2448 accesses; 480/480 barriers; current paired 1.38x | 🟧 Compute-active through 1800 seconds; no verdict | Dense relay hosts may no longer move gfx1250 VGPR-bank transitions. Exact-size sharding replaces the obsolete monolithic timeout path while preserving all six configured problem sizes. |
| P1 | Bounded `gfx1250_tensile_streamk_smoke` | 🟩 Current exact numeric row is accepted in 7.73 s with complete 320/320 access coverage and a complete dynamic verdict | 🟩 Current exact numeric row is accepted in 14.15 s with complete 320/320 accesses, 22/22 barriers, and 4/4 fences; complete dynamic verdict and zero diagnostics | 🟩 Current exact numeric row is accepted in 7.87 s with complete 320/320 accesses and 20/20 barriers; complete dynamic verdict | 🟩 Current exact numeric row is accepted in 27.86 s with complete 320/320 accesses and 11/11 barriers; the former strict-placement rejection is fixed | One Stream-K mode-3 solution requests four fixed workgroups across six output tiles and two K iterations. The current baseline passes its exact row in 7.22 s. The runner requires exactly one numeric row, a positive device-timing canary, rejects malformed rows and wrong hardware, verifies the fixed-grid runtime control still exists, and verifies every emitted object declares gfx1250. |
| P2 | `000_sk_sgemm_quick` | 🟨 First problem: 12/12 exact numeric rows; 640/640 accesses; static/dynamic complete | 🟨 First problem exact and fully covered; aggregate host analysis fixed; full client is intrinsically execution-bound | 🟨 First problem: 12/12 exact numeric rows; 640/640 accesses; 40/40 barrier members | 🟧 First problem: 12/12 exact rows and complete static coverage; interrupted second problem leaves dynamic analysis incomplete | The first problem is validated; the full multi-problem client remains execution-bound. |
| P2 | `005_sk_f8gemm_quick` | 🟩 Exact oracle; 1772/1772 accesses; current paired 1.43x; reviewed fault and health accepted | 🟩 Exact oracle; 1772/1772 accesses; 44/44 barriers; 16/16 fences; current paired 8.00x | 🟧 Current clean execution remains compute-active through 900 seconds; no verdict or measured overhead | 🟧 Current tip executes 49 exact rows with zero failures before the fixed 180-second bound | SuperCollider and Record/Replay are accepted; Sampled and Inline Shadow lack a full-client verdict. |
| P2 | `006_sk_hgemm_quick` | 🟧 136 exact numeric passes with zero failures; first 143-solution problem remains active at 300 seconds | 🟩 Exact oracle; 8162/8162 accesses; 292/292 barriers; 80/80 fences; current paired 2.02x | 🟩 Exact oracle; 8162/8162 accesses; 544/544 barriers; current paired 2.24x | 🟧 Current tip executes 189 exact rows with zero failures before the fixed 180-second bound | Record/Replay and Sampled are accepted; SuperCollider and Inline Shadow lack a full-client verdict. |
| P3 | `015_spmm_f8_ml` stress | 🟧 First contraction exact numeric pass; 298/4316 accesses; second orientation active at 120 seconds | 🟨 Current clean E2E accepts all eight generated clients with 172,468/172,468 accesses and 3,060/3,060 barriers. Multi-block exact-size sharding now preserves all eight clients in each of three concurrent size slices, but the large-size baseline's first client remains compute-active beyond five minutes; paired and reviewed-fault bundles remain. A new quick target-native sparse-FP8 behavioral pair covers packed low/D16-high stores, byte/transposed metadata loads, live SWMMAC publication, and the missing-edge diagnostic in all ten baseline/engine rows. | 🟩 All eight generated clients exact; 172,468/172,468 accesses and 6,120/6,120 barriers; paired 4.88x; reviewed fault, containment, and health accepted | 🟧 Exact failing kernel fixed; standard run has 8 passes and zero failures before its bound | Sampled is accepted; Record/Replay lacks paired/fault evidence, and the other profiles remain bounded. |
| P2 | `019_spmm_f16_sb` closure | 🟧 9,546/9,546 accesses patched; first client exceeds 300 seconds without a numeric row | 🟩 Four exact orientations; 31,265/31,265 accesses; current paired 2.48x | 🟧 9,546/9,546 accesses and 646/646 applicable barriers patched; first client exceeds 300 seconds without a numeric row | 🟧 9,546/9,546 accesses and 323/323 barriers patched; first client exceeds 300 seconds without a numeric row | Sampled is accepted; the other profiles retain the bounded partial results shown in their cells. |
| Survey | Remaining Tensile configurations | 🟩 Architecture-level decoded opcode union covered by accepted selected rows | 🟩 Architecture-level decoded opcode union covered by accepted selected rows, including full `019_spmm_f16_sb` bundle | 🟩 Architecture-level decoded opcode union covered by accepted selected rows | 🟩 Architecture-level decoded opcode union covered by accepted selected rows | Survey complete; selected high-signal rows above define the executable denominator. |

### 2026-08-22 `015_spmm_f8_ml` Record/Replay sharding diagnostic

The source configuration has eight benchmark blocks, not seven, and every
block repeats the same three Exact problem sizes. The validation harness now
fails closed unless all block inventories are identical, filters every block
into each size shard, runs the three shards concurrently, and requires all
eight generated clients plus every emitted numerical row to pass. Focused
host regressions cover multi-block filtering, inconsistent source inventories,
the complete command/fault policy, per-shard timing canaries, and the explicit
eight-client denominator. Fixed numerical-row counts are intentionally not
used: repeated clean generation produced 13--16, 23--29, and 40--45 printed
winners for the same three source sizes while every generated client and row
passed.

Standalone baseline diagnostics for the three slices complete in roughly 73,
82, and 117 seconds. Concurrent paired attempt
`/home/ossci/xx/consan-validation/prep-20260822-gfx1250-spmm-f8ml-rr-sharded-693-v3`
finishes the two smaller baseline slices in 83 and 89 seconds, each with all
eight clients passing. The large slice's first client remains compute-active
after five minutes, so the attempt is terminated before Record/Replay rather
than spending a second long cycle. This is bounded duration evidence, not a
paired result or promotion; the cell remains yellow and rotates. The earlier
clean full-client coverage result remains valid.

Two fresh current-tip inventory attempts,
`prep-20260822-gfx1250-spmm-f8ml-inventory-current-c131` and
`prep-20260822-gfx1250-spmm-f8ml-inventory-current-c131-t180`, reached their
30- and 180-second bounds while still performing active 64-way Tensile code
generation, before loading the first object or collecting coverage. This is
generation latency rather than evidence of a RocJitsu or ConSan hang, so the
full-client cell rotates instead of consuming another long iteration.
Previously generated code establishes the relevant target-native idiom:
`TensileLibrary_gfx1250.co` contains 620
`v_swmmac_f32_16x16x128_fp8_fp8` operations, 1,920 each of `ds_store_b8` and
`ds_store_b8_d16_hi`, 14,080 `ds_load_u8`, 240 `ds_load_tr8_b64`, 5,120
`ds_bpermute_b32`, and 421 split-barrier signal/wait pairs.

The checked-in `Gfx1250SparseFp8Pipeline` correct/incorrect pair now extracts
the device-observable contract without depending on prototype placement. Both
members stage packed FP8 bytes and sparse metadata and execute the live native
SWMMAC; the correct member requires packed `0x5aa5`, exact 64.0f matrix output,
control values and canaries with no diagnostic, while the incorrect member
removes only the producer/consumer barrier and requires the intended conflict.
All ten baseline and four-engine rows pass. The direct host unit
`Gfx1250ExecutionTest.DsStoreB8D16HiSelectsUpperHalf` also requires the exact
D16-high payload byte at the encoded offset. This materially closes a quick
coverage gap but does not supply paired overhead or reviewed-fault E2E
evidence; Record/Replay therefore remains yellow.

### 2026-08-21 `007_sk_mxf4gemm_tdm` Record/Replay far-fence fix

The focused E2E failure combines dense LDS accesses, an addressed buffer
acquire and device-scope global invalidate, and adjacent split-barrier signal
and wait operations. Record/Replay could patch the accesses and barriers but
not a far-routed fence when no ordinary entry island was reachable. Its
reusable dense relay was rejected for three independent reasons: it used the
code-object-wide scalar ABI rather than the owning kernel's assignment, it
rejected a valid eight-word displaced host, and its reservation search did not
recognize the Record/Replay barrier patch kind.

The checked-in host regression synthesizes two independent 33,000-word
gfx1250 owners with different scalar live ranges. It requires 18/18 access
patches, 4/4 barrier records, and 2/2 far-fence patches, verifies each fence
anchor uses its owner's call-return register, and requires final validation to
pass. The focused regression and all 178 nearby Record/Replay and dense-fence
tests pass.

Artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx1250-mxf4-tdm-record-replay-dense-fence-fix-v2`
runs the full clean client through the explicit RocJitsu `gfx1250_mi455x.json`
launcher. It timed out after 300.126 seconds before producing either a coverage
record or a numeric verdict. This proves no E2E promotion, so the matrix cell
remains orange pending a completed run.

### 2026-08-22 `007_sk_mxf4gemm_tdm` device contract

The checked-in `FenceBarrierPublication` pair distills the device-observable
part of the E2E failure without encoding the prototype's relay placement. Both
members execute an independent target-native workgroup barrier beside an LDS
publication mediated by an addressed VGLOBAL release/acquire sequence. The
correct member includes the final acquire-side invalidation and requires the
exact barrier and published values with no diagnostic. The incorrect member
removes only that final invalidation, preserves the exact barrier/control
result, and requires the intended conflict diagnostic from every ConSan
engine.

gfx1201 and gfx1250 use their native adjacent `s_barrier_signal -1` and
`s_barrier_wait -1`; gfx942, gfx950, and gfx1100 use the target-equivalent
monolithic `s_barrier`. Baseline and all four engines pass both members on all
five RocJitsu targets (50 rows), and the same ten rows pass on physical gfx950.
The original VGLOBAL publication pair plus this composite pair pass together
as 100 simulator and 20 physical rows. The full 1,386-row simulator matrix also
passes in 55.82 seconds. This closes the checked-in behavioral-coverage gap but
does not promote the orange E2E cell, whose complete 75-row client has not yet
produced a post-fix verdict.

The retained P0 and first P1 artifacts confirm the tensor-data-mover control
shape used by these configurations: tensor work is followed by
`s_wait_tensorcnt` and workgroup synchronization before LDS consumption.  The
current acceptance claim remains deliberately scoped to the decoded and
patched LDS accesses plus their surrounding barriers; a wait instruction by
itself is not counted as a raceable memory access.

### 2026-08-22 `007_sk_mxf4gemm_tdm` full-bound replay result

Artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx1250-mxf4-tdm-rr-full`
uses the active worktree based on `437d92825b98` and records hook SHA-256
`c06d4371a7a8de0a063e7fb27d7cc79a98142ddb23a62cb241cdb9bd661ea6a7`.
The complete clean Record/Replay client reached the manifest's declared
1,800-second workload bound and returned a timeout after 1,800.15 seconds,
before emitting either a coverage record or a numeric verdict. Consequently
there is also no complete ConSan analysis verdict. This supersedes the earlier
300-second bounded result but proves no promotion: the cell remains orange and
is deferred as execution-bound rather than consuming repeated long iterations.

### 2026-08-22 `007_sk_mxf4gemm_tdm` sharded high-bank replay closure

Artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx1250-mxf4-tdm-rr-highbank-spill-v3`
supersedes the monolithic timeout above for clean Record/Replay correctness and
coverage. The runner verifies the source configuration's complete six-size
matrix, executes one exact problem size per process with at most four processes
in flight, and requires exactly 16 numerical rows from each shard. All six
shards return zero and pass all 96/96 rows. Their individual elapsed times range
from 344.86 to 998.26 seconds; the bounded batch completes in approximately
22 minutes 35 seconds. This is an E2E workload duration, not the checked-in
device-suite latency budget.

The artifact records hook SHA-256
`a8e8542ead3314e27df0f9c5d23f5d42572843262b3835719bc81b092bfe5662`,
which matches the rebuilt current source. Every shard is statically and
dynamically complete at 2448/2448 access sites, 544/544 barrier members, and
64/64 fences. Across the six independent replays, all 5,281,152 published
accesses and 108,992 barriers are processed with zero drops, unsupported
events, diagnostics, metadata exhaustion, or unresolved provenance. Each
replay uses 50,684 shadow entries rather than the bogus approximately
830-million-entry range that previously caused replay to be skipped.

The underlying gfx1250 defect was that a native LDS operand could name physical
`v286` through encoded `v30` while Record/Replay selected a low instrumentation
bank before preserving the address. Under spill pressure, an initial repair
then overwrote the low-bank spill victim before saving it. The patch now saves
the spill window first, captures the physical high-bank address under the
incoming SRC0 mode, returns to the low scratch bank, and restores the complete
guest bank mode around the displaced instruction. Two focused host tests pin
appended capture placement and the seven-VGPR save-before-capture ordering. An
adjacent checked-in device pair uses the same encoded high-bank LDS address in
correct-barrier and missing-barrier workloads; baseline and Record/Replay pass
all four rows with exact clean results and the required conflict diagnostic.

A follow-up harness audit found that inventory and contained fault execution
still bypassed the exact-size shard policy and would have retraced the obsolete
monolithic 96-row command. They now use an explicit manifest-selected
representative shard with its complete 16-row numerical oracle; clean and
paired-overhead qualification continue to require all six shards and all 96
rows. Host regressions prove that both inventory and fault use the bounded
command and that missing or out-of-range shard policies fail closed.

The same frozen clean source, `da6167f4fbd686b0737c43dbd02e1a597634b0b6`,
now also has the paired-overhead artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx1250-mxf4-tdm-rr-overhead-highbank-current-da616-v2`.
All 18 processes and all 288 numerical rows pass: six baseline-before shards,
six Record/Replay shards, and six baseline-after shards. The mean of the two
baseline medians is 20,428.11 ms, the Record/Replay median is 397,813.07 ms,
and the paired slowdown is 19.47x. Every instrumented shard retains complete
2448/2448 access, 544/544 barrier, and 64/64 fence coverage and lossless
replay.

The reviewed inventory artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx1250-mxf4-tdm-rr-inventory-fault-shard-v1`
contains 544 barrier members and 272 split-barrier pairs for the representative
exact `120x120x1x1024` shard. A prospectively frozen mutation of the first
tensor-to-LDS publication barrier was correctly rejected as evidence: the
oracle passed but the expected diagnostic was absent, and there was no
precommitted reach witness. That observed result was not relabeled after the
fact. A distinct, previously untried adjacent second split barrier was then
reviewed from final ISA before execution. The immediately preceding
publication pair remains intact and only `s_wait_dscnt` separates it from the
selected pair, so dropping exactly that second signal is expected to be a
semantically redundant qualified miss.

Artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx1250-mxf4-tdm-rr-fault-redundant-tensor-lds-barrier-da616-v4`
accepts that prospective contract at the same source revision. It requests,
plans, reserves, and applies exactly one mutation with complete installation
and process evidence. The reviewed site is reached; all 16 exact rows pass;
the expected no-diagnostic qualified miss is observed; and surviving coverage
is complete at 2448/2448 accesses, 542/542 barrier members, and 64/64 fences.
The run publishes 2,097,152 access events and 12,512 barrier events without
overflow, and both pre- and post-run health checks pass. The checked-in correct
member now mirrors this adjacent-barrier tail with split barriers separated by
`s_wait_dscnt`; the incorrect member still removes the publication edge and
must diagnose. All four gfx1250 baseline/Record/Replay rows pass. This closes
the paired, fault, containment, health, cleanup, and quick-regression gates and
promotes the Record/Replay cell from yellow to green.

## PyTorch expansion

This is the staging ledger for broadening validation with the gfx1250 PyTorch
nightly stack.  The initial survey used PyTorch
`2.11.0+rocm7.15.0a20260719`.  Its target-specific `libtorch_hip` archive
contains 232 code-object fragments and 4,760 symbols with at least one static
barrier, LDS, atomic, or cache operation.  The static inventory includes
32,300 barrier signals, 32,292 barrier waits, substantial LDS traffic, and
4,649 decoded global, flat, or buffer atomics.  These are archive totals, not
dynamic denominators for any proposed workload.

Every workload has a separate cell for SuperCollider, Record/Replay, Sampled,
and Inline Shadow.  Flavor-specific evidence belongs in its corresponding
cell, so an advanced diagnostic result in one engine cannot visually promote
the other three.

The survey also inspected the 707 gfx1250 code-object fragments packaged for
BLAS, collectives, random-number generation, and FFT.  None of those 939
precompiled PyTorch and library fragments contains a decoded
`tensor_load_to_lds`, `tensor_store_from_lds`, `cluster_load_*`,
`s_wait_tensorcnt`, or `s_wait_asynccnt`.  Consequently, ordinary eager
PyTorch calls provide excellent synchronization and spill coverage but do not
by themselves add TDM or cluster coverage.

The installed PyTorch/Triton stack can generate that missing target-specific
coverage.  A small prototype using three `tl.make_tensor_descriptor` objects
and `num_ctas=2` compiled for gfx1250 to two `tensor_load_to_lds`, one
`tensor_store_from_lds`, three tensor waits, and six barrier pairs.  Its
metadata requests a two-CTA cluster.  This proves a practical route to TDM and
clustered-dispatch coverage from a workload whose inputs and numeric oracle
are PyTorch tensors.  It does not yet prove cluster-memory or inter-workgroup
synchronization: the prototype contains no `cluster_load_*` instruction, so
that remains a separate discovery target rather than an implied result.

### 2026-07-22 Record/Replay revalidation

After rebasing `users/bjacob/sanitizers`, all 27 cells that were green in the
Record/Replay column received a fresh one-repetition clean assessment.  The
the norm/softmax evidence hook SHA-256 is
`c4be4b6fb29f63817a18bb3044def3dca1df4f2af4014639510a74289df3c368`.
The shared architecture gate passes 764/764 ConSan tests and the validation
Python suite passes 71/71.  This retains the RDNA4 cases while qualifying the
gfx1250 changes.

The rebase exposed and the current branch fixes two gfx1250 placement defects:
singleton spill-backed scalar placement and reuse of relocated dense access
hosts by far fence probes.  The latter restores `001_sk_mxf8f4gemm_tdm` and
`004_sk_mxf8gemm_tdm` from 8/24 to 24/24 fences without changing the RDNA4
path.

The norm/softmax regression is fixed.  Five full-pressure reduction kernels
could not preserve the code-object-wide hardware dispatch-ID pair even though
they had a complete owner-local scalar ABI.  Record/Replay now uses its
defined zero-generation fallback only for those proven-safe owner components.
The current exact run is complete at 4,756/4,756 accesses and 2,352/2,352
barriers, and its paired one-repetition bundle measures 211.06x.  The full
shared ConSan suite, including the RDNA4 cases, remains green.

A later cheap-row audit at source revision `b6143a2ce1` and hook SHA-256
`fd67f1ce7ee438952d8f798b395ce1c75ea87e86fdead53eae5327e8a587fd57`
reran four native Record/Replay rows with one process repetition.  WMMA
attention, Stream-K arrival, and tree atomic-OR are exact, statically and
dynamically complete, and finish in 7.81, 2.18, and 2.55 seconds respectively.
D128 pressure reproducibly passes its first three exact cases, but the fourth
case crosses the ordinary 30-second aggregate row deadline; that cell is
yellow pending a complete current-tip verdict.

`007_sk_mxf4gemm_tdm` is the one prior green claim that does not survive this
audit.  It is statically complete and passes its first five exact rows; the
matching uninstrumented client completes all rows.  The failure is isolated to
the PGR2 MT128x128 kernel: access-only and fence-plus-access instrumentation
pass, while enabling barriers makes access candidate 145, the `ds_load_b128`
at `0x257d8`, the first numerical corruption.  The immediately preceding
barriers are at `0x25790` and `0x2579c`.  The cell is orange while this bounded
barrier/access composition defect is repaired.  The access-only and combined
objects use the same anchor and relay, and their normalized access bodies are
instruction-equivalent.  A temporary diagnostic retaining the barriers,
normal VGPR spill/restore, relocated guest `ds_load_b128`, and return while
removing all Record/Replay publication and EXEC logic still corrupts the same
row.  That edit was reverted; the remaining frontier is therefore the
barrier-plus-relocation/spill execution interaction, not epoch arithmetic or
record publication.  The independent emulator cannot currently adjudicate
this case: its uninstrumented corpus run stops on a later 202,752-byte LDS
solution above the modeled 163,840-byte limit, while the instrumented run
fails earlier in emulator dispatch-private handling.

### Shared-branch merge revalidation closeout

The 2026-07-21 revalidation is complete at source revision `949199f096` and
hook SHA-256
`2a599bb9ff2399a677fab0ca8aee4e96ba47cf02db7a49c6e601d7f2a688227e`.
All eight local patches range-diff unchanged from the preserved pre-rebase
stack onto `59bf98ccc9`.  A clean 648-action rebuild completed, the focused
ConSan/MOI suite passed 726/726, and the shared waitcheck smoke suite passed
20/20.

Current-tip, one-repetition execution reconfirmed the green Qwen and TP1
profiles, every compact attention and atomic-regression profile, every green
selected Tensile profile, and the green PyTorch families.  In particular,
Qwen SuperCollider and Record/Replay retain exact output oracles with
1,000/1,000 accesses; Record/Replay also retains 46/46 barriers.  TP1 prefill
retains all four profiles, while TP1 decode/combined retains its three green
profiles.  Quick-F8 Record/Replay completed its long software-GPU run with
1,772/1,772 accesses, 44/44 barriers, and 16/16 fences.

One genuine merge-era behavioral regression was found: Record/Replay reused
one atomic report slot for repeated executions of a static site, producing
false conflicts in Stream-K arrival and tree atomic-OR.  Commit `4600166d3d`
publishes bounded dynamic atomic events; both workloads now pass all four
profiles at the rebased tip.  The apparent failures listed below reproduce
with the preserved pre-rebase hook and are therefore stale historical claims
or retained defects, not regressions introduced by the shared-branch merge.
An initially rejected D128-pressure Sampled artifact was also transient: exact
direct reruns with both hooks pass all four workload tests with 40/40 accesses,
8/8 applicable barrier members, and complete analysis.

### Post-merge revalidation exceptions

This small override ledger takes precedence over stale green cells in the
larger table below.  The merge-revalidation failures were first recorded at
`66586a47b2` and reproduced with the pre-rebase `58379f3c1a` hook; cells that
have since advanced name their newer committed revision and retained evidence.

| Tracking unit | SuperCollider | Record/Replay | Sampled | Inline Shadow |
|---|---|---|---|---|
| `torch.mode` | 🟩 At `23236d897d`: strict exact oracle; 28,195/28,195 ordinary accesses; paired 116.51x; reviewed exact-one qualified miss and both health gates accepted | 🟩 At `6491647e31`: exact oracle; 28,939/28,939 accesses, 2/2 atomics, and 4,446/4,446 barriers; paired 208.78x; reviewed exact-one qualified miss and both health gates accepted | 🟩 At `96ecd9024a`: exact oracle; 28,939/28,939 accesses, 2/2 atomics, and 8,892/8,892 barrier members; paired 120.09x; reviewed exact-one qualified miss and both health gates accepted | 🟩 Exact oracle; 28,939/28,939 accesses and 4,446/4,446 barriers |
| `torch.histc` | 🟩 Current dual-precision exact oracle; 133/133 ordinary accesses; paired 33.54x; reviewed causal fault bundle retained | 🟩 Current dual-precision exact oracle; 175/175 accesses and 168/168 split-barrier members; paired 28.53x; reviewed causal fault bundle retained | 🟩 Current dual-precision exact oracle; 175/175 accesses and 168/168 barriers; paired 24.82x | 🟩 Current dual-precision exact oracle; 175/175 accesses and 84/84 barriers; paired 81.25x. Current-pass dispatcher/body overlap fixed and unit-tested |
| `001_sk_mxf8f4gemm_tdm` | 🟩 Exact oracle; 768/768 accesses | 🟩 Exact oracle; 768/768 accesses, 204/204 barriers, and 24/24 fences | 🟩 Exact oracle; 768/768 accesses and 180/180 barriers | 🟩 Exact oracle; 768/768 accesses and 102/102 barriers; current paired 13.38x; reviewed exact-one fault and health accepted |
| `006_sk_hgemm_quick` | 🟧 Existing bounded result retained | 🟩 At `82a0a1dd8b`: exact oracle; 8,162/8,162 accesses, 292/292 barriers, and 80/80 fences; paired 2.02x; reviewed exact-one fault and both health gates accepted | 🟩 Exact oracle; 8,162/8,162 accesses and 544/544 barriers | 🟧 Existing bounded result retained |

The initial PyTorch agent-discovery and baseline-copy failures were setup
issues rather than ConSan regressions.  Staging the matching runtime first in
`LD_LIBRARY_PATH` restores discovery; disabling software-model SDMA avoids a
baseline-only host-to-device-copy crash.  Accepted PyTorch revalidation uses
that workaround consistently.

| Priority | Tracking unit | SuperCollider | Record/Replay | Sampled | Inline Shadow | Notes |
|---|---|---|---|---|---|---|
| P0 | PyTorch/Triton tensor-descriptor add, one-CTA and two-CTA variants | 🟩 Exact `a + b`; 29/29 accesses; current paired 1.19x | 🟩 Exact `a + b`; 29/29 accesses; 12/12 barriers; current paired 1.33x | 🟩 Exact `a + b`; 29/29 accesses; 20/20 applicable barriers; current paired 2.16x | 🟩 Exact `a + b`; 29/29 accesses; 12/12 barriers; current paired 4.16x | All profiles accepted; proves tensor-descriptor and clustered dispatch, not cluster-memory opcodes. |
| P0 | `torch.mode`, large rows | 🟩 Exact values/indices; 28,195/28,195 accesses; current paired 120.67x | 🟩 Exact values/indices; 28,939/28,939 accesses and 4,446/4,446 barriers; current paired 232.20x | 🟩 Exact values/indices; 28,939/28,939 accesses and 8,892/8,892 barrier members; current paired 203.53x | 🟩 Exact values/indices; 28,939/28,939 accesses and 4,446/4,446 barriers; current paired 341.90x | All four profile bundles accepted. |
| P0 | `torch.topk`, FP64 spill and BF16 coverage cases | 🟩 Exact FP64/BF16 values and indices; 160,956/160,956 accesses; current paired 903.20x maximum; reviewed exact-one fault and health accepted | 🟨 The retained exact FP64/BF16 run passes in 75.02 seconds with dynamic completeness and zero diagnostics. Current admission work eliminates all 16,856 resource failures, but a bounded diagnostic still omits 1,777/113,020 accesses and 2,340/10,782 barriers at branch placement and times out at 240 seconds after growing the large replacement from 281 MB to 322 MB; a compact complete route and fresh exact verdict remain | 🟨 Current exact FP64/BF16 oracles pass in 93.52 seconds with dynamic completeness, 102,598/161,136 accesses, and 15,182/15,182 barriers; spill-backed scalar support does not address the remaining owner resource failures | 🟨 Current exact FP64/BF16 oracles pass in 89.29 seconds with dynamic completeness and zero diagnostics; static coverage remains resource-incomplete at 113,760/161,244 accesses and 9,148/11,423 barriers. The smaller BF16 object is complete at 48,224/48,224 accesses and 6,032/6,032 barriers | The InlineShadow architectural-SGPR exclusion fixes the all-zero BF16 result. Artifact `rebase-20260821-gfx1250-pytorch-topk-inline-architectural-sgprs` supersedes the red frontiers retained in `rebase-20260821-gfx1250-pytorch-topk-inline-branch-only-fix` and `rebase-20260821-gfx1250-pytorch-topk-inline-composite-Y7nTVO`. Record/Replay now has placement and execution-scaling debt rather than a resource-admission gap; Sampled and InlineShadow remain resource-incomplete. |
| P1 | `torch.sort` over segmented rows | 🟩 Exact values/indices; 48,224/48,224 accesses; current paired 184.68x | 🟩 Exact values/indices; 48,224/48,224 accesses and 6,032/6,032 barriers; current paired 370.29x | 🟩 Exact values/indices; 48,224/48,224 accesses and 12,064/12,064 barrier members; current paired 171.77x; reviewed noncausal fault accepted | 🟩 Exact values/indices; 48,224/48,224 accesses and 6,032/6,032 barriers; current paired 416.22x | All four profile bundles accepted. |
| P1 | Collision-heavy `torch.scatter_reduce` (`sum`, BF16 and FP32) | 🟩 Exact collision sums; 23/23 accesses; current paired 24.37x | 🟩 Exact collision sums; 23/23 accesses; current paired 42.30x | 🟩 Exact collision sums; 23/23 accesses; current paired 41.91x | 🟩 Exact collision sums; 23/23 accesses; current paired 40.17x | All profiles accepted; ordered-atomic fault modes are typed N/A for this relaxed singleton reduction. |
| P1 | `torch.histc` with a shared-memory-sized bin count | 🟩 Exact FP32/FP64 counts; 133/133 supported accesses; current paired 33.54x | 🟩 Exact FP32/FP64 counts in 6.33 seconds with 175/175 accesses and 168/168 split-barrier members; current paired 28.53x; prior causal fault bundle retained | 🟩 Exact FP32/FP64 counts in 3.29 seconds with 175/175 accesses and 168/168 applicable barriers; current paired 24.82x | 🟩 Exact FP32/FP64 counts in 3.69 seconds with 175/175 accesses and 84/84 barriers; current paired 81.25x after repairing same-pass dispatcher/body overlap | All four profiles remain green on the expanded workload. Artifacts `prep-20260822-gfx1250-pytorch-histc-dual-precision-all-v3` and `prep-20260822-gfx1250-pytorch-histc-dual-precision-inline-fix-v1` record the refresh; the earlier reviewed causal fault evidence remains applicable. |
| P2 | `torch.linalg.vector_norm` and large-row `torch.softmax` | 🟩 Exact norm/softmax; 4,756/4,756 accesses; current paired 315.57x | 🟩 Exact norm/softmax; 4,756/4,756 accesses and 2,352/2,352 barriers; current paired 211.06x; reviewed exact-one fault and health accepted | 🟩 Exact norm/softmax; 4,756/4,756 accesses and 4,572/4,572 barriers; current paired 534.97x | 🟩 Exact norm/softmax; 4,756/4,756 accesses and 2,352/2,352 barriers; current paired 317.24x | Record/Replay uses owner-local zero-generation records where full-pressure kernels cannot preserve the global dispatch-ID pair; all profiles have accepted bundles. |
| P1 | PyTorch cluster synchronization | 🟩 Exact oracle; 25/25 applicable accesses; current paired 1.02x | 🟩 Exact oracle; 25/25 accesses and 2/2 barriers; current paired 1.03x | 🟩 Exact oracle; 25/25 accesses and 4/4 barrier members; current paired 1.07x | 🟩 Exact oracle; 25/25 accesses and 2/2 barriers; current paired 1.24x | All profiles accepted for the causal cluster-scope synchronization workload. |
| Survey | Cluster-memory and inter-workgroup synchronization from PyTorch | 🟩 Executable cluster-scope synchronization full bundle accepted | 🟩 Executable cluster-scope synchronization full bundle accepted | 🟩 Executable cluster-scope synchronization full bundle accepted | 🟩 Executable cluster-scope synchronization full bundle accepted | Cluster-scope synchronization is covered; no distinct cluster-memory opcode is claimed. |

### 2026-08-22 dual-precision `torch.histc` refresh

Artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx1250-pytorch-histc-dual-precision-all-v3`
runs the expanded FP32 and FP64 exact-bin-count oracles through RocJitsu's
checked-in B0 `gfx1250_mi455x.json` configuration. Baseline, SuperCollider,
Record/Replay, and Sampled accept both precision rows. SuperCollider patches
133/133 supported accesses; Record/Replay and Sampled each patch 175/175
accesses plus 168/168 split-barrier members. Their maximum paired device
slowdowns are 33.54x, 28.53x, and 24.82x respectively, and all clean rows have
complete static, analysis, and dynamic verdicts.

The first Inline Shadow row rejected the object before execution. Final
validation showed that its reused access dispatcher had grown from
`.text+196516` across a 24-byte local barrier body already emitted at
`.text+196828` in the same incremental pass. The reuse search considered only
the globally committed patch inventory, so it missed the nearer current-pass
boundary and overwrote both code and patch ownership. Dispatcher growth now
uses the nearest nonempty trampoline from both inventories and falls back to
the independent barrier router when necessary. A focused host regression pins
that two-inventory boundary, and the neighboring dense-reuse/fallback/router
tests remain green.

Artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx1250-pytorch-histc-dual-precision-inline-fix-v1`
accepts the repaired Inline Shadow row in 3.69 seconds with exact FP32 and FP64
counts, complete 175/175 access plus 84/84 barrier coverage, complete
static/analysis/dynamic evidence, and an 81.25x maximum paired device
slowdown. The cell therefore remains green on the expanded current workload;
the previously accepted causal fault bundle is retained rather than rerun or
reinterpreted.

### 2026-08-21 `torch.topk` Record/Replay simulator fix

Artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx1250-pytorch-topk-rr-bcnt-fix-timeout180`
runs the target-specific PyTorch environment through the explicit RocJitsu
`gfx1250_mi455x.json` launcher.  The exact FP64 and BF16 value/index oracles now pass
both uninstrumented in 1.83 seconds and under Record/Replay in 75.02 seconds.
This supersedes
`/home/ossci/xx/consan-validation/rebase-20260821-gfx1250-pytorch-topk-rr-current`,
where both paths returned only four BF16 values per row.

The simulator had implemented `v_bcnt_u32_b32` as `popcount(src0)` and dropped
its explicit `src1` accumulator.  PyTorch's TopK gather uses that accumulator
to combine each lane's wave-local rank with the global prefix reserved for its
wave; dropping it made all 32 waves overwrite the first few output slots.  The
fix is guarded by generator, scalar/SIMD, exact-instruction, and paired
gfx1250 device tests.

Both transformed objects are dynamically complete and replay 58,056 published
access records plus 512 barrier records without drops, unsupported replay
events, conflicts, metadata exhaustion, or diagnostics.  Static coverage is
complete for the second object at 48,224/48,224 accesses and 12,064/12,064
barriers.  The first object is incomplete at 105,524/113,020 accesses and
1,422/10,782 barriers; all 16,856 omissions are typed resource failures.  The
aggregate row is therefore 153,748/161,244 accesses and 13,486/22,846
barriers.  The loaded hook SHA-256 is
`7b64edf75d40fb14de7178be673bfc0b03da37ce9df85eabdb8984e0bf561b7b`.

### 2026-08-22 `torch.topk` Record/Replay resource-admission follow-up

Artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx1250-pytorch-topk-rr-branch-only-v1`
tests the fixed-private full-scalar-pressure route through the explicit
RocJitsu gfx1250 launcher with hook SHA-256
`62889da94fbfdcef1406ff99f5501bfc6f91812e6f5a91f0a44488ca50251eaf`.
The first transformed object has no resource failures: 41,497 sites use dead
VGPRs and 82,305 use descriptor growth. It patches 111,243/113,020 accesses
and 8,442/10,782 barriers, improving the aggregate static coverage to
159,467/161,244 accesses and 20,506/22,846 barriers. The remaining 1,777
accesses and 2,340 barriers are typed placement/lowering failures, not resource
failures. The smaller 48,224-access object remains statically complete.

This is not an accepted promotion. The large replacement grows from the prior
281,317,752 bytes to 321,544,568 bytes, and the row reaches its 240-second
bound after the smaller object's complete 141,502,104-byte transform and first
dispatch. No final exact oracle, teardown verdict, or analysis verdict is
available. Transform timing itself remains bounded at 29.6 and 8.1 seconds;
the new delay is execution/code-size amplification. Focused host regressions
now require gfx1250 full-pressure fixed-stack owners to receive a state-safe
route, and grouped qualification telemetry makes any future resource failure
actionable. Completion still requires a compact route with zero placement
omissions followed by a fresh accepted exact run; repeatedly extending this
timeout is not useful evidence.

### 2026-08-21 `torch.topk` InlineShadow architectural-SGPR fix

Artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx1250-pytorch-topk-inline-architectural-sgprs`
runs the exact FP64 and BF16 value/index oracles through the explicit RocJitsu
`gfx1250_mi455x.json` launcher.  Both pass under InlineShadow in 89.29 seconds with
dynamic completeness, zero diagnostics, and loaded hook SHA-256
`669981f6e21299650871f0e8e1e18433f2c2a46733f07ac82c04e82b41f40e2f`.

The corrupting access was an address/data-aliasing `ds_store_b96`.  Its
relocated call path selected first `s[102:103]`, then `s[104:105]`, for the
generated `s_call_i64` return pair.  Both pairs are encodable ordinary SGPRs,
but gfx1250 aliases them to the architectural `FLAT_SCRATCH` and `XNACK_MASK`
registers.  The common gfx1250 scalar-selection and validation paths now
exclude the complete `s[102:105]` range.  Host regressions cover both explicit
and automatic selection, while the paired correct/incorrect B96 device
contract preserves the address/data alias found in PyTorch.

Static coverage remains resource-incomplete overall at 113,760/161,244
accesses and 9,148/11,423 barriers.  The smaller BF16 object is fully covered
at 48,224/48,224 accesses and 6,032/6,032 barriers; the larger object admits
65,536/113,020 accesses and 3,116/5,391 barriers.

## Environment baseline

This table establishes that the development environment can execute target
code.  It is not instrumentation acceptance evidence.

| Item | Current evidence |
|---|---|
| Port branch | `users/bjacob/sanitizers` |
| Rebased foundation | `origin/users/bjacob/sanitizers` at `6420276d7bdc` |
| ROCm distribution | `$WORKSPACE_ROOT/TheRock/build/dist/rocm` |
| Toolchain | workspace TheRock HIP compiler targeting `gfx1250`; host Clang 21.1.8 |
| Execution | software GPU environment initialized one gfx1250 node |
| Dispatch smoke | 1,024-element HIP vector add passed with zero errors; a separate HIP device test passed 1/1 |
| Code-object proof | executed test contains a `hipv4-amdgcn-amd-amdhsa--gfx1250` offload bundle |

## Promotion requirements

A green workload/profile cell requires retained evidence that:

1. ordinary `standard-v1` defaults instrument every admitted supported site;
2. the uninjected workload passes its independent oracle;
3. static and dynamic completeness are explicit and accepted, with typed
   exclusions and no hidden coverage-limiting knobs;
4. every admitted mutation reaches exactly one reviewed final executed byte
   sequence and its diagnostic, qualified miss, or typed non-applicability
   matches the precommitted flavor contract;
5. execution terminates within its bound and the device remains healthy;
6. paired baseline/profile latency and instrumentation-owned peak memory are
   retained; and
7. the command, environment, hashes, source identities, hook identity, target,
   and artifact directory all refer to the same frozen committed tip.

A crash, trap, timeout, oracle mismatch, or device loss is never counted as a
ConSan diagnostic.  A flavor may be green with an honest qualified miss when
that outcome was precommitted and the mutation, containment, and independent
oracle evidence are valid.

## Implementation evidence

Prerequisite implementation results will be recorded here as they land.  They
do not promote matrix cells without the end-to-end evidence above.

| Area | Status | Evidence |
|---|---|---|
| Generated gfx1250 decode | 🟨 Hook-integrated baseline | Generated decoding and builders are hook-integrated; ConSan-specific qualification remains active. |
| ConSan instruction emission | 🟨 Active | All four engines emit validated target instructions across the current matrix; remaining synchronization gaps are visible in non-green cells. |
| Register allocation and spilling | 🟨 Active | Wave32 allocation, spill-private growth, dense relay placement, and shared persistent state are workload-proven; remaining pressure gaps are tracked in the matrix. |
| Validation target | 🟨 Green expansion active | The target registry and doctor pass; the matrix is the authoritative runnable-cell ledger. |
| Four focused flavor verticals | 🟨 Four-engine bootstrap | The cooperative-LDS bootstrap executes in all four modes; broader acceptance is tracked per workload above. |
