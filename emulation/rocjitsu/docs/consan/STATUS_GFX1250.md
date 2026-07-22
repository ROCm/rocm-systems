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
| **P0 Qwen3-0.6B prefill** | 🟩 Current paired 1.94x; exact oracle and 1000/1000 accesses | 🟩 Current paired 5.33x; exact oracle, 1000/1000 accesses, and 46/46 barriers | 🟧 Full-object isolated run signals at ~255 seconds and the independent software-GPU path has no verdict through 600 seconds; a diagnostic restricted to the final 151,936-workgroup initializer is exact and complete at 3/3 accesses plus 4/4 barrier members, localizing the blocker to cumulative full-object cost; no accepted overhead | 🟧 Current isolated run signals at the final large-output dispatch after 552 seconds; no verdict or accepted overhead |
| **P1 Sharktank TP1 prefill** | 🟩 Exact prefill oracle; 352/352 accesses; current paired 1.17x | 🟩 Exact prefill oracle; 352/352 accesses, 37/37 barriers; current paired 1.25x | 🟩 Exact prefill oracle; 352/352 accesses, 64/64 applicable barriers; current paired 1.51x | 🟩 Exact prefill oracle; 352/352 accesses, 37/37 barriers; current paired 2.11x |
| **P1 Sharktank TP1 decode/combined** | 🟩 Exact decode/combined oracles; 704/704 accesses; current paired 1.09x | 🟩 Exact decode/combined oracles; 704/704 accesses, 74/74 barriers; current paired 1.16x | 🟩 Exact decode/combined oracles; 704/704 accesses, 128/128 applicable barriers; current paired 1.28x | 🟧 Compute-active through 600 seconds; no verdict or accepted overhead |
| **P2 Sharktank TP2 family** | 🟧 Current uninstrumented all-mode baseline exceeds 600 seconds; prior frozen bundle retained | 🟧 Current uninstrumented all-mode baseline exceeds 600 seconds; prior frozen bundle retained | 🟧 Current uninstrumented all-mode baseline exceeds 600 seconds; prior frozen bundle retained | 🟧 Current uninstrumented all-mode baseline exceeds 600 seconds; prior frozen bundle retained |
| **P4 hip-moi D128 block attention** | 🟩 Current paired 1.56x; 18/18 accesses | 🟩 At `fff5f3597b`: spot rerun exact in 10.89 seconds; 18/18 accesses and 4/4 barriers; paired 1.11x retained | 🟩 Current paired 1.13x; 18/18 accesses, 8/8 applicable barriers | 🟩 Current paired 1.09x; 18/18 accesses, 4/4 barriers |
| **P4 hip-moi D128 pressure attention** | 🟩 Current paired 1.84x; 40/40 accesses | 🟨 Fresh clean run passes three of four exact cases with 40/40 accesses and 4/4 barriers statically complete; the fourth exceeds the 30-second row deadline | 🟩 Current paired 1.12x; 40/40 accesses, 8/8 applicable barriers | 🟩 Current paired 1.29x; 40/40 accesses, 4/4 barriers |
| **P4 hip-moi WMMA attention** | 🟩 Current paired 1.78x; 18/18 accesses | 🟩 At `fff5f3597b`: spot rerun exact in 7.81 seconds; 18/18 accesses and 4/4 barriers; paired 1.17x retained | 🟩 Current paired 1.15x; 18/18 accesses, 8/8 applicable barriers | 🟩 Current paired 1.17x; 18/18 accesses, 4/4 barriers |
| **P4 hip-moi Stream-K arrival** | 🟩 Current paired 7.38x; 4/4 accesses | 🟩 Fresh clean run exact and complete at 4/4 accesses, 4/4 barriers, 10/10 atomics, and 16/16 fences; prior paired 2.41x retained | 🟩 Current paired 2.72x; 4/4 accesses, 8/8 applicable barriers, 10/10 atomics | 🟩 Current paired 2.62x; 4/4 accesses, 4/4 barriers, 10/10 atomics |
| **P4 hip-moi tree atomic-OR** | 🟩 Current paired 6.55x; 4/4 accesses | 🟩 Fresh clean run exact and complete at 4/4 accesses, 4/4 barriers, 10/10 atomics, and 16/16 fences; prior paired 2.04x retained | 🟩 Current paired 2.57x; 4/4 accesses, 8/8 applicable barriers, 10/10 atomics | 🟩 Current paired 2.19x; 4/4 accesses, 4/4 barriers, 10/10 atomics |
| **P4 Jakub attention variants** | 🟩 Current paired 2.52x; 31/31 accesses | 🟩 Current paired 1.44x; 62/62 accesses, 4/4 barriers | 🟩 Current paired 1.56x; 62/62 accesses, 8/8 applicable barriers | 🟩 Current paired 1.55x; 62/62 accesses, 4/4 barriers |

CLIP BF16 is intentionally omitted from the current acceptance matrix.  Its
uninstrumented execution is not presently practical in the software GPU
environment: the default multi-executor configuration can stall before model
inference, and a single-executor baseline reaches inference but remains too
slow for useful iteration.  Existing static gfx1250 qualification evidence is
not sufficient for promotion; CLIP remains outside the matrix denominator
until baseline execution becomes suitable for end-to-end validation.

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
| P1 | `007_sk_mxf4gemm_tdm` | 🟩 2448/2448 accesses; current paired 1.35x | 🟧 Current instrumentation is statically complete (2448 accesses, 272 barriers, 64 fences); PGR1 and PGR2 access/fence-only pass, but PGR2 access candidate 145 first corrupts results when barriers are enabled | 🟩 2448/2448 accesses; 480/480 barriers; current paired 1.38x | 🟧 Compute-active through 1800 seconds; no verdict | Record/Replay is localized to barrier/access composition around `ds_load_b128` at `0x257d8`, after barriers at `0x25790` and `0x2579c`; Inline Shadow remains bounded. |
| P2 | Reduced `sk_sgemm_runtime_smoke` | 🟩 Exact numeric oracle; 640/640 accesses; current paired 1.07x | 🟩 Exact numeric oracle; 640/640 accesses; 22/22 barriers; 8/8 fences; current paired 1.32x | 🟩 Exact numeric oracle; 640/640 accesses; 40/40 barriers; current paired 1.34x | 🟩 Exact numeric oracle; 640/640 accesses; 22/22 barriers; paired 1.33x; causal fault diagnosed | Exact numeric oracle; all profiles accepted, including a causal Inline Shadow fault. |
| P2 | `000_sk_sgemm_quick` | 🟨 First problem: 12/12 exact numeric rows; 640/640 accesses; static/dynamic complete | 🟨 First problem exact and fully covered; aggregate host analysis fixed; full client is intrinsically execution-bound | 🟨 First problem: 12/12 exact numeric rows; 640/640 accesses; 40/40 barrier members | 🟧 First problem: 12/12 exact rows and complete static coverage; interrupted second problem leaves dynamic analysis incomplete | The first problem is validated; the full multi-problem client remains execution-bound. |
| P2 | `005_sk_f8gemm_quick` | 🟩 Exact oracle; 1772/1772 accesses; current paired 1.43x; reviewed fault and health accepted | 🟩 Exact oracle; 1772/1772 accesses; 44/44 barriers; 16/16 fences; current paired 8.00x | 🟧 Current clean execution remains compute-active through 900 seconds; no verdict or measured overhead | 🟧 Current tip executes 49 exact rows with zero failures before the fixed 180-second bound | SuperCollider and Record/Replay are accepted; Sampled and Inline Shadow lack a full-client verdict. |
| P2 | `006_sk_hgemm_quick` | 🟧 136 exact numeric passes with zero failures; first 143-solution problem remains active at 300 seconds | 🟩 Exact oracle; 8162/8162 accesses; 292/292 barriers; 80/80 fences; current paired 2.02x | 🟩 Exact oracle; 8162/8162 accesses; 544/544 barriers; current paired 2.24x | 🟧 Current tip executes 189 exact rows with zero failures before the fixed 180-second bound | Record/Replay and Sampled are accepted; SuperCollider and Inline Shadow lack a full-client verdict. |
| P3 | `015_spmm_f8_ml` stress | 🟧 First contraction exact numeric pass; 298/4316 accesses; second orientation active at 120 seconds | 🟨 Current clean E2E accepts all seven clients with 172,468/172,468 accesses and 3,060/3,060 barriers; paired and reviewed-fault bundle pending | 🟩 All seven clients exact; 172,468/172,468 accesses and 6,120/6,120 barriers; paired 4.88x; reviewed fault, containment, and health accepted | 🟧 Exact failing kernel fixed; standard run has 8 passes and zero failures before its bound | Sampled is accepted; Record/Replay lacks paired/fault evidence, and the other profiles remain bounded. |
| P2 | `019_spmm_f16_sb` closure | 🟧 9,546/9,546 accesses patched; first client exceeds 300 seconds without a numeric row | 🟩 Four exact orientations; 31,265/31,265 accesses; current paired 2.48x | 🟧 9,546/9,546 accesses and 646/646 applicable barriers patched; first client exceeds 300 seconds without a numeric row | 🟧 9,546/9,546 accesses and 323/323 barriers patched; first client exceeds 300 seconds without a numeric row | Sampled is accepted; the other profiles retain the bounded partial results shown in their cells. |
| Survey | Remaining Tensile configurations | 🟩 Architecture-level decoded opcode union covered by accepted selected rows | 🟩 Architecture-level decoded opcode union covered by accepted selected rows, including full `019_spmm_f16_sb` bundle | 🟩 Architecture-level decoded opcode union covered by accepted selected rows | 🟩 Architecture-level decoded opcode union covered by accepted selected rows | Survey complete; selected high-signal rows above define the executable denominator. |

The retained P0 and first P1 artifacts confirm the tensor-data-mover control
shape used by these configurations: tensor work is followed by
`s_wait_tensorcnt` and workgroup synchronization before LDS consumption.  The
current acceptance claim remains deliberately scoped to the decoded and
patched LDS accesses plus their surrounding barriers; a wait instruction by
itself is not counted as a raceable memory access.

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
| `torch.histc` | 🟩 At `502b286cfc`: strict exact oracle; 133/133 ordinary accesses; paired 85.67x; reviewed exact-one causal barrier mutation and both health gates accepted | 🟩 Exact oracle; 175/175 accesses and 84/84 barriers | 🟩 Exact oracle; 175/175 accesses and 168/168 barriers | 🟩 Exact oracle; 175/175 accesses and 84/84 barriers |
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
| P0 | `torch.topk`, FP64 spill and BF16 coverage cases | 🟩 Exact FP64/BF16 values and indices; 160,956/160,956 accesses; current paired 903.20x maximum; reviewed exact-one fault and health accepted | 🟧 Current tip transforms both large objects, then signals before an oracle; older exact run covered 160,345/161,136 accesses and all 11,423 barriers | 🟨 Current exact FP64/BF16 oracles pass in 93.52 seconds with dynamic completeness, 102,598/161,136 accesses, and 15,182/15,182 barriers; spill-backed scalar support does not address the remaining owner resource failures | 🟧 Both large objects finish patching; client signals during execution before an oracle | SuperCollider is accepted; Record/Replay and Inline Shadow signal before an oracle, while Sampled remains resource-incomplete. |
| P1 | `torch.sort` over segmented rows | 🟩 Exact values/indices; 48,224/48,224 accesses; current paired 184.68x | 🟩 Exact values/indices; 48,224/48,224 accesses and 6,032/6,032 barriers; current paired 370.29x | 🟩 Exact values/indices; 48,224/48,224 accesses and 12,064/12,064 barrier members; current paired 171.77x; reviewed noncausal fault accepted | 🟩 Exact values/indices; 48,224/48,224 accesses and 6,032/6,032 barriers; current paired 416.22x | All four profile bundles accepted. |
| P1 | Collision-heavy `torch.scatter_reduce` (`sum`, BF16 and FP32) | 🟩 Exact collision sums; 23/23 accesses; current paired 24.37x | 🟩 Exact collision sums; 23/23 accesses; current paired 42.30x | 🟩 Exact collision sums; 23/23 accesses; current paired 41.91x | 🟩 Exact collision sums; 23/23 accesses; current paired 40.17x | All profiles accepted; ordered-atomic fault modes are typed N/A for this relaxed singleton reduction. |
| P1 | `torch.histc` with a shared-memory-sized bin count | 🟩 Exact counts; 133/133 supported accesses; current paired 60.11x | 🟩 Exact counts; 175/175 accesses and 84/84 barriers; current paired 72.00x | 🟩 Exact counts; 175/175 accesses and 168/168 applicable barriers; current paired 67.37x | 🟩 Exact counts; 175/175 accesses and 84/84 barriers; current paired 85.86x | All four profile bundles accepted, including causal barrier-fault evidence. |
| P2 | `torch.linalg.vector_norm` and large-row `torch.softmax` | 🟩 Exact norm/softmax; 4,756/4,756 accesses; current paired 315.57x | 🟩 Exact norm/softmax; 4,756/4,756 accesses and 2,352/2,352 barriers; current paired 211.06x; reviewed exact-one fault and health accepted | 🟩 Exact norm/softmax; 4,756/4,756 accesses and 4,572/4,572 barriers; current paired 534.97x | 🟩 Exact norm/softmax; 4,756/4,756 accesses and 2,352/2,352 barriers; current paired 317.24x | Record/Replay uses owner-local zero-generation records where full-pressure kernels cannot preserve the global dispatch-ID pair; all profiles have accepted bundles. |
| P1 | PyTorch cluster synchronization | 🟩 Exact oracle; 25/25 applicable accesses; current paired 1.02x | 🟩 Exact oracle; 25/25 accesses and 2/2 barriers; current paired 1.03x | 🟩 Exact oracle; 25/25 accesses and 4/4 barrier members; current paired 1.07x | 🟩 Exact oracle; 25/25 accesses and 2/2 barriers; current paired 1.24x | All profiles accepted for the causal cluster-scope synchronization workload. |
| Survey | Cluster-memory and inter-workgroup synchronization from PyTorch | 🟩 Executable cluster-scope synchronization full bundle accepted | 🟩 Executable cluster-scope synchronization full bundle accepted | 🟩 Executable cluster-scope synchronization full bundle accepted | 🟩 Executable cluster-scope synchronization full bundle accepted | Cluster-scope synchronization is covered; no distinct cluster-memory opcode is claimed. |

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
