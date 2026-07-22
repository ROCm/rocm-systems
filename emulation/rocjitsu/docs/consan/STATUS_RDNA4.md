# ConSan RDNA4 (`gfx1201`) status

This is the current-tip workload × instrumentation catch-up ledger for
`gfx1201`.  It deliberately does not inherit a green cell, coverage
denominator, machine-code identity, fault expectation, timing result, or
provenance record from `gfx1250`.

The last complete gfx1201 certificate was produced on 2026-07-16 at executable
commit `640e575da2`, with hook SHA-256
`c45aa0fece5a9aa7ef8b3ad24bcbb2077e477586df6b4eecf12990f7fafa693d`.  It is
retained below as historical evidence, but it does not qualify the current
branch.  After rebasing onto the 2026-07-21 sanitizer tip, the current
certificate has advanced through validation checkpoint `6ddab1a730`, using
rebuilt hook SHA-256
`ae96b66a43b6017570784c3136373340368f27205d1b9a058805c411727160a9`.
Current-tip qualification is in progress; cells below name retained artifacts
when post-rebase execution evidence has replaced the initial gray state.
Intermediate `aff4853917` and `da7af06ef7` artifacts predate that rebase and
remain diagnostic evidence only; they cannot promote a current cell.

The executable authority is
[`consan_validation.py`](../../tests/dbi/consan/consan_validation.py), with the
experiment contract described by [VALIDATION.md](VALIDATION.md).  The expanded
architecture-local comparison is [STATUS_GFX1250.md](STATUS_GFX1250.md).

End-to-end LLM evidence remains the primary project metric.  Focused decoder,
builder, relocation, spill, and resource tests are prerequisites and debugging
tools; they cannot promote a workload cell by themselves.

## Status legend

Every cell uses the same totally ordered maturity scale as the other
architecture ledgers:

- 🩶 **unseen / unassessed:** no useful current-tip gfx1201 execution evidence;
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

`N/A` is used only when a fresh gfx1201 inventory proves semantic absence and
records a typed reason.  A gray cell says that current-tip evidence is missing;
it does not assert that the implementation has regressed.

## Catch-up snapshot

- **Immediate requalification:** the portable manifest now has 17 workloads ×
  4 profiles = 68 current-tip cells, including six independently selected
  native gfx1201 PyTorch workloads.  Rebuild the exact hook, run
  clean and paired rows, regenerate target identities, review fault specs,
  execute contained faults, and freeze one provenance bundle.
- **Historical comparator:** commit `640e575da2` accepted 55 clean
  baseline/profile rows, 14 reviewed fault policies, and 66 paired-overhead
  rows.  Do not copy its colors or selectors to the current tip.
- **Native corpus expansion:** survey `rocjitsu-test-corpus`'s gfx1201 cases,
  inventory their executed code objects, and admit independently useful
  oracle-backed workloads.  The discovery table tracks all four profiles
  separately even before admission.
- **Native PyTorch expansion:** use the fresh checkout at
  `$CONSAN_VALIDATION_WORKSPACE_DIR/pytorch` to discover shapes that naturally
  select interesting gfx1201 kernels, then freeze small exact clients.  The
  discovery table tracks each profile independently.
- **gfx1250 comparison:** use its workload diversity and proof quality as a
  model, but do not port its configurations, shapes, selectors, denominators,
  or expected outcomes to RDNA4.

Thus the immediate regression campaign contains 68 cells.  The expanded
gfx1201 denominator is intentionally unknown until discovery produces concrete
native workloads with independent oracles.  Survey rows and baselines are not
counted as instrumentation cells.

The current `gfx1201 manifest --json` exposes the original 11 workloads plus
`pytorch-rdna4-compiled-softmax`, `pytorch-rdna4-split-softmax`,
`pytorch-rdna4-llm-topk`, `pytorch-rdna4-sdpa`, and the independently
inventoried target-native `pytorch-scatter-reduce` and
`pytorch-torch-histc`.  All other
expanded PyTorch and Tensile
workloads remain declared with `targets=("gfx1250",)`.  Their configuration
paths, generated kernels, and in one case intrinsic are target-specific; they
are examples to follow, not an RDNA4 porting queue.

## Current-tip requalification matrix

Every cell began gray because the intervening work changed shared decoding,
instrumentation builders, register allocation and spilling, branch-relay
planning, synchronization admission, report sizing, hook composition, runtime
cleanup, and the validation protocol.  Promotion must use one rebuilt current
tip; a successful smoke or an old artifact is insufficient.

| Priority workload | SuperCollider | Record/Replay | Sampled | Inline Shadow | First current-tip gate |
|---|---|---|---|---|---|
| **P0 Qwen3-0.6B prefill** | 🟩 Exact oracle; clean 20/20; exact barrier drop emits an attributed diagnostic and breaks the oracle; overhead 1.044x | 🟩 Exact oracle; clean 20/20 + 14/14; exact drop is a precommitted qualified miss and breaks the oracle; overhead 0.988x | 🟩 Exact oracle; clean 20/20 + 26/26; deterministic 32-offset fault sweep detects 17/32 and every exact mutation breaks the oracle; overhead 1.003x | 🟧 Complete 20/20 accesses + 14/14 barriers, but a diagnostic rejects the correct workload | Checkpoint `9c5ff65778`: clean `...-105`, paired overhead `...-098` through `...-100`, inventory `...-101`, and accepted faults `...-102` through `...-104`; the Sampled offset sweep is fault-sensitivity evidence, not clean-workload tuning |
| **P1 Sharktank TP1 prefill** | 🟩 Exact oracle; clean 352/352; exact drop is a precommitted qualified miss; overhead 1.26x | 🟩 Exact oracle; clean 352/352 + 46/46; exact drop emits a replay diagnostic; overhead 2.09x | 🟩 Exact oracle; clean 352/352 + 86/86; exact drop is a precommitted qualified miss; overhead 1.14x | 🟩 Exact oracle; clean 352/352 + 46/46; exact drop emits an attributed diagnostic; overhead 2.75x | One-tip checkpoint `178d16dcee`: clean `...-040`, overhead `...-035`, inventory `...-036`, accepted fault rows `...-037` through `...-039`; initial historical expectations were retained as failed discovery evidence |
| **P1 Sharktank TP1 decode/combined** | 🟩 Exact oracle; clean 704/704; exact drop is a precommitted qualified miss; overhead 1.18x | 🟧 Complete 704/704 + 92/92 and overhead 1.49x; exact drop is a qualified miss, but one of the retained correct-workload runs reports a false replay conflict | 🟩 Exact oracle; clean 704/704 + 172/172; exact drop is a precommitted qualified miss; overhead 1.01x | 🟩 Exact oracle; clean 704/704 + 92/92; exact drop emits an attributed diagnostic; overhead 1.70x | Checkpoint `b189b3bbe7`: clean `...-047`, overhead `...-041`, inventory `...-042`, accepted fault rows `...-043` through `...-046`; earlier clean `...-033` retains the Record/Replay false positive |
| **P1 PyTorch collision-heavy scatter-reduce** | 🟨 Exact BF16/FP32 collision oracle and clean-complete 27/27 LDS accesses; overhead and reviewed atomic fault pending | 🟨 Exact oracle and clean-complete 27/27 accesses; overhead and reviewed atomic fault pending | 🟧 Exact oracle; 14/27 supported accesses patch, with 13 helper-fill sites reporting `instrumentation_patch_missing` | 🟧 Exact oracle; the same 14/27 helper-fill accesses patch | Checkpoint `6b8d2db5f3`, hook `68bea625…`: clean artifact `rdna4-pytorch-scatter-clean-001`; inventory `rdna4-pytorch-scatter-inventory` records 656 global-atomic sites for both order and scope mutation families. The executed reductions use relaxed singleton global atomics, so MOI correctly reports `atomic=0/0`; the 27 LDS sites belong to a bundled boolean-fill helper |
| **P2 PyTorch/Inductor compiled softmax** | 🟩 Exact oracle; clean 4/4 LDS accesses; exact third barrier drop is a precommitted qualified miss; overhead 0.960x | 🟩 Exact oracle; clean 4/4 + 3/3; exact third barrier drop emits an attributed replay diagnostic; overhead 1.052x | 🟧 Exact oracle and complete 4/4 accesses + 6/6 barrier members, but reports one conflict on the correct workload | 🟧 Exact oracle and static 4/4 accesses + 3/3 barriers now patch, but the 128-workgroup execution exceeds the 32-bank external-shadow partition and records 1,536 dynamic-incomplete events | Inline checkpoint `d47f1f85b6`: clean `...-126` fixes opaque dynamic-LDS report sizing and isolates the workgroup-bank limit. Other profiles retain checkpoint `b6be4081e1`: clean `...-110`, paired overhead `...-111` and `...-112`, inventory `...-107`, and accepted faults `...-108` and `...-109` |
| **P2 PyTorch split online softmax** | 🟨 Exact CPU-derived BF16 oracle and clean-complete 8/8 LDS accesses across two generated stages; overhead and reviewed fault pending | 🟨 Exact oracle and clean-complete 8/8 accesses + 6/6 barriers; overhead and reviewed fault pending | 🟧 Exact oracle and static-complete 8/8 accesses + 12/12 barrier members, but reports 16 sampled conflicts on the correct workload | 🟧 Exact oracle and static-complete 8/8 accesses + 6/6 barriers, but the two-stage execution records 18,576 dynamic-incomplete events at the 32-bank external-shadow limit | Checkpoint artifact `rdna4-pytorch-split-softmax-clean-001`, hook `68bea625…`. The deterministic client freezes upstream `test_online_softmax.py::test_split_reduction` shape `1x(2^20+13)` without loading the unrelated 856-kernel RNG object used by the test harness |
| **P2 PyTorch Qwen-vocabulary top-k** | 🟥 Exact baseline values and indices pass; LDS atomics are no longer a false admission blocker, but the 3,153-kernel, 1.27-million-instruction rocPRIM object reaches the end of Waitcheck and not a ConSan verdict within 30 seconds | 🟥 Correct-workload analysis of that fat code object exceeds even a 60-second discriminator before a patch verdict | 🟥 Shared fat-code-object construction gate | 🟥 Shared fat-code-object construction gate | Checkpoint `48fb0737bb`, hook `6d7b789d…`, artifact `rdna4-topk-sc-atomic-admission` removes the former 40-kernel atomic rejection and exposes the remaining whole-object scalability gate. Historical artifact `rdna4-pytorch-topk-strict-rejection-committed` retains the superseded rejection; Record/Replay artifact `rdna4-pytorch-topk-rr-discriminator` reaches the end of Waitcheck but remains inside ConSan inventory at 60 seconds |
| **P2 PyTorch causal SDPA** | 🟩 Independent CPU oracle; clean 158/158; exact barrier drop emits an attributed diagnostic and breaks the oracle; overhead 1.946x | 🟨 Independent CPU oracle; clean-complete 158/158 accesses + 22/22 barriers + 2/2 atomics + 2/2 fences; overhead 7.795x; two reviewed exact drops break the oracle through unattributed hardware traps but emit no replay diagnostic | 🟥 Full-pressure attention kernel has no persistent dispatch-ID SGPR pair or fresh automatic EXEC-save window; only 14/27 accesses in a separate bundled fill kernel patch | 🟥 Same full-pressure SGPR blockers; strict fail-closed exit after only the separate fill kernel patches 14/27 accesses | Current no-tuning Sampled artifact `rdna4-sdpa-sampled-current`, hook `8c64e5ee…`, reproduces the two typed scalar-state blockers in 10.15 seconds; this is a genuine scalar spill/continuation boundary, not stale setup. Earlier checkpoint `8a5669d5e5` retains paired SC/RR overhead, inventory, and accepted SC fault artifacts. Rejected RR fault rows retain two distinct unattributed-trap outcomes. Record/Replay remains dynamically complete after automatic report headroom was raised; no user tuning knobs are used |
| **P2 Sharktank TP2 family** | 🟩 Exact oracle; clean 2976/2976; exact drop is a precommitted qualified miss; overhead 1.28x | 🟩 Exact oracle; clean 2976/2976 + 228/228; exact drop emits a replay diagnostic; overhead 1.83x | 🟩 Exact oracle; clean 2976/2976 + 420/420; exact drop is a precommitted qualified miss; overhead 1.24x | 🟩 Exact oracle; clean 2976/2976 + 228/228; exact drop detected 16/16; overhead 2.17x | One-tip checkpoint `08b15c6d91`: clean `...-048`, overhead `...-049`, inventory `...-050`, accepted fault rows `...-053` through `...-056`; historical expectations remain as failed discovery rows `...-051` and `...-052` |
| **P3 CLIP BF16** | 🟩 Exact oracle; clean 85/85; exact drop and move are precommitted qualified misses; overhead 0.98x | 🟩 Exact oracle; clean 85/85 + 36/36; exact drop and subtle move are precommitted qualified misses; overhead 1.236x | 🟩 Exact oracle; clean 85/85 + 72/72; exact drop and move are precommitted qualified misses; overhead 0.97x | 🟩 Exact oracle; clean 85/85 + 36/36; exact move emits an attributed diagnostic while exact drop is a qualified miss; overhead 1.51x | Record/Replay checkpoint `38f77d6601`, hook `bb08fa7c…`: clean `rdna4-clip-rr-scalar-007`, paired overhead `...-overhead-001`, inventory `...-inventory-001`, and accepted faults `...-fault-drop-001` and `...-fault-move-001`. gfx1201 now keeps owner/epoch in persistent SGPRs instead of loading a private-memory epoch at every hot access. Other profiles remain at checkpoint `3033f30f2a` in `...-057` through `...-070` |
| **P3 PyTorch native histogram** | 🟨 Exact oracle; clean-complete 135/135 ordinary LDS accesses; relaxed LDS atomics remain untouched; overhead 1.033x; reviewed fault pending | 🟧 Exact oracle and complete 135/135 ordinary LDS accesses; 42/84 supported barriers patch, with the rest lacking reachable entry islands | 🟧 Exact oracle and clean execution; 122/135 accesses and 86/168 barrier members patch, while unreachable relays remain explicit coverage gaps | 🟧 Exact oracle; 100/135 accesses and 43/84 barriers patch with no clean diagnostic | SuperCollider checkpoint `48fb0737bb`, hook `6d7b789d…`: clean `rdna4-histc-sc-clean-frozen`, paired overhead `rdna4-histc-sc-overhead-frozen`, and complete inventory `rdna4-histc-inventory-after` (84 barrier sequences, 104 atomic sites). Exact executed fault selection remains to be reviewed. Sampled checkpoint `ab8f0ea4c5`, hook `95e7ddd3…`, artifact `rdna4-histc-sampled-partial-002`; earlier checkpoint `0a7b607134` retains the other profiles. The relaxed accumulations are not qualified MOI ordering events and correctly remain `atomic=0/0` |
| **P4 hip-moi D128 block attention** | 🟩 Exact oracle; clean 12/12; exact barrier drop is a precommitted qualified miss and breaks the oracle; overhead 164.77x | 🟩 Exact oracle; clean 12/12 + 4/4; exact barrier drop emits an attributed diagnostic and breaks the oracle; overhead 11.21x | 🟨 Exact oracle and clean-complete 12/12 accesses + 8/8 barrier members; overhead and reviewed fault pending | 🟧 Complete 12/12 accesses + 4/4 barriers, but a diagnostic rejects the correct workload | Sampled checkpoint `e96efba818`: clean `...-140` uses a reachable local relay after the end-of-text island became unreachable. SuperCollider checkpoint `ca1eb5456e`: `...-132` through `...-135`. Record/Replay checkpoint `baed32a85e`: `...-114` through `...-118` |
| **P4 hip-moi D128 pressure attention** | 🟨 Exact oracle and clean-complete 12/12 accesses in 3.48 seconds; overhead and reviewed fault pending | 🟩 Exact oracle; clean 12/12 + 4/4; exact barrier drop emits an attributed diagnostic and breaks the oracle; overhead 14.30x | 🟧 Exact oracle and complete 12/12 accesses + 8/8 barrier members, but four sampled conflicts reject the correct workload | 🟩 Exact oracle; clean 12/12 + 4/4; exact barrier drop emits an attributed diagnostic; overhead 13.72x | SuperCollider checkpoint `cde553c12c`, hook `20334d0f…`, artifact `rdna4-d128-pressure-sc-maxrefs-cache`: caching immutable owner register bounds removes a quadratic shared-function scan and replaces the prior 30-second planner timeout. Sampled checkpoint `e96efba818`: clean `...-141`. Record/Replay checkpoint `baed32a85e`; Inline checkpoint `79aea7420c` |
| **P4 hip-moi WMMA attention** | 🟩 Exact oracle; clean 12/12; exact barrier drop is a precommitted qualified miss and breaks the oracle; overhead 158.07x | 🟩 Exact oracle; clean 12/12 + 4/4; exact barrier drop emits an attributed diagnostic and breaks the oracle; overhead 11.99x | 🟨 Exact oracle and clean-complete 12/12 accesses + 8/8 barrier members; overhead and reviewed fault pending | 🟧 Complete 12/12 + 4/4 and one clean run passes, but two of three repeated correct-workload runs emit false diagnostics | Sampled checkpoint `e96efba818`: clean `...-142`. SuperCollider checkpoint `ca1eb5456e`: `...-128` through `...-131`. Record/Replay checkpoint `baed32a85e`: `...-114`, `...-115`, `...-116`, `...-120`, and `...-125`. Inline repeated attempt `...-078` retains its intermittent false positive |
| **P4 hip-moi Stream-K arrival** | 🟩 Exact oracle; clean 4/4 accesses; exact order and scope weakenings are precommitted qualified misses; overhead 558.83x | 🟩 Exact oracle; clean 4/4 accesses + 15/15 atomics + 4/4 barriers + 16/16 fences; exact order and scope weakenings are precommitted qualified misses; overhead 53.00x | 🟨 Exact oracle and clean-complete 4/4 accesses + 15/15 atomics + 8/8 barrier members; overhead and reviewed faults pending | 🟧 Complete 4/4 + 15/15 + 4/4, but a diagnostic rejects the correct workload | Sampled checkpoint `e96efba818`: clean `...-143`. Record/Replay checkpoint `baed32a85e`: `...-114` through `...-116`, `...-122`, and `...-123`. SuperCollider checkpoint `79aea7420c`: `...-073`, `...-082`, `...-087`, `...-091`, and `...-092` |
| **P4 hip-moi tree atomic-OR** | 🟩 Exact oracle; clean 4/4 accesses; exact order and scope weakenings are precommitted qualified misses; overhead 591.81x | 🟧 Exact oracle and complete 4/4 accesses + 15/15 atomics + 4/4 barriers, but four supported fences fail placement at 12/16 | 🟨 Exact oracle and clean-complete 4/4 accesses + 15/15 atomics + 8/8 barrier members; overhead and reviewed faults pending | 🟧 Complete 4/4 + 15/15 + 4/4, but a diagnostic rejects the correct workload | Sampled checkpoint `e96efba818`: clean `...-144`. Current Record/Replay discriminator `...-114` isolates four `instrumentation_patch_missing` fences. Other retained evidence is at checkpoint `79aea7420c` |
| **P4 hip-moi Jakub attention variants** | 🟩 Exact oracle; clean 31/31 accesses; exact barrier drop is a precommitted qualified miss; overhead 103.75x | 🟩 Exact oracle; clean 31/31 + 4/4; exact barrier drop is a precommitted qualified miss; overhead 11.69x | 🟨 Exact oracle and clean-complete 31/31 accesses + 8/8 barrier members; overhead and reviewed fault pending | 🟩 Exact oracle; clean 31/31 + 4/4; exact barrier drop emits an attributed diagnostic; overhead 11.03x | Sampled checkpoint `e96efba818`: clean `...-139`. Record/Replay checkpoint `baed32a85e`: `...-113`, `...-115`, `...-116`, and `...-121`. SuperCollider and Inline checkpoint `79aea7420c`: `...-076`, `...-084`, `...-085`, `...-089`, `...-096`, and `...-097` |

## Native gfx1201 corpus discovery and qualification

The local `rocjitsu-test-corpus` checkout at revision `0fccdd2c58d8` is already
organized around target-native discovery.  A collection-only survey for
`gfx1201` finds 105 runnable cases: 7 IREE cases, 3 kernel cases, and 95 CTS
cases.  This is a discovery pool, not 105 ConSan acceptance rows.  Most cases
will have no relevant executed synchronization or will duplicate a stronger
workload.

The gfx1250 Tensile directory is explicitly marked unsupported on gfx1201.
Do not copy, retarget, or author lookalike versions of those configurations
merely to make the ledgers symmetric.  Search the native IREE, kernel, and CTS
suites for workloads that exercise the same broad difficulty classes.

A discovery row uses `— Not admitted` after target-native evidence rules it out
of the validation denominator.  These are resolved selection decisions, not
support results and not gray instrumentation cells.  Once a candidate has an
exact baseline and current-hook execution evidence, its four cells use the
ordinary ordered color scale independently.  Colored discovery cells are
provisional qualification until the command is promoted into
`consan_validation.py` and frozen with overhead and fault evidence.

| Discovery order and native gfx1201 source | SuperCollider | Record/Replay | Sampled | Inline Shadow | Admission decision and evidence |
|---|---|---|---|---|---|
| **D0** `kernels.gfx1201.llama.cpp.llama_mul_mat_vec_q.default` | 🟨 Exact CPU oracle; clean-complete 462/462 LDS accesses; overhead and reviewed fault pending | 🟥 Strict fail-closed exit 92: ordered atomics and fences in ownerless bundled device functions cannot be placed, so the workload never launches | 🟥 Same required-instrumentation-missing rejection before launch | 🟧 Exact oracle and useful partial execution: 135/462 accesses + 7/44 barriers patch, but one diagnostic rejects the clean run | Current Record/Replay and Sampled artifacts `rdna4-matvec-rr-current` and `rdna4-matvec-sampled-current`, hook `8c64e5ee…`, localize the red to ownerless `__ockl_printf_*` and `no_device_code` functions, not the matvec kernels or setup. Safely advancing those cells requires reachability/dispatch-owner proof; guessing that bundled functions are dead would weaken strict coverage. Exact baseline artifact `.pytest-artifacts-rdna4-llama-baseline`; SuperCollider passes the 0.01 CPU tolerance. Inline retains its earlier 135/462 + 7/44 result and one clean diagnostic. |
| **D0** `kernels.gfx1201.llama.cpp.llama_rms_norm.default` | 🟨 Exact CPU oracle; clean-complete 22/22 LDS accesses; no mismatch; 1.306x process overhead; reviewed effective fault pending | 🟨 Exact oracle; clean-complete 22/22 accesses + 11/11 barriers; visible replay evidence and no conflict; 1.308x overhead; reviewed effective fault pending | 🟨 Exact oracle; clean-complete 22/22 accesses + 22/22 barrier members; sampled evidence and no clean diagnostic; 1.307x overhead; reviewed effective fault pending | 🟨 Exact 128-element CPU oracle; static- and dynamic-complete 22/22 accesses + 11/11 barriers; 80 visible events and no malformed or incomplete snapshots; overhead and reviewed effective fault pending | Inline checkpoint `6ddab1a730`, hook `ae96b66a…`, artifact `rdna4-rms-inline-retry-sgpr-fix-full`; exact baseline artifact `.pytest-artifacts-rdna4-llama-baseline`. The versioned exact-shadow retry counter had aliased the saved low half of guest VCC, restoring the decremented `0x7ff` counter as the application's lane mask. Moving the counter from EXEC-base `+8` to the diagnostic-temporary `+20` slot fixes the exact clean workload without a user knob. The earlier `45f8b118f3` dispatch-ID-zero fix remains required. Other profiles retain five-process medians of 0.163896 s baseline, 0.213989 s SuperCollider, 0.214359 s Record/Replay, and 0.214191 s Sampled. |
| **D0** `kernels.gfx1201.hip-matmul.hip_matmul_matvec.m256_n1_k1024` | — Not admitted | — Not admitted | — Not admitted | — Not admitted | A standalone build with the unrelated llama.cpp backend disabled passes all nine exact matvec variants. Fresh SuperCollider inventory finds no decoded LDS, barriers, or atomics in the nine workload kernels; only eight ambiguous flat maybe-group sites fail placement, so this does not add a sound or nonredundant ConSan workload. |
| **D0** `llama.cpp` noncontiguous batched-matmul and hazard metadata | — Not admitted | — Not admitted | — Not admitted | — Not admitted | The gfx1201 config explicitly skips compiling both variants. The hazard overlay restores llama.cpp PR #13155's pre-fix noncontiguous-stride conversion and expects a deterministic output-validation failure; it is a data-layout correctness reproducer, not a concurrency oracle. |
| **D1** Three collected IREE direct-tile matmuls: F16, FP8, and I8 | — Not admitted | — Not admitted | — Not admitted | — Not admitted | The three gfx1201 compilations pass, but their corpus records explicitly set `compile_only=true`; they dispatch no workload and provide no runtime oracle. Normalize a calls/support-module wrapper before reconsidering them. |
| **D1** IREE `argmax`, strided extract, and map-load/map-store cases | — Not admitted | — Not admitted | — Not admitted | — Not admitted | All four exact baselines pass. Record/Replay inventory in `.pytest-artifacts-rdna4-iree-inventory` reports zero supported accesses, barriers, atomics, and fences for every executed code object, so these global-only shapes add no ConSan coverage. |
| **D2** RDNA4 WMMA/SWMMAC, wave32/wave64, atomic, and lane/DS CTS families | — Not admitted | — Not admitted | — Not admitted | — Not admitted | Seven exact baseline cases pass, but the arithmetic and lane-operation cases add no admitted ConSan synchronization traffic. The bundled atomic case is useful as an unsupported-transform stress object, not as a compact or nonredundant acceptance workload. Retain this family as an engine-specific reproducer pool. |
| **D3** Remaining corpus families | — Not admitted | — Not admitted | — Not admitted | — Not admitted | Collection and representative execution now account for all 105 gfx1201 cases. Remaining FPSan host-only/arithmetic cases either dispatch no GPU work or reproduce the already-retained 2,560-atomic support object; remaining integer-ISA arithmetic variants add no distinct ordering family, and the reduction representative currently crashes this SDK's compiler. Keep these as compiler/engine reproducer pools rather than inflating the acceptance denominator. |

Any future selected case must be added to `consan_validation.py` with its exact
target-native command and independent oracle. Its four cells then advance
independently: evidence from one engine never promotes another.

The 2026-07-21 D0 retry enabled `THEROCK_ENABLE_BLAS=ON`, initialized TheRock's
pinned `rocm-libraries` submodule, and built the focused `hipBLAS+stage` target.
The corpus configuration uses `TheRock-build/dist/rocm` as `ROCM_PATH` and
adds both `TheRock-build/math-libs/BLAS/hipBLAS/dist` and
`TheRock-build/math-libs/BLAS/hipBLAS-common/dist` to `CMAKE_PREFIX_PATH`; the
BLAS `lib` directory is prepended to `LD_LIBRARY_PATH`.  This removes the old
setup blocker without copying libraries or changing the corpus source.  Both
exact CPU/GPU baseline comparisons pass.  Current-hook GPU runs additionally
set `HSA_TOOLS_ROCPROFILER_V1_TOOLS=1`, as required by this modern HSA runtime.
The same corpus CMake configuration unnecessarily enables llama.cpp while
building the independent HIP-matmul case.  Configuring that case alone with
`KERNEL_CORPUS_ENABLE_LLAMA_HIP=OFF` establishes its exact baseline and permits
inventory without weakening the workload.

The first RMS fault discriminator exactly drops the wait-side singleton at
`pc=0x1830` in the executed 256-thread, non-fused kernel.  Exact-one mutation
enforcement passes, but the output remains within the 1e-5 oracle tolerance and
SuperCollider reports no mismatch.  Keep that result as a reviewed-candidate
miss, not detection acceptance; select a mutation with an observable semantic
effect before promoting any RMS cell to green.

The noncontiguous llama cases are absent from collection because
`corpus/kernels/configs/gfx1201.json` names both cases in
`skip_compile_tests`; the originating corpus commit described these as
temporarily skipped "unsure kernels."  Static comparison of the hazard overlay
with the fixed vendored source shows that it converts a noncontiguous view as
one contiguous span while retaining the old batched-GEMM strides.  Its case
manifest requires exit code 1 for the trigger because CPU/output validation is
supposed to fail.  That makes it a useful llama.cpp layout-regression
reproducer, but not a ConSan concurrency-validation workload.

The D2 CTS survey ran serially with a 30-second per-case bound.  Exact
baselines pass for the atomic, gfx12 WMMA and SWMMAC wave32/wave64, and DS
swizzle wave32/wave64 cases.  The two DS-permute variants do not compile with
the current SDK because `__builtin_amdgcn_wave_shuffle` is unavailable.  A
Record/Replay inventory confirms that the admitted arithmetic and swizzle
executions do not provide the LDS-memory or ordering coverage sought here.
The atomic executable instead bundles a large object with 2,560 atomic
capacity sites and reaches an unsupported transform/Waitcheck path.  That is
valuable focused-debugging evidence, but admitting it would duplicate stronger
atomic workload rows while weakening the exact-oracle acceptance standard.
The retained external artifacts are `.pytest-artifacts-rdna4-cts-sync` and
`.pytest-artifacts-rdna4-cts-inventory` in the `rocjitsu-test-corpus` checkout.
The completion survey additionally retains
`.pytest-artifacts-rdna4-hip-matvec-standalone`,
`.pytest-artifacts-rdna4-cts-d3-baseline`, and
`.pytest-artifacts-rdna4-cts-d3-device-baseline`.  `fpsan_core_test` passes but
is host-only; `fpsan_hip_device_test` reaches the same 2,560-atomic bundled
support-object rejection already represented by D2; and the integer reduction
representative crashes Clang 23 during instruction selection before ConSan can
run.  Those outcomes close corpus *selection* without creating spurious
four-engine acceptance rows.

## Native gfx1201 PyTorch discovery

PyTorch is checked out separately at
`$CONSAN_VALIDATION_WORKSPACE_DIR/pytorch`, revision `50779302a8fd`.  It is not
vendored by RocJITsu or `rocjitsu-test-corpus`.  A separate prebuilt-wheel
environment now provides `torch 2.14.0.dev20260720+rocm7.1`, HIP 7.1.52802,
and Triton 3.8.0.  It imports Triton Gluon, sees the Radeon RX 9070 as
`gfx1201`, and passes an exact elementwise device oracle.  A direct ordinary
`torch.topk` setup smoke also passes exact BF16 and FP64 values-and-indices
oracles.  The runner now carries six independently selected gfx1201 clients.
Ordinary `torch.softmax` compiled by Inductor at shape `128x256`
generates a compact Triton kernel with four supported LDS accesses and three
split-barrier pairs, and records both the source revision and installed
PyTorch/ROCm build identity.  Ordinary `torch.topk` now models decode-time
sampling over one 151,936-element Qwen vocabulary row with `k=50`; its exact
values-and-indices oracle passes, while its native rocPRIM implementation
exposes the large precompiled-object gate recorded below.  Ordinary causal
`torch.nn.functional.scaled_dot_product_attention` selects a full-pressure
native attention kernel; its output is checked against an independent CPU
implementation rather than against a second GPU path.  Ordinary `torch.histc`
selects a native precompiled histogram object whose actual gfx1201 machine code
contains LDS loads/stores, split barriers, and relaxed LDS atomics.
Upstream split online-softmax shape `1x(2^20+13)` produces two generated
reduction stages.  Its frozen client replaces the test harness's random input
with a deterministic host pattern and checks the BF16 output exactly against
an independently computed FP32 CPU softmax rounded once to BF16.

The official wheel bundles its own modern HSA runtime.  That runtime skips
legacy `HSA_TOOLS_LIB` loading after successful rocprofiler registration unless
`HSA_TOOLS_ROCPROFILER_V1_TOOLS=1` is set.  The validation runner supplies and
audits this only for instrumented PyTorch rows; it is runtime plumbing, not a
coverage or workload-tuning exception.  The doctor now proves this with a real
numeric GPU dispatch, a target check, and an exact-hook process-mapping check;
an importable wheel that silently skips ConSan no longer passes preflight.

The gfx1250 PyTorch rows demonstrate useful selection principles—dense
control flow, spill pressure, barriers, atomics, dynamic LDS, and exact
oracles—but their tensor shapes and compiled kernels are target-local evidence.
For gfx1201, start from upstream operations and tests, execute representative
native shapes, inventory the resulting code objects, and select cases only
after seeing what the RDNA4 stack actually dispatches.

| Discovery order and PyTorch source area | SuperCollider | Record/Replay | Sampled | Inline Shadow | Selection rule |
|---|---|---|---|---|---|
| **D0** `test/test_sort_and_select.py` and `test/test_reductions.py` | 🟥 Promoted; typed strict rejection exits 92 | 🟥 Promoted; 60-second fat-object analysis gate | 🟥 Promoted; shared fat-object construction gate | 🟥 Promoted; shared fat-object construction gate | A decode-style `topk` over one 151,936-element Qwen vocabulary row with `k=50` is now in the main matrix. Its exact baseline passes. The native rocPRIM path loads 3,153 kernels and exposes shared whole-code-object scalability and unsupported-site handling that the compact Inductor client does not. |
| **D0** Attention and model paths, including `test/inductor/test_fused_attention.py` and `test/nn/test_multihead_attention.py` | 🟨 Promoted: independent oracle; clean 158/158 | 🟨 Promoted: clean 158/158 + 22/22 + 2/2 + 2/2 | 🟥 Promoted: full-pressure persistent/transient SGPR placement | 🟥 Promoted: same full-pressure SGPR placement | Ordinary causal `torch.nn.functional.scaled_dot_product_attention` is now in the main matrix.  It selects a real native attention kernel and exposed the RDNA4 scalar-spilling and relay-placement work needed to make default Record/Replay clean-complete. |
| **D1** `test/test_scatter_gather_ops.py` | 🟨 Promoted: exact oracle, clean 27/27 | 🟨 Promoted: exact oracle, clean 27/27 | 🟧 Promoted: exact oracle, 14/27 | 🟧 Promoted: exact oracle, 14/27 | Collision-heavy BF16/FP32 `scatter_reduce` is now in the main matrix. Artifact `rdna4-pytorch-scatter-inventory` records 656 global-atomic sites for both mutation families; clean artifact `rdna4-pytorch-scatter-clean-001` classifies all four engines. The relaxed singleton reductions are atomicity operations, not qualified MOI ordering edges. |
| **D1** `test/test_reductions.py` histogram | 🟨 Promoted: exact oracle, clean 135/135, 1.033x | 🟧 Promoted: exact oracle, 135/135 accesses, 42/84 barriers | 🟧 Promoted: exact oracle, 122/135 accesses, 86/168 barrier members | 🟧 Promoted: exact oracle, 100/135 accesses, 43/84 barriers | Ordinary `torch.histc` is now in the main matrix. Its native object adds a real precompiled-library placement and relaxed-atomic difficulty class. SuperCollider now leaves those atomics unchanged while instrumenting every supported ordinary LDS site; the relaxed atomics remain accesses rather than qualified memory-ordering events. Sampled preserves explicit partial coverage instead of rejecting the entire object when one far barrier has no relay island. |
| **D1** `torch.compile` softmax selected from the reduction/softmax survey | 🟩 Promoted: clean 4/4; exact drop qualified miss; 0.960x | 🟩 Promoted: clean 4/4 + 3/3; exact drop detected; 1.052x | 🟧 Promoted but clean false conflict | 🟧 Promoted: static 4/4 + 3/3; dynamic workgroup-bank limit | The `128x256` exact client is in the main matrix.  Record/Replay detects the third exact barrier-pair drop even though the numeric oracle remains schedule-masked. Inline now instruments every site and exposes a distinct 32-bank external-shadow scalability limit; a different native shape remains desirable for Sampled. |
| **D2** `test/inductor/test_cooperative_reductions.py` and `test/inductor/test_online_softmax.py` | 🟨 Promoted: exact oracle, clean 8/8 across two stages | 🟨 Promoted: clean 8/8 + 6/6 across two stages | 🟧 Promoted: static-complete, but 16 clean false conflicts | 🟧 Promoted: static-complete, but 18,576 dynamic-incomplete events | Upstream split online softmax is now in the main matrix. Its deterministic client avoids the unrelated 856-kernel RNG object while retaining the exact target-native `1x(2^20+13)` split-reduction shape. |
| **D3** Broader PyTorch model/test survey | — Not admitted | — Not admitted | — Not admitted | — Not admitted | The retained generic mode, sort, top-k, and norm/softmax clients all pass their exact gfx1201 baselines. They add no stronger LLM semantic class than the admitted Qwen-vocabulary top-k and target-native softmax rows; norm/softmax Record/Replay also fails to reach a verdict within a 30-second discovery bound. Keep these as reproducer clients, not four more redundant acceptance rows. |

The gfx1250 tensor-descriptor and cluster-synchronization cases are not
presumed to have RDNA4 equivalents.  They remain examples of architecture-
specific breadth, not TODOs to recreate.  A genuinely native RDNA4 feature may
take their conceptual place; otherwise no N/A cell is needed because they were
never admitted to the gfx1201 denominator.

The final D3 selection pass executes `torch.mode`, segmented `torch.sort`, the
generic FP64/BF16 `torch.topk` pair, and `torch.linalg.vector_norm` plus
large-row `torch.softmax` through `consan_pytorch_validation.py`. All exact
baselines pass on gfx1201. The norm/softmax Record/Replay discriminator
`rdna4-norm-softmax-discovery-001` reaches its 30-second bound before emitting
an analysis verdict; pursuing that whole precompiled object would duplicate
the admitted compiled- and split-softmax semantics instead of expanding useful
coverage. Thus the broader survey is a resolved selection decision, not an
unbounded gray promise.

## Qualification order

Track each cell through the same ordered gates; do not jump directly from gray
to green after a clean launch:

1. freeze the source commit and rebuilt hook identity; run `doctor`, retain
   `manifest --json`, and prove a target-native dispatch smoke;
2. establish the independent uninstrumented oracle with no inherited
   `RJ_CONSAN_*` settings;
3. run the standard profile with strict static and dynamic completeness and no
   coverage-limiting or workload-tuning controls;
4. retain baseline-before/profile/baseline-after latency and
   instrumentation-owned peak memory;
5. regenerate every admitted target-specific fault identity, review and
   precommit detector/oracle expectations, then run exact-one contained faults;
6. require termination, report cleanup, and target health before and after;
   and
7. publish the artifact root, exact commands, settings, hashes, denominators,
   diagnostics or qualified misses, and unrounded overhead at the same tip.

The first execution wave should be P0 Qwen, then both P1 TP1 rows.  They give
the earliest direct answer to whether current ConSan still works for real
RDNA4 LLM execution.  The P4 pressure and atomic rows should follow immediately
as compact discriminators for shared spill/resource and synchronization
regressions.  Native corpus and PyTorch discovery can then expand breadth
without delaying the LLM regression verdict.

## RDNA4 regression focus

Changes since the historical certificate touched shared code as well as
gfx1250-specific paths.  Current-tip qualification should explicitly exercise:

- gfx1201 decode and emission, including reserved operands, RDNA4 indirect
  calls, `s_clause` anchor exclusion, and forward/backward relay routing;
- dead-register selection and spill-backed VGPR/SGPR state under full pressure,
  including descriptor private growth and dynamic or oversized LDS topology;
- full access, split-barrier, atomic, and fence admission with target-correct
  offsets, scopes, and wait behavior;
- dense-object planning and bounded report allocation without patch caps,
  kernel filters, manual registers, or forced-spill controls;
- HSA hook composition with waitcheck, queue/resource lifetime, timeout
  cleanup, DSO unload, and repeated-dispatch report cleanup; and
- fault rewriting against final relocated bytes, with stale selectors rejected
  rather than silently redirected.

## Historical 2026-07-16 certificate

The following table is intentionally retained as a regression comparator.  Its
green cells describe executable commit `640e575da2`, not the current branch.

| Priority workload | SuperCollider | Record/Replay | Sampled | Inline Shadow |
|---|---|---|---|---|
| **P0 Qwen3-0.6B prefill** | 🟩 20/20 clean; exact barrier drop corrupts the oracle and produces a measured mismatch diagnostic; overhead: 1.0x | 🟩 20/20 clean + 28/28 barriers; exact drop is a qualified replay miss; overhead: 1.0x | 🟩 20/20 clean + 26/26 applicable barriers; stride-256 sensitivity sweep detects 17/32 exact drops; overhead: 1.0x | 🟩 20/20 clean + 28/28 barriers; exact drop corrupts the oracle and emits an attributed diagnostic; overhead: 1.0x |
| **P1 Sharktank TP1 prefill** | 🟩 Clean 352/352; exact pair removal emits a measured-instability diagnostic while the oracle remains schedule-masked; overhead: 1.1x | 🟩 Clean 352/352 + 92/92 barriers; exact pair removal corrupts oracle, qualified replay miss; overhead: 1.6x | 🟩 Clean 352/352; exact pair removal is a schedule-masked qualified miss; four unpatched supported barriers are decode-only; overhead: 1.1x | 🟩 Clean 352/352 + 92/92; exact pair removal emits an attributed diagnostic; overhead: 3.7x |
| **P1 Sharktank TP1 decode/combined** | 🟩 Clean 704/704; exact pair removal emits a measured-instability diagnostic while the oracle remains schedule-masked; overhead: 1.0x | 🟩 Clean 704/704 + 184/184 barriers; exact pair removal corrupts the decode oracle, qualified replay miss; overhead: 1.2x | 🟩 Clean 704/704 + 36/36 applicable barriers; ordinary defaults complete repeated calls; exact pair removal is a schedule-masked qualified miss; overhead: 1.0x | 🟩 Clean and dynamically complete at 704/704 + 184/184; exact pair removal is a schedule-masked qualified diagnostic miss; overhead: 2.0x |
| **P2 Sharktank TP2 family** | 🟩 Clean 2976/2976; exact pair removal emits a measured-instability diagnostic while the oracle remains schedule-masked; overhead: 1.0x | 🟩 Clean 2976/2976 + 456/456 barriers; exact pair removal corrupts the prefill oracle, qualified replay miss; overhead: 1.0x | 🟩 Clean 2976/2976 + 36/36 applicable barriers; exact pair removal is schedule/sampling-masked; overhead: 1.0x | 🟩 Clean and dynamically complete at 2976/2976 + 456/456; exact pair removal detects 8/16 with a precommitted minimum of one; overhead: 1.3x |
| **P3 CLIP BF16** | 🟩 Clean 85/85; exact drop and subtle move both emit measured-instability diagnostics while preserving the oracle; overhead: 1.0x | 🟩 Clean 85/85 + 72/72 barriers; exact drop/move qualified misses; moved image remains complete and terminates; overhead: 1.5x | 🟩 Clean 85/85 + 70/70 applicable barriers; exact drop/move qualified misses; moved image terminates with typed exclusions; overhead: 1.2x | 🟩 Clean and dynamically complete at 85/85 + 72/72; exact drop/move qualified misses; moved image remains complete and terminates; overhead: 1.7x |
| **P4 hip-moi D128 block attention** | 🟩 Clean 8/8; exact pair removal corrupts oracle, qualified SC miss; overhead: 7.7x | 🟩 Clean 8/8 + 8/8 barriers; exact pair removal corrupts oracle, qualified replay miss; overhead: 2.1x | 🟩 Clean 8/8; no applicable sampled barrier window; exact pair removal corrupts oracle, qualified sampling miss; overhead: 2.1x | 🟩 Clean 8/8 + 8/8 barriers; exact pair removal corrupts oracle, qualified diagnostic miss; overhead: 2.1x |
| **P4 hip-moi D128 pressure attention** | 🟩 Clean 8/8; exact pair removal corrupts oracle, qualified SC miss; overhead: 16.0x | 🟩 Clean 8/8 + 8/8 barriers; exact pair removal corrupts oracle, qualified replay miss; overhead: 3.1x | 🟩 Clean 8/8; exact pair removal corrupts oracle, qualified sampling miss; overhead: 3.0x | 🟩 Clean and dynamically complete at 8/8 + 8/8; exact pair removal emits attributed diagnostics while oracle corruption is schedule-dependent; overhead: 3.1x |
| **P4 hip-moi WMMA attention** | 🟩 Clean 8/8; exact pair removal corrupts oracle, qualified SC miss; overhead: 7.0x | 🟩 Clean 8/8 + 8/8 barriers; exact pair removal corrupts oracle, qualified replay miss; overhead: 2.1x | 🟩 Clean 8/8; exact pair removal corrupts oracle, qualified sampling miss; overhead: 2.0x | 🟩 Clean and dynamically complete at 8/8 + 8/8 barriers; exact pair removal corrupts oracle, qualified diagnostic miss; overhead: 2.1x |
| **P4 hip-moi Stream-K arrival** | 🟩 Clean 4/4; exact atomic order/scope mutations are schedule-masked qualified misses; overhead: 7.9x | 🟩 Clean and dynamically complete at 4/4 + 8/8 barriers + 15/15 atomics + 16/16 fences; exact order/scope mutations are qualified misses; overhead: 2.6x | 🟩 Clean and dynamically complete at 4/4 + 15/15 atomics; exact order/scope mutations are qualified misses; overhead: 2.7x | 🟩 Clean and dynamically complete at 4/4 + 8/8 barriers + 15/15 atomics; exact order/scope mutations are qualified misses; overhead: 2.8x |
| **P4 hip-moi tree atomic-OR** | 🟩 Clean 4/4; exact atomic order/scope mutations are schedule-masked qualified misses; overhead: 9.1x | 🟩 Clean and dynamically complete at 4/4 + 8/8 barriers + 15/15 atomics + 16/16 fences; exact order/scope mutations are qualified misses; overhead: 2.7x | 🟩 Clean and dynamically complete at 4/4 + 15/15 atomics; exact order/scope mutations are qualified misses; overhead: 2.8x | 🟩 Clean and dynamically complete at 4/4 + 8/8 barriers + 15/15 atomics; exact order/scope mutations are qualified misses; overhead: 2.9x |
| **P4 Jakub attention variants** | 🟩 Clean 3/3; 31/31 accesses; exact barrier drop is a schedule-masked qualified miss; overhead: 4.6x | 🟩 Clean 3/3; 31/31 accesses + 8/8 barriers; exact barrier drop is a qualified miss; overhead: 2.2x | 🟩 Clean 3/3; 31/31 accesses; exact barrier drop is a diagnostic miss and its oracle outcome is schedule-dependent; overhead: 2.2x | 🟩 Clean and dynamically complete 3/3 at 31/31 accesses + 8/8 barriers; 15 dynamic-stack sites spill; exact barrier drop emits 31 diagnostics; overhead: 2.3x |

That campaign accepted all 55 clean baseline/profile rows, all 14 exact fault
policies, and all 66 paired-overhead rows.  Its Qwen barrier drop was the
primary end-to-end sensitivity result: SuperCollider detected 1/1,
Record/Replay retained a qualified miss, Sampled detected 17/32 in the
stride-256 sweep, and Inline Shadow detected 1/1.  These remain useful expected
behaviors to review before the new campaign, but the old machine-code selectors
must not be executed or promoted without fresh inventory.

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
ConSan diagnostic.  A flavor may be green with an honest qualified miss only
when that outcome was precommitted and the mutation, containment, independent
oracle, and coverage evidence are valid.

The historical Qwen Sampled clean result used automatic `standard-v1`
defaults: runtime stride 16,384 and offset zero, with no workload-specific
setting.  Its separate stride-256, 32-offset fault-sensitivity experiment was a
declared experiment policy, not clean-workload tuning.  A current campaign must
audit its own effective settings rather than inheriting that conclusion.

## Reproduction and evidence

Start by inspecting what is actually runnable for gfx1201:

```sh
export CONSAN_VALIDATION_TARGET=gfx1201

python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" doctor

python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" manifest --json
```

[VALIDATION.md](VALIDATION.md) defines clean, overhead, inventory, reviewed
fault, containment, and provenance execution.  [DESIGN.md](DESIGN.md),
[SPILLING.md](SPILLING.md), and [MALFORMED_INPUT.md](MALFORMED_INPUT.md)
describe the implementation and safety boundaries.

Update this document frequently during the catch-up campaign.  Change a cell
as soon as stronger evidence warrants it, record the exact artifact identity,
and never leave the tables implying that work is current merely because it was
once green on an older commit.
