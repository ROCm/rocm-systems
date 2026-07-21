# ConSan RDNA4 (`gfx1201`) status

This is the current-tip workload × instrumentation catch-up ledger for
`gfx1201`.  It deliberately does not inherit a green cell, coverage
denominator, machine-code identity, fault expectation, timing result, or
provenance record from `gfx1250`.

The last complete gfx1201 certificate was produced on 2026-07-16 at executable
commit `640e575da2`, with hook SHA-256
`c45aa0fece5a9aa7ef8b3ad24bcbb2077e477586df6b4eecf12990f7fafa693d`.  It is
retained below as historical evidence, but it does not qualify the current
branch.  The current certificate is being assembled for executable revision
`aff4853917`, using hook SHA-256
`876765e78818db9026f3670f1b48a920c9d784fda9c61b496f95dad79ff56272`.
Current-tip qualification is in progress; cells below name retained artifacts
when execution evidence has replaced the initial gray state.

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

- **Immediate requalification:** the existing portable manifest has 11
  workloads × 4 profiles = 44 current-tip cells.  Rebuild the exact hook, run
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

Thus the immediate regression campaign contains 44 cells.  The expanded
gfx1201 denominator is intentionally unknown until discovery produces concrete
native workloads with independent oracles.  Survey rows and baselines are not
counted as instrumentation cells.

The current `gfx1201 manifest --json` still exposes only the original 11
workloads.  Every expanded workload is declared with `targets=("gfx1250",)`.
The Tensile configuration paths and launcher also name `gfx1250`, and the
cluster workload uses a gfx1250-specific intrinsic.  These rows are not an
RDNA4 porting queue.  They are examples of the kinds of real, dense, atomic,
spill-heavy, and synchronization-heavy behavior for which independent gfx1201
workloads should be found.

## Current-tip requalification matrix

Every cell began gray because the intervening work changed shared decoding,
instrumentation builders, register allocation and spilling, branch-relay
planning, synchronization admission, report sizing, hook composition, runtime
cleanup, and the validation protocol.  Promotion must use one rebuilt current
tip; a successful smoke or an old artifact is insufficient.

| Priority workload | SuperCollider | Record/Replay | Sampled | Inline Shadow | First current-tip gate |
|---|---|---|---|---|---|
| **P0 Qwen3-0.6B prefill** | 🟩 Clean 20/20; 1.004x; exact barrier drop detected and oracle fails | 🟩 Clean 20/20 + 14/14 barriers; 0.998x; exact drop is a qualified diagnostic miss and oracle fails | 🟩 Clean 20/20 + 26/26 applicable barriers; 1.001x; declared 32-offset sweep detects 15/32 and every oracle fails | 🟧 Complete 20/20 + 14/14, but a false-positive diagnostic rejects clean execution, preventing paired timing and fault qualification | Clean `rdna4-aff4853917-clean-llm-013`; overhead `-overhead-007`; inventory `-inventory-008`; faults `-fault-qwen-*-011` and `-012` |
| **P1 Sharktank TP1 prefill** | 🟨 Clean 352/352; 1.238x; exact drop leaves oracle correct but expected instability diagnostic is absent | 🟨 Clean 352/352 + 46/46 barriers; 2.127x; exact drop unexpectedly diagnoses while oracle stays correct | 🟩 Clean 352/352 + 86/86 applicable barriers; 1.143x; exact drop is the precommitted qualified miss and oracle passes | 🟩 Clean 352/352 + 46/46 barriers; 2.735x; exact drop produces the precommitted diagnostic and oracle passes | Clean `rdna4-aff4853917-clean-llm-013`; overhead `-overhead-007`; inventory `-inventory-008`; fault `-fault-tp1-prefill-009` |
| **P1 Sharktank TP1 decode/combined** | 🟨 Clean 704/704; 1.180x; exact drop leaves oracle correct but expected instability diagnostic is absent | 🟨 Clean 704/704 + 92/92 barriers; 1.580x; diagnostic miss matches but expected oracle corruption is absent | 🟩 Clean 704/704 + 172/172 applicable barriers; 1.132x; exact drop is the precommitted qualified miss and both oracles pass | 🟨 Clean 704/704 + 92/92 barriers; 1.841x; exact drop diagnoses where the frozen policy expected a qualified miss | Clean `rdna4-aff4853917-clean-llm-013`; overhead `-overhead-007`; inventory `-inventory-008`; fault `-fault-tp1-decode-010` |
| **P2 Sharktank TP2 family** | 🩶 Rerun required | 🩶 Rerun required | 🩶 Rerun required | 🩶 Rerun required | Establish an untuned current baseline, then retain all-mode clean completeness and paired timing |
| **P3 CLIP BF16** | 🩶 Rerun required | 🩶 Rerun required | 🩶 Rerun required | 🩶 Rerun required | Confirm the baseline remains practical on hardware; inventory both barrier-drop and barrier-move identities anew |
| **P4 hip-moi D128 block attention** | 🟥 Clean patching crashes after `flat check/trap proof could not encode readback` | 🟧 Both exact host-reference tests pass and 12/12 accesses + 4/4 barriers patch; static analysis remains incomplete after wait-hazard preflight | 🟥 Clean patching crashes because sampled barrier sync cannot use its reserved entry island | 🟧 Both exact host-reference tests pass with complete 12/12 accesses + 4/4 barriers, but a false-positive diagnostic rejects the clean workload | `rdna4-c91323fc27-clean-001`; fix or qualify each distinct failure, then paired timing and contained barrier drop |
| **P4 hip-moi D128 pressure attention** | 🟥 Clean patching crashes | 🟧 Oracle passes and 12/12 accesses + 4/4 barriers patch; static analysis incomplete | 🟥 Clean patching crashes | 🟨 Clean accepted with complete 12/12 accesses + 4/4 barriers | `rdna4-c91323fc27-clean-p4-002`; retain spill/resource proof, then timing and fault |
| **P4 hip-moi WMMA attention** | 🟥 Clean patching crashes | 🟧 Oracle passes and 12/12 accesses + 4/4 barriers patch; static analysis incomplete | 🟥 Clean patching crashes | 🟧 Complete 12/12 accesses + 4/4 barriers, but a clean diagnostic rejects execution | `rdna4-c91323fc27-clean-p4-002`; then timing and fault |
| **P4 hip-moi Stream-K arrival** | 🟨 Clean accepted with complete 4/4 applicable accesses | 🟧 Oracle passes and 4/4 accesses + 4/4 barriers + 15/15 atomics + 16/16 fences patch; static analysis incomplete | 🟥 Clean patching crashes | 🟧 Complete 4/4 accesses + 4/4 barriers + 15/15 atomics, but a clean diagnostic rejects execution | `rdna4-c91323fc27-clean-p4-002`; then timing and atomic fault inventory |
| **P4 hip-moi tree atomic-OR** | 🟨 Clean accepted with complete 4/4 applicable accesses | 🟧 All 4/4 accesses + 4/4 barriers + 15/15 atomics + 16/16 fences patch, but static incompleteness and a clean diagnostic reject execution | 🟥 Clean patching crashes | 🟧 Complete 4/4 accesses + 4/4 barriers + 15/15 atomics, but a clean diagnostic rejects execution | `rdna4-c91323fc27-clean-p4-002`; then timing and atomic fault inventory |
| **P4 Jakub attention variants** | 🟨 Clean accepted with complete 31/31 applicable accesses | 🟧 Oracle passes and 31/31 accesses + 4/4 barriers patch; static analysis incomplete | 🟥 Clean patching crashes | 🟨 Clean accepted with complete 31/31 accesses + 4/4 barriers | `rdna4-c91323fc27-clean-p4-002`; retain dynamic-stack spill proof, then timing and fault |

## Native gfx1201 corpus discovery

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

| Discovery order and native gfx1201 source | SuperCollider | Record/Replay | Sampled | Inline Shadow | Admission decision still needed |
|---|---|---|---|---|---|
| **D0** `kernels.gfx1201.llama.cpp.llama_mul_mat_vec_q.default` | 🩶 Inventory pending | 🩶 Inventory pending | 🩶 Inventory pending | 🩶 Inventory pending | Real LLM quantized matvec with an existing validation result.  Inventory executed LDS, barriers, atomics, pressure, and dispatch count; admit only if it adds meaningful coverage beyond Qwen. |
| **D0** `kernels.gfx1201.llama.cpp.llama_rms_norm.default` | 🩶 Inventory pending | 🩶 Inventory pending | 🩶 Inventory pending | 🩶 Inventory pending | Real LLM normalization with an existing numeric path.  Measure synchronization relevance and choose a useful native shape without turning it into synthetic stress. |
| **D0** `llama.cpp` noncontiguous batched-matmul and hazard metadata | 🩶 Collection pending | 🩶 Collection pending | 🩶 Collection pending | 🩶 Collection pending | Resolve why the correct and hazardous gfx1201 variants are absent from the current collection, then determine whether the hazard is a concurrency oracle or an unrelated layout bug. |
| **D1** Three collected IREE direct-tile matmuls: F16, FP8, and I8 | 🩶 Inventory pending | 🩶 Inventory pending | 🩶 Inventory pending | 🩶 Inventory pending | Inventory final code objects and retain only distinct access/synchronization shapes with their exact output files. |
| **D1** IREE `argmax`, strided extract, and map-load/map-store cases | 🩶 Inventory pending | 🩶 Inventory pending | 🩶 Inventory pending | 🩶 Inventory pending | Prefer exact-output cases with actual LDS/barrier/atomic traffic; reject trivial global-only cases from the ConSan ledger. |
| **D2** RDNA4 WMMA/SWMMAC, wave32/wave64, atomic, and lane/DS CTS families | 🩶 Survey pending | 🩶 Survey pending | 🩶 Survey pending | 🩶 Survey pending | Use as engine-specific prerequisites or compact reproducers, not as substitutes for end-to-end workload cells. |
| **D3** Remaining 105-case corpus inventory | 🩶 Survey pending | 🩶 Survey pending | 🩶 Survey pending | 🩶 Survey pending | Cluster by executed event families, code-object shape, and engine applicability before selecting a small nonredundant set. |

Each selected case must be added to `consan_validation.py` with its exact
target-native command and independent oracle.  Its four gray cells advance
independently: evidence from one engine never promotes another.

## Native gfx1201 PyTorch discovery

PyTorch is checked out separately at
`$CONSAN_VALIDATION_WORKSPACE_DIR/pytorch`, revision `50779302a8fd`.  It is not
vendored by RocJITsu or `rocjitsu-test-corpus`, and it is not yet a required
path in the validation doctor.  The eventual validation client should use the
workspace checkout and record both its revision and the installed PyTorch/ROCm
build identity.

The gfx1250 PyTorch rows demonstrate useful selection principles—dense
control flow, spill pressure, barriers, atomics, dynamic LDS, and exact
oracles—but their tensor shapes and compiled kernels are target-local evidence.
For gfx1201, start from upstream operations and tests, execute representative
native shapes, inventory the resulting code objects, and select cases only
after seeing what the RDNA4 stack actually dispatches.

| Discovery order and PyTorch source area | SuperCollider | Record/Replay | Sampled | Inline Shadow | Selection rule |
|---|---|---|---|---|---|
| **D0** `test/test_sort_and_select.py` and `test/test_reductions.py` | 🩶 Native case pending | 🩶 Native case pending | 🩶 Native case pending | 🩶 Native case pending | Seek gfx1201 `topk`, sort, or mode kernels with dense barriers, large code objects, or spill pressure.  Choose shapes from observed native structure, require exact values and indices, and do not copy gfx1250 shapes. |
| **D0** Attention and model paths, including `test/inductor/test_fused_attention.py` and `test/nn/test_multihead_attention.py` | 🩶 Native case pending | 🩶 Native case pending | 🩶 Native case pending | 🩶 Native case pending | Prefer a real attention or small-model path with an independent PyTorch reference that complements Qwen/TP1 over isolated synthetic Triton code. |
| **D1** `test/test_scatter_gather_ops.py` | 🩶 Atomic inventory pending | 🩶 Atomic inventory pending | 🩶 Atomic inventory pending | 🩶 Atomic inventory pending | Select a collision-heavy reduction only if gfx1201 inventory proves a meaningful atomic synchronization role; retain exact collision results. |
| **D1** `test/test_reductions.py` and shared-memory softmax tests in `test/test_nn.py` | 🩶 Native case pending | 🩶 Native case pending | 🩶 Native case pending | 🩶 Native case pending | Sweep ordinary histogram, reduction, normalization, and softmax shapes only until distinct native kernel families appear; freeze the smallest representative exact client. |
| **D2** `test/inductor/test_cooperative_reductions.py` and `test/inductor/test_online_softmax.py` | 🩶 Native case pending | 🩶 Native case pending | 🩶 Native case pending | 🩶 Native case pending | Admit only target-native generated kernels that terminate reliably and add dynamic/shared-memory or multi-stage coverage absent from D0/D1. |
| **D3** Broader PyTorch model/test survey | 🩶 Survey pending | 🩶 Survey pending | 🩶 Survey pending | 🩶 Survey pending | Inventory first, then cluster by executed event, resource shape, and engine applicability to prevent a large redundant matrix. |

The gfx1250 tensor-descriptor and cluster-synchronization cases are not
presumed to have RDNA4 equivalents.  They remain examples of architecture-
specific breadth, not TODOs to recreate.  A genuinely native RDNA4 feature may
take their conceptual place; otherwise no N/A cell is needed because they were
never admitted to the gfx1201 denominator.

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
