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
certificate has advanced through validation checkpoint `971a5a347e`, using
rebuilt hook SHA-256
`897e52a34a0abc107afbda6eb9a9a83d8d5a06cc6f9f2c46247faa684d3da10a`.
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

🟦 **resolved discovery exclusion** is deliberately outside the ordered
maturity scale.  It means target-native evidence established that the candidate
is not an acceptance workload (for example, it is compile-only, executes no
supported synchronization, or duplicates a stronger retained workload).  It
must not be read as an unassessed gray cell or counted in the validation
denominator.

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
| **P0 Qwen3-0.6B prefill** | 🟩 Exact oracle; clean 20/20; exact barrier drop emits an attributed diagnostic and breaks the oracle; overhead 1.044x | 🟩 Exact oracle; clean 20/20 + 14/14; exact drop is a precommitted qualified miss and breaks the oracle; overhead 0.988x | 🟩 Exact oracle; clean 20/20 + 26/26; deterministic 32-offset fault sweep detects 17/32 and every exact mutation breaks the oracle; overhead 1.003x | 🟩 Exact oracle; clean 20/20 + 14/14; exact barrier drop emits a diagnostic and breaks the oracle; overhead 1.005x | Inline checkpoint `913fc2e7a6`, hook `71cc2454…`: generation-qualified workgroup-local LDS cells replace the unreliable eager-clear assumption. Frozen artifacts `rdna4-qwen-inline-generation-frozen-clean`, `...-overhead`, and `...-fault` are accepted at one clean commit; an additional ten consecutive clean processes all pass without diagnostics. The paired dispatch medians are 25.790 ms instrumented versus 25.658 ms baseline (1.005128x). Current-tip SuperCollider regression artifact `rdna4-qwen-sc-after-fat-fastpath`, hook `403c35a3…`, reconfirms exact 20/20 clean coverage in 0.532 seconds after the fat-object optimization. Other cells retain checkpoint `9c5ff65778`: clean `...-105`, paired overhead `...-098` through `...-100`, inventory `...-101`, and accepted faults `...-102` through `...-104`; the Sampled offset sweep is fault-sensitivity evidence, not clean-workload tuning |
| **P1 Sharktank TP1 prefill** | 🟩 Exact oracle; clean 352/352; exact drop is a precommitted qualified miss; overhead 1.26x | 🟩 Exact oracle; clean 352/352 + 46/46; exact drop emits a replay diagnostic; overhead 2.09x | 🟩 Exact oracle; clean 352/352 + 86/86; exact drop is a precommitted qualified miss; overhead 1.14x | 🟩 Exact oracle; clean 352/352 + 46/46; exact drop emits an attributed diagnostic; overhead 2.75x | One-tip checkpoint `178d16dcee`: clean `...-040`, overhead `...-035`, inventory `...-036`, accepted fault rows `...-037` through `...-039`; initial historical expectations were retained as failed discovery evidence |
| **P1 Sharktank TP1 decode/combined** | 🟩 Exact oracle; clean 704/704; exact drop is a precommitted qualified miss; overhead 1.18x | 🟩 Exact oracle; clean 704/704 accesses + 92/92 barriers; exact drop emits attributed replay diagnostics in 2/8 contained trials while the oracle is schedule-masked; overhead 1.537x combined / 1.445x decode | 🟩 Exact oracle; clean 704/704 + 172/172; exact drop is a precommitted qualified miss; overhead 1.01x | 🟩 Exact oracle; clean 704/704 + 92/92; exact drop emits an attributed diagnostic; overhead 1.70x | Record/Replay checkpoint `1f29d401de`, hook `71cc2454…`: clean `rdna4-tp1-decode-rr-final-clean-env`, paired overhead `...-overhead`, inventory `...-inventory`, and accepted eight-trial statistical fault `...-fault-statistical`. The first discovery fault produced a replay diagnostic, the next was schedule-masked, and the frozen statistical gate independently reproduced 2/8 detections without oracle corruption or device-health loss |
| **P1 PyTorch collision-heavy scatter-reduce** | 🟩 Exact BF16/FP32 collision oracle; clean-complete 27/27 LDS accesses; exact BF16 global-atomic scope weakening is a precommitted qualified miss; overhead 1.010x | 🟩 Exact oracle; clean-complete 27/27 accesses; exact BF16 global-atomic scope weakening is a precommitted qualified miss; overhead 0.998x | 🟩 Exact oracle; clean-complete 27/27 accesses; exact BF16 global-atomic scope weakening is a precommitted qualified miss; overhead 0.993x | 🟩 Exact oracle; clean-complete 27/27 accesses; exact BF16 global-atomic scope weakening is a precommitted qualified miss; overhead 1.031x | Inline hook `bedb65a3…`: clean `rdna4-scatter-inline-dense-relay-final-clean`, paired overhead `...-overhead`, inventory `...-inventory`, and accepted fault `...-fault`; all 720 ConSan unit tests pass. The conservative relay-sizing follow-up hook `2f91fcf1…` reconfirms complete scatter and Qwen Inline execution in `rdna4-{scatter,qwen}-inline-sized20-regression`. The later composition-envelope regression `rdna4-scatter-inline-qualified135-regression` reconfirms clean-complete 27/27 execution. Sampled hook `4606c057…` uses the corresponding `rdna4-scatter-sampled-dense-relay-*` artifacts. Both engines now share an RDNA4 relay with an explicit per-site key, recovering the 13 adjacent sub-byte helper-fill sites that formerly collided with independent 28-byte entry islands; no user tuning is used. The selected `global_atomic_pk_add_bf16` scope mutation applies exactly once and leaves the exact collision oracle unchanged, so it remains a qualified miss. SuperCollider checkpoint `8a8ebaf3c5`, hook `2dec0c65…`; Record/Replay checkpoint `99061989a8`, hook `71cc2454…`. The executed reductions use relaxed singleton global atomics, so MOI correctly reports `atomic=0/0`; the LDS sites belong to a bundled boolean-fill helper |
| **P2 PyTorch/Inductor compiled softmax** | 🟩 Exact oracle; clean 4/4 LDS accesses; exact third barrier drop is a precommitted qualified miss; overhead 0.960x | 🟩 Exact oracle; clean 4/4 + 3/3; exact third barrier drop emits an attributed replay diagnostic; overhead 1.052x | 🟨 Exact oracle; clean-complete 4/4 accesses + 6/6 barrier members; 24 visible synchronization records and no conflict; overhead and reviewed fault pending | 🟧 Exact oracle and static 4/4 accesses + 3/3 barriers now patch, but the 128-workgroup execution exceeds the 32-bank external-shadow partition and records 1,536 dynamic-incomplete events | Sampled checkpoint `c5d9c59672`, hook `87db680e…`, artifact `rdna4-compiled-softmax-sampled-address-aligned-clean-20260722`: runtime sampling selects LDS addresses, so qualified barriers no longer apply an unrelated workgroup-residue gate. Every reaching workgroup now advances its epoch and inspects its banked causal window; the former six clean conflicts and zero synchronization records become zero conflicts and 24 records without a user knob. Inline checkpoint `d47f1f85b6`: clean `...-126` fixes opaque dynamic-LDS report sizing and isolates the workgroup-bank limit. Other profiles retain checkpoint `b6be4081e1`: clean `...-110`, paired overhead `...-111` and `...-112`, inventory `...-107`, and accepted faults `...-108` and `...-109` |
| **P2 PyTorch split online softmax** | 🟩 Exact CPU-derived BF16 oracle; clean-complete 8/8 LDS accesses across two generated stages; exact third barrier drop is a precommitted qualified miss; overhead 0.977x | 🟩 Exact oracle; clean-complete 8/8 accesses + 6/6 barriers; exact third barrier drop is a qualified replay miss; overhead 1.037x | 🟨 Exact oracle; clean-complete 8/8 accesses + 12/12 barrier members across two generated stages; 25 visible synchronization records and no conflict; overhead and reviewed fault pending | 🟧 Exact oracle and static-complete 8/8 accesses + 6/6 barriers, but the two-stage execution records 18,576 dynamic-incomplete events at the 32-bank external-shadow limit | Sampled checkpoint `c5d9c59672`, hook `87db680e…`, artifact `rdna4-split-softmax-sampled-address-aligned-clean-20260722`: the shared address/workgroup sampling fix eliminates all former clean conflicts and publishes 22 + 3 synchronization records across the two instrumented stage objects, without a user knob. Other profiles retain checkpoint `a199956578`, hook `2dec0c65…`: clean artifacts `rdna4-split-softmax-{sc,rr}-final-clean`, paired overhead `...-final-overhead`, inventory `rdna4-split-softmax-final-inventory`, accepted SuperCollider fault `...-sc-final-fault`, and accepted Record/Replay miss `...-rr-final-fault-accepted`. Both exact pair removals leave the oracle unchanged; the initial rejected replay expectation is retained in `...-rr-final-fault`. The deterministic client freezes upstream `test_online_softmax.py::test_split_reduction` shape `1x(2^20+13)` without loading the unrelated 856-kernel RNG object used by the test harness |
| **P2 PyTorch Qwen-vocabulary top-k** | 🟥 Exact values and indices pass and clean analysis reaches a typed verdict in 54.00 seconds, but only 2,039/63,474 supported accesses patch | 🟥 Both semantic passes now complete and a 109,229,808-byte automatic report is allocated without tuning, but all-site lowering still exceeds a clean 75-second discriminator before a patch verdict | 🟥 Semantic inventory completes and a 61,466,864-byte automatic report is allocated without tuning, but all-site lowering exceeds 45 seconds | 🟥 Semantic inventory completes and a 10,978,736-byte automatic report is allocated without tuning, but all-site lowering exceeds 45 seconds | SuperCollider checkpoint `512c09ea0a`, artifact `rdna4-topk-sc-threshold-regression-65`, hook `2dec0c65…`, reconfirms exactly 941/42,652 and 1,098/20,822 patched accesses after the moderate-object relay fix. The two 23 MiB and 40 MiB objects retain preflight-candidate-only CFG construction, but 61,435 sites still lack reachable relay placement, far below usable coverage. Record/Replay checkpoint `3c599609cf`, hook `78bb43d9…`: offset-indexing removes the quadratic ordinary acquire/release fence scan, reducing the formerly unbounded acquire association to about 1.9 seconds and the complete semantic inventory to about 19 seconds. Automatic sizing retains the largest power-of-two dynamic headroom that fits the ordinary 128 MiB ceiling; this object gets 256 records per static barrier and a 109,229,808-byte report without a user knob. Clean artifact `rdna4-topk-rr-indexed-adaptive-clean-20260722` reaches lowering before its explicit 75-second harness cap; the identical implementation also remained in lowering at a bounded 120 seconds. Checkpoint `ff1afe848c` replaces per-site candidate and kernel-context scans with indexed lookup and caches per-kernel private layouts. A bounded stage diagnostic now reaches candidate inventory in 0.004 seconds, kernel CFG/liveness construction in 4.456 seconds, initial resource plans in 4.818 seconds, and the rebuilt persistent-VGPR plans in 5.514 seconds. The remaining 60-second boundary occurs in repeated whole-object semantic reconstruction before the second lowering pass, not in the now-bounded resource-plan lookup. Clean artifacts `rdna4-topk-{sampled,inline}-indexed-discriminator-20260722` independently show that both online engines complete semantic inventory and automatic allocation before their 45-second caps. The compact Qwen Record/Replay regression remains complete at 20/20 accesses and 14/14 barriers at committed checkpoint `ff1afe848c` in `rdna4-planner-committed-qwen-rr-20260722-01`; 723/723 ConSan and 20/20 ConSan hook tests pass |
| **P2 PyTorch causal SDPA** | 🟩 Independent CPU oracle; clean 158/158; exact barrier drop emits an attributed diagnostic and breaks the oracle; overhead 1.946x | 🟨 Independent CPU oracle; clean-complete 158/158 accesses + 22/22 barriers + 2/2 atomics + 2/2 fences; overhead 7.795x; two reviewed exact drops break the oracle through unattributed hardware traps but emit no replay diagnostic | 🟥 Full-pressure attention kernel has no persistent dispatch-ID SGPR pair or fresh automatic EXEC-save window; only 14/27 accesses in a separate bundled fill kernel patch | 🟥 Same full-pressure SGPR blockers; strict fail-closed exit after only the separate fill kernel patches 14/27 accesses | Committed checkpoint `3868bdf4ee`, hook `7e3cf6c6…`, artifact `rdna4-sdpa-sampled-ff1-recheck-20260722` reproduces the Sampled blockers in 10.01 seconds after an exact baseline passes in 5.43 seconds. The attention kernel declares 107 SGPRs and 13 compiler SGPR spills; it needs both a persistent dispatch-ID pair and a seven-register transient Sampled window. Inline needs a 28-register transient window and the same persistent identity for external-shadow/report attribution. This is genuine scalar-state spilling and continuation work, not stale setup or a removable sampling requirement: dispatch identity prevents causal-window aliasing across concurrent or repeated dispatches. The prior Inline artifact `rdna4-sdpa-inline-current-red-recheck` independently records its matching fail-closed outcome. A bounded spill prototype statically covered all 131 attention accesses and ran through 62 accumulated sites, but adding the 63rd produced a GPU aperture fault even though that site passed alone; the prototype was reverted, separating the placement work from a subsequent Sampled composition/record-index bug. Earlier checkpoint `8a5669d5e5` retains paired SC/RR overhead, inventory, and accepted SC fault artifacts. Rejected RR fault rows retain two distinct unattributed-trap outcomes. Record/Replay remains dynamically complete after automatic report headroom was raised; no user tuning knobs are used |
| **P2 Sharktank TP2 family** | 🟩 Exact oracle; clean 2976/2976; exact drop is a precommitted qualified miss; overhead 1.28x | 🟩 Exact oracle; clean 2976/2976 + 228/228; exact drop emits a replay diagnostic; overhead 1.83x | 🟩 Exact oracle; clean 2976/2976 + 420/420; exact drop is a precommitted qualified miss; overhead 1.24x | 🟩 Exact oracle; clean 2976/2976 + 228/228; exact drop detected 16/16; overhead 2.17x | One-tip checkpoint `08b15c6d91`: clean `...-048`, overhead `...-049`, inventory `...-050`, accepted fault rows `...-053` through `...-056`; historical expectations remain as failed discovery rows `...-051` and `...-052` |
| **P3 CLIP BF16** | 🟩 Exact oracle; clean 85/85; exact drop and move are precommitted qualified misses; overhead 0.98x | 🟩 Exact oracle; clean 85/85 + 36/36; exact drop and subtle move are precommitted qualified misses; overhead 1.236x | 🟩 Exact oracle; clean 85/85 + 72/72; exact drop and move are precommitted qualified misses; overhead 0.97x | 🟩 Exact oracle; clean 85/85 + 36/36; exact move emits an attributed diagnostic while exact drop is a qualified miss; overhead 1.51x | Record/Replay checkpoint `38f77d6601`, hook `bb08fa7c…`: clean `rdna4-clip-rr-scalar-007`, paired overhead `...-overhead-001`, inventory `...-inventory-001`, and accepted faults `...-fault-drop-001` and `...-fault-move-001`. gfx1201 now keeps owner/epoch in persistent SGPRs instead of loading a private-memory epoch at every hot access. Other profiles remain at checkpoint `3033f30f2a` in `...-057` through `...-070` |
| **P3 PyTorch native histogram** | 🟩 Exact oracle; clean-complete 135/135 ordinary LDS accesses; exact post-accumulation barrier drop is a precommitted qualified miss; overhead 0.966x | 🟧 Exact oracle and complete 135/135 ordinary LDS accesses; 42/84 supported barriers patch, with the rest lacking reachable entry islands | 🟨 Exact oracle; clean-complete 135/135 accesses + 168/168 barrier members; overhead 1.094x; reviewed effective fault pending | 🟨 Exact oracle; clean-complete 135/135 accesses + 84/84 barriers; overhead 5.301x; reviewed effective fault pending | Current checkpoint `971a5a347e`, hook `897e52a3…`: the dense barrier routers use the gfx1201 `s_call_b64` equivalent of gfx1250's `s_call_i64`. Sampled artifact `rdna4-histc-sampled-at-inline-tip-clean-20260722` is exact-complete at 135/135 accesses + 168/168 barrier members; paired artifact `rdna4-histc-sampled-at-inline-tip-overhead-20260722` measures 1.094381x. Inline artifact `rdna4-histc-inline-dense-barrier-committed-20260722` is exact-complete at 135/135 accesses + 84/84 barriers; paired artifact `rdna4-histc-inline-dense-barrier-overhead-20260722` measures 5.300524x. Synthetic far-pair regressions cover both RDNA4 routes; all 725 ConSan and 20 ConSan hook tests pass. Record/Replay remains separate: its five-register RDNA4 scalar layout cannot safely host the router's eight-register call envelope. The gfx1201 reference spec has no histogram fault contract, so a freshly reviewed effective mutation remains the yellow→green gate for both online engines. SuperCollider hook `2dec0c65…`: clean `rdna4-histc-sc-final-clean`, paired overhead `...-final-overhead`, fresh inventory `rdna4-histc-final-inventory` (84 barrier sequences, 104 atomic sites), and accepted fault `...-final-fault-accepted`. The exact float32 post-accumulation barrier-pair removal applies once but is schedule-masked. Relaxed accumulations are not qualified MOI ordering events and correctly remain `atomic=0/0` |
| **P4 hip-moi D128 block attention** | 🟩 Exact oracle; clean 12/12; exact barrier drop is a precommitted qualified miss and breaks the oracle; overhead 164.77x | 🟩 Exact oracle; clean 12/12 + 4/4; exact barrier drop emits an attributed diagnostic and breaks the oracle; overhead 11.21x | 🟩 Exact oracle; clean-complete 12/12 accesses + 8/8 barrier members; exact barrier drop is a precommitted qualified miss and breaks the oracle; overhead 14.03x | 🟩 Exact oracle; clean 12/12 accesses + 4/4 barriers; exact barrier drop emits a diagnostic and breaks the oracle; overhead 12.83x | Current hook `2dec0c65…`: Sampled clean `rdna4-d128-block-sampled-final-clean`, paired overhead `...-final-overhead`, shared fresh inventory `rdna4-d128-block-inline-final-inventory`, and accepted fault `...-sampled-final-fault`. Inline uses the same inventory plus paired overhead `...-inline-final-overhead` and accepted fault `...-inline-final-fault-accepted`; rejected historical expectation `...-inline-final-fault` preserves the improved detector outcome after generation-qualified local shadows removed the former clean diagnostic. SuperCollider checkpoint `ca1eb5456e`; Record/Replay checkpoint `baed32a85e` |
| **P4 hip-moi D128 pressure attention** | 🟩 Exact oracle; clean-complete 12/12 accesses; exact barrier drop is a precommitted qualified miss and breaks the oracle; overhead 11.25x | 🟩 Exact oracle; clean 12/12 + 4/4; exact barrier drop emits an attributed diagnostic and breaks the oracle; overhead 14.30x | 🟧 Exact oracle and complete 12/12 accesses + 8/8 barrier members, but four sampled conflicts reject the correct workload | 🟩 Exact oracle; clean 12/12 + 4/4; exact barrier drop emits an attributed diagnostic; overhead 13.72x | Current hook `2dec0c65…`: SuperCollider clean `rdna4-d128-pressure-sc-final-clean`, paired overhead `...-final-overhead`, fresh inventory `...-final-inventory`, and accepted contained fault `...-final-fault`. Caching immutable owner register bounds removes the former quadratic shared-function scan. Sampled checkpoint `e96efba818`: clean `...-141`. Record/Replay checkpoint `baed32a85e`; Inline checkpoint `79aea7420c` |
| **P4 hip-moi WMMA attention** | 🟩 Exact oracle; clean 12/12; exact barrier drop is a precommitted qualified miss and breaks the oracle; overhead 158.07x | 🟩 Exact oracle; clean 12/12 + 4/4; exact barrier drop emits an attributed diagnostic and breaks the oracle; overhead 11.99x | 🟩 Exact oracle; clean-complete 12/12 accesses + 8/8 barrier members; exact barrier drop is a precommitted qualified miss and breaks the oracle; overhead 14.50x | 🟩 Exact oracle; clean 12/12 accesses + 4/4 barriers in five consecutive processes; exact barrier drop emits a diagnostic and breaks the oracle; overhead 13.24x | Current hook `2dec0c65…`: Sampled clean `rdna4-wmma-sampled-final-clean`, paired overhead `...-final-overhead`, shared fresh inventory `rdna4-wmma-inline-final-inventory`, and accepted fault `...-sampled-final-fault`. Inline uses the same inventory plus five clean processes `rdna4-generation-wmma-repeat-{1..5}`, paired overhead `...-inline-final-overhead`, and accepted fault `...-inline-final-fault-accepted`; rejected historical expectation `...-inline-final-fault` records the improved detector outcome. SuperCollider checkpoint `ca1eb5456e`; Record/Replay checkpoint `baed32a85e` |
| **P4 hip-moi Stream-K arrival** | 🟩 Exact oracle; clean 4/4 accesses; exact order and scope weakenings are precommitted qualified misses; overhead 558.83x | 🟩 Exact oracle; clean 4/4 accesses + 15/15 atomics + 4/4 barriers + 16/16 fences; exact order and scope weakenings are precommitted qualified misses; overhead 53.00x | 🟩 Exact oracle; clean-complete 4/4 accesses + 15/15 atomics + 8/8 barrier members; exact order and scope weakenings are precommitted qualified misses; overhead 48.21x | 🟩 Exact oracle; clean-complete 4/4 accesses + 15/15 atomics + 4/4 barriers; exact order and scope weakenings each emit a diagnostic while preserving the oracle; overhead 50.93x | Current Inline checkpoint `137bcfefe5`, hook `62a17bf5…`: clean `rdna4-inline-durable-token-final3-clean`, paired overhead `...-final3-overhead`, inventory `...-final3-inventory`, and accepted detected faults `...-final3-fault-atomic-weaken-{order,scope}`. The initial fault contract expected a miss and is retained as rejected evidence that the detector outcome was not silently fitted after observation. Five additional clean processes pass. Sampled artifacts remain `rdna4-streamk-sampled-final-*`; Record/Replay checkpoint `baed32a85e`; SuperCollider checkpoint `79aea7420c` |
| **P4 hip-moi tree atomic-OR** | 🟩 Exact oracle; clean 4/4 accesses; exact order and scope weakenings are precommitted qualified misses; overhead 591.81x | 🟨 Exact oracle and clean-complete execution with all 4/4 accesses + 15/15 atomics + 4/4 barriers and 12/16 fences; only four adjacent fences exhaust reachable entry islands | 🟩 Exact oracle; clean-complete 4/4 accesses + 15/15 atomics + 8/8 barrier members; exact order and scope weakenings are precommitted qualified misses; overhead 49.91x | 🟧 Exact oracle and complete 4/4 accesses + 15/15 atomics + 4/4 barriers, but repeated-process qualification still intermittently diagnoses a correct owner-4 read after owner 2 | Current Inline checkpoint `137bcfefe5`, hook `62a17bf5…`: ten isolated clean processes pass, but paired artifact `rdna4-inline-durable-token-final3-overhead` reproduces the diagnostic in 2/3 instrumented processes, so no overhead is claimed. Committed inherited tokens now remain authoritative without re-reading a mutable source slot, deleting 354 hot-path lines and shrinking these access probes from about 3,960 to 2,480 bytes; a second access-time token-visibility gap remains. Current-tip Record/Replay artifact `rdna4-tree-rr-current-recheck`, hook `c9f19b1b…`, retains the isolated four-fence placement gap. Sampled artifacts remain `rdna4-tree-sampled-final-*`; other retained evidence is at checkpoint `79aea7420c` |
| **P4 hip-moi Jakub attention variants** | 🟩 Exact oracle; clean 31/31 accesses; exact barrier drop is a precommitted qualified miss; overhead 103.75x | 🟩 Exact oracle; clean 31/31 + 4/4; exact barrier drop is a precommitted qualified miss; overhead 11.69x | 🟩 Exact oracle; clean-complete 31/31 accesses + 8/8 barrier members; exact barrier drop is a precommitted qualified miss with the oracle schedule-masked; overhead 10.87x | 🟩 Exact oracle; clean 31/31 + 4/4; exact barrier drop emits an attributed diagnostic; overhead 11.03x | Current hook `2dec0c65…`: Sampled clean `rdna4-jakub-sampled-final-clean`, paired overhead `...-final-overhead`, fresh inventory `...-final-inventory`, and accepted contained fault `...-final-fault`. Record/Replay checkpoint `baed32a85e`; SuperCollider and Inline checkpoint `79aea7420c` |

## Native gfx1201 corpus discovery and qualification (zero gray cells)

The local `rocjitsu-test-corpus` checkout at revision `0fccdd2c58d8` is already
organized around target-native discovery.  A collection-only survey for
`gfx1201` finds 105 runnable cases: 7 IREE cases, 3 kernel cases, and 95 CTS
cases.  This is a discovery pool, not 105 ConSan acceptance rows.  Most cases
will have no relevant executed synchronization or will duplicate a stronger
workload.

**Selection state: zero gray cells.** Two llama.cpp candidates are admitted and
colored independently per engine below; six candidate families are resolved
blue exclusions.  Red, orange, and yellow cells are real qualification work,
not undiscovered corpus scope.

| Native selection outcome | Count |
|---|---:|
| Admitted four-engine workload rows | 2 |
| Resolved target-native exclusions | 6 |
| Unassessed gray rows or cells | **0** |

The gfx1250 Tensile directory is explicitly marked unsupported on gfx1201.
Do not copy, retarget, or author lookalike versions of those configurations
merely to make the ledgers symmetric.  Search the native IREE, kernel, and CTS
suites for workloads that exercise the same broad difficulty classes.

A discovery row uses `🟦 Excluded` after target-native evidence rules it out of
the validation denominator.  Blue is a resolved selection decision, not a
support result and not a gray instrumentation cell.  Once a candidate has an
exact baseline and current-hook execution evidence, its four cells use the
ordinary ordered color scale independently.  Colored discovery cells are
provisional qualification until the command is promoted into
`consan_validation.py` and frozen with overhead and fault evidence.

| Discovery order and native gfx1201 source | SuperCollider | Record/Replay | Sampled | Inline Shadow | Admission decision and evidence |
|---|---|---|---|---|---|
| **D0** `kernels.gfx1201.llama.cpp.llama_mul_mat_vec_q.default` | 🟨 Exact CPU oracle; clean-complete 462/462 LDS accesses; overhead and reviewed fault pending | 🟧 Exact 0.01 CPU oracle and useful partial execution: 462/462 accesses, 8/44 barriers, 63/63 supported atomics, and 63/63 associated fences patch without a clean diagnostic | 🟧 Exact 0.01 CPU oracle and useful partial execution: 241/462 accesses + 10/88 barrier members patch, but two sampled conflicts reject the clean run | 🟧 Exact oracle and useful partial execution: 132/462 accesses + 6/44 barriers patch, but one diagnostic rejects the clean run | Current Record/Replay artifact `rdna4-matvec-rr-b64-relay10`, hook `f32d0529…`, extends exact-address reconstruction to 64-bit global/flat atomics and reserves enough RDNA4 dense-dispatch relay space; it exits zero and differs from the CPU result by at most 0.00403. Current Sampled artifact `rdna4-matvec-sampled-no-access-atomics`, hook `95f96eae…`, no longer rejects 63 standalone printf-helper atomics that have no LDS causal-window consumer; it reaches the real matvec object and passes the oracle with the same maximum error, but its two clean conflicts and placement gaps prevent yellow. Earlier artifacts `rdna4-matvec-rr-current` and `rdna4-matvec-sampled-current`, hook `8c64e5ee…`, retain the pre-fix strict rejections. Exact baseline artifact `.pytest-artifacts-rdna4-llama-baseline`; SuperCollider passes the same tolerance. Inline artifact `rdna4-matvec-inline-dense-budget` passes the same oracle while retaining one clean diagnostic. The automatic 135-candidate RDNA4 dense-routing qualification envelope prevents the newly recovered shared-relay path from composing an unqualified 462-site relocation layout; owner-filtered sweeps show that individual 42-site owners are sound, while the combined layout corrupts output. This is a conservative internal admission rule, not a user knob. |
| **D0** `kernels.gfx1201.llama.cpp.llama_rms_norm.default` | 🟨 Exact CPU oracle; clean-complete 22/22 LDS accesses; no mismatch; 1.306x process overhead; reviewed effective fault pending | 🟨 Exact oracle; clean-complete 22/22 accesses + 11/11 barriers; visible replay evidence and no conflict; 1.308x overhead; reviewed effective fault pending | 🟨 Exact oracle; clean-complete 22/22 accesses + 22/22 barrier members; sampled evidence and no clean diagnostic; 1.307x overhead; reviewed effective fault pending | 🟨 Exact 128-element CPU oracle; static- and dynamic-complete 22/22 accesses + 11/11 barriers; 80 visible events and no malformed or incomplete snapshots; overhead and reviewed effective fault pending | Inline checkpoint `6ddab1a730`, hook `ae96b66a…`, artifact `rdna4-rms-inline-retry-sgpr-fix-full`; exact baseline artifact `.pytest-artifacts-rdna4-llama-baseline`. The versioned exact-shadow retry counter had aliased the saved low half of guest VCC, restoring the decremented `0x7ff` counter as the application's lane mask. Moving the counter from EXEC-base `+8` to the diagnostic-temporary `+20` slot fixes the exact clean workload without a user knob. The earlier `45f8b118f3` dispatch-ID-zero fix remains required. Other profiles retain five-process medians of 0.163896 s baseline, 0.213989 s SuperCollider, 0.214359 s Record/Replay, and 0.214191 s Sampled. |
| **D0** `kernels.gfx1201.hip-matmul.hip_matmul_matvec.m256_n1_k1024` | 🟦 Excluded | 🟦 Excluded | 🟦 Excluded | 🟦 Excluded | A standalone build with the unrelated llama.cpp backend disabled passes all nine exact matvec variants. Fresh SuperCollider inventory finds no decoded LDS, barriers, or atomics in the nine workload kernels; only eight ambiguous flat maybe-group sites fail placement, so this does not add a sound or nonredundant ConSan workload. |
| **D0** `llama.cpp` noncontiguous batched-matmul and hazard metadata | 🟦 Excluded | 🟦 Excluded | 🟦 Excluded | 🟦 Excluded | The gfx1201 config explicitly skips compiling both variants. The hazard overlay restores llama.cpp PR #13155's pre-fix noncontiguous-stride conversion and expects a deterministic output-validation failure; it is a data-layout correctness reproducer, not a concurrency oracle. |
| **D1** Three collected IREE direct-tile matmuls: F16, FP8, and I8 | 🟦 Excluded | 🟦 Excluded | 🟦 Excluded | 🟦 Excluded | The three gfx1201 compilations pass, but their corpus records explicitly set `compile_only=true`; they dispatch no workload and provide no runtime oracle. Normalize a calls/support-module wrapper before reconsidering them. |
| **D1** IREE `argmax`, strided extract, and map-load/map-store cases | 🟦 Excluded | 🟦 Excluded | 🟦 Excluded | 🟦 Excluded | All four exact baselines pass. Record/Replay inventory in `.pytest-artifacts-rdna4-iree-inventory` reports zero supported accesses, barriers, atomics, and fences for every executed code object, so these global-only shapes add no ConSan coverage. |
| **D2** RDNA4 WMMA/SWMMAC, wave32/wave64, atomic, and lane/DS CTS families | 🟦 Excluded | 🟦 Excluded | 🟦 Excluded | 🟦 Excluded | Seven exact baseline cases pass, but the arithmetic and lane-operation cases add no admitted ConSan synchronization traffic. The bundled atomic case is useful as an unsupported-transform stress object, not as a compact or nonredundant acceptance workload. Retain this family as an engine-specific reproducer pool. |
| **D3** Remaining corpus families | 🟦 Excluded | 🟦 Excluded | 🟦 Excluded | 🟦 Excluded | Collection and representative execution now account for all 105 gfx1201 cases. Remaining FPSan host-only/arithmetic cases either dispatch no GPU work or reproduce the already-retained 2,560-atomic support object; remaining integer-ISA arithmetic variants add no distinct ordering family, and the reduction representative currently crashes this SDK's compiler. Keep these as compiler/engine reproducer pools rather than inflating the acceptance denominator. |

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

The llama.cpp runner's `--validate` switch selects its CPU backend; it does not
run a GPU workload and then validate it.  Reproducible qualification therefore
uses two processes: an instrumented GPU run without `--validate`, and an
uninstrumented CPU run with `--validate`, each writing its output for an exact
comparison.  Running only the latter silently exercises no HSA code and cannot
qualify a ConSan cell.

The first RMS fault discriminator exactly dropped the wait-side singleton at
`pc=0x1830` in the executed 256-thread, non-fused kernel.  Current-hook artifact
`rdna4-rms-native-inventory-ae96` additionally drops the complete signal/wait
pair with exact-one enforcement at both the 128- and 1,024-element shapes.  All
three mutations apply, but the outputs remain bitwise equal to their independent
CPU results and SuperCollider reports no mismatch.  Keep these as reviewed
candidate misses, not detection acceptance; select a mutation with an
observable semantic effect before promoting any RMS cell to green.

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
| **D0** `test/test_sort_and_select.py` and `test/test_reductions.py` | 🟥 Promoted; typed 2,039/63,474 verdict in 54.12 seconds | 🟥 Promoted; semantic inventory and automatic allocation complete, all-site lowering exceeds 75 seconds | 🟥 Promoted; semantic inventory and automatic allocation complete, all-site lowering exceeds 45 seconds | 🟥 Promoted; semantic inventory and automatic allocation complete, all-site lowering exceeds 45 seconds | A decode-style `topk` over one 151,936-element Qwen vocabulary row with `k=50` is now in the main matrix. Its exact baseline passes. The native rocPRIM path loads two large operator objects and exposes both whole-code-object scalability and a dominant far-relay placement gap that the compact Inductor client does not. Checkpoint `3c599609cf` removes the untyped semantic-analysis and report-capacity blockers for all MOI engines and isolates their remaining host-side lowering cost. |
| **D0** Attention and model paths, including `test/inductor/test_fused_attention.py` and `test/nn/test_multihead_attention.py` | 🟩 Promoted: independent oracle; clean 158/158; exact fault detected; 1.946x | 🟨 Promoted: clean 158/158 + 22/22 + 2/2 + 2/2; reviewed traps remain unattributed; 7.795x | 🟥 Promoted: full-pressure persistent/transient SGPR placement | 🟥 Promoted: same full-pressure SGPR placement | Ordinary causal `torch.nn.functional.scaled_dot_product_attention` is now in the main matrix.  It selects a real native attention kernel and exposed the RDNA4 scalar-spilling and relay-placement work needed to make default Record/Replay clean-complete. |
| **D1** `test/test_scatter_gather_ops.py` | 🟩 Promoted: exact oracle, clean 27/27, exact fault qualified miss, 1.010x | 🟩 Promoted: exact oracle, clean 27/27, exact fault qualified miss, 0.998x | 🟩 Promoted: exact oracle, clean 27/27, exact fault qualified miss, 0.993x | 🟩 Promoted: exact oracle, clean 27/27, exact fault qualified miss, 1.031x | Collision-heavy BF16/FP32 `scatter_reduce` is now green across all four engines. The current Sampled and Inline explicit-key shared relays restore all adjacent sub-byte helper sites without a user knob. Fresh artifacts named in the main row supersede the discovery-era `rdna4-pytorch-scatter-*` evidence. The relaxed singleton reductions are atomicity operations, not qualified MOI ordering edges. |
| **D1** `test/test_reductions.py` histogram | 🟩 Promoted: exact oracle, clean 135/135, exact fault qualified miss, 0.966x | 🟧 Promoted: exact oracle, 135/135 accesses, 42/84 barriers | 🟨 Promoted: exact-complete 135/135 accesses + 168/168 barrier members, 1.094x | 🟨 Promoted: exact-complete 135/135 accesses + 84/84 barriers, 5.301x | Ordinary `torch.histc` is now in the main matrix. Its native object adds a real precompiled-library placement and relaxed-atomic difficulty class. Sampled and Inline use gfx1201 dense barrier routing and now instrument every supported ordinary LDS and barrier site; Record/Replay retains an isolated scalar-envelope placement gap. The relaxed atomics remain accesses rather than qualified memory-ordering events. The reviewed exact post-accumulation barrier drop is schedule-masked by this launch and retained as a qualified miss. |
| **D1** `torch.compile` softmax selected from the reduction/softmax survey | 🟩 Promoted: clean 4/4; exact drop qualified miss; 0.960x | 🟩 Promoted: clean 4/4 + 3/3; exact drop detected; 1.052x | 🟧 Promoted but clean false conflict | 🟧 Promoted: static 4/4 + 3/3; dynamic workgroup-bank limit | The `128x256` exact client is in the main matrix.  Record/Replay detects the third exact barrier-pair drop even though the numeric oracle remains schedule-masked. Inline now instruments every site and exposes a distinct 32-bank external-shadow scalability limit; a different native shape remains desirable for Sampled. |
| **D2** `test/inductor/test_cooperative_reductions.py` and `test/inductor/test_online_softmax.py` | 🟩 Promoted: exact oracle, clean 8/8, exact fault qualified miss, 0.977x | 🟩 Promoted: clean 8/8 + 6/6, exact replay fault qualified miss, 1.037x | 🟧 Promoted: static-complete, but 16 clean false conflicts | 🟧 Promoted: static-complete, but 18,576 dynamic-incomplete events | Upstream split online softmax is now in the main matrix. Its deterministic client avoids the unrelated 856-kernel RNG object while retaining the exact target-native `1x(2^20+13)` split-reduction shape. |
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
