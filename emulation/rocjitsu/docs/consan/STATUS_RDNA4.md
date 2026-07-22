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
certificate has advanced through rebuilt-hook checkpoint `5643227561`, using
hook SHA-256
`9dc9695d241114b74c00597878caf2e1c07afa4311d85296a6c48a5b13f05c34`.
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

`N/A — RESOLVED EXCLUSION` is deliberately outside the ordered maturity scale.
It means target-native evidence established that the candidate is not an
acceptance workload (for example, it is compile-only, executes no supported
synchronization, or duplicates a stronger retained workload).  This textual
label is intentional: it must not be mistaken for an unassessed gray cell or
counted in the validation denominator, even in renderers where colored emoji
are indistinguishable.

## Catch-up snapshot

- **Immediate requalification:** the portable manifest now has 19 workloads ×
  4 profiles = 76 current-tip cells, including six independently selected
  native gfx1201 PyTorch workloads and two native llama.cpp workloads. Rebuild
  the exact hook, run
  clean and paired rows, regenerate target identities, review fault specs,
  execute contained faults, and freeze one provenance bundle.
- **Historical comparator:** commit `640e575da2` accepted 55 clean
  baseline/profile rows, 14 reviewed fault policies, and 66 paired-overhead
  rows.  Do not copy its colors or selectors to the current tip.
- **Native corpus expansion:** survey `rocjitsu-test-corpus`'s gfx1201 cases,
  inventory their executed code objects, and admit independently useful
  oracle-backed workloads.  The discovery table tracks all four profiles
  separately even before admission.  This survey is now selection-complete:
  both admitted llama.cpp rows have non-gray per-engine evidence and every
  other corpus family has a target-native resolved exclusion.  hipBLAS is
  enabled in the external TheRock build and is no longer a setup blocker.
- **Native PyTorch expansion:** use the fresh checkout at
  `$CONSAN_VALIDATION_WORKSPACE_DIR/pytorch` to discover shapes that naturally
  select interesting gfx1201 kernels, then freeze small exact clients.  The
  discovery table tracks each profile independently.
- **gfx1250 comparison:** use its workload diversity and proof quality as a
  model, but do not port its configurations, shapes, selectors, denominators,
  or expected outcomes to RDNA4.

Thus the immediate regression campaign contains 76 cells.  The expanded
gfx1201 denominator is intentionally unknown until discovery produces concrete
native workloads with independent oracles.  Survey rows and baselines are not
counted as instrumentation cells.

The current `gfx1201 manifest --json` exposes the original 11 workloads plus
`pytorch-rdna4-compiled-softmax`, `pytorch-rdna4-split-softmax`,
`pytorch-rdna4-llm-topk`, `pytorch-rdna4-sdpa`, and the independently
inventoried target-native `pytorch-scatter-reduce` and
`pytorch-torch-histc`, plus `llama-rdna4-mul-mat-vec-q` and
`llama-rdna4-rms-norm`. All other
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
| **P2 PyTorch/Inductor compiled softmax** | 🟩 Exact oracle; clean 4/4 LDS accesses; exact third barrier drop is a precommitted qualified miss; overhead 0.960x | 🟩 Exact oracle; clean 4/4 + 3/3; exact third barrier drop emits an attributed replay diagnostic; overhead 1.052x | 🟩 Exact oracle; clean-complete 4/4 accesses + 6/6 barrier members; exact third barrier-pair drop emits a causal diagnostic while the oracle remains correct; overhead 1.204x | 🟩 Exact oracle; clean-complete 4/4 accesses + 3/3 barriers; exact third barrier-pair drop emits attributed diagnostics while the oracle remains correct; overhead 0.937x | Sampled checkpoint `fe455546dd`, hook `d5f9ba58…`: retained clean artifact `rdna4-compiled-softmax-sampled-address-aligned-clean-20260722`, paired overhead `rdna4-compiled-softmax-sampled-overhead-20260722`, current inventory `rdna4-compiled-softmax-current-inventory-20260722`, and accepted contained fault `rdna4-compiled-softmax-sampled-fault-20260722`. Ten timed dispatches measure 0.060313 ms instrumented versus a 0.050091 ms paired baseline (1.204063x). The precommitted third-pair removal applies exactly once as two NOP rewrites, preserves the independent CPU oracle, and produces four Sampled causal conflicts; remaining static coverage is complete at 4/4 accesses and 4/4 barrier members, and cleanup returns live report memory to zero. The address-aligned synchronization checkpoint `c5d9c59672` made this possible: runtime sampling selects LDS addresses, so every reaching workgroup advances its epoch and inspects its banked causal window rather than applying an unrelated workgroup-residue gate. Inline checkpoint `5373a186a5`, hook `d5f9ba58…`: clean-tip artifact `rdna4-compiled-softmax-inline-256banks-committed-20260722`, paired artifact `...-overhead-20260722`, and accepted fault `...-fault-20260722`. Automatic 256-bank exact-shadow allocation eliminates all 576 clean publication-claim losses from the original 32-bank layout. Its 100,671,536-byte workload report remains below the 128 MiB per-buffer ceiling, cleanup returns live report memory to zero, and no user knob is used. The historical `inline_unsupported_workgroups` log label was misleading: it exposed `inline_undercoverage_count`, not a packed-workgroup-identity exclusion; the current log and parser use the accurate `inline_undercoverage` name. Inline's ten timed dispatches measure 0.069966 ms instrumented versus a 0.074677 ms paired baseline (0.936911x); the same reviewed pair removal produces 242 Inline diagnostics. Other profiles retain checkpoint `b6be4081e1`: clean `...-110`, paired overhead `...-111` and `...-112`, inventory `...-107`, and accepted faults `...-108` and `...-109` |
| **P2 PyTorch split online softmax** | 🟩 Exact CPU-derived BF16 oracle; clean-complete 8/8 LDS accesses across two generated stages; exact third barrier drop is a precommitted qualified miss; overhead 0.977x | 🟩 Exact oracle; clean-complete 8/8 accesses + 6/6 barriers; exact third barrier drop is a qualified replay miss; overhead 1.037x | 🟩 Exact oracle; clean-complete 8/8 accesses + 12/12 barrier members across two generated stages; exact third barrier-pair drop emits a causal diagnostic while the oracle remains correct; overhead 1.257x | 🟩 Exact oracle; clean-complete 8/8 accesses + 6/6 barriers across both stages; independently confirmed exact third barrier-pair drop emits diagnostics while the oracle remains correct; overhead 4.012x | Sampled checkpoint `61916f3183`, hook `d5f9ba58…`: retained clean artifact `rdna4-split-softmax-sampled-address-aligned-clean-20260722`, paired overhead `rdna4-split-softmax-sampled-overhead-20260722`, current inventory `rdna4-split-softmax-current-inventory-20260722`, and accepted contained fault `rdna4-split-softmax-sampled-fault-20260722`. Ten timed dispatches measure 0.091581 ms instrumented versus a 0.072864 ms paired baseline (1.256887x). The precommitted third-pair removal applies exactly once as two NOP rewrites, preserves the exact independent BF16 oracle, and produces one Sampled causal conflict; remaining static coverage is complete at 8/8 accesses and 10/10 barrier members, and cleanup returns live report memory to zero. The address-aligned synchronization checkpoint `c5d9c59672` eliminates all former clean conflicts and publishes 22 + 3 synchronization records across the two stage objects without a user knob. Inline checkpoint `5373a186a5`, hook `d5f9ba58…`: clean-tip artifact `rdna4-split-softmax-inline-256banks-committed-20260722`, paired artifact `...-overhead-20260722`, and rejected discovery fault `...-fault-20260722`. Increasing exact-shadow banking from 32 to 128 reduced clean claim losses from 468 to 10, and the automatic 256-bank layout eliminates them. Its two 100,671,536-byte workload reports plus the runtime support report total 201,351,248 live bytes, within the 256 MiB process ceiling; cleanup returns live memory to zero and no user knob is used. Ten Inline dispatches measure 0.289021 ms instrumented versus a 0.072043 ms paired baseline (4.011819x). The original precommitted qualified-miss discovery remains rejected after the exact third-pair removal preserved the oracle and produced 1,033 Inline diagnostics. Independent accepted confirmation `rdna4-split-softmax-inline-detected-confirm-20260722`, reviewed-spec SHA `81ab57e2…`, precommits detection and reproduces it without retroactively fitting the discovery row. Other profiles retain checkpoint `a199956578`, hook `2dec0c65…`: clean artifacts `rdna4-split-softmax-{sc,rr}-final-clean`, paired overhead `...-final-overhead`, inventory `rdna4-split-softmax-final-inventory`, accepted SuperCollider fault `...-sc-final-fault`, and accepted Record/Replay miss `...-rr-final-fault-accepted`. Both exact pair removals leave the oracle unchanged; the initial rejected replay expectation is retained in `...-rr-final-fault`. The deterministic client freezes upstream `test_online_softmax.py::test_split_reduction` shape `1x(2^20+13)` without loading the unrelated 856-kernel RNG object used by the test harness |
| **P2 PyTorch Qwen-vocabulary top-k** | 🟥 Exact values and indices pass and clean analysis reaches a typed verdict in 54.00 seconds, but only 2,039/63,474 supported accesses patch | 🟥 Both semantic passes now complete and a 109,229,808-byte automatic report is allocated without tuning, but all-site lowering still exceeds a clean 75-second discriminator before a patch verdict | 🟧 Exact values-and-indices oracle and clean execution pass without diagnostics or tuning; useful partial coverage reaches 57,153/63,474 accesses + 12,978/14,200 barrier members in 102.05 seconds; overhead and fault pending | 🟧 Exact oracle passes and useful partial execution reaches 38,365/63,474 accesses + 3,450/7,100 barriers in 99.84 seconds without tuning, but one clean diagnostic rejects the strict run | Sampled checkpoint `551c284c40`, hook `f28c56eb…`, artifact `rdna4-topk-sampled-relay-scaled-final-20260722`: the committed trace-free hook exits zero with the exact oracle, no clean diagnostic, dynamic completeness, 90.04% supported-access coverage, and 91.39% supported-barrier-member coverage. The 23 MiB object is complete at 20,822/20,822 accesses + 4,200/4,200 barrier members; the 40 MiB object reaches 36,331/42,652 + 8,778/10,000, leaving only typed placement/lowering gaps. Lowering now reuses the pristine resource-planning CFG and indexes local relay-island searches by exact SOPP reach, with no user knob. Inline artifact `rdna4-topk-inline-indexed-final-20260722` uses the same committed hook and reaches a typed verdict rather than timing out: 60.44% of supported accesses and 48.59% of supported barriers patch, and the exact oracle passes, but an owner-1 store versus owner-4 load conflict over LDS `[4096,4104)` produces one visible clean diagnostic and the required strict exit 89. That false positive and incomplete placement keep Inline orange. The automatic report-sizing and instrumentation passes still repeat about 19 seconds of whole-object semantic analysis for the larger object; fault and paired-overhead evidence remain the Sampled orange-to-yellow gate. All 727 ConSan unit tests pass, and current compact Qwen regressions remain complete at 20/20 + 26/26 for Sampled and 20/20 + 14/14 for Record/Replay in `rdna4-topk-optimizer-qwen{-rr}-regression-20260722`. SuperCollider checkpoint `512c09ea0a`, artifact `rdna4-topk-sc-threshold-regression-65`, hook `2dec0c65…`, reconfirms exactly 941/42,652 and 1,098/20,822 patched accesses after the moderate-object relay fix. The two 23 MiB and 40 MiB objects retain preflight-candidate-only CFG construction, but 61,435 sites still lack reachable relay placement, far below usable coverage. Record/Replay checkpoint `3c599609cf`, hook `78bb43d9…`: offset-indexing removes the quadratic ordinary acquire/release fence scan, reducing the formerly unbounded acquire association to about 1.9 seconds and the complete semantic inventory to about 19 seconds. Automatic sizing retains the largest power-of-two dynamic headroom that fits the ordinary 128 MiB ceiling; this object gets 256 records per static barrier and a 109,229,808-byte report without a user knob. Clean artifact `rdna4-topk-rr-indexed-adaptive-clean-20260722` reaches lowering before its explicit 75-second harness cap; the identical implementation also remained in lowering at a bounded 120 seconds. Earlier checkpoint `ff1afe848c` replaced per-site candidate and kernel-context scans with indexed lookup and cached per-kernel private layouts. Clean artifacts `rdna4-topk-{sampled,inline}-indexed-discriminator-20260722` show that both online engines complete semantic inventory and automatic allocation before their old 45-second caps |
| **P2 PyTorch causal SDPA** | 🟩 Independent CPU oracle; clean 158/158; exact barrier drop emits an attributed diagnostic and breaks the oracle; overhead 1.946x | 🟨 Independent CPU oracle; clean-complete 158/158 accesses + 22/22 barriers + 2/2 atomics + 2/2 fences; overhead 7.795x; two reviewed exact drops break the oracle through unattributed hardware traps but emit no replay diagnostic | 🟥 Literal dispatch-ID fallback now removes the persistent-pair blocker, but the full-pressure attention kernel still has no safe transient scalar probe/router state; only the separate 27/27-access fill object patches | 🟥 Literal dispatch identity and gfx1201 transient-window preservation are now available, but this kernel has no common dead scalar pair for its indirect router; the attention object's 131 accesses + 22 barriers + 2 atomics remain unpatched and strict execution fails closed | Checkpoint `a4dfe2e49f`, hook `70d13b68…`: committed artifacts `rdna4-sdpa-{sampled-literal,inline-scalar-fallback}-committed-20260722` both reach a typed verdict in under 9.9 seconds without a knob. They prove that the hardware dispatch-ID pair is no longer load-bearing on gfx1201. Inline additionally proves that all 155 attention-object resource plans have spillable VGPR windows, but no scalar entry/return router state is dead across every site; fully spill-backed indirect routing is the remaining distinct capability, not ordinary VGPR spilling. All 728 ConSan and 20 ConSan hook tests pass. Current Record/Replay artifact `rdna4-sdpa-rr-safe-entry-return-20260722`, checkpoint `e49f6b0c33`, reconfirms exact static and dynamic completeness in 9.59 seconds. It also caught a real regression: cumulative cave growth pushed the owner/epoch entry prologue beyond SOPP reach, and the old long return reused a probe-dead SGPR pair that still held kernel-entry ABI state. RDNA-family entry returns now use entry-undefined VCC instead. Earlier checkpoint `8a5669d5e5` retains paired SC/RR overhead, inventory, and accepted SC fault artifacts. Rejected RR fault rows retain two distinct unattributed-trap outcomes; no user tuning knobs are used |
| **P2 Sharktank TP2 family** | 🟩 Exact oracle; clean 2976/2976; exact drop is a precommitted qualified miss; overhead 1.28x | 🟩 Exact oracle; clean 2976/2976 + 228/228; exact drop emits a replay diagnostic; overhead 1.83x | 🟩 Exact oracle; clean 2976/2976 + 420/420; exact drop is a precommitted qualified miss; overhead 1.24x | 🟩 Exact oracle; clean 2976/2976 + 228/228; exact drop detected 16/16; overhead 2.17x | One-tip checkpoint `08b15c6d91`: clean `...-048`, overhead `...-049`, inventory `...-050`, accepted fault rows `...-053` through `...-056`; historical expectations remain as failed discovery rows `...-051` and `...-052` |
| **P3 CLIP BF16** | 🟩 Exact oracle; clean 85/85; exact drop and move are precommitted qualified misses; overhead 0.98x | 🟩 Exact oracle; clean 85/85 + 36/36; exact drop and subtle move are precommitted qualified misses; overhead 1.236x | 🟩 Exact oracle; clean 85/85 + 72/72; exact drop and move are precommitted qualified misses; overhead 0.97x | 🟩 Exact oracle; clean 85/85 + 36/36; exact move emits an attributed diagnostic while exact drop is a qualified miss; overhead 1.51x | Record/Replay checkpoint `38f77d6601`, hook `bb08fa7c…`: clean `rdna4-clip-rr-scalar-007`, paired overhead `...-overhead-001`, inventory `...-inventory-001`, and accepted faults `...-fault-drop-001` and `...-fault-move-001`. gfx1201 now keeps owner/epoch in persistent SGPRs instead of loading a private-memory epoch at every hot access. Other profiles remain at checkpoint `3033f30f2a` in `...-057` through `...-070` |
| **P3 PyTorch native histogram** | 🟩 Exact oracle; clean-complete 135/135 ordinary LDS accesses; exact post-accumulation barrier drop is a precommitted qualified miss; overhead 0.966x | 🟩 Exact oracle; clean-complete 135/135 accesses + 84/84 barriers; exact initialization-barrier drop is a reviewed qualified miss and breaks the oracle; overhead 1.014x | 🟩 Exact oracle; clean-complete 135/135 accesses + 168/168 barrier members; exact initialization-barrier drop is a reviewed qualified miss and is schedule-masked; overhead 1.094x | 🟩 Exact oracle; clean-complete 135/135 accesses + 84/84 barriers; exact initialization-barrier drop is a reviewed qualified miss and breaks the oracle; overhead 5.301x | Current fault inventory `rdna4-histc-moi-fault-inventory-20260722`, checkpoint `5643227561`, hook `9dc9695d…`, proves the selected float32 initialization pair exists exactly once. Finalized reviewed spec SHA `139300a8…` is accepted by `rdna4-histc-{rr,sampled,inline}-init-miss-confirm-20260722`: each run applies one logical mutation as exactly two NOP rewrites, remains statically and dynamically complete, passes before/after device-health containment, and emits no diagnostic. Record/Replay and Inline independently fail the exact histogram oracle; Sampled's run is schedule-masked. Rejected precommit artifacts retain both the initially expected detections and the earlier post-accumulation selector, so the qualified misses are explicit confirmations rather than fitted discovery rows. The limitation is semantic: the relaxed LDS accumulation remains outside the qualified MOI atomic-event model (`atomic=0/0`), so removing its initialization barrier does not create a modeled release/acquire edge. Record/Replay clean artifact `rdna4-histc-rr-conditional-router-20260722` is exact at 135/135 + 84/84; paired artifact `rdna4-histc-rr-dense-barrier-overhead-20260722` measures 1.013901x. Sampled clean `rdna4-histc-sampled-at-inline-tip-clean-20260722` is exact at 135/135 + 168/168; paired overhead is 1.094381x. Inline clean `rdna4-histc-inline-dense-barrier-committed-20260722` is exact at 135/135 + 84/84; paired overhead is 5.300524x. SuperCollider retains its independently accepted post-accumulation qualified miss and 0.966x overhead |
| **P4 hip-moi D128 block attention** | 🟩 Exact oracle; clean 12/12; exact barrier drop is a precommitted qualified miss and breaks the oracle; overhead 164.77x | 🟩 Exact oracle; clean 12/12 + 4/4; exact barrier drop emits an attributed diagnostic and breaks the oracle; overhead 11.21x | 🟩 Exact oracle; clean-complete 12/12 accesses + 8/8 barrier members; exact barrier drop is a precommitted qualified miss and breaks the oracle; overhead 14.03x | 🟩 Exact oracle; clean 12/12 accesses + 4/4 barriers; exact barrier drop emits a diagnostic and breaks the oracle; overhead 12.83x | Current hook `2dec0c65…`: Sampled clean `rdna4-d128-block-sampled-final-clean`, paired overhead `...-final-overhead`, shared fresh inventory `rdna4-d128-block-inline-final-inventory`, and accepted fault `...-sampled-final-fault`. Inline uses the same inventory plus paired overhead `...-inline-final-overhead` and accepted fault `...-inline-final-fault-accepted`; rejected historical expectation `...-inline-final-fault` preserves the improved detector outcome after generation-qualified local shadows removed the former clean diagnostic. SuperCollider checkpoint `ca1eb5456e`; Record/Replay checkpoint `baed32a85e` |
| **P4 hip-moi D128 pressure attention** | 🟩 Exact oracle; clean-complete 12/12 accesses; exact barrier drop is a precommitted qualified miss and breaks the oracle; overhead 11.25x | 🟩 Exact oracle; clean 12/12 + 4/4; exact barrier drop emits an attributed diagnostic and breaks the oracle; overhead 14.30x | 🟩 Exact oracle; clean-complete 12/12 accesses + 8/8 barrier members; independently confirmed exact drop emits a diagnostic and breaks the oracle; overhead 18.531x | 🟩 Exact oracle; clean 12/12 + 4/4; exact barrier drop emits an attributed diagnostic; overhead 13.72x | Current Sampled hook `897e52a3…`: clean `rdna4-d128-pressure-sampled-current-20260722`, paired overhead `rdna4-d128-pressure-sampled-current-paired-20260722`, and inventory `rdna4-d128-pressure-current-inventory-20260722`. The recent address-aligned synchronization fix removes all four former clean conflicts without a user knob. The fresh inventory exactly matches the historical selector. Rejected discovery artifact `rdna4-d128-pressure-sampled-current-fault-20260722` preserves the original qualified-miss precommit. Independent accepted confirmation `rdna4-d128-pressure-sampled-detected-confirm-20260722`, reviewed-spec SHA `9c8724b6…`, instead precommits the stronger outcome and reproduces one exact mutation, a Sampled diagnostic, and oracle failure. Other engines retain hook `2dec0c65…`: SuperCollider clean `rdna4-d128-pressure-sc-final-clean`, paired overhead `...-final-overhead`, inventory `...-final-inventory`, and accepted contained fault `...-final-fault`. Record/Replay checkpoint `baed32a85e`; Inline checkpoint `79aea7420c` |
| **P4 hip-moi WMMA attention** | 🟩 Exact oracle; clean 12/12; exact barrier drop is a precommitted qualified miss and breaks the oracle; overhead 158.07x | 🟩 Exact oracle; clean 12/12 + 4/4; exact barrier drop emits an attributed diagnostic and breaks the oracle; overhead 11.99x | 🟩 Exact oracle; clean-complete 12/12 accesses + 8/8 barrier members; exact barrier drop is a precommitted qualified miss and breaks the oracle; overhead 14.50x | 🟩 Exact oracle; clean 12/12 accesses + 4/4 barriers in five consecutive processes; exact barrier drop emits a diagnostic and breaks the oracle; overhead 13.24x | Current hook `2dec0c65…`: Sampled clean `rdna4-wmma-sampled-final-clean`, paired overhead `...-final-overhead`, shared fresh inventory `rdna4-wmma-inline-final-inventory`, and accepted fault `...-sampled-final-fault`. Inline uses the same inventory plus five clean processes `rdna4-generation-wmma-repeat-{1..5}`, paired overhead `...-inline-final-overhead`, and accepted fault `...-inline-final-fault-accepted`; rejected historical expectation `...-inline-final-fault` records the improved detector outcome. SuperCollider checkpoint `ca1eb5456e`; Record/Replay checkpoint `baed32a85e` |
| **P4 hip-moi Stream-K arrival** | 🟩 Exact oracle; clean 4/4 accesses; exact order and scope weakenings are precommitted qualified misses; overhead 558.83x | 🟩 Exact oracle; clean 4/4 accesses + 15/15 atomics + 4/4 barriers + 16/16 fences; exact order and scope weakenings are precommitted qualified misses; overhead 53.00x | 🟩 Exact oracle; clean-complete 4/4 accesses + 15/15 atomics + 8/8 barrier members; exact order and scope weakenings are precommitted qualified misses; overhead 48.21x | 🟩 Exact oracle; clean-complete 4/4 accesses + 15/15 atomics + 4/4 barriers; exact order and scope weakenings each emit a diagnostic while preserving the oracle; overhead 50.93x | Current Inline checkpoint `137bcfefe5`, hook `62a17bf5…`: clean `rdna4-inline-durable-token-final3-clean`, paired overhead `...-final3-overhead`, inventory `...-final3-inventory`, and accepted detected faults `...-final3-fault-atomic-weaken-{order,scope}`. The initial fault contract expected a miss and is retained as rejected evidence that the detector outcome was not silently fitted after observation. Five additional clean processes pass. Sampled artifacts remain `rdna4-streamk-sampled-final-*`; Record/Replay checkpoint `baed32a85e`; SuperCollider checkpoint `79aea7420c` |
| **P4 hip-moi tree atomic-OR** | 🟩 Exact oracle; clean 4/4 accesses; exact order and scope weakenings are precommitted qualified misses; overhead 591.81x | 🟩 Exact oracle; clean-complete 4/4 accesses + 15/15 atomics + 4/4 barriers + 16/16 fences; exact order weakening is a reviewed qualified miss, while exact scope weakening emits an attributed replay diagnostic; both preserve the oracle; overhead 50.395x | 🟩 Exact oracle; clean-complete 4/4 accesses + 15/15 atomics + 8/8 barrier members; exact order and scope weakenings are precommitted qualified misses; overhead 49.91x | 🟧 Exact oracle and complete 4/4 accesses + 15/15 atomics + 4/4 barriers, but repeated-process qualification still intermittently diagnoses a correct owner-4 read after owner 2 | Record/Replay checkpoint `5643227561`, hook `9dc9695d…`: clean artifact `rdna4-tree-rr-fence-relocation-clean-20260722` is exact-complete and exits in 2.289 seconds without diagnostics or tuning. Paired artifact `rdna4-tree-rr-fence-relocation-paired-overhead-20260722` measures 2,167 ms instrumented versus a 43 ms before/after baseline (50.395349x). Fresh inventory `rdna4-tree-rr-fence-relocation-inventory-20260722` reviewed one system-scope returning `flat_atomic_or_b32`; finalized spec SHA `09edc931…` is accepted by `...-fault-order-confirm-20260722` as a qualified miss and by `...-fault-scope-confirm-20260722` as one replay detection. The scope diagnostic attributes owner 0's store versus owner 3's load over exact LDS `[0,4)` while the independent oracle passes. The initial rejected scope-discovery artifact is retained separately, making the stronger expectation and independent confirmation explicit rather than retrofitting the observed row. The fix relocates a decoded fence prefix into its appended helper and uses the vacated seven words for an SCC-liveness-proven compact entry when ordinary islands are exhausted; a compiler-layout regression joins the complete 730-test ConSan suite. Current Inline checkpoint `137bcfefe5`, hook `62a17bf5…`: ten isolated clean processes pass, but paired artifact `rdna4-inline-durable-token-final3-overhead` reproduces the diagnostic in 2/3 instrumented processes, so no overhead is claimed. Committed inherited tokens now remain authoritative without re-reading a mutable source slot, deleting 354 hot-path lines and shrinking these access probes from about 3,960 to 2,480 bytes; a second access-time token-visibility gap remains. Sampled artifacts remain `rdna4-tree-sampled-final-*`; other retained evidence is at checkpoint `79aea7420c` |
| **P4 hip-moi Jakub attention variants** | 🟩 Exact oracle; clean 31/31 accesses; exact barrier drop is a precommitted qualified miss; overhead 103.75x | 🟩 Exact oracle; clean 31/31 + 4/4; exact barrier drop is a precommitted qualified miss; overhead 11.69x | 🟩 Exact oracle; clean-complete 31/31 accesses + 8/8 barrier members; exact barrier drop is a precommitted qualified miss with the oracle schedule-masked; overhead 10.87x | 🟩 Exact oracle; clean 31/31 + 4/4; exact barrier drop emits an attributed diagnostic; overhead 11.03x | Current hook `2dec0c65…`: Sampled clean `rdna4-jakub-sampled-final-clean`, paired overhead `...-final-overhead`, fresh inventory `...-final-inventory`, and accepted contained fault `...-final-fault`. Record/Replay checkpoint `baed32a85e`; SuperCollider and Inline checkpoint `79aea7420c` |

The latest top-k Record/Replay discriminators retain the fixed 75-second cap.
Checkpoints `5c3c99c49a` and `da733901b4` replace two quadratic dense-relay
searches with the indexes already owned by the resource planner: merged local
instrumentation ranges plus kernel-local CFG ranges, and exact descriptor plus
block-set lookup for scalar-window liveness.  All 729 ConSan C++ tests pass.
Artifacts `rdna4-topk-rr-dense-indexed-final-20260722` and
`rdna4-topk-rr-context-indexed-final-20260722` still complete semantic report
planning, allocate the 109,229,808-byte automatic report without a knob, and
enter the 40 MiB object's instrumentation pass without producing a typed
verdict before the cap.  These generally useful indexing fixes do not promote
the cell.  This cell has now consumed its bounded optimization attempt: do not
extend its timeout or resume lowering work without a profiler identifying a
third distinct cost.

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

A discovery row uses `N/A — RESOLVED EXCLUSION` after target-native evidence
rules it out of the validation denominator.  This is a resolved selection
decision, not a support result and not a gray instrumentation cell.  Once a
candidate has an exact baseline and current-hook execution evidence, its four
cells use the ordinary ordered color scale independently. Both admitted llama.cpp commands
are now first-class `consan_validation.py` workloads. Their wrapper runs the
instrumented GPU client and an uninstrumented CPU backend in separate
processes, compares their binary F32 outputs, and emits a machine-readable
oracle and timing record.

| Discovery order and native gfx1201 source | SuperCollider | Record/Replay | Sampled | Inline Shadow | Admission decision and evidence |
|---|---|---|---|---|---|
| **D0** `llama-rdna4-mul-mat-vec-q` | **YELLOW** 🟨 Exact 0.01 CPU oracle; clean-complete 462/462 LDS accesses; 17.881x process overhead; reviewed fault pending | **YELLOW** 🟨 Exact oracle; clean-complete 462/462 accesses + 44/44 barriers + 63/63 atomics + 72/72 fences; 15.967x process overhead; reviewed effective fault pending | **YELLOW** 🟨 Exact oracle; clean-complete 462/462 accesses + 88/88 barrier members; 17.077x process overhead; reviewed fault pending | **ORANGE** 🟧 Exact oracle passes; useful partial execution reaches 132/462 accesses + 18/44 barriers, but one clean diagnostic rejects the strict run | Record/Replay checkpoint `42f4ff3c0e`, hook `1cc6bde4…`, artifact `rdna4-llama-matvec-rr-ordinary-address-final-20260722`, passes the independent CPU oracle with static and dynamic completeness in 2.756 seconds. Its paired artifact `rdna4-llama-matvec-rr-overhead-final-20260722` measures 2,584.806 ms instrumented against a 161.882 ms bracketed baseline (15.967262x), with all five instrumented repetitions complete. The exact `0xb8cc/0xb8d0` barrier drop was frozen with a precommitted detection expectation at checkpoint `c5c1e1ecb3`; rejected artifact `rdna4-llama-matvec-rr-fault-oracle-fixed-20260722`, checkpoint `67a55accf6`, applies it once, stays healthy, preserves the exact oracle, and emits no replay diagnostic. It is schedule-masked evidence, not a retrofitted qualified miss. That trial also exposed and fixed a validation-harness gap: the llama wrapper printed its independent oracle but did not export the structured result consumed by contained fault rows. The native corpus separately exposed a real lowering gap: nine ordinary global acquire loads were semantically associated with fences, but the fence emitter rejected the fully supported RDNA4 classification and then failed to carry its scalar-address encoding into the shared effective-address planner. Current scripted artifact `rdna4-llama-matvec-all-scripted-20260722`, hook `f28c56eb…`, retains SuperCollider and Sampled; every result differs from CPU by at most 0.004023. Paired artifacts `rdna4-llama-matvec-{sc,sampled}-overhead-scripted-20260722` use five instrumented processes bracketed by five-process baselines. Current Inline artifact `rdna4-llama-matvec-inline-current-20260722`, hook `1cc6bde4…`, identifies its lone owner-33 store versus owner-1 load diagnostic as a consequence of the intervening barrier being one of 26 placement misses. Rejected experiment `rdna4-llama-matvec-inline-dense462-experiment-20260722` proves that merely raising the qualified dense-routing envelope is unsafe: it reaches 462/462 accesses + 44/44 barriers and removes the diagnostic, but corrupts the exact output with 0.8198 maximum error. The 135-site safety cap therefore remains; no user knob narrows any accepted row. |
| **D0** `llama-rdna4-rms-norm` | **YELLOW** 🟨 Exact CPU oracle; clean-complete 22/22 LDS accesses; 1.194x process overhead; reviewed effective fault pending | **YELLOW** 🟨 Exact oracle; clean-complete 22/22 accesses + 11/11 barriers; 1.350x overhead; reviewed effective fault pending | **YELLOW** 🟨 Exact oracle; clean-complete 22/22 accesses + 22/22 barrier members; 1.404x overhead; reviewed effective fault pending | **YELLOW** 🟨 Exact oracle; clean-complete 22/22 accesses + 11/11 barriers; 1.375x overhead; reviewed effective fault pending | Current scripted clean and paired evidence is `rdna4-llama-rms-{all,overhead}-scripted-20260722`, hook `f28c56eb…`. All four engines pass the independent 128-element GPU-versus-CPU binary-output oracle without a diagnostic or user knob; every one of five overhead repetitions remains complete. The strong precommitted drop discriminator at checkpoint `e990e90858` removes the executed `0x1824/0x1830` pair exactly once for every engine, but rejected artifact `rdna4-llama-rms-all-fault-strong-20260722` stays bitwise correct and produces no diagnostic in all four trials. Follow-up inventory `rdna4-llama-rms-move-inventory-20260722` finds 262 legal move destinations only in the unexecuted `group_norm` kernel and none in the dispatched RMS specialization, so barrier move is not admitted as a misleading fault option. The earlier Inline retry-counter and dispatch-ID-zero fixes remain load-bearing. A reviewed fault with an observable semantic effect is still required for green. |
| **D0** `kernels.gfx1201.hip-matmul.hip_matmul_matvec.m256_n1_k1024` | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | A standalone build with the unrelated llama.cpp backend disabled passes all nine exact matvec variants. Fresh SuperCollider inventory finds no decoded LDS, barriers, or atomics in the nine workload kernels; only eight ambiguous flat maybe-group sites fail placement, so this does not add a sound or nonredundant ConSan workload. |
| **D0** `llama.cpp` noncontiguous batched-matmul and hazard metadata | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | The gfx1201 config explicitly skips compiling both variants. The hazard overlay restores llama.cpp PR #13155's pre-fix noncontiguous-stride conversion and expects a deterministic output-validation failure; it is a data-layout correctness reproducer, not a concurrency oracle. |
| **D1** Three collected IREE direct-tile matmuls: F16, FP8, and I8 | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | The three gfx1201 compilations pass, but their corpus records explicitly set `compile_only=true`; they dispatch no workload and provide no runtime oracle. Normalize a calls/support-module wrapper before reconsidering them. |
| **D1** IREE `argmax`, strided extract, and map-load/map-store cases | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | All four exact baselines pass. Record/Replay inventory in `.pytest-artifacts-rdna4-iree-inventory` reports zero supported accesses, barriers, atomics, and fences for every executed code object, so these global-only shapes add no ConSan coverage. |
| **D2** RDNA4 WMMA/SWMMAC, wave32/wave64, atomic, and lane/DS CTS families | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | Seven exact baseline cases pass, but the arithmetic and lane-operation cases add no admitted ConSan synchronization traffic. The bundled atomic case is useful as an unsupported-transform stress object, not as a compact or nonredundant acceptance workload. Retain this family as an engine-specific reproducer pool. |
| **D3** Remaining corpus families | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | N/A — RESOLVED EXCLUSION | Collection and representative execution now account for all 105 gfx1201 cases. Remaining FPSan host-only/arithmetic cases either dispatch no GPU work or reproduce the already-retained 2,560-atomic support object; remaining integer-ISA arithmetic variants add no distinct ordering family, and the reduction representative currently crashes this SDK's compiler. Keep these as compiler/engine reproducer pools rather than inflating the acceptance denominator. |

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
| **D0** `test/test_sort_and_select.py` and `test/test_reductions.py` | 🟥 Promoted; typed 2,039/63,474 verdict in 54.12 seconds | 🟥 Promoted; semantic inventory and automatic allocation complete, all-site lowering exceeds 75 seconds | 🟧 Promoted; exact clean execution, 57,153/63,474 accesses + 12,978/14,200 barrier members, no diagnostic or tuning | 🟧 Promoted; exact oracle and useful 38,365/63,474 access coverage, but partial barrier coverage and one clean diagnostic | A decode-style `topk` over one 151,936-element Qwen vocabulary row with `k=50` is now in the main matrix. Its exact baseline passes. The native rocPRIM path loads two large operator objects and exposes both whole-code-object scalability and far-relay placement pressure that the compact Inductor client does not. Checkpoint `551c284c40` reuses the pristine planning CFG and bounds island searches by SOPP reach; the final committed Sampled artifact completes in 102.05 seconds with useful 90% access and 91% barrier coverage. The committed Inline artifact completes in 99.84 seconds with 60% access and 49% barrier coverage but exposes a clean-input causal conflict. Remaining work is typed placement closure, removal of that Inline false positive, overhead, and reviewed-fault evidence—not a hang or setup blocker. |
| **D0** Attention and model paths, including `test/inductor/test_fused_attention.py` and `test/nn/test_multihead_attention.py` | 🟩 Promoted: independent oracle; clean 158/158; exact fault detected; 1.946x | 🟨 Promoted: current clean 158/158 + 22/22 + 2/2 + 2/2; reviewed traps remain unattributed; 7.795x | 🟥 Promoted: current full-pressure persistent/transient SGPR placement failure | 🟥 Promoted: current matching full-pressure SGPR placement failure | Ordinary causal `torch.nn.functional.scaled_dot_product_attention` is now in the main matrix. Current artifacts `rdna4-sdpa-{rr,sampled,inline}-safe-entry-return-20260722` reconfirm both Record/Replay completeness and the two online engines' precise resource boundaries after fixing the out-of-range entry return's ABI-state clobber. |
| **D1** `test/test_scatter_gather_ops.py` | 🟩 Promoted: exact oracle, clean 27/27, exact fault qualified miss, 1.010x | 🟩 Promoted: exact oracle, clean 27/27, exact fault qualified miss, 0.998x | 🟩 Promoted: exact oracle, clean 27/27, exact fault qualified miss, 0.993x | 🟩 Promoted: exact oracle, clean 27/27, exact fault qualified miss, 1.031x | Collision-heavy BF16/FP32 `scatter_reduce` is now green across all four engines. The current Sampled and Inline explicit-key shared relays restore all adjacent sub-byte helper sites without a user knob. Fresh artifacts named in the main row supersede the discovery-era `rdna4-pytorch-scatter-*` evidence. The relaxed singleton reductions are atomicity operations, not qualified MOI ordering edges. |
| **D1** `test/test_reductions.py` histogram | 🟩 Promoted: exact oracle, clean 135/135, exact fault qualified miss, 0.966x | 🟩 Promoted: exact-complete 135/135 accesses + 84/84 barriers; exact initialization drop is a qualified miss and breaks the oracle; 1.014x | 🟩 Promoted: exact-complete 135/135 accesses + 168/168 barrier members; exact initialization drop is a schedule-masked qualified miss; 1.094x | 🟩 Promoted: exact-complete 135/135 accesses + 84/84 barriers; exact initialization drop is a qualified miss and breaks the oracle; 5.301x | Ordinary `torch.histc` is now in the main matrix. Its native object adds a real precompiled-library placement and relaxed-atomic difficulty class. All three MOI engines now route every supported ordinary LDS and barrier site; independently confirmed reviewed faults now close their green gates. The relaxed atomics remain accesses rather than qualified memory-ordering events. |
| **D1** `torch.compile` softmax selected from the reduction/softmax survey | 🟩 Promoted: clean 4/4; exact drop qualified miss; 0.960x | 🟩 Promoted: clean 4/4 + 3/3; exact drop detected; 1.052x | 🟩 Promoted: clean 4/4 + 6/6 barrier members; exact drop detected; 1.204x | 🟩 Promoted: clean 4/4 + 3/3; exact drop detected; 0.937x | The `128x256` exact client is green across all four engines in the main matrix. Automatic 256-bank Inline exact shadow removes the 576 clean claim losses exposed by the original 32-bank layout without a user knob. The reviewed third-pair removal preserves the independent oracle while producing four Sampled causal conflicts and 242 Inline diagnostics. |
| **D2** `test/inductor/test_cooperative_reductions.py` and `test/inductor/test_online_softmax.py` | 🟩 Promoted: exact oracle, clean 8/8, exact fault qualified miss, 0.977x | 🟩 Promoted: clean 8/8 + 6/6, exact replay fault qualified miss, 1.037x | 🟩 Promoted: clean 8/8 + 12/12 barrier members; exact drop detected; 1.257x | 🟩 Promoted: clean 8/8 + 6/6 across both stages; independently confirmed exact drop detected with intact oracle; 4.012x | Upstream split online softmax is now in the main matrix. Its deterministic client avoids the unrelated 856-kernel RNG object while retaining the exact target-native `1x(2^20+13)` split-reduction shape. Sampled's precommitted drop produces a causal conflict while preserving the exact oracle. Automatic 256-bank Inline exact shadow removes the former 468 clean claim losses while remaining inside the process memory ceiling; Inline's rejected discovery precommit is retained, while a separate reviewed-spec confirmation independently reproduces the stronger detected result. |
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
