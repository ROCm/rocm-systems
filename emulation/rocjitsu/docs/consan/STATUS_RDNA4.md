# ConSan RDNA4 (`gfx1201`) status

This is the current workload × instrumentation ledger for `gfx1201`.  It does
not inherit colors, selectors, denominators, or fault expectations from another
architecture.  End-to-end LLM workloads are the primary project metric;
focused kernels are retained as compact regression discriminators.

The executable authority is
[`consan_validation.py`](../../tests/dbi/consan/consan_validation.py), and
[VALIDATION.md](VALIDATION.md) defines the experiment contract.  The latest
Record/Replay checkpoint is `5af82ade33`, using hook SHA-256
`0edfe1985a2ee4512b65185a6ad625fe1160e0d080be5edd160d2a12b75dd82b`.
Individual cells may cite another retained current-tip checkpoint in their
artifact provenance.

## Status legend

The colors form a strict progression:

- 🩶 **unseen / unassessed:** no useful current-tip execution evidence;
- 🟥 **does not work:** execution fails before establishing useful behavior;
- 🟧 **some things work:** useful behavior is demonstrated, but substantial
  correctness, coverage, completion, or acceptance gaps remain;
- 🟨 **most things work:** the clean workload and important instrumentation
  path work, with a limited final-acceptance gap; and
- 🟩 **everything works:** clean oracle, coverage, reviewed fault, containment,
  overhead, memory, timeout, health, and provenance gates are all retained.

A crash, trap, timeout, oracle mismatch, or device loss is not a ConSan
diagnostic.  A green cell may contain an honest qualified miss only when that
outcome was precommitted and the exact mutation, containment, independent
oracle, and coverage evidence are valid.

## Current matrix

All 19 workloads × 4 engines are assessed.  Coverage counts use each engine's
own admitted-site model; in particular, Sampled reports split-barrier members
where other engines report logical barriers.

| Priority workload | SuperCollider | Record/Replay | Sampled | Inline Shadow |
|---|---|---|---|---|
| **P0 Qwen3-0.6B prefill** | 🟩 Exact oracle; clean 20/20; exact barrier drop emits an attributed diagnostic and breaks the oracle; overhead 1.044x | 🟩 Exact oracle; clean 20/20 + 14/14; exact drop is a precommitted qualified miss and breaks the oracle; overhead 1.004x | 🟩 Exact oracle; clean 20/20 + 26/26; deterministic 32-offset fault sweep detects 17/32 and every exact mutation breaks the oracle; overhead 1.003x | 🟩 Exact oracle; clean 20/20 + 14/14; exact barrier drop emits a diagnostic and breaks the oracle; overhead 1.005x |
| **P1 Sharktank TP1 prefill** | 🟩 Exact oracle; clean 352/352; exact drop is a precommitted qualified miss; overhead 1.26x | 🟩 Exact oracle; clean 352/352 + 46/46; exact drop emits a replay diagnostic; overhead 2.002x | 🟩 Exact oracle; clean 352/352 + 86/86; exact drop is a precommitted qualified miss; overhead 1.14x | 🟩 Exact oracle; clean 352/352 + 46/46; exact drop emits an attributed diagnostic; overhead 2.75x |
| **P1 Sharktank TP1 decode/combined** | 🟩 Exact oracle; clean 704/704; exact drop is a precommitted qualified miss; overhead 1.18x | 🟩 Exact oracle; clean 704/704 accesses + 92/92 barriers; exact drop emits replay diagnostics in 2/8 contained trials while the oracle is schedule-masked; overhead 1.542x combined / 1.517x decode | 🟩 Exact oracle; clean 704/704 + 172/172; exact drop is a precommitted qualified miss; overhead 1.01x | 🟩 Exact oracle; clean 704/704 + 92/92; exact drop emits an attributed diagnostic; overhead 1.70x |
| **P1 PyTorch collision-heavy scatter-reduce** | 🟩 Exact BF16/FP32 oracle; clean-complete 27/27 LDS accesses; exact global-atomic scope weakening is a precommitted qualified miss; overhead 1.010x | 🟩 Exact oracle; clean-complete 27/27 accesses; exact scope weakening is a precommitted qualified miss; overhead 1.006x (0.996x BF16) | 🟩 Exact oracle; clean-complete 27/27 accesses; exact scope weakening is a precommitted qualified miss; overhead 0.993x | 🟩 Exact oracle; clean-complete 27/27 accesses; exact scope weakening is a precommitted qualified miss; overhead 1.031x |
| **P2 PyTorch/Inductor compiled softmax** | 🟩 Exact oracle; clean 4/4; exact third barrier drop is a precommitted qualified miss; overhead 0.960x | 🟩 Exact oracle; clean 4/4 + 3/3; exact drop emits an attributed replay diagnostic; overhead 1.065x | 🟩 Exact oracle; clean 4/4 + 6/6 barrier members; exact drop emits a causal diagnostic; overhead 1.204x | 🟩 Exact oracle; clean 4/4 + 3/3; exact drop emits attributed diagnostics; overhead 0.937x |
| **P2 PyTorch split online softmax** | 🟩 Exact CPU-derived BF16 oracle; clean 8/8 across two stages; exact drop is a precommitted qualified miss; overhead 0.977x | 🟩 Exact oracle; clean 8/8 + 6/6; exact drop is a qualified replay miss; overhead 0.982x | 🟩 Exact oracle; clean 8/8 + 12/12 barrier members; exact drop emits a causal diagnostic; overhead 1.257x | 🟩 Exact oracle; clean 8/8 + 6/6; independently confirmed exact drop emits diagnostics; overhead 4.012x |
| **P2 PyTorch Qwen-vocabulary top-k** | 🟥 Exact oracle and typed verdict in 54.00 seconds, but only 2,039/63,474 supported accesses patch | 🟨 Exact oracle; clean-complete 63,474/63,474 accesses + 7,100/7,100 barriers in 89.39 seconds without diagnostics or tuning; overhead 1.316x; reviewed fault pending | 🟧 Exact oracle and clean execution; 57,153/63,474 accesses + 12,978/14,200 barrier members in 102.05 seconds; overhead and fault pending | 🟧 Exact oracle; 38,365/63,474 accesses + 3,450/7,100 barriers in 99.84 seconds, but one clean diagnostic rejects strict execution |
| **P2 PyTorch causal SDPA** | 🟩 Independent CPU oracle; clean 158/158; exact barrier drop emits an attributed diagnostic and breaks the oracle; overhead 1.946x | 🟨 Independent CPU oracle; clean-complete 158/158 accesses + 22/22 barriers + 2/2 atomics + 2/2 fences; overhead 7.894x; reviewed drops cause unattributed traps | 🟥 The attention kernel lacks safe transient scalar probe/router state; only the separate 27/27-access fill object patches | 🟥 The attention kernel lacks a common dead scalar pair for its indirect router; 131 accesses + 22 barriers + 2 atomics remain unpatched |
| **P2 llama.cpp quantized matvec** | 🟩 Independent CPU oracle; clean-complete 462/462 accesses; reviewed exact drop is a qualified miss; overhead 19.225x | 🟩 Independent CPU oracle; clean-complete 462/462 accesses + 44/44 barriers + 63/63 atomics + 72/72 fences; reviewed drop breaks the oracle; overhead 14.493x | 🟩 Independent CPU oracle; clean-complete 462/462 accesses + 88/88 barrier members; reviewed drop breaks the oracle; overhead 17.548x | 🟧 Exact oracle; 132/462 accesses + 18/44 barriers, but one clean diagnostic and 221,184 unsupported dynamic events reject strict execution |
| **P2 Sharktank TP2 family** | 🟩 Exact oracle; clean 2,976/2,976; exact drop is a precommitted qualified miss; overhead 1.28x | 🟨 Exact oracle; five clean-complete trials at 2,976/2,976 + 228/228 with no diagnostics; exact drop detected in 3/5 contained trials; overhead 1.806x prefill / 1.319x combined / 1.270x decode | 🟩 Exact oracle; clean 2,976/2,976 + 420/420; exact drop is a precommitted qualified miss; overhead 1.24x | 🟩 Exact oracle; clean 2,976/2,976 + 228/228; exact drop detected 16/16; overhead 2.17x |
| **P3 CLIP BF16** | 🟩 Exact oracle; clean 85/85; exact drop and move are precommitted qualified misses; overhead 0.98x | 🟩 Exact oracle; clean 85/85 + 36/36; exact drop and move are qualified misses; overhead 1.380x | 🟩 Exact oracle; clean 85/85 + 72/72; exact drop and move are qualified misses; overhead 0.97x | 🟩 Exact oracle; clean 85/85 + 36/36; exact move emits a diagnostic and drop is a qualified miss; overhead 1.51x |
| **P3 PyTorch native histogram** | 🟩 Exact oracle; clean-complete 135/135; exact drop is a qualified miss; overhead 0.966x | 🟩 Exact oracle; clean-complete 135/135 + 84/84; exact initialization drop breaks the oracle and is a qualified miss; overhead 1.041x | 🟩 Exact oracle; clean-complete 135/135 + 168/168 barrier members; exact drop is schedule-masked; overhead 1.094x | 🟩 Exact oracle; clean-complete 135/135 + 84/84; exact drop breaks the oracle and is a qualified miss; overhead 5.301x |
| **P3 llama.cpp RMS norm** | 🟨 Exact CPU oracle; clean-complete 22/22 accesses; overhead 1.194x; reviewed effective fault pending | 🟨 Exact oracle; clean-complete 22/22 + 11/11; overhead 1.327x; reviewed effective fault pending | 🟨 Exact oracle; clean-complete 22/22 + 22/22 barrier members; overhead 1.404x; reviewed effective fault pending | 🟨 Exact oracle; clean-complete 22/22 + 11/11; overhead 1.375x; reviewed effective fault pending |
| **P4 hip-moi D128 block attention** | 🟩 Exact oracle; clean 12/12; exact drop breaks the oracle and is a qualified miss; overhead 164.77x | 🟩 Exact oracle; clean 12/12 + 4/4; exact drop emits a diagnostic and breaks the oracle; overhead 13.216x | 🟩 Exact oracle; clean 12/12 + 8/8 barrier members; exact drop breaks the oracle and is a qualified miss; overhead 14.03x | 🟩 Exact oracle; clean 12/12 + 4/4; exact drop emits a diagnostic and breaks the oracle; overhead 12.83x |
| **P4 hip-moi D128 pressure attention** | 🟩 Exact oracle; clean 12/12; exact drop breaks the oracle and is a qualified miss; overhead 11.25x | 🟨 Exact oracle; clean-complete 12/12 + 4/4; exact drop breaks the oracle but produced no replay diagnostic in 5/5 contained trials; overhead 17.354x | 🟩 Exact oracle; clean 12/12 + 8/8 barrier members; confirmed exact drop emits a diagnostic and breaks the oracle; overhead 18.531x | 🟩 Exact oracle; clean 12/12 + 4/4; exact drop emits a diagnostic; overhead 13.72x |
| **P4 hip-moi WMMA attention** | 🟩 Exact oracle; clean 12/12; exact drop breaks the oracle and is a qualified miss; overhead 158.07x | 🟩 Exact oracle; clean 12/12 + 4/4; exact drop emits a diagnostic and breaks the oracle; overhead 13.856x | 🟩 Exact oracle; clean 12/12 + 8/8 barrier members; exact drop breaks the oracle and is a qualified miss; overhead 14.50x | 🟩 Exact oracle; clean 12/12 + 4/4; exact drop emits a diagnostic and breaks the oracle; overhead 13.24x |
| **P4 hip-moi Stream-K arrival** | 🟩 Exact oracle; clean 4/4; exact order/scope weakenings are qualified misses; overhead 558.83x | 🟩 Exact oracle; clean 4/4 + 15/15 atomics + 4/4 barriers + 16/16 fences; exact weakenings are qualified misses; overhead 48.889x | 🟩 Exact oracle; clean 4/4 + 15/15 atomics + 8/8 barrier members; exact weakenings are qualified misses; overhead 48.21x | 🟩 Exact oracle; clean 4/4 + 15/15 atomics + 4/4 barriers; exact weakenings emit diagnostics; overhead 50.93x |
| **P4 hip-moi tree atomic-OR** | 🟩 Exact oracle; clean 4/4; exact order/scope weakenings are qualified misses; overhead 591.81x | 🟩 Exact oracle; clean 4/4 + 15/15 atomics + 4/4 barriers + 16/16 fences; order weakening is a qualified miss and scope weakening emits a replay diagnostic; overhead 50.635x | 🟩 Exact oracle; clean 4/4 + 15/15 atomics + 8/8 barrier members; exact weakenings are qualified misses; overhead 49.91x | 🟧 Exact oracle and complete 4/4 accesses + 15/15 atomics + 4/4 barriers, but repeated processes intermittently diagnose a correct owner-4 read after owner 2 |
| **P4 hip-moi Jakub attention variants** | 🟩 Exact oracle; clean 31/31; exact drop is a qualified miss; overhead 103.75x | 🟩 Exact oracle; clean 31/31 + 4/4; exact drop is a qualified miss; overhead 10.255x | 🟩 Exact oracle; clean 31/31 + 8/8 barrier members; exact drop is a qualified miss; overhead 10.87x | 🟩 Exact oracle; clean 31/31 + 4/4; exact drop emits a diagnostic; overhead 11.03x |

## Non-green handoff

Only current blockers and the most useful retained evidence are recorded here.
Do not raise a timeout or change a fault expectation merely to promote a cell.
The source-matched Record/Replay campaigns for this checkpoint are
`rdna4-rr-final-clean-20260722`, `rdna4-rr-final-faults-20260722`, and
`rdna4-rr-final-overhead-20260722`.  All 19 clean and all 57 paired-overhead
rows pass; the reviewed-fault campaign accepts 17/19 rows.

### PyTorch Qwen-vocabulary top-k

- **SuperCollider:** the two 23 MiB and 40 MiB rocPRIM objects expose 63,474
  supported accesses, but only 2,039 patch.  The next task is scalable far-relay
  placement for SuperCollider, not another clean retry.
- **Record/Replay:** the current clean and overhead campaigns above establish
  complete coverage with no tuning and 1.315819x paired overhead.  The exact
  inventory retains 5,000 barrier-pair identities.
  `rdna4-topk-rr-fault-reused-inventory-20260722` still spends the fixed
  180-second bound in initial semantic fault planning, before mutation or GPU
  execution.  Optimize initial exact fault planning; do not increase the
  timeout or call this an applied miss.
- **Sampled:** the 23 MiB object is complete, while the 40 MiB object reaches
  36,331/42,652 accesses and 8,778/10,000 barrier members.  Close its remaining
  placement/lowering gaps before collecting overhead and reviewed-fault
  evidence.  Artifact: `rdna4-topk-sampled-relay-scaled-final-20260722`.
- **Inline Shadow:** `rdna4-topk-inline-indexed-final-20260722` reaches a typed
  verdict, but coverage is about 60% of accesses and 49% of barriers, and a
  clean owner-1 store versus owner-4 load conflict over LDS `[4096,4104)` is a
  false positive.  Fix that ownership conflict before expanding coverage.

### PyTorch causal SDPA

- **Record/Replay:** `rdna4-sdpa-rr-safe-entry-return-20260722` is clean and
  complete.  Two reviewed exact drops break the oracle through hardware traps
  but emit no replay diagnostic; traps are not detections.  A bounded late-pair
  trial (`rdna4-sdpa-rr-late18-discovery-20260722b`) also times out at 30
  seconds.  The next useful step is either report attribution before abnormal
  termination or a reviewed effective mutation that terminates normally.
- **Sampled and Inline Shadow:** all 155 attention-object resource plans have
  spillable VGPR windows, but the kernel has no common dead scalar pair for the
  indirect router.  The missing capability is fully spill-backed indirect
  SGPR entry/return state, not ordinary VGPR spilling.  Artifact:
  `rdna4-sdpa-{sampled-literal,inline-scalar-fallback}-committed-20260722`.

### Sharktank TP2 family

Record/Replay is clean-complete and false-positive-free in five consecutive
trials, but the reviewed exact barrier drop is diagnosed in only three of five
contained trials.  Static first-light slots are claimed once for the lifetime
of a loaded code object; the hardware dispatch ID now prevents invalid
cross-dispatch comparisons, but a multi-dispatch workload can consequently
retain a mixed-generation snapshot with no comparable pair for the fault.
Artifacts: `rdna4-rr-final-tp2-repeat-{2..5}-20260722` and
`rdna4-rr-final-fault-repeat-tp2-{2..5}-20260722`, in addition to the main
clean and fault campaigns above.  Restore deterministic detection with a
bounded dispatch-coherent capture design; do not revert generation filtering
or use unbounded dynamic append as an ordinary-mode workaround.

### llama.cpp quantized matvec

Inline Shadow is the only non-green engine.  The 1,024-element shape passes its
independent GPU/CPU oracle but reaches only 132/462 accesses and 18/44 barriers,
then reports one clean conflict and 221,184 unsupported dynamic events.  Use
`rdna4-llama-matvec-1024-all-clean-20260722` to separate the false positive
from the placement shortfall before adding more fault work.

### llama.cpp RMS norm

All four engines are clean-complete with modest process overhead in
`rdna4-llama-rms-{all,overhead}-scripted-20260722`.  The executed
`0x1824/0x1830` barrier-pair deletion is bitwise schedule-masked at both the
default and a larger `1024x4` discriminator.  Rewriting the same pair to legal
cluster and trap/workgroup IDs is also bitwise masked.  Barrier-move inventory
finds destinations only in the unexecuted `group_norm` kernel.  Green requires
a reviewed mutation with an observable semantic effect; do not keep varying
the already-disqualified pair.  Retained evidence:
`rdna4-llama-rms-all-fault-strong-20260722`,
`rdna4-llama-rms-move-inventory-20260722`, and
`rdna4-llama-rms-rr-barrier-id-{cluster,trap}-proof-fixed-20260722`.

### hip-moi D128 pressure attention

Record/Replay remains clean-complete and the exact barrier drop breaks the
independent oracle, but five consecutive contained trials produce no replay
diagnostic.  This is the same dispatch-coherence limitation exposed by TP2,
not a coverage, mutation, timeout, containment, or device-health failure.
Artifacts: `rdna4-rr-final-faults-20260722` and
`rdna4-rr-final-fault-repeat-pressure-{2..5}-20260722`.  Requalify this cell
after the bounded dispatch-coherent capture work tracked in
[FUTURE_WORK.md](FUTURE_WORK.md).

### hip-moi tree atomic-OR

Inline Shadow remains intermittently false-positive on a correct owner-4 read
after owner 2.  Ten isolated clean processes pass, but
`rdna4-inline-durable-token-final3-overhead` reproduces the diagnostic in two
of three instrumented processes.  Committed inherited tokens no longer reread
a mutable source slot, but an access-time token-visibility gap remains.  Fix
that clean-input race before claiming overhead or repeating fault acceptance.

## Green requirements

A green workload/engine cell requires retained evidence that:

1. ordinary `standard-v1` defaults instrument every admitted supported site;
2. the uninjected workload passes an independent oracle;
3. static and dynamic completeness are explicit, with typed exclusions and no
   hidden coverage-limiting or workload-tuning controls;
4. each admitted mutation reaches exactly one reviewed final byte sequence and
   its diagnostic or precommitted qualified miss matches the engine contract;
5. execution terminates within its bound and the device remains healthy;
6. paired baseline/instrumented latency and instrumentation-owned peak memory
   are retained; and
7. commands, environment, hashes, source identity, hook identity, target, and
   artifact directory refer to one frozen checkpoint.

## Reproduction

Start by validating the workspace and inspecting the executable manifest:

```sh
export CONSAN_VALIDATION_TARGET=gfx1201

python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" doctor

python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py \
  --target "$CONSAN_VALIDATION_TARGET" manifest --json
```

[VALIDATION.md](VALIDATION.md) defines clean, paired-overhead, inventory,
reviewed-fault, containment, and provenance execution.  [DESIGN.md](DESIGN.md),
[SPILLING.md](SPILLING.md), and [MALFORMED_INPUT.md](MALFORMED_INPUT.md)
describe implementation and safety boundaries.

Update this file whenever evidence changes a cell.  Keep the matrix focused on
the current result; retain detailed investigation notes only while they help
the next person close a non-green cell.
