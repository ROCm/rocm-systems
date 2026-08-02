# ConSan RDNA4 (`gfx1201`) status

This is the current workload × instrumentation ledger for `gfx1201`.  It does
not inherit colors, selectors, denominators, or fault expectations from another
architecture.  End-to-end LLM workloads are the primary project metric;
focused kernels are retained as compact regression discriminators.

The executable authority is
[`consan_validation.py`](../../tests/dbi/consan/consan_validation.py), and
[VALIDATION.md](VALIDATION.md) defines the experiment contract. Individual
cells cite their retained artifact provenance. The current large-object
SuperCollider, Sampled, and Inline Shadow paired rows use hook SHA-256
`2113c773bdcac837e58cbd0cc0e784d7d99d971de3a8e08c377743cd245f8e6e`;
the separately retained Record/Replay row uses the immediately preceding hook
build and is unaffected by the Sampled-only scalar-layout refactor. This is not
yet a frozen release checkpoint because the larger nightly top-k object's
reviewed-fault and containment gates remain to be refreshed.

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

Overhead cells are warm steady-state ratios unless they explicitly say
**cold first-operation**. Cold and warm ratios have different denominators and
must not be compared cell-to-cell.

A crash, trap, timeout, oracle mismatch, or device loss is not a ConSan
diagnostic.  A green cell may contain an honest qualified miss only when that
outcome was precommitted and the exact mutation, containment, independent
oracle, and coverage evidence are valid.

## Current matrix

19 of 21 workloads × 4 engines are assessed. The newly admitted production
FP16 and FP8 matmul rows are intentionally gray until the empirical campaign
tracked by `bd-2wsf` completes. Coverage counts use each engine's own
admitted-site model; in particular, Sampled reports split-barrier members where
other engines report logical barriers.

This checkpoint also admits native `ds_load_b96` and `ds_store_b96` on both
gfx1201 and gfx1250 across Record/Replay, Sampled, and Inline Shadow. gfx1100
and CDNA3/4 remain explicitly unsupported. This changes the static
supported-site denominator when a generated object contains those
instructions; the much larger top-k counts also reflect a changed nightly
object and are not attributed to B96 admission alone. The current top-k
diagnostic classification must nevertheless disassemble the retained sites and
rule in or out B96 admission/lowering as a third hypothesis alongside a rocPRIM
race and a generic ConSan model false positive.

| Priority workload | SuperCollider | Record/Replay | Sampled | Inline Shadow |
|---|---|---|---|---|
| **P0 production FP16 RDNA4 matmul** | 🩶 Empirical admission pending (`bd-2wsf`) | 🩶 Empirical admission pending (`bd-2wsf`) | 🩶 Empirical admission pending (`bd-2wsf`) | 🩶 Empirical admission pending (`bd-2wsf`) |
| **P0 production FP8 RDNA4 matmul** | 🩶 Empirical admission pending (`bd-2wsf`) | 🩶 Empirical admission pending (`bd-2wsf`) | 🩶 Empirical admission pending (`bd-2wsf`) | 🩶 Empirical admission pending (`bd-2wsf`) |
| **P0 Qwen3-0.6B prefill** | 🟩 Exact oracle; clean 20/20; exact barrier drop emits an attributed diagnostic and breaks the oracle; overhead 1.044x | 🟩 Exact oracle; clean 20/20 + 14/14; exact drop is a precommitted qualified miss and breaks the oracle; overhead 1.004x | 🟩 Exact oracle; clean 20/20 + 26/26; deterministic 32-offset fault sweep detects 17/32 and every exact mutation breaks the oracle; overhead 1.003x | 🟩 Exact oracle; clean 20/20 + 14/14; exact barrier drop emits a diagnostic and breaks the oracle; overhead 1.005x |
| **P0 PyTorch torch.mode** | 🟥 Current 12,284,128-byte object is rejected during relay planning because SOPP relay coordinates are not globally unique | 🟩 Exact value/index oracle; 4/4 strict clean runs complete at 140,334/140,334 accesses + 21,002/21,002 barriers + 38/38 fences with zero diagnostics in 18.97–21.76 seconds | 🟧 Exact oracle; 131,326/140,334 accesses patch, with 9,008 placement/lowering failures and 4,640/4,640 admitted barrier members | 🟥 Complete inventory and report planning, then the large-object patch exceeds the ordinary 30-second bound |
| **P1 Sharktank TP1 prefill** | 🟩 Exact oracle; clean 352/352; exact drop is a precommitted qualified miss; overhead 1.26x | 🟩 Exact oracle; clean 352/352 + 46/46; exact drop emits a replay diagnostic; overhead 2.002x | 🟩 Exact oracle; clean 352/352 + 86/86; exact drop is a precommitted qualified miss; overhead 1.14x | 🟩 Exact oracle; clean 352/352 + 46/46; exact drop emits an attributed diagnostic; overhead 2.75x |
| **P1 Sharktank TP1 decode/combined** | 🟩 Exact oracle; clean 704/704; exact drop is a precommitted qualified miss; overhead 1.18x | 🟩 Exact oracle; clean 704/704 accesses + 92/92 barriers; exact drop emits replay diagnostics in 2/8 contained trials while the oracle is schedule-masked; overhead 1.542x combined / 1.517x decode | 🟩 Exact oracle; clean 704/704 + 172/172; exact drop is a precommitted qualified miss; overhead 1.01x | 🟩 Exact oracle; clean 704/704 + 92/92; exact drop emits an attributed diagnostic; overhead 1.70x |
| **P1 PyTorch collision-heavy scatter-reduce** | 🟩 Exact BF16/FP32 oracle; clean-complete 27/27 LDS accesses; exact global-atomic scope weakening is a precommitted qualified miss; overhead 1.010x | 🟩 Exact oracle; clean-complete 27/27 accesses; exact scope weakening is a precommitted qualified miss; overhead 1.006x (0.996x BF16) | 🟩 Exact oracle; clean-complete 27/27 accesses; exact scope weakening is a precommitted qualified miss; overhead 0.993x | 🟩 Exact oracle; clean-complete 27/27 accesses; exact scope weakening is a precommitted qualified miss; overhead 1.031x |
| **P2 PyTorch/Inductor compiled softmax** | 🟩 Exact oracle; clean 4/4; exact third barrier drop is a precommitted qualified miss; overhead 0.960x | 🟩 Exact oracle; clean 4/4 + 3/3; exact drop emits an attributed replay diagnostic; overhead 1.065x | 🟩 Exact oracle; clean 4/4 + 6/6 barrier members; exact drop emits a causal diagnostic; overhead 1.204x | 🟩 Exact oracle; clean 4/4 + 3/3; exact drop emits attributed diagnostics; overhead 0.937x |
| **P2 PyTorch split online softmax** | 🟩 Exact CPU-derived BF16 oracle; clean 8/8 across two stages; exact drop is a precommitted qualified miss; overhead 0.977x | 🟩 Exact oracle; clean 8/8 + 6/6; exact drop is a qualified replay miss; overhead 0.982x | 🟩 Exact oracle; clean 8/8 + 12/12 barrier members; exact drop emits a causal diagnostic; overhead 1.257x | 🟩 Exact oracle; clean 8/8 + 6/6; independently confirmed exact drop emits diagnostics; overhead 4.012x |
| **P2 PyTorch Qwen-vocabulary top-k** | 🟨 Current exact clean and paired rows are complete at 58,992/58,992 accesses; clean execution takes 98.587 seconds and cold first-operation slowdown is 448.435x; current reviewed-fault refresh pending | 🟨 Current exact clean and paired rows are complete at 418,292/418,292 accesses plus 100,916/100,916 barriers with zero diagnostics; clean execution takes 108.626 seconds and cold first-operation slowdown is 433.798x; current reviewed-fault refresh pending | 🟨 Current exact clean and paired rows are complete at 418,292/418,292 accesses plus 100,916/100,916 barrier members; clean execution takes 96.725 seconds and cold first-operation slowdown is 416.660x; current reviewed-fault refresh pending | 🟨 Current exact clean and paired rows are complete at 418,292/418,292 accesses plus 50,458/50,458 barriers with zero incomplete state; clean execution takes 146.873 seconds and cold first-operation slowdown is 461.646x; current reviewed-fault refresh pending |
| **P2 llama.cpp quantized matvec** | 🟩 Independent CPU oracle; clean-complete 462/462 accesses; reviewed exact drop is a qualified miss; overhead 19.225x | 🟩 Independent CPU oracle; clean-complete 462/462 accesses + 44/44 barriers + 63/63 atomics + 72/72 fences; reviewed drop breaks the oracle; overhead 14.493x | 🟩 Independent CPU oracle; clean-complete 462/462 accesses + 88/88 barrier members; reviewed drop breaks the oracle; overhead 17.548x | 🟧 Exact oracle and zero diagnostics; 81/462 accesses + 44/44 barriers, but 49,152 unsupported dynamic events reject strict execution |
| **P2 Sharktank TP2 family** | 🟩 Exact oracle; clean 2,976/2,976; exact drop is a precommitted qualified miss; overhead 1.28x | 🟨 Exact oracle; five clean-complete trials at 2,976/2,976 + 228/228 with no diagnostics; exact drop detected in 3/5 contained trials; overhead 1.806x prefill / 1.319x combined / 1.270x decode | 🟩 Exact oracle; clean 2,976/2,976 + 420/420; exact drop is a precommitted qualified miss; overhead 1.24x | 🟩 Exact oracle; clean 2,976/2,976 + 228/228; exact drop detected 16/16; overhead 2.17x |
| **P3 CLIP BF16** | 🟩 Exact oracle; clean 85/85; exact drop and move are precommitted qualified misses; overhead 0.98x | 🟩 Exact oracle; clean 85/85 + 36/36; exact drop and move are qualified misses; overhead 1.380x | 🟩 Exact oracle; clean 85/85 + 72/72; exact drop and move are qualified misses; overhead 0.97x | 🟩 Exact oracle; clean 85/85 + 36/36; exact move emits a diagnostic and drop is a qualified miss; overhead 1.51x |
| **P3 PyTorch native histogram** | 🟩 Exact oracle; clean-complete 135/135; exact drop is a qualified miss; overhead 0.966x | 🟩 Exact oracle; clean-complete 135/135 + 84/84; exact initialization drop breaks the oracle and is a qualified miss; overhead 1.041x | 🟩 Exact oracle; clean-complete 135/135 + 168/168 barrier members; exact drop is schedule-masked; overhead 1.094x | 🟩 Exact oracle; clean-complete 135/135 + 84/84; exact drop breaks the oracle and is a qualified miss; overhead 5.301x |
| **P3 llama.cpp RMS norm** | 🟨 Exact CPU oracle; clean-complete 22/22 accesses; overhead 1.194x; reviewed effective fault pending | 🟨 Exact oracle; clean-complete 22/22 + 11/11; overhead 1.327x; reviewed effective fault pending | 🟨 Exact oracle; clean-complete 22/22 + 22/22 barrier members; overhead 1.404x; reviewed effective fault pending | 🟨 Exact oracle; clean-complete 22/22 + 11/11; overhead 1.375x; reviewed effective fault pending |
| **P4 hip-moi D128 block attention** | 🟩 Exact oracle; clean 12/12; exact drop breaks the oracle and is a qualified miss; overhead 164.77x | 🟩 Exact oracle; clean 12/12 + 4/4; exact drop emits a diagnostic and breaks the oracle; overhead 13.216x | 🟩 Exact oracle; clean 12/12 + 8/8 barrier members; exact drop breaks the oracle and is a qualified miss; overhead 14.03x | 🟩 Exact oracle; clean 12/12 + 4/4; exact drop emits a diagnostic and breaks the oracle; overhead 12.83x |
| **P4 hip-moi D128 pressure attention** | 🟩 Exact oracle; clean 12/12; exact drop breaks the oracle and is a qualified miss; overhead 11.25x | 🟨 Exact oracle; bounded-capture candidate is clean-complete at 12/12 + 4/4 in 5/5 trials and diagnoses the exact drop with oracle failure in 5/5; frozen-tip overhead and memory requalification pending | 🟩 Exact oracle; clean 12/12 + 8/8 barrier members; confirmed exact drop emits a diagnostic and breaks the oracle; overhead 18.531x | 🟩 Exact oracle; clean 12/12 + 4/4; exact drop emits a diagnostic; overhead 13.72x |
| **P4 hip-moi WMMA attention** | 🟩 Exact oracle; clean 12/12; exact drop breaks the oracle and is a qualified miss; overhead 158.07x | 🟩 Exact oracle; clean 12/12 + 4/4; exact drop emits a diagnostic and breaks the oracle; overhead 13.856x | 🟩 Exact oracle; clean 12/12 + 8/8 barrier members; exact drop breaks the oracle and is a qualified miss; overhead 14.50x | 🟩 Exact oracle; clean 12/12 + 4/4; exact drop emits a diagnostic and breaks the oracle; overhead 13.24x |
| **P4 hip-moi Stream-K arrival** | 🟩 Exact oracle; clean 4/4; exact order/scope weakenings are qualified misses; overhead 558.83x | 🟩 Exact oracle; clean 4/4 + 15/15 atomics + 4/4 barriers + 16/16 fences; exact weakenings are qualified misses; overhead 48.889x | 🟩 Exact oracle; clean 4/4 + 15/15 atomics + 8/8 barrier members; exact weakenings are qualified misses; overhead 48.21x | 🟩 Exact oracle; clean 4/4 + 15/15 atomics + 4/4 barriers; exact weakenings emit diagnostics; overhead 50.93x |
| **P4 hip-moi tree atomic-OR** | 🟩 Exact oracle; clean 4/4; exact order/scope weakenings are qualified misses; overhead 591.81x | 🟩 Exact oracle; clean 4/4 + 15/15 atomics + 4/4 barriers + 16/16 fences; order weakening is a qualified miss and scope weakening emits a replay diagnostic; overhead 50.635x | 🟩 Exact oracle; clean 4/4 + 15/15 atomics + 8/8 barrier members; exact weakenings are qualified misses; overhead 49.91x | 🟩 Exact oracle; 10/10 current-checkpoint processes clean-complete at 4/4 accesses + 15/15 atomics + 4/4 barriers; relaxed producer bits emit the expected diagnostic |
| **P4 hip-moi Jakub attention variants** | 🟩 Exact oracle; clean 31/31; exact drop is a qualified miss; overhead 103.75x | 🟩 Exact oracle; clean 31/31 + 4/4; exact drop is a qualified miss; overhead 10.255x | 🟩 Exact oracle; clean 31/31 + 8/8 barrier members; exact drop is a qualified miss; overhead 10.87x | 🟩 Exact oracle; clean 31/31 + 4/4; exact drop emits a diagnostic; overhead 11.03x |

## Non-green handoff

Only current blockers and the most useful retained evidence are recorded here.
Do not raise a timeout or change a fault expectation merely to promote a cell.
The source-matched Record/Replay campaigns for this checkpoint are
`rdna4-rr-final-clean-20260722`, `rdna4-rr-final-faults-20260722`, and
`rdna4-rr-final-overhead-20260722`.  Those historical 19-workload campaigns use
a different set: they predate `torch.mode` and include the now-retired SDPA
row. They have all 19 clean and all 57 paired-overhead rows passing; the
reviewed-fault campaign accepts 17/19 rows.
`rdna4-rr-final-faults-20260722` predates
accounting schema v2 and its installation-evidence records, so it remains
historical evidence but requires a rerun for current qualification rather than
re-parsing.  The subsequently added `torch.mode` row is qualified separately
below.

### PyTorch torch.mode

- **Record/Replay:** the current ordinary `torch.mode` operation loads a
  12,284,128-byte object containing 2,250 kernels.  Four separate physical-host
  runner invocations, each under the explicit 30-second manifest bound, pass
  the exact value/index oracle, patch all 140,334 accesses, 21,002 barriers,
  and 38 fences, and emit zero diagnostics.  This is the large-object clean
  gate.
- **Other engines:** SuperCollider hits a duplicate-coordinate relay-planning
  error, Sampled leaves 9,008 accesses unpatched, and Inline Shadow exceeds
  the ordinary bound during patching.  These general large-object gaps are
  tracked by `bd-1w9.6.4`; do not special-case the workload or relax strict
  coverage.

### PyTorch Qwen-vocabulary top-k

- **Current clean gate:**
  `consan-validation-large-objects-gfx1201-topk-clean-0bf1d17-20260731`
  passes the exact sorted value/index oracle in every engine. SuperCollider
  patches 58,992/58,992 accesses. Record/Replay and Sampled each patch
  418,292/418,292 accesses plus 100,916/100,916 barriers; Inline Shadow patches
  418,292/418,292 accesses plus 50,458/50,458 barriers. Every row is statically,
  dynamically, and analytically complete. The current object is pinned as
  `fnv1a64:3833562345afa454`; the former 63,474-access SuperCollider denominator
  belonged to a different retained object, so it is not a partial-coverage
  denominator for this row.
- **Cold paired gate:** each physical profile uses ten fresh one-repetition
  processes so a sample has one bounded report-buffer lifetime. The retained
  artifacts are
  `consan-validation-large-objects-gfx1201-topk-sc-overhead-cold-final-0bf1d17-20260731`,
  `consan-validation-large-objects-gfx1201-topk-rr-overhead-isolated-v2-0bf1d17-20260731`,
  `consan-validation-large-objects-gfx1201-topk-sampled-overhead-cold-final-0bf1d17-20260731`,
  and
  `consan-validation-large-objects-gfx1201-topk-inline-overhead-cold-final-0bf1d17-20260731`.
  All 40 instrumented processes pass their exact oracle and strict coverage
  contract. The reported 448.435x, 433.798x, 416.660x, and 461.646x slowdowns
  are cold end-to-end first-operation ratios and include object transformation.
- **Sampled state preservation:** the selected sampled access body restores
  SCC from its planner-owned post-body snapshot rather than the publication
  EXEC scratch. Compact physical regressions cover both the expanded return
  shapes and an SCC-dependent branch after a staged LDS completion sequence.
  The exact-site artifact
  `consan-validation-topk-sampled-scc-fix-0bf1d17-20260731` and the standard
  clean/paired artifacts above pass without workload-specific placement or
  result exceptions.
- **Remaining maturity gate:** current reviewed-fault and containment evidence
  has not been refreshed for this larger nightly object, so all four cells stay
  yellow rather than inheriting green from older denominators.

### Retired PyTorch causal SDPA

Tracking: `bd-1w9.26`.

The July 21 TheRock nightly reproduces the default fused-backend failure on
physical gfx1201 before ConSan is loaded: the maximum error is
`2.407407522201538`. Explicit fused backend selections are also incorrect for
this case, and forcing the math backend is not an acceptable sanitizer gate.
A correct compiled decomposition was rejected as a replacement because its
generated kernels expose no sanitizer-visible barrier sites.

The invalid PyTorch row is no longer executable qualification evidence. The
target-native `d128-block`, `d128-pressure`, and `wmma-attention` rows replace
its exact-oracle attention LDS/barrier mechanism coverage, but gfx1201 now has
no real-framework causal-attention row. The older SDPA artifacts remain
historical diagnostics only and do not establish current acceptance. If a
future nightly restores a correct default fused backend, reinstate the row from
commit `1003eba026` and requalify it before treating it as current evidence.

### Sharktank TP2 family

The retained checkpoint is clean-complete and false-positive-free in five
consecutive trials, but diagnoses the reviewed exact barrier drop in only three
of five contained trials.  Artifacts: `rdna4-rr-final-tp2-repeat-{2..5}-20260722` and
`rdna4-rr-final-fault-repeat-tp2-{2..5}-20260722`, in addition to the main
clean and fault campaigns above.

The current candidate replaces the lifetime-static slot with a bounded
4-dispatch × 4-owner capture while keeping direct caller layouts unchanged.
Its host, model, emission, gfx1250, gfx942, gfx950, and physical gfx1201 D128
gates pass.  TP2 could not be requalified because neither `iree-test-suites`
nor `iree-test-suites-build` is present in this workspace.  Retain the
historical 3/5 result until those dependencies are restored; do not infer TP2
acceptance from the smaller D128 workload.

### llama.cpp quantized matvec

Inline Shadow is the only non-green engine.  The 1,024-element shape passes its
independent GPU/CPU oracle with zero diagnostics.  Preventing access entry
relays from consuming adjacent synchronization sites restores all 44/44
barriers; access placement remains incomplete at 81/462, with 49,152
unsupported dynamic events.  Artifact:
`gfx1201-inline-owner-token-sync-guard-20260726`.  The remaining blocker is
access placement, not ownership correctness.

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

The retained release checkpoint is clean-complete and its exact barrier drop
breaks the independent oracle, but five consecutive contained trials produce
no replay diagnostic.  Artifacts: `rdna4-rr-final-faults-20260722` and
`rdna4-rr-final-fault-repeat-pressure-{2..5}-20260722`.

The bounded-capture candidate closes that detection gap:
`consan-validation-gfx1201-clean-owner-banks-final-20260725-{1..5}` is accepted
with 12/12 accesses, 4/4 barriers, complete dynamic evidence, and no
diagnostics;
`consan-validation-gfx1201-fault-owner-banks-final-60s-5x-20260725` diagnoses
the exact drop with oracle failure in all five contained trials.  Both use hook
SHA-256
`07bba71e8826d33fa6328b2e79268df926f7d8b2e189bb2a0cf6e95635fc0839`.
The fault campaign retains a 60-second bound; an immediately preceding
30-second campaign accepted four trials before one trial timed out during
transform startup with `requested=planned=applied=0`, and both post-timeout
health probes passed.  Keep the cell yellow until a frozen checkpoint also
carries paired overhead and memory evidence.

## Resolved Inline Shadow ownership checkpoint

The tree atomic-OR workload is clean-complete in 10/10 final physical `gfx1201`
processes with 4/4 accesses, 15/15 atomics, and 4/4 barriers.  The relaxed
producer-bits discriminator still emits one exact access diagnostic.  An
earlier 20/20 clean campaign exercised deferred teardown qualification in four
processes, proving that a stable token published after the device-side
diagnostic is qualified only when its dispatch, workgroup, owner direction,
producer epoch, and acquire-established consumer segment all match.  A later
acquire therefore cannot retroactively suppress an earlier access diagnostic;
incomplete token snapshots and saturated consumer epochs remain fail-closed.

The implementation also keeps access placement independent from
synchronization placement: a far access entry relay cannot relocate a decoded
barrier, atomic, or fence as an uninstrumented tail.  This removes the llama
clean-input ownership diagnostic while leaving its incomplete access count
explicit.  No workload instruction offsets, owner IDs, register numbers, or
fixed runtime LDS sizes participate in either decision.

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
