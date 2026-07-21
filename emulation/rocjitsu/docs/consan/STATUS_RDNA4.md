# ConSan RDNA4 status

| Priority workload | SuperCollider | Record/Replay | Sampled | Inline Shadow |
| --- | --- | --- | --- | --- |
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

This is the current gfx1201 result ledger as of 2026-07-16. End-to-end LLM
prefill and decode outrank isolated kernels. Every admitted workload has source
or artifacts in the validation workspace and a concrete runnable definition.

The one-tip release campaign for executable commit `640e575da2`, using rebuilt
hook SHA-256 `c45aa0fece5a9aa7`, accepts all 55 clean baseline/profile rows, all
14 exact fault policies, and all 66 paired-overhead result rows across the 11
workloads. The Qwen barrier drop remains the primary e2e sensitivity result:
SuperCollider detects 1/1, Record/Replay records its qualified miss, Sampled
detects 17/32 in the stride-256 sweep, and Inline Shadow detects 1/1.

The five temporary reds exposed by the earlier final-tip campaign were not
introduced by the code-sharing refactor. They were stronger evidence against
claims accumulated at older checkpoints. Full-site execution uncovered two
RDNA4 encoders that set reserved operand bits and an entry relay that separated
`s_clause` from its memory-instruction run; focused fixes plus the exact-tip
rerun restored TP2 and tree-atomic-OR clean qualification. Parser, Sampled
selection, and Inline evidence fixes restored the three Qwen fault contracts.
The complete rerun also replaced stale conservative SuperCollider misses with
observed detections for TP1 prefill, TP1 decode, TP2, and both CLIP mutations;
it quantified TP2 Inline as an 8/16 statistical detection result and recorded
the schedule-dependent D128-pressure Inline and Jakub Sampled oracle outcomes
rather than turning bookkeeping or oracle failure into a diagnostic.

## What green means

A green cell is a provenance-bound clean-and-fault experiment, not merely a
successful process launch. It establishes that:

1. ordinary defaults instrument every admitted supported relevant site;
2. the uninjected workload passes its independent result oracle;
3. static and dynamic completeness are explicit and accepted;
4. every admitted mutation is proven to reach executed final bytes and its
   observed diagnostic, qualified miss, or typed non-applicability matches the
   precommitted flavor contract;
5. execution terminates within its bound and the GPU remains healthy;
6. no-fault baseline and instrumented latency produce the displayed slowdown
   factor, and instrumentation-owned peak memory is retained; and
7. commands, identities, hashes, and evidence are recorded.

Green does not mean that every flavor detects every injected fault. A bounded
Record/Replay snapshot, a probabilistic Sampled run, or a schedule-masked
mutation can be green with an honest, reproducible qualified miss. A timeout,
crash, output mismatch, or GPU reset is never counted as a diagnostic.

The Qwen Sampled clean result uses automatic `standard-v1` defaults: runtime
stride 16,384 and offset zero, with no workload-specific sampling setting. The
separately declared stride-256, 32-offset fault-sensitivity experiment supplies
the retained statistical claim and detects 16/32 on the current executable.
Those controls are experiment parameters, not clean-workload tuning.

The large slowdown factors on short P4 micro-workloads are real measurements,
not representative LLM overhead claims. The P0–P2 end-to-end rows are the
primary usability result.

Each cell is bound to the same exact committed hook and final executable used
for this release campaign. Improvements beyond this completed gfx1201
certificate are tracked in [FUTURE_WORK.md](FUTURE_WORK.md).

## Reproduction and evidence

- [VALIDATION.md](VALIDATION.md) describes the executable experiment contract;
  `consan_validation.py` owns commands, ordinary settings, faults, expectations,
  overhead calculation, provenance, and device-health gates.
- [DESIGN.md](DESIGN.md), [SPILLING.md](SPILLING.md), and
  [MALFORMED_INPUT.md](MALFORMED_INPUT.md) describe the implementation and its
  safety boundaries.
- [FUTURE_WORK.md](FUTURE_WORK.md) tracks improvements beyond this completed
  status matrix.

Any regression or stronger contrary evidence must immediately move an affected
cell out of green. New workloads are not added until their assets and runnable
validation contract exist in the workspace.
