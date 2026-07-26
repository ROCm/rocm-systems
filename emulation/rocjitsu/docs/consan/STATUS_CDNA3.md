# ConSan CDNA3 (`gfx942`) status

This is the `gfx942` workload × instrumentation evidence ledger. It uses the
same acceptance contract as the [gfx950](STATUS_CDNA4.md),
[gfx1201](STATUS_RDNA4.md), and [gfx1250](STATUS_GFX1250.md) ledgers, but it
inherits no artifact, coverage denominator, machine-code identity, fault
expectation, timing, provenance, or green cell from another target.

The executable authority is
[`consan_validation.py`](../../tests/dbi/consan/consan_validation.py), and the
experiment contract is described by [VALIDATION.md](VALIDATION.md). Runtime LDS
capacity comes from the active runtime agent. RocJITsu JSON configuration is
authoritative only for simulator and offline tests.

End-to-end evidence is the primary metric. Decoder, builder, spill, and
RocJITsu simulator tests are prerequisites and debugging tools; they cannot
promote a cell in this ledger.

## Status legend

- 🩶 **unseen / unassessed:** no qualifying current-tip gfx942 workload
  evidence. A cell labeled **simulator prerequisite only** remains gray: the
  target-native build and simulator oracle pass, but physical qualification is
  still pending;
- 🟥 **does not work:** current target-native execution establishes a product
  defect;
- 🟧 **some things work:** useful execution evidence exists but major
  correctness, completeness, or acceptance gaps remain;
- 🟨 **most things work:** clean execution and the important instrumentation
  path work, with limited final evidence missing;
- 🟩 **everything works:** one frozen revision retains clean correctness,
  complete coverage, a reviewed fault outcome, containment, paired overhead,
  bounded memory and time, health, cleanup, and provenance.

`N/A` is allowed only after a fresh gfx942 inventory proves semantic absence
and records a typed reason.

## Current matrix

Every selected workload remains gray. The validator resolves its six gfx942
gtest campaign roles only to explicitly named CDNA3 artifacts, and all six
artifacts pass their target-specific suites under the gfx942 simulator. The
broader offline gate also runs the seven register-handoff, no-score, ping-pong,
and LDS-alias suites, for 13 binaries and 33 tests in total. That is a build and
simulator prerequisite, not physical target qualification. The other five
campaign workloads have not completed a current-tip gfx942 campaign. No status
below is promoted from a simulator smoke.

| Workload | SuperCollider | Record/Replay | Sampled | Inline Shadow |
|---|---|---|---|---|
| **P0 Qwen3-0.6B prefill** | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| **P1 Sharktank TP1 prefill** | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| **P1 Sharktank TP1 decode + combined** | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| **P2 Sharktank TP2 family** | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| **P3 CLIP BF16** | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed | 🩶 unassessed |
| **P4 hip-moi D128 block attention** | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |
| **P4 hip-moi D128 pressure attention** | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |
| **P4 hip-moi MFMA attention** | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |
| **P4 hip-moi Stream-K arrival** | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |
| **P4 hip-moi tree atomic-OR** | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |
| **P4 hip-moi Jakub attention** | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only | 🩶 simulator prerequisite only |

## Executable contract

The non-gtest rows use the same target-independent source roles as the other
ledgers, but their VMFBs and executions must still identify gfx942. The six
gtest campaign roles deliberately require these target-native definitions:

| Validation ID | Required gfx942 definition |
|---|---|
| `d128-block` | `hip_moi_instrumented_cdna3_d128_attention_block_test`; `HipMoiCdna3D128AttentionBlock.*` |
| `d128-pressure` | `hip_moi_instrumented_cdna3_d128_attention_pressure_test`; `HipMoiCdna3D128AttentionPressure.*` |
| `wmma-attention` | `hip_moi_instrumented_cdna3_mfma_attention_block_test`; `HipMoiCdna3MfmaAttentionBlock.*` |
| `streamk-arrival` | `hip_moi_instrumented_cdna3_mfma_streamk_arrival_counter_test`; ordering-oracle test only |
| `tree-atomic-or` | `hip_moi_instrumented_cdna3_mfma_streamk_tree_atomic_or_test`; ordering-oracle test only |
| `jakub-attention` | `hip_moi_reference_cdna3_jakub_matmul`; `SafeFp16Packed/JakubCdna3MatmulReference.MatchesHostReference/*` |

At hip-moi source revision `29a1c212183b`, the validator's workload-scoped
`doctor` command resolves all six exact paths directly under the retained
`hip-moi-build-gfx942-tests` build. It must never substitute a mutable generic
build link or an RDNA4 or CDNA4 executable.

## Retained simulator prerequisites

Manual runs through
`rocjitsu --config emulation/rocjitsu/configs/gfx942_cdna3_kmd.json` pass all
33 tests in the 13 target-native hip-moi binaries, including binaries from a
build configured for `gfx942:xnack-`. RocJITsu exposes each binary as an
independent bounded CTest entry from the shared offline suite registry.

Separately, the focused in-repo CTest gate exercises gfx942 simulator coverage
for:

- SuperCollider clean execution;
- flat-LDS and MOI Record/Replay forced spilling;
- Sampled, Inline Shadow, shared-helper, barrier, and atomic forced spilling;
- Inline Shadow dynamic-stack forced spilling.

These tests establish native CDNA3 rewriting and bounded simulator execution.
They are not real-workload qualification and do not change the gray matrix.

## Promotion checklist

Promote one cell only from a retained result bundle at one revision containing:

1. target and executable identity proving gfx942 code;
2. an independent clean correctness oracle;
3. complete static and dynamic ConSan coverage for every applicable object;
4. a reviewed exact fault with diagnosis, or a documented qualified miss;
5. baseline-before, instrumented, and baseline-after paired timing;
6. containment, timeout, memory, cleanup, and health results; and
7. source, hook, workload, runtime, toolchain, command, and environment
   provenance.

The CDNA3 hip-moi artifact enablement is complete and retained under
`bd-1w9.1.1`. Physical gfx942 execution and four-engine result bundles remain
open qualification work; this ledger continues to fail closed rather than
infer those results from simulator or other-target evidence.
