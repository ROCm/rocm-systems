# ConSan CDNA4 (`gfx950`) status

This is the native `gfx950` workload × instrumentation evidence ledger.  It
follows the acceptance standard of [STATUS_RDNA4.md](STATUS_RDNA4.md) and the
expanded corpus structure of [STATUS_GFX1250.md](STATUS_GFX1250.md), but it
inherits no coverage denominator, machine-code identity, fault expectation,
timing, provenance, or green cell from another architecture.

The executable authority is
[`consan_validation.py`](../../tests/dbi/consan/consan_validation.py), with the
experiment contract described by [VALIDATION.md](VALIDATION.md).  This status
matrix is the authoritative progress tracker for gfx950.

End-to-end evidence is the primary project metric.  Focused decoder, builder,
spill, and resource tests are prerequisites and debugging tools; they cannot
promote a workload cell by themselves.

All benchmark and workload measurements use exactly one repetition
(`--benchmark_repetitions=1`, or `--repetitions 1` for the PyTorch runner).

## Status legend

Every cell uses the same maturity scale as the other architecture ledgers:

- 🩶 **unseen / unassessed:** no useful current-tip gfx950 execution evidence
  yet;
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

`N/A` is used only when a fresh gfx950 inventory proves semantic absence and
records a typed reason.

## Current matrix

Every cell began gray at the current branch tip because the implementation and
workload changed substantially since the first gfx950 campaign.  Text
following a gray heart records historical evidence or a known prerequisite
only.  The first current-tip D128 runs below promote only the profiles they
actually revalidated; every other profile must still be revalidated from
scratch before promotion.

| Workload | SuperCollider | Record/Replay | Sampled | Inline Shadow |
|---|---|---|---|---|
| **P0 Qwen3-0.6B prefill** | 🟩 Current exact clean and paired rows are complete at 628/628 accesses; clean execution takes 182.176 seconds and paired slowdown is 1.34x. A prospectively reviewed exact-one drop of the final output-store convergence barrier is reached and accepted as `not_detected/pass`, with the exact expected output, zero diagnostics, bounded teardown, containment, health, hook hashing, and current provenance | 🟩 Current clean-source exact oracle, complete 658/658 access plus 46/46 barrier coverage, zero diagnostics, and 2.08x paired slowdown; a prospectively reviewed exact-one reader-retirement barrier drop fails the oracle and emits 120,034 Record/Replay diagnostics with bounded memory, cleanup, containment, health, and clean provenance | 🟩 Current exact clean and paired rows are complete at 628/628 accesses plus 51/51 barrier members; clean execution takes 583.067 seconds and paired slowdown is 1.70x. Current-tip targeted revalidation covers the selected output matmul's 76/76 accesses and 4/4 barriers; a prospectively reviewed exact-one reader-retirement drop preserves the exact oracle, emits 32 Sampled conflicts with 76/76 access and 3/3 surviving-barrier coverage, and passes bounded cleanup, containment, health, hook hashing, and current provenance | 🟨 Current exact clean and paired rows are complete at 628/628 accesses plus 52/52 barriers with zero incomplete state; clean execution takes 2364.519 seconds and paired slowdown is 21.75x; current reviewed-fault and containment refresh pending |
| **P1 Sharktank TP1 prefill** | 🟩 Current clean-revision exact oracle in 3.62 seconds with complete 120/120 access coverage; prior accepted paired and reviewed-fault bundle retained | 🟩 Current clean-revision exact oracle in 171.29 seconds with complete 120/120 accesses plus 31/31 barriers, 9,216 visible events, zero diagnostics, and a complete dynamic verdict under the target's bounded validation stride; prior accepted paired and reviewed-fault bundle retained | 🟩 Current clean-revision exact oracle in 3.92 seconds with complete 120/120 accesses plus 24/24 applicable barriers, zero diagnostics, and a complete dynamic verdict | 🟩 Current clean-revision exact oracle in 16.77 seconds with complete 120/120 accesses plus 31/31 barriers and a complete dynamic verdict; prior accepted paired and reviewed-fault bundle retained |
| **P1 Sharktank TP1 decode/combined** | 🟩 Current clean-revision exact decode/combined oracles in 3.35 seconds with complete 240/240 access coverage; prior accepted paired and reviewed-fault bundle retained | 🟩 Current clean-revision exact oracles in 166.03 seconds with complete 240/240 accesses plus 62/62 barriers, 11,520 visible events, zero diagnostics, and a complete dynamic verdict under the target's bounded validation stride; prior accepted paired and reviewed-fault bundle retained | 🟩 Current clean-revision exact oracles in 4.44 seconds with complete 240/240 accesses plus 48/48 applicable barriers, zero diagnostics, and a complete dynamic verdict | 🟩 Current clean-revision exact oracles in 29.67 seconds with complete 240/240 accesses plus 62/62 barriers and a complete dynamic verdict; prior accepted paired and reviewed-fault bundle retained |
| **P2 Sharktank TP2 family** | 🟩 current accepted bundle: all three exact clean and paired oracles, complete 936/936 access coverage, reviewed exact-one attention publish/read barrier fault with one instability diagnosis, bounded execution, cleanup, health, and clean provenance | 🟩 current accepted bundle: all three exact clean and paired oracles, complete 936/936 accesses plus 168/168 barriers, 1.57x combined paired slowdown, reviewed exact-one DPP-phase qualified miss, bounded execution, cleanup, health, and clean provenance | 🟩 Current-source physical bundle passes all three exact prefill, decode, and combined oracles with complete 1524/1524 access plus 150/150 applicable-barrier coverage and a 1.69x maximum paired slowdown; a prospectively frozen exact-one DPP-phase barrier drop is reached and accepted as `not_detected/pass` with 149/149 surviving barriers, 240 sampled windows, bounded memory, cleanup, containment, health, hook hashing, and exact provenance | 🟩 current VCC-safe spill-backed bundle: all three exact clean and paired oracles, complete 936/936 accesses plus 168/168 barriers, 167.0x maximum slowdown, reviewed exact-one fail/no-diagnosis fault, containment, health, and clean provenance |
| **P3 CLIP BF16** | 🟩 Current clean-revision cosine oracle in 1.65 seconds with complete 45/45 access coverage; the compact B128 auto-report regression at the ordinary-VGPR/AccVGPR boundary is fixed; prior accepted paired and reviewed-fault bundle retained | 🟩 Current clean-revision cosine oracle in 161.03 seconds with complete 45/45 accesses plus 24/24 barriers, zero diagnostics, and a complete dynamic verdict under the target's bounded validation stride; prior accepted paired and reviewed-fault bundle retained | 🟩 Current clean-revision cosine oracle in 1.80 seconds with complete 45/45 accesses plus 24/24 applicable barriers, zero diagnostics, and a complete dynamic verdict; prior accepted paired and reviewed-fault bundle retained | 🟩 Current clean-revision cosine oracle in 9.99 seconds with complete 45/45 accesses plus 24/24 barriers and a complete dynamic verdict; prior accepted paired and reviewed-fault bundle retained |
| **P4 hip-moi D128 block attention** | 🟩 current accepted bundle: exact clean, 12/12 coverage, paired overhead, reviewed exact-one fault, containment, health, and clean provenance | 🟩 post-rebase accepted bundle: exact oracle, complete 12/12 accesses plus 4/4 barriers, paired 16.78x, and a reviewed exact-one barrier drop that fails the oracle and produces one Record/Replay diagnosis; containment, health, cleanup, and clean provenance pass | 🟩 Current exact oracle, complete 128/128 access plus 117/117 applicable-barrier coverage, zero diagnostics, and 9.41x paired slowdown; a prospectively reviewed exact-one half of an adjacent FastContext barrier pair is reached and accepted as a qualified miss with bounded memory, cleanup, containment, health, and exact provenance | 🟩 current generation-qualified bundle: both exact clean oracles, zero diagnostics, complete 12/12 accesses plus 4/4 barriers, paired 15.76x, reviewed exact-one qualified miss, bounded memory and cleanup, containment, health, and clean provenance |
| **P4 hip-moi D128 pressure attention** | 🟩 Current physical paired row passes the exact full-KV oracle with complete 252/252 access coverage and complete static, analysis, and dynamic verdicts; the bracketing baselines take 79 and 77 ms versus 1,331 ms for SuperCollider (17.06x). A prospectively reviewed exact-one drop of the first of two consecutive FastContext barriers is reached and accepted as a `not_detected/pass` qualified miss with all four exact oracles passing, zero diagnostics, complete 252/252 access coverage, zero report-memory leakage, cleanup, containment, health, hook hashing, and exact provenance. The earlier prospectively frozen K/V-publication `not_detected/fail` hypothesis remains rejected without relabeling | 🟩 Current clean-source physical bundle passes all four exact oracles with complete 252/252 access and 28/28 barrier coverage, zero diagnostics, and complete static, analysis, and dynamic verdicts. The paired bracketing baselines take 66 and 85 ms versus 1,463 ms for Record/Replay (19.38x), well inside the ordinary 30-second bound. A distinct prospectively reviewed exact-one drop of the first of two consecutive FastContext barriers is reached and accepted as a qualified miss with a passing oracle, bounded memory, cleanup, and healthy pre/post probes | 🟩 Current exact oracle, complete 252/252 access plus 24/24 applicable-barrier coverage, zero diagnostics, and 17.32x paired slowdown; a prospectively reviewed exact-one first FastContext barrier drop leaves its adjacent barrier intact and is reached and accepted as a qualified miss with complete coverage, bounded memory, cleanup, containment, health, hook hashing, and exact provenance | 🟨 Current physical paired row passes the exact full-KV oracle with complete 252/252 access and 28/28 barrier coverage, zero forbidden diagnostics, and complete static, analysis, and dynamic verdicts; both bracketing baselines take 64 ms versus 1,483 ms for Inline Shadow (23.17x); reviewed-fault and clean-provenance acceptance remain |
| **P4 hip-moi MFMA attention** | 🟩 Current candidate-tree physical run passes both exact oracles in 0.23 seconds with complete 58/58 access coverage; CDNA4 B16 store readback is now covered by a focused host regression; prior paired and reviewed-fault evidence retained | 🟩 Current clean-source physical bundle passes the exact paired oracles with complete 58/58 access and 14/14 barrier coverage, zero diagnostics, and complete static, analysis, and dynamic verdicts; the paired slowdown is 4.22x. A prospectively reviewed exact-one score-publication barrier drop is reached and accepted as a qualified miss with a passing oracle, complete surviving coverage, cleanup, and healthy pre/post probes | 🟩 Current exact oracle, complete 58/58 access plus 12/12 applicable-barrier coverage, zero diagnostics, and 2.57x paired slowdown; a prospectively reviewed exact-one first score-publication barrier drop leaves its adjacent publication barrier intact and is reached and accepted as a qualified miss with complete surviving coverage, bounded memory, cleanup, containment, health, hook hashing, and exact provenance | 🟨 Current physical paired row passes the exact oracle with complete 58/58 access and 14/14 barrier coverage, 8,704 visible exact-shadow cells, zero forbidden diagnostics or incomplete state, and complete static, analysis, and dynamic verdicts; the bracketing baselines take 80 and 68 ms versus 246 ms for Inline Shadow (3.32x); reviewed-fault and clean-provenance acceptance remain |
| **P4 hip-moi Stream-K arrival** | 🟩 current accepted bundle: exact clean, 4/4 coverage, paired 143.70x, reviewed exact-one CDNA4 atomic-order fault, containment, health, and clean provenance | 🟩 frozen accepted bundle: exact clean and paired oracles, complete 4/4 accesses plus 4/4 barriers, 10/10 atomics, and 16/16 fences, zero diagnostics, 34.8x paired slowdown, and a reviewed exact-one release-order fault with pass/qualified-miss outcome, containment, health, cleanup, and clean provenance | 🟩 Current clean-revision accepted bundle: exact clean and paired oracles, complete 32/32 access plus 4/4 applicable-barrier and 1/1 atomic coverage, 7.57x paired slowdown, and a precommitted stride-1 exact-one release-order fault that preserves the oracle and produces four Sampled conflicts; containment, health, cleanup, hook hashing, and clean provenance pass | 🟩 Current clean-revision accepted bundle: exact clean and paired oracles, complete 32/32 access plus 6/6 barrier and 1/1 atomic coverage, 7.34x paired slowdown, and a precommitted exact-one release-order fault that preserves the oracle and produces 256 Inline Shadow diagnostics; containment, health, cleanup, hook hashing, and clean provenance pass |
| **P4 hip-moi tree atomic-OR** | 🟩 current accepted bundle: both exact clean tests, 4/4 coverage, paired 185.5x, reviewed exact-one producer-release atomic-order fault, containment, health, and clean provenance | 🟩 frozen accepted bundle: exact clean and paired oracles, complete 4/4 accesses plus 4/4 barriers, 10/10 atomics, and 16/16 fences, zero diagnostics, 47.08x paired slowdown, and a reviewed exact-one producer-release atomic-order fault with pass/no-diagnosis outcome, containment, health, and clean provenance | 🟩 Current exact MFMA-partials oracle, complete 48/48 access plus 4/4 barrier and 3/3 atomic coverage, zero clean diagnostics, and 15.89x paired slowdown; a prospectively selected exact-one weakening of the producer release at audited stride one preserves the oracle and produces 12 Sampled conflicts with complete surviving coverage, bounded memory, cleanup, containment, health, hook hashing, and exact provenance | 🟨 Current physical paired row passes the exact MFMA-partials oracle with complete 48/48 access, 6/6 barrier, and 3/3 atomic coverage, zero forbidden diagnostics, and complete static, analysis, and dynamic verdicts; the bracketing baselines take 60 and 98 ms versus 1,165 ms for Inline Shadow (14.75x); reviewed-fault and clean-provenance acceptance remain |
| **P4 hip-moi Jakub attention** | 🟩 Current physical paired bundle passes all four exact oracles with complete 338/338 access coverage and complete static, analysis, and dynamic verdicts; the bracketing baselines take 83 and 97 ms versus 266 ms for SuperCollider (2.96x). A prospectively reviewed exact-one PipelinedProd16x8 publication-barrier drop is reached and accepted under its frozen `not_detected/pass` schedule-masked policy, with all four exact oracles passing, zero diagnostics, complete coverage, exact hook/executable hashing, containment, and healthy pre/post probes | 🟩 Current clean-source physical bundle passes all four exact oracles with complete 338/338 access and 35/35 barrier coverage, zero diagnostics, and complete static, analysis, and dynamic verdicts. The paired bracketing baselines take 92 and 96 ms versus 411 ms for Record/Replay (4.37x). A distinct prospectively reviewed exact-one pipelined publication-barrier drop is reached and accepted under its schedule-masked qualified-miss policy, with 338/338 access and 34/34 surviving-barrier coverage, bounded memory, cleanup, and healthy pre/post probes | 🟩 Current exact four-oracle bundle, complete 338/338 access plus 35/35 applicable-barrier coverage, zero diagnostics, and 2.95x paired slowdown; a prospectively reviewed distinct pipelined publication-barrier drop is reached and accepted as a qualified miss with complete 338/338 access plus 34/34 surviving-barrier coverage, bounded memory, cleanup, containment, health, hook hashing, and exact provenance; the separate stride-1 producer-skew hypothesis remains rejected without relabeling | 🟨 Current physical paired row passes all four exact oracles with complete 338/338 access and 35/35 barrier coverage, zero forbidden diagnostics, and complete static, analysis, and dynamic verdicts; the bracketing baselines take 68 and 120 ms versus 299 ms for Inline Shadow (3.18x); reviewed-fault and clean-provenance acceptance remain |

### 2026-08-22 Qwen SuperCollider final-output qualification

Fresh current-object inventory at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-qwen-sc-inventory-v3`
records fingerprint `fnv1a64:8bc7e0c7622852d6`. The rebuilt ROCm runtime,
current Qwen VMFB, parameters, input, expected output, IREE tools, hook, and
physical gfx950 all pass the workload-scoped doctor. Final-ISA review of
`main$async_dispatch_562_batch_matmul_1x5x151936x1024_f32` selected occurrence
3 at code-object PC `0xae44`: every per-lane final global output-store path
rejoins at this unconditional barrier immediately before `s_endpgm`, with no
later memory consumer.

Before execution, artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-qwen-sc-final-output-v1`
froze `not_detected/pass` for that exact site. The mutation is requested,
planned, reserved, installed, and applied exactly once. The exact IREE expected
output passes, SuperCollider emits zero diagnostics, execution completes in
7.544 seconds without timeout, report-memory allocation and leakage remain
zero, and both pre/post physical-device discovery and dispatch probes pass.
The hook SHA-256 is
`7dff2c41b3038c3f9f8a42f7e317492491fbac15e1a78e5fb66068a3743bbb88`.
Provenance pins `iree-test-suites` at
`49f46d6d4370e5aa0a6367751474e20c6c4e95c0`; the rocm-systems tree is dirty
only because of the preserved user-owned HIP header and core-dump paths.

This no-later-consumer qualified miss is containment evidence, not a new race
contract. The checked-in all-target `ReusedLdsGemmPipeline` correct/incorrect
pair already preserves Qwen's stronger same-tile reader-retirement and
replacement-publication behavior and requires the diagnostic when that
semantic edge is removed. Adding a final-store barrier pair would make the
incorrect member behaviorally correct, violating the adjacent behavioral-pair
rule. The SuperCollider cell is green.

### 2026-08-22 D128-pressure SuperCollider qualification

The current inventory at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-d128-pressure-sc-inventory-v1`
records 36 singleton barriers across the four full-KV and double-buffered,
ExactContext and FastContext kernel specializations. It retains fingerprint
`fnv1a64:1cc23eaceeb81f21`, executable SHA-256
`0047616e6ef3bf1e325bfdcd4595a05d40dc893ca51b37c2767bab30bf26292f`,
and hook SHA-256
`7dff2c41b3038c3f9f8a42f7e317492491fbac15e1a78e5fb66068a3743bbb88`.

Final-ISA and source review first selected ExactContext occurrence 1 at
code-object PC `0x14dc`, the unconditional first K/V publication barrier, and
prospectively froze `not_detected/fail`. Artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-d128-pressure-sc-kv-publication-fault-v1`
proves that the mutation was requested, planned, reserved, applied, and
installed exactly once, but all four independent host-reference oracles still
pass. That trial is rejected and its expectation is not relabeled.

A distinct prospective trial selected FastContext occurrence 1 at PC
`0x1c394`, the first of two consecutive unconditional `s_barrier`
instructions. Because occurrence 2 remains, the policy was frozen as the
semantically redundant qualified miss `not_detected/pass` before execution.
Artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-d128-pressure-sc-redundant-barrier-v2`
accepts the exact-one mutation with a reviewed unconditional-final-ISA reach
proof. All four exact oracles pass, SuperCollider emits zero diagnostics,
static, analysis, and dynamic verdicts are complete at 252/252 accesses, and
pre/post physical-GPU discovery and dispatch probes pass. SuperCollider uses
no report buffer in this mode; allocation, capacity, cleanup, and live-byte
counts are all zero. Provenance records the exact hook and executable above
and source heads for all three repositories; the rocm-systems tree is marked
dirty only by the preserved unrelated user-owned HIP header and core-dump
paths, not by a ConSan candidate change.

The architecture-general `DoubleBufferedPipeline` correct member owns the
observable adjacent-barrier behavior on all five RocJitsu targets and physical
gfx950; its incorrect member removes the complete publication edge and
requires the sanitizer diagnostic. Adding an incorrect device workload that
only drops one redundant barrier would violate the behavioral-pair rule, so
this qualified miss remains E2E containment evidence rather than a duplicate
device fixture. The SuperCollider cell is green.

### 2026-08-22 Jakub-attention SuperCollider qualification

Paired artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-jakub-sc-paired-v1`
passes all four exact host-reference oracles with complete 338/338 access
coverage and complete static, analysis, and dynamic verdicts. The bracketing
baselines take 83 and 97 ms while SuperCollider takes 266 ms, for a 90 ms
paired baseline and 2.96x slowdown. The loaded hook SHA-256 is
`7dff2c41b3038c3f9f8a42f7e317492491fbac15e1a78e5fb66068a3743bbb88`.

Fresh inventory
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-jakub-sc-inventory-v1`
records fingerprint `fnv1a64:120a833f9c345de1` and 35 singleton barriers.
Source and final-ISA review selected the executed PipelinedProd16x8 kernel's
unconditional first load-to-compute publication barrier at PC `0x8e4`.
Completed LDS stores and an `lgkm` wait precede it, and consumer LDS reads
follow it. Unlike ProducerSkew, this variant has no intentional producer
delay, so the policy was prospectively frozen as the physical schedule-masked
qualified miss `not_detected/pass`.

Artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-jakub-sc-pipelined-miss-v1`
accepts that exact-one mutation with requested, planned, reserved, applied,
installed, and reviewed-unconditional-final-ISA reach evidence. All four
oracles pass, SuperCollider emits no diagnosis, 338/338 access coverage and
all verdicts remain complete, and pre/post discovery and dispatch health probes
pass. Provenance pins executable SHA-256
`7bc0aa5de9ec4953cf6d2490f7d2f1e4639ab2f47cc87586fb24774390d11fbe`
and the hook above; rocm-systems dirt consists only of the preserved unrelated
user-owned HIP header and core-dump paths.

`DoubleBufferedPipeline` already owns the adjacent correct/missing-publication
contract on all five RocJitsu targets and physical gfx950, while
`CdnaMfmaPipeline` adds native CDNA3/CDNA4 matrix and accumulator pressure.
Their combined 90-row all-engine gate is green. This schedule-masked qualified
miss is containment evidence rather than a valid incorrect workload, so no
duplicate device pair is added. The SuperCollider cell is green.

### 2026-08-22 TP2-family Sampled DPP-phase qualification

The current inventory at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-tp2-sampled-inventory-v2`
records 28 singleton barrier sites in the six executed TP2 code objects and the
current fingerprint `fnv1a64:51a5c50079c3ccba`. An earlier inventory attempt
without the required IREE/Sharktank environment is retained as an environment
failure, not ConSan evidence.

Current paired artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-tp2-sampled-dpp-phase-v3`
passes the prefill, decode, and combined exact oracles. Sampled has complete
1524/1524 access and 150/150 applicable-barrier coverage, zero forbidden
diagnostics, and complete static, analysis, and dynamic verdicts. The paired
device-time medians are 1.980/3.347/1.976 ms for prefill, 40.149/51.475/39.110
ms for decode, and 22.168/32.993/21.447 ms for combined in
baseline-before/Sampled/baseline-after order. The maximum slowdown against the
bracketing mean is 1.69x in prefill.

Before fault execution, current inventory and final-ISA review selected the
first unconditional barrier of the prefill attention dispatch at PC `0x7db4`.
This is DPP-phase synchronization rather than the later LDS publication edge;
an independent prior Record/Replay campaign had reached the same current
code-object identity. The Sampled policy was prospectively frozen as the
schedule-masked qualified miss `not_detected/pass`. The mutation is requested,
planned, reserved, installed, and applied exactly once. It preserves all three
exact oracles, emits no diagnosis, retains complete 1524/1524 access and
149/149 surviving-barrier coverage, and commits 240 sampled windows. The
command takes 6.427 seconds, peaks at 704,528 live report bytes, releases all
report memory with no allocation, capacity, or cleanup failure, and passes
both pre/post physical-GPU health probes. The exact hook SHA-256 is
`6072161a2b5a01c28b26b59d5b6f14dacd60fbd10a4a0f0e3d34fceb9050aab7`.

The architecture-general `DoubleBufferedPipeline` pair owns the observable
attention publication behavior on all five simulator targets and physical
gfx950, while `CdnaMfmaPipeline` adds native MFMA/AccVGPR pressure on
CDNA3/CDNA4. Their combined 90-row baseline/all-engine gate is green. The
qualified miss itself is containment evidence rather than a distinct incorrect
behavior, so no prototype-shaped duplicate is added. The Sampled cell is
green.

### 2026-08-22 Jakub-attention Sampled publication qualification

Current paired artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-jakub-sampled-producer-skew-v1`
passes all four exact host-reference oracles with complete 338/338 access and
35/35 applicable-barrier coverage, zero diagnostics, and complete static,
analysis, and dynamic verdicts. Baseline-before and baseline-after take 88 and
96 ms while Sampled takes 271 ms, for a 92 ms paired baseline and 2.95x
slowdown. The exact hook SHA-256 is
`6072161a2b5a01c28b26b59d5b6f14dacd60fbd10a4a0f0e3d34fceb9050aab7`;
the recorded source head is `cd6786f9a8fe4c478c8a4ba5f5941d4bb370c492`.
The source is marked dirty only by the preserved user-owned modified header
and two core-dump paths, not by a ConSan implementation change.

The first fault prospectively selected the ProducerSkewProd16x8 kernel's
unconditional initial publication barrier at PC `0x1808`, predeclared stride
one, and froze `detected/fail`: the last producer wave deliberately sleeps
before publishing its B fragment. The exact-one mutation is reached and
applied with complete evidence, but physical scheduling still preserves every
oracle and Sampled emits no diagnosis. This trial is rejected and retained at
the paired artifact above; neither expectation was revised.

A distinct prospectively reviewed trial at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-jakub-sampled-pipelined-miss-v2`
selects the PipelinedProd16x8 kernel's unconditional initial publication
barrier at PC `0x8e4`. That kernel has no intentional producer delay, so its
frozen policy was the schedule-masked qualified miss `not_detected/pass`. The
mutation is requested, planned, reserved, installed, and applied exactly once
and reaches under reviewed final-ISA evidence. It preserves all four exact
oracles and complete 338/338 access plus 34/34 surviving-barrier coverage,
with complete analysis and no overflow. The command completes in 0.418
seconds, peaks at 735,640 live report bytes, releases all report memory without
allocation, capacity, or cleanup failure, and passes both pre/post physical-GPU
health probes.

The architecture-general `DoubleBufferedPipeline` pair owns the observable
publication contract on all five simulator targets and physical gfx950, while
`CdnaMfmaPipeline` adds native MFMA/AccVGPR pressure on CDNA3/CDNA4. Their
combined 90-row baseline/all-engine gate is green. The Sampled cell is green.

### 2026-08-22 tree atomic-OR Sampled producer-release qualification

Fresh inventory
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-tree-sampled-inventory-v1`
identifies the correct kernel specialization's producer-only release sequence:
the `global_atomic_or` at PC `0x18db8` is preceded by `buffer_wbl2` and has no
acquire suffix, unlike the final subgroup's separate acquire-release RMW. From
that inventory, the fault policy prospectively selected exact-one weakening of
this producer release, required `detected/pass`, and predeclared stride one as
fault-only audited Sampled tuning.

Current bundle
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-tree-sampled-producer-release-v1`
passes the exact MFMA-partials oracle with complete 48/48 access, 4/4
applicable-barrier, and 3/3 atomic coverage, zero diagnostics, and complete
static, analysis, and dynamic verdicts. Baseline-before and baseline-after
take 69 and 73 ms while Sampled takes 1,128 ms, for a 71 ms paired baseline
and 15.89x slowdown. The exact hook SHA-256 is
`6072161a2b5a01c28b26b59d5b6f14dacd60fbd10a4a0f0e3d34fceb9050aab7`;
the recorded source head is `a305b6cf641b12bb10cf35147714194c9d9e299f`.
The source is marked dirty only by the preserved user-owned modified header
and two core-dump paths, not by a ConSan implementation change.

The fault requests, plans, reserves, installs, and applies its mutation exactly
once. The numerical oracle still passes, while the detector emits 12 Sampled
conflicts and thereby supplies the dynamic reach proof. The row retains
complete 48/48 access, 4/4 barrier, and 2/2 surviving-atomic coverage with
complete analysis and no overflow. Its command completes in 1.618 seconds,
peaks at 927,016 live report bytes, releases all report memory without
allocation, capacity, or cleanup failure, and passes both pre/post physical-GPU
health probes.

The checked-in `TreeAtomicOr` correct/incorrect pair is the direct behavioral
reduction: only the producer-release edge differs, the correct member requires
the exact aggregate and no diagnostic, and the incorrect member requires the
conflict. All 60 baseline/all-engine rows pass on all five simulator targets
and physical gfx950. The Sampled cell is green.

### 2026-08-22 Stream-K two-tile Sampled exact-byte qualification

Current clean and fault artifacts are
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-streamk-two-tile-sampled-byte-range-v2`
and
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-streamk-two-tile-sampled-byte-range-fault-v2`.
The source-matched clean row passes its exact oracle with complete 80/80
access, 5/5 applicable-barrier, and 2/2 atomic coverage and zero diagnostics.
Sampled takes 558.432 ms against a 275.763 ms paired baseline, for 2.025x
slowdown.

The prospectively reviewed exact-one removal of the publication barrier at
`.text+0x1810` is requested, planned, reserved, installed, applied, and reached
once. It preserves the exact numerical oracle and covers 80/80 accesses, 4/4
surviving barriers, and 2/2 atomics while emitting six Sampled conflicts. The
fault commits 28 sampled windows, peaks at 1,844,584 live report bytes, returns
to zero live bytes without allocation, capacity, or cleanup failure, and
passes both physical-GPU health probes. The detector-owned fault uses the
prospectively frozen diagnostic oracle rather than revising its expectation
after execution.

This workload exposed Sampled's compression of LDS accesses to four-byte
cells: distinct producer waves writing the adjacent FP16 halves of one dword
could falsely conflict. Sampled now retains exact byte ranges. Focused host
tests pin range packing, overlap, generation, and logging, while the new
checked-in `AdjacentSubwordWriters` correct/incorrect behavioral pair avoids
asserting the representation. Its 60 baseline/all-engine rows pass across all
five RocJitsu targets and physical gfx950: the correct member requires the
exact combined result and no diagnostic, and the incorrect member removes only
the third-wave consumer publication barrier and requires the conflict.

The artifacts pin the executed hook hash and source revisions, but record the
candidate ConSan edits and preserved unrelated workspace changes as a dirty
source tree. They are valid current-candidate behavioral evidence, not a claim
of clean-provenance acceptance; a clean-revision refresh remains part of the
eventual global green revalidation.

### 2026-08-23 HIP Stream-K Sampled current-tip refresh

Current paired and fault artifacts for the simple row are
`/home/ossci/xx/consan-validation/prep-20260823-gfx950-streamk-simple-sampled-overhead-v1`
and
`/home/ossci/xx/consan-validation/prep-20260823-gfx950-streamk-simple-sampled-fault-v1`.
The paired row preserves the exact `Errors: 0` oracle with complete 32/32
access, 3/3 barrier, and 2/2 atomic coverage and zero diagnostics. The two
bracketing baselines take 153.964 and 128.093 ms; Sampled takes 258.367 ms,
for a 141.029-ms mean baseline and 1.83x slowdown. Final-ISA review freezes
occurrence 1 at PC `0x12a4`, the reached loop phase-publication barrier before
the next buffer state and peer `ds_read2_b32` consumers. Its exact-one removal
produces the required detector-owned Sampled conflict with complete 32/32
access, 2/2 surviving-barrier, and 2/2 atomic coverage. All 192 selected
windows are claimed, report memory peaks at 924,904 bytes and returns to zero,
and the pre/post physical-GPU health probes pass.

The two-tile paired and fault artifacts are
`/home/ossci/xx/consan-validation/prep-20260823-gfx950-streamk-two-tile-sampled-overhead-v1`
and
`/home/ossci/xx/consan-validation/prep-20260823-gfx950-streamk-two-tile-sampled-fault-v1`.
The exact clean oracle passes with complete 80/80 access, 5/5 barrier, and 2/2
atomic coverage and zero diagnostics. Its 286.178- and 299.277-ms baselines
bracket a 566.342-ms Sampled row, for a 292.727-ms mean baseline and 1.94x
slowdown. The prospectively reviewed exact-one publication-barrier removal at
PC `0x1810` produces six required Sampled conflicts with complete 80/80 access,
4/4 surviving-barrier, and 2/2 atomic coverage. All 384 selected windows are
claimed, report memory peaks at 1,844,584 bytes and returns to zero, and both
health probes pass. Both fault runs expose expected sampling saturation but no
dropped, malformed, overflowed, or incomplete evidence; they therefore make
no stronger lossless-sampling claim.

No implementation repair was needed. The architecture-general
`StreamKLastArriver` pair owns the simple publication contract on all five
simulator targets plus physical gfx950; `CdnaStreamkTwoTile` owns the
access-heavy native CDNA shape on gfx942/gfx950 simulation plus physical
gfx950; and `AdjacentSubwordWriters` transports the two-tile exact-byte
publication idea to all five targets plus physical gfx950. The focused
baseline/Sampled correct/incorrect gate passes all 30 rows. These current-tip
artifacts supersede the older dirty-candidate return point and promote both
Sampled cells to green; the eventual one-revision global refresh remains an
exit criterion.

### 2026-08-22 D128-pressure Sampled fault qualification

Current artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-d128-pressure-sampled-redundant-barrier-v1`
passes the exact full-KV host-reference oracle with complete 252/252 access and
24/24 applicable-barrier coverage, zero diagnostics, and complete static,
analysis, and dynamic verdicts. Baseline-before and baseline-after take 66 and
81 ms while Sampled takes 1,273 ms, for a 73.5 ms paired baseline and 17.32x
slowdown. The exact hook SHA-256 is
`6072161a2b5a01c28b26b59d5b6f14dacd60fbd10a4a0f0e3d34fceb9050aab7`;
the recorded source head is `61d82d5ddee427635e06a1a752ea8e8d7115f486`.
The source is marked dirty only by the preserved user-owned modified header
and two core-dump paths, not by a ConSan implementation change.

Before execution, final-ISA review prospectively selected occurrence one of
the first of two consecutive unconditional FastContext barriers at PC
`0x1c394`. The adjacent occurrence-two barrier remains, so the frozen policy
was the semantically redundant qualified miss `not_detected/pass`. The
mutation is requested, planned, reserved, installed, and applied exactly once
and reaches under reviewed unconditional final-ISA evidence. It preserves the
exact oracle and complete 252/252 access plus 24/24 applicable-barrier
coverage, with complete analysis and no overflow. The fault command completes
in 2.219 seconds, peaks at 328,792 live report bytes, releases all report
memory without allocation, capacity, or cleanup failure, and passes both
pre/post physical-GPU health probes.

The architecture-general `DoubleBufferedPipeline` pair already owns adjacent
publication barriers in its correct member and removes the complete semantic
edge in its incorrect member across all five simulator targets and physical
gfx950. This qualified miss therefore strengthens E2E containment evidence
without warranting a prototype-layout duplicate. The Sampled cell is green.

### 2026-08-22 MFMA-attention Sampled fault qualification

Current artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-mfma-sampled-redundant-score-barrier-v1`
passes the exact host-reference oracle with complete 58/58 access and 12/12
applicable-barrier coverage, zero diagnostics, and complete static, analysis,
and dynamic verdicts. Baseline-before and baseline-after take 81 and 77 ms,
while Sampled takes 203 ms, for a 79 ms paired baseline and 2.57x slowdown.
The exact hook SHA-256 is
`6072161a2b5a01c28b26b59d5b6f14dacd60fbd10a4a0f0e3d34fceb9050aab7`;
the recorded source head is `4794ed438cdbf8dfb946b042a73c7b6a68904307`.
The source is marked dirty only by the preserved user-owned modified header
and two core-dump paths recorded in the worktree status, not by a ConSan
implementation change.

Before execution, final-ISA review prospectively selected occurrence three of
the first score-publication barrier at PC `0x1f88`. The adjacent second
source-level publication barrier remains, so the frozen policy was the
semantically redundant qualified miss `not_detected/pass`. The mutation is
requested, planned, reserved, installed, and applied exactly once and reaches
under reviewed unconditional final-ISA evidence. It preserves the exact
oracle and complete 58/58 access plus 11/11 surviving-barrier coverage, with
complete analysis and no overflow. The fault command completes in 0.317
seconds, peaks at 75,816 live report bytes, releases all report memory, and
passes both pre/post physical-GPU health probes.

The common `OrderedTileHandoff` pair and target-native `CdnaMfmaPipeline` pair
already own the positive publication-diagnostic contract across every
applicable backend, including physical gfx950; their focused 90-row gate is
green. This E2E qualified miss does not justify a duplicate fixture tied to
the prototype's adjacent-barrier layout. The Sampled cell is green.

### 2026-08-22 D128-block Sampled fault qualification

Current artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-d128-block-sampled-redundant-barrier-v3`
passes the exact FastContext host-reference oracle with complete 128/128 access
and 117/117 applicable-barrier coverage, zero diagnostics, and complete static,
analysis, and dynamic verdicts. Baseline-before and baseline-after take 77 and
86 ms while Sampled takes 767 ms, for an 81.5 ms paired baseline and 9.41x
slowdown. The exact hook SHA-256 is
`6072161a2b5a01c28b26b59d5b6f14dacd60fbd10a4a0f0e3d34fceb9050aab7`;
the recorded source head is `4a7a76105a33211f227787d6ff27237a23d509fc`.

Two prospective attempts at the ExactContext K-tile publication edge remain
rejected. The standard-cadence trial at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-d128-block-sampled-k-publication-v1`
and the distinct stride-1 trial at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-d128-block-sampled-k-publication-stride1-v2`
both requested, planned, installed, and applied their exact-one mutations, but
physical scheduling preserved the oracle and Sampled emitted no diagnostic.
Their prospectively frozen `detected` expectations were not relabeled.

The accepted distinct trial prospectively selects the first of consecutive
unconditional FastContext barriers at final-ISA PCs `0x25e40` and `0x25e44`.
Dropping exactly the first leaves the adjacent second barrier intact, so its
frozen policy is the source-reviewed qualified miss `not_detected/pass`. The
mutation is requested, planned, reserved, installed, and applied exactly once
and reaches under reviewed final-ISA evidence. It preserves the exact oracle,
complete 128/128 access and 117/117 static barrier coverage, and complete
analysis without overflow. The fault process completes in 1.319 seconds,
peaks at 167,096 live report bytes, releases all report memory, and passes both
pre/post physical-GPU health probes.

The common `OrderedTileHandoff` pair and target-native `CdnaMfmaPipeline` pair
remain the positive publication-diagnostic contracts across every applicable
backend; `CdnaSampledIndependentScalarProofs` retains the D128-specific
two-owner resource shape. No implementation-shaped duplicate device fixture
is added for this qualified miss. The Sampled cell is green.

### 2026-08-22 Qwen Record/Replay reader-retirement qualification

Current-tip artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-qwen-rr-qual-77c-v1`
passes the exact Qwen prefill oracle with complete 658/658 access and 46/46
barrier coverage, zero diagnostics, and complete static, analysis, and dynamic
verdicts. The selected final-output matmul
`main$async_dispatch_562_batch_matmul_1x5x151936x1024_f32` takes 13.350 ms
under Record/Replay versus a 6.410 ms paired baseline, for 2.08x slowdown. The
full clean command completes in 0.912 seconds without workload-specific clean
tuning or filtering.

The prospective fault removes occurrence one of the reader-retirement barrier
at final-ISA PC `0xaa6c`: all readers of the first LDS tile must retire before
the producer overwrites the same storage, while the replacement tile's
publication barrier remains. A standard-cadence trial in
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-qwen-rr-inventory-96169-v1`
reached and applied its exact-one mutation and failed the exact oracle, but a
different initializer dispatch was sampled and no diagnosis was emitted. That
prospectively frozen `detected` expectation remains rejected and was not
relabeled.

A separate targeted trial predeclared stride one and the selected final-output
kernel as fault-only audited settings. Its mutation is requested, planned,
reserved, installed, and applied exactly once; the exact oracle fails and
Record/Replay emits 120,034 detector-owned diagnostics. The large fault trace
intentionally saturates its barrier-report bank, so its dynamic analysis is
incomplete and is not claimed as lossless fault coverage. It nevertheless
completes without timeout, peaks at 216,973,496 live report bytes, releases all
report memory, and passes pre/post physical-GPU health probes. Clean coverage,
oracle, overhead, source and object provenance, containment, and cleanup all
remain in the frozen accepted bundle, so the cell is green.

The checked-in `ReusedLdsGemmPipeline` correct/incorrect pair distills this
same-storage publication and reader-retirement contract across all five
RocJitsu targets and physical gfx950. Its 60 baseline/all-engine rows pass.
Validation host regressions additionally pin fault-only policy auditing,
ASLR-independent loader-linkage provenance, and conservative reuse of only a
completed exact-match fault row after interruption.

### 2026-08-22 Qwen Sampled kernarg-preload qualification

Qwen's selected final-output matmul exposed an AMDHSA entry-ABI defect that
was hidden by the ordinary Sampled cadence. With stride one, Sampled has no
dispatch-ID consumer; the private-epoch prologue therefore emitted only its
primary entry even though the compiler's kernarg-preload descriptor permits
firmware to enter at descriptor entry plus 256 bytes. The physical gfx950
stopped on an illegal instruction at exactly that secondary entry. ConSan now
plans the paired private-state entries from the descriptor independently of
dispatch-ID capture. A focused clean run of
`main$async_dispatch_562_batch_matmul_1x5x151936x1024_f32` passes the exact
Qwen oracle with complete 76/76 access and 4/4 barrier coverage, zero
diagnostics, and complete static, analysis, and dynamic verdicts.

The new checked-in `CdnaKernargPreloadPrivateState` device pair compiles the
existing mixed fixed/dynamic-owner workload with a real compiler-generated
kernarg preload. Its first run exposed a second independent gap: the
dynamic-stack owner/epoch path rejected the same secondary hardware entry.
That path now uses the existing paired descriptor-redirection abstraction so
both firmware entries initialize persistent state while the guest dynamic-stack
code and symbols remain at their original addresses. Focused host tests cover
both stride-one private-state and dynamic-stack owner paths on CDNA3 and
CDNA4. The adjacent correct/incorrect pair contributes 12 green baseline and
Sampled rows on gfx942/gfx950 simulation and physical gfx950; the correct
member requires exact results and no diagnostic, while the incorrect member
requires the publication conflict.

Fresh contained artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-qwen-sampled-reader-retirement-v3`
prospectively selects occurrence one of the output matmul's reader-retirement
barrier at final-ISA PC `0xaa6c`, predeclares stride one, and limits fault-only
instrumentation to that reviewed kernel. The mutation is requested, planned,
reserved, installed, applied, and reached exactly once. The exact Qwen oracle
still passes, while Sampled emits 32 detector-owned conflicts with complete
76/76 access and 3/3 surviving-barrier coverage, 1,088 committed sampled
windows, complete analysis, and no overflow. The contained command completes
in 6.892 seconds, peaks at 177,528 live report bytes, releases all report
memory without allocation, capacity, or cleanup failure, and passes both
physical-GPU health probes. The hook SHA-256 is
`0f1e554192be7f324cd54e201fff207aad9367f1534ec02adfd075b201bdbbb5`.
The earlier v1/v2 illegal-instruction attempts remain rejected artifacts; no
failed outcome or expectation was relabeled. The Sampled cell is green.

### 2026-08-22 Jakub-attention Record/Replay qualification

Clean-source paired artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-jakub-rr-current-ffb-v2`
passes all four exact host-reference oracles with complete 338/338 access and
35/35 barrier coverage, zero diagnostics, and complete static, analysis, and
dynamic verdicts. Baseline-before and baseline-after take 92 and 96 ms, while
Record/Replay takes 411 ms, for a 94 ms paired baseline and 4.37x slowdown.
This supersedes the retained 189-second result after the independently
regression-tested coarse-report allocation work.

The first prospective fault trial in that artifact selected the
`ProducerSkewProd16x8` publication barrier and froze `detected` plus failing
oracle before execution. Its exact-one mutation reached under reviewed source
and final-ISA evidence, but physical scheduling masked the fault: no diagnosis
was emitted and all exact oracles passed. The validator correctly rejects that
trial, and neither expectation nor site is relabeled.

A separate, distinct trial at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-jakub-rr-fault-pipeline-ffb-v3`
prospectively selected the ordinary pipelined kernel's first load-to-compute
barrier. Because that variant has no intentional producer delay, its frozen
policy was the schedule-masked qualified miss `not_detected` plus passing
oracle. The mutation is requested, planned, reserved, installed, and applied
exactly once and reaches under reviewed unconditional final-ISA evidence. The
accepted run retains 338/338 access and 34/34 surviving-barrier coverage,
records 2,097,152 access and 208 barrier events without diagnostic or overflow,
peaks at 220,045,496 report bytes, releases all report memory, passes pre/post
health probes, retains the hook hash, and records clean source revision
`ffbafe1e16e61d41d1e166656bdf6438ad521410`. The Record/Replay cell is green.
The architecture-general checked-in double-buffered pipeline pair already
protects the corresponding correct publication and missing-edge behavior on
all five RocJitsu targets and physical gfx950.

### 2026-08-21 D128-pressure Record/Replay fault review

Fresh inventory artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-d128-pressure-rr-inventory-postmerge-v1`
accepts 36 exact singleton-barrier identities for the rebuilt physical gfx950
workload. Three separate prospective Record/Replay trials each froze
`detected` plus failing-oracle expectations before execution: the initial
K/V-staging barrier, the score-publication barrier before lane-0 reduction,
and the weight-publication barrier before all-lane consumption.

All three mutations were requested, planned, installed, and applied exactly
once; each run retained complete 252/252 access and 27/27 surviving-barrier
coverage, bounded report state, teardown evidence, and healthy pre/post GPU
probes. All three nevertheless produced zero replay conflicts and passed the
exact oracle. The third trial at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-d128-pressure-rr-weight-fault-postmerge-v3`
also carried a predeclared reviewed-unconditional-final-ISA reach proof, so it
is admitted evidence of a schedule-masked qualified miss rather than an
unreached mutation. Because its frozen detector/oracle policy expected a
detection and failure, the validator correctly rejects it and the matrix cell
remains yellow. Further site cycling is deferred in favor of other cells; no
expectation was revised after observation.

A later, distinct site was frozen prospectively from final-ISA review at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-d128-pressure-rr-current-881-v1`.
It is the first of two consecutive barriers in the unconditional
`FullKvDoubleBuffered16Key/FastContext` loop; the adjacent second barrier
remains, so the predeclared policy was the semantically redundant qualified
miss `not_detected` plus passing oracle. The exact-one mutation is requested,
planned, reserved, installed, and applied once, reaches by its reviewed proof,
and is accepted without revising any earlier expectation. It retains complete
252/252 access and 28/28 static barrier coverage, records 2,097,152 access and
152 barrier events with no diagnostic or overflow, peaks at 259,768,504 report
bytes, releases all report memory, and passes both health probes. This accepted
trial supersedes the yellow conclusion above without relabeling or reusing any
of the three rejected experiments.

### 2026-08-21 MFMA-attention Record/Replay refresh

Fresh paired artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-mfma-record-replay-overhead-postmerge-v1`
passes both exact host-reference oracles with complete 58/58 access and 14/14
barrier coverage, zero diagnostics, and complete static, analysis, and dynamic
verdicts. Baseline-before and baseline-after take 68 and 76 ms; Record/Replay
takes 298 ms, for a 72 ms paired baseline and 4.14x slowdown. This also shows
that the previously retained 150-second clean runtime was not reproduced.

Fresh inventory artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-mfma-record-replay-inventory-postmerge-v1`
maps the exact-context kernel's nine barriers to initialization and the two
physical barriers surrounding each of four source-level epoch transitions.
Before mutation, review froze the first score-publication barrier as a reached
qualified miss: dropping exactly that barrier leaves its adjacent second
barrier intact, so the expected contract was no diagnosis and a passing
numerical oracle. Contained artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-mfma-record-replay-fault-postmerge-v1`
accepts that exact-one requested/planned/installed/applied mutation, its
reviewed-unconditional-final-ISA reach proof, zero Record/Replay diagnoses,
the passing oracle, complete surviving coverage, teardown evidence, and
healthy pre/post probes.

The clean-source refresh at artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-mfma-record-replay-current-622-v1`
closes that final provenance gate at source revision
`622039cb430e94614b7b96272f0e5dd324aba151`. The baseline-before and
baseline-after rows take 75 and 73 ms, while Record/Replay takes 312 ms, for a
74 ms paired baseline and 4.22x slowdown. Both exact oracles pass with complete
58/58 access and 14/14 barrier coverage and zero diagnostics. The contained
fault requests, plans, installs, and applies exactly one prospectively reviewed
score-publication barrier drop. It reaches the selected unconditional site,
preserves the exact oracle, produces the expected qualified miss, and retains
complete 58/58 access and 13/13 surviving-barrier coverage. The report peaks
at 251,986,104 bytes and has zero live bytes or cleanup failures afterward;
both physical-GPU health probes pass. All recorded source checkouts are clean,
so the Record/Replay cell is green.

### 2026-08-22 post-merge Hip-MOI tree atomic-OR prerequisite repair

The post-merge broad non-device gate originally selected 1,581 tests and passed
1,580, with `ConSanGfx950HipMoiSim.TreeAtomicOr` as its only failure. The
external Hip-MOI correct workload produced 512 false conflicts in RocJitsu
simulation even though its exact numerical oracle passed; its relaxed negative
companion and both workloads on physical gfx950 behaved as expected.

The failure was in Hip-MOI's metadata protocol rather than in ConSan or the
simulator. All producer release RMWs were visible and all four metadata records
were eventually published, but the final subgroup imported only producer 0's
record. The generic acquiring-RMW path had a bounded four-probe window to cover
post-RMW metadata publication, yet returned as soon as *any* matching producer
was found. Under RocJitsu's legal interleaving, that return occurred before
producers 1 and 2 published, yielding per-producer epoch tokens `[2, 0, 0, 0]`
instead of `[2, 2, 2, 0]`.

Hip-MOI commit `eef1431` (`Fix multi-producer atomic acquire retries`) now
consumes the entire bounded retry window and accumulates every producer that
publishes during it. This preserves the fixed bound and does not constrain
RocJitsu scheduling. After rebuilding the gfx950 Hip-MOI fixture, the simulated
tree atomic-OR correct/incorrect pair passes 2/2 in 0.70 seconds. Four focused
physical gfx950 atomic suites, including that same pair, pass 4/4 under the
physical-GPU lock. The checked-in ConSan `TreeAtomicOr` correct/incorrect pair
continues to guard the sanitizer behavior across the baseline and all four
engines. The integration-gate failure is resolved without a ConSan
implementation change.

### 2026-08-21 Sampled independent scalar-proof repair

Paired physical artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-d128-block-sampled-overhead-independent-abi-v2`
uses hook SHA-256
`dcb50737106a15c068da43ebbc439b83cfddaf08032b280792ddb804b7638712`.
Its baseline-before, Sampled, and baseline-after rows are all accepted. Sampled
passes the exact oracle in 0.828 seconds with complete 128/128 access and
117/117 barrier coverage, zero diagnostics, and complete static, analysis, and
dynamic verdicts. The baseline-before, Sampled, and baseline-after process
medians are 56, 745, and 70 ms respectively.

The regression had two independent causes. Dispatch-only planning replaced a
previously selected owner-local EXEC assignment instead of overlaying only the
owner's literal dispatch identity. It also leaked the temporary
partial-EXEC-planning marker, strengthening a site-liveness proof into an
owner-wide reservation and rejecting guest SGPR uses outside instrumentation
sites. The repair preserves the prior assignment and planning classification
while merging only the newly proven dispatch representation.

Two new focused host tests pin those placement invariants. The existing
heterogeneous-owner regression additionally proves that dispatch planning does
not resurrect an owner deliberately excluded by a partial EXEC proof. A
separate checked-in Sampled device workload distills the same two-owner shape into
correct/incorrect LDS synchronization pairs: correct runs require exact
results and no diagnostics, while incorrect runs require the race diagnostic
and preserve independent controls. Its 12 baseline and Sampled cases pass on
gfx942 and gfx950 simulation plus physical gfx950. Deliberately restoring the
leaked marker makes five of its six instrumented cases fail with 38/40 access
resource failures, proving that the device regression is behaviorally
sensitive to the fix. The pre-existing all-engine dense-SCC fixture remains
unchanged and passes all 24 cases.

Neighboring current-hook physical refreshes also pass: artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-d128-pressure-sampled-independent-abi-v2`
passes all four D128-pressure oracles in 1.386 seconds with 252/252 accesses
and 24/24 barriers; artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-mfma-attention-sampled-independent-abi-v2`
passes both MFMA-attention oracles in 0.270 seconds with 58/58 accesses and
12/12 barriers. Both have zero forbidden diagnostics and complete static,
analysis, and dynamic verdicts.

The paired D128-pressure follow-up at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-d128-pressure-sampled-overhead-independent-abi-v2`
is also accepted. Its 88 ms baseline-before, 1,243 ms Sampled row, and 74 ms
baseline-after all pass the exact full-KV oracle. Sampled retains complete
252/252 access and 24/24 applicable-barrier coverage with zero forbidden
diagnostics; the paired process slowdown is 15.35x. Reviewed-fault and clean
provenance gates remain open.

The paired MFMA-attention follow-up at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-mfma-attention-sampled-overhead-independent-abi-v2`
is accepted as well. Its 74 ms baseline-before, 206 ms Sampled row, and 71 ms
baseline-after pass the exact oracle. Sampled retains complete 58/58 access
and 12/12 applicable-barrier coverage, zero forbidden diagnostics, and a
2.84x paired process slowdown. Reviewed-fault and clean-provenance gates
remain open.

The paired Stream-K-arrival follow-up at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-streamk-sampled-overhead-independent-abi-v2`
is accepted too. Its 80 ms baseline-before, 473 ms Sampled row, and 71 ms
baseline-after pass the exact MFMA-partials oracle. Sampled retains complete
32/32 access, 4/4 applicable-barrier, and 1/1 atomic coverage, zero forbidden
diagnostics, and a 6.26x paired process slowdown. Reviewed-fault and
clean-provenance gates remain open.

The paired tree-atomic-OR follow-up at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-tree-sampled-overhead-independent-abi-v2`
is accepted. Its 90 ms baseline-before, 1,162 ms Sampled row, and 67 ms
baseline-after pass the exact MFMA-partials oracle. Sampled retains complete
48/48 access, 4/4 applicable-barrier, and 3/3 atomic coverage, zero forbidden
diagnostics, and a 14.80x paired process slowdown. Reviewed-fault and
clean-provenance gates remain open.

The paired Jakub-attention follow-up at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-jakub-sampled-overhead-independent-abi-v2`
is accepted. Its 104 ms baseline-before, 274 ms Sampled row, and 118 ms
baseline-after pass all four exact oracles. Sampled retains complete 338/338
access and 35/35 applicable-barrier coverage, zero forbidden diagnostics, and
a 2.47x paired process slowdown. Reviewed-fault and clean-provenance gates
remain open.

The paired TP2-family follow-up at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-tp2-sampled-overhead-independent-abi-v2-with-iree-env`
is accepted. The baseline-before, Sampled, and baseline-after rows pass the
prefill, decode, and combined exact oracles. Sampled retains complete 936/936
access and 120/120 applicable-barrier coverage with zero forbidden
diagnostics; its maximum paired slowdown is 1.45x (prefill), with 1.09x decode
and 1.40x combined. Reviewed-fault and clean-provenance gates remain open.

The paired Stream-K Inline Shadow follow-up at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-streamk-inline-overhead-independent-abi-v2`
is accepted. Its 84 ms baseline-before, 530 ms Inline Shadow row, and 71 ms
baseline-after pass the exact MFMA-partials oracle. Inline Shadow retains
complete 32/32 access, 6/6 barrier, and 1/1 atomic coverage, zero forbidden
diagnostics, and a 6.84x paired process slowdown. Reviewed-fault and
clean-provenance gates remain open.

The paired tree-atomic-OR Inline Shadow follow-up at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-tree-inline-overhead-independent-abi-v2`
is accepted. Its 60 ms baseline-before, 1,165 ms Inline Shadow row, and 98 ms
baseline-after pass the exact MFMA-partials oracle. Inline Shadow retains
complete 48/48 access, 6/6 barrier, and 3/3 atomic coverage, zero forbidden
diagnostics, and a 14.75x paired process slowdown. Reviewed-fault and
clean-provenance gates remain open.

The paired D128-pressure Inline Shadow follow-up at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-d128-pressure-inline-overhead-independent-abi-v2`
is accepted. Both bracketing baselines take 64 ms and the Inline Shadow row
takes 1,483 ms, with the exact full-KV oracle passing throughout. Inline
Shadow retains complete 252/252 access and 28/28 barrier coverage, zero
forbidden diagnostics, and a 23.17x paired process slowdown. Reviewed-fault
and clean-provenance gates remain open.

The paired MFMA-attention Inline Shadow follow-up at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-mfma-inline-overhead-independent-abi-v2`
is accepted. Its 80 ms baseline-before, 246 ms Inline Shadow row, and 68 ms
baseline-after pass the exact oracle. Inline Shadow retains complete 58/58
access and 14/14 barrier coverage, 8,704 visible exact-shadow cells, zero
forbidden diagnostics or incomplete state, and a 3.32x paired process
slowdown. Reviewed-fault and clean-provenance gates remain open.

The paired Jakub-attention Inline Shadow follow-up at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-jakub-inline-overhead-independent-abi-v2`
is accepted. Its 68 ms baseline-before, 299 ms Inline Shadow row, and 120 ms
baseline-after pass all four exact oracles. Inline Shadow retains complete
338/338 access and 35/35 barrier coverage, zero forbidden diagnostics, and a
3.18x paired process slowdown. Reviewed-fault and clean-provenance gates
remain open.

The paired D128-pressure SuperCollider follow-up at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-d128-pressure-supercollider-overhead-independent-abi-v2`
is accepted. Its 79 ms baseline-before, 1,331 ms SuperCollider row, and 77 ms
baseline-after pass the exact full-KV oracle. SuperCollider retains complete
252/252 access coverage and complete static, analysis, and dynamic verdicts;
the paired process slowdown is 17.06x. Reviewed-fault and clean-provenance
gates remain open.

A fresh D128-block barrier inventory at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-d128-block-inventory-independent-abi-v2`
also invalidates the old schema-v1 fault selector and retains 138 exact
current-ISA barrier identities. The precommitted first Fast-context barrier
hypothesis was then rejected, rather than relabeled after execution: artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-d128-block-sampled-fault-independent-abi-v2`
has complete exact-one requested/planned/applied and installation evidence,
bounded execution, and healthy before/after discovery and dispatch probes, but
the exact numerical oracle still passes and Sampled emits no diagnostic. The
source-level reason is that `sampled_watchpoint_context::init_workgroup()`'s
initial barrier has no preceding producer/consumer boundary. This does not
qualify the reviewed-fault gate, which remains open in the matrix.

### 2026-08-21 CDNA VGLOBAL atomic-order fault repair

A fresh physical Stream-K inventory at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-streamk-inventory-global-atomic-fix-v3`
exposed a fault-injection coverage bug. ConSan's synchronization analysis had
already proven the exact compiler-emitted
`buffer_wbl2; wait; global_atomic_add; wait; buffer_inv` acquire-release
sequence at PC `0x6520`, but the fault-site encoding filter admitted only the
equivalent CDNA `flat_atomic` form. Consequently the old inventory reported no
qualified atomic-order boundary. The repair admits the identically sized CDNA3
and CDNA4 VGLOBAL form while retaining the existing restrictions: order faults
remain supported, while scope and address rewriting remain unsupported.

Both test tiers guard the repair. A focused host unit test pins the exact gfx950
compiler words and proves that the release cache operation is removed while the
wait, VGLOBAL atomic, acquire cache operation, and program termination remain
unchanged. The checked-in atomic-arrival device fixture now models the protocol
with a release-only producer and acquire-only consumer. Six explicit
correct/faulted Inline Shadow cases pass on gfx942 and gfx950 simulation plus
physical gfx950; every faulted case requires a visible diagnostic and rejects a
missing exact-one mutation. The pair therefore covers CDNA3 as well as CDNA4
and is behaviorally sensitive to this host-side selector bug.

The first reviewed physical fault run was deliberately retained as a rejected
pilot rather than relabeled after observation. Artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-streamk-inline-global-atomic-fault-v3`
precommitted `pass/not-detected`, but the exact-one mutation preserved the
numerical oracle and produced a detector-owned Inline Shadow diagnostic. A
fresh inventory and independently precommitted confirmation at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-streamk-inline-global-atomic-fault-confirm-v4`
accepted `pass/detected`: one mutation was requested, planned, applied, and
installed; the run emitted 256 visible diagnostics with complete 32/32 access,
6/6 barrier, and 1/1 atomic coverage; pre/post discovery and dispatch health,
bounded execution, cleanup, and hook hash containment all passed. The hook
SHA-256 is
`14363d31ea87fc629dc501687d30d484c01d8a9b75f419d1dc906199d6db790e`.
The post-commit inventory at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-streamk-inventory-global-atomic-postcommit-v5`
and accepted fault bundle at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-streamk-inline-global-atomic-fault-postcommit-v5`
refresh the result at clean source revision `36fb947af9`. The independently
precommitted `pass/detected` trial again requests, plans, applies, and installs
exactly one mutation, preserves the exact oracle, and emits 256 Inline Shadow
diagnostics. Discovery and dispatch health pass before and after the workload;
cleanup returns live report memory to zero; and the hook hash is unchanged.
The paired artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-streamk-inline-global-atomic-overhead-postcommit-v5`
records clean 56 ms and 83 ms bracketing baselines around a 510 ms Inline
Shadow row. All three rows pass the exact oracle, and Inline Shadow retains
complete 32/32 access, 6/6 barrier, and 1/1 atomic coverage with zero forbidden
diagnostics and complete static, analysis, and dynamic verdicts. All retained
source repositories are clean, so the Inline Shadow cell is now green.

The first Sampled follow-up is retained as a rejected pilot at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-streamk-sampled-global-atomic-fault-postcommit-v5`.
Its precommitted `pass/detected` expectation did not hold because the standard
stride 256 selected no workgroup: the exact-one mutation and oracle passed,
but `visible_sampled=0` left no detector-owned runtime witness. The separately
precommitted campaign at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-streamk-sampled-global-atomic-fault-stride1-postcommit-v6`
discloses a fault-only stride-1 override without changing the ordinary clean
profile. It applies and installs the same exact-one release mutation, preserves
the exact oracle, commits 24 sampled windows, and produces four Sampled
conflicts. Pre/post health, cleanup to zero live bytes, hook containment, and
clean provenance pass. The paired standard-profile artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-streamk-sampled-overhead-postcommit-v6`
records 59 ms and 67 ms baselines around a 477 ms Sampled row. It preserves the
exact oracle with complete 32/32 access, 4/4 applicable-barrier, and 1/1 atomic
coverage and complete static, analysis, and dynamic verdicts. The Sampled cell
is therefore green while retaining the standard untuned clean contract.

### 2026-08-21 Sampled full-pressure routing refresh

Three physical-gfx950 artifacts record the repaired Sampled clean rows:
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-d128-block-sampled-route-scc-fixed`
passes both exact D128-block oracles in 0.880 seconds with complete 128/128
access and 117/117 applicable-barrier coverage;
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-d128-pressure-sampled-owner-route-fixed`
passes all four D128-pressure oracles in 1.359 seconds with complete 252/252
access and 24/24 applicable-barrier coverage; and
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-mfma-sampled-far-route-fixed`
passes both MFMA-attention oracles in 0.250 seconds with complete 58/58 access
and 12/12 applicable-barrier coverage. All three runs have zero forbidden
diagnostics and complete static, analysis, and dynamic verdicts.

The repair gives Sampled an owner-local literal dispatch identity when the
code-object-wide hardware dispatch pair is live in one high-pressure owner,
without discarding an independently safe global EXEC-save window. Dense
access routes restore the captured guest SCC after their successful-match
comparison. Far barrier entry islands now use the same owner-local scalar ABI
as their bodies, instead of clobbering live guest SGPRs with the global route.
Focused host tests reproduce and pin each planning and emission invariant.

Strengthening the checked-in CDNA B16 SCC correct/incorrect device pair to
carry a false SCC across a routed store exposed the same direct-arm omission
in Inline Shadow. Its ordinary and tagged-key dense ABIs now both restore the
guest predicate before entering the access body. The corresponding structural
host tests pass, and the behavioral pair passes all 30 baseline and four-engine
instances across gfx942 and gfx950 simulation plus physical gfx950. Paired
overhead and reviewed-fault evidence remain, so these three cells are yellow
rather than green. The complete host gate passes 5,040 tests with four
intentional skips. The complete checked-in device gate passes all 1,511 cases
across five emulated targets and physical gfx950 in 552.39 seconds wall time
and 2,139.09 seconds aggregate test-process time at `ctest -j64`; measured CPU
time is 955.44 seconds user plus 805.94 seconds system.

### 2026-08-20 Stream-K arrival Inline Shadow clean refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-streamk-inline-private-dispatch-fix`
records a physical-gfx950 baseline and Inline Shadow run from the candidate
tree. The baseline exact oracle passes in 0.154 seconds. Inline Shadow passes
the same exact oracle in 9.050 seconds with complete 32/32 LDS-access, 6/6
barrier, and 1/1 acquire-release atomic coverage, zero forbidden diagnostics,
and complete static, analysis, and dynamic verdicts. The loaded hook SHA-256 is
`9b01bf1ff91b811edc70e53720f02f8122a23540eabca0cd9d82fcf6ed5170c5`.

The prior rejection combined two implementation gaps. When no persistent CDNA
dispatch-ID SGPR pair fit, automatic placement requested owner-private dispatch
state but did not activate its entry-captured private owner/epoch prologue; the
claimed acquire-release transaction consequently reached preflight without a
dispatch identity. Dynamic-stack owners now choose the corresponding persistent
dispatch-ID VGPR representation instead. A separately exposed descriptor bug
also treated an empty gap below `ACCUM_OFFSET` as already allocated instead of
growing the unified VGPR count. Focused host regressions cover private
non-dynamic dispatch capture, the dynamic-stack VGPR alternative, the
acquire-release atomic consumer, and unified allocation growth inside an empty
accumulator gap. Paired-overhead and reviewed-fault evidence remain, so the cell
is yellow rather than green.

### 2026-08-20 tree atomic-OR Inline Shadow clean refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-tree-inline-publication-wait-final`
records a physical-gfx950 baseline and Inline Shadow run from the candidate
tree.  The 0.13-second baseline and 18.13-second Inline Shadow row both pass
the exact MFMA-partials oracle.  Inline Shadow has complete 48/48 access, 6/6
barrier, and 3/3 atomic coverage, zero forbidden diagnostics, and complete
static, analysis, and dynamic verdicts.

The repair adds owner-private dispatch-ID state for full-pressure CDNA owners,
spill-safe access and barrier routing, and a liveness-proven dense relay for a
far ordered atomic.  Dense CDNA keys preserve the incoming SCC even when the
only available key and SCC-save scalar is the same register.  The acquired
token table now uses separate authorization and release-sequence namespaces,
so the two causal edges that alias even in a 1,024-slot undivided table cannot
overwrite each other.  Competing publishers and acquiring consumers now share
a bounded wait sized for the winner's full causal scan, rather than
misclassifying or ignoring a healthy in-progress publication.  Focused host
regressions cover each placement, scalar-state, token-namespace, and wait-policy
failure on every supported architecture.  The checked-in tree atomic-OR
correct/incorrect device pair now runs three producer releases concurrently;
all 60 instances pass across the five emulated targets, all engines, and
physical gfx950.  The current full checked-in device suite passes all 1,355
instances in 839.76 seconds wall time at `ctest -j64`.
Paired-overhead and reviewed-fault acceptance remain before this cell can
become green.  The loaded hook SHA-256 is
`eadc8a11a7ad3cfd89d934467f2073e0b8b9ad062b5316c3ae7261d6661108c4`.

### 2026-08-20 tree atomic-OR Sampled clean refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-tree-sampled-router-fix-final`
records a physical-gfx950 baseline and Sampled run from the candidate tree
based on revision `5b22a23dba`.  The 0.10-second baseline and 1.29-second
Sampled run both pass the exact MFMA-partials oracle.  Sampled has complete
48/48 access, 4/4 applicable-barrier, and 3/3 atomic coverage, zero forbidden
diagnostics, and complete static, analysis, and dynamic verdicts.

The repair gives CDNA3/4 Sampled accesses, barriers, and atomics complete far
relay coverage; preserves spill-backed scalar state in atomic caves; and
qualifies both dense and ordinary atomic-router bootstrap SGPRs at the actual
atomic anchor.  The ordinary producer route had selected `s0:s1` even though
the generated kernel retained its saved EXEC mask there, preventing Hip-MOI's
release metadata from being published.  Focused host regressions cover dense
relays, spill-backed route keys, scalar preservation, and the live-bootstrap
case.  The checked-in tree atomic-OR correct/incorrect device pair passes for
gfx942, gfx950 simulation, physical gfx950, and gfx1250 simulation.  The loaded
hook SHA-256 is
`d3946367d10998c23238a50ab6e3e35b3ff93a911eb60a897cf8a784567adb68`.

### 2026-08-20 Stream-K Sampled clean refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-streamk-sampled-relay-fix`
records a physical-gfx950 baseline and Sampled run from the candidate tree
based on revision `28b83770dd`.  The 0.11-second baseline and 0.57-second
Sampled run both pass the exact MFMA-partials oracle.  Sampled has complete
32/32 access, 4/4 applicable-barrier, and 1/1 atomic coverage, zero forbidden
diagnostics, and complete static, analysis, and dynamic verdicts.

Two focused host regressions cover the readiness fixes.  The first checks that
Sampled can use its versioned report's stable reader-identity literal on
full-scalar-pressure CDNA3 and CDNA4 owners.  The second builds a far CDNA4
access-plus-atomic-plus-barrier object and requires distinct reserved relay
slots for its atomic and barrier probes.  This supersedes both the stale
dynamic-stack rejection and the intermediate partially-overlapping relay
failure.  The cell is now yellow; paired-overhead and reviewed-fault
acceptance remain.  The loaded hook SHA-256 is
`83b9ce72571bcb20e498f2bfae1af489ccf69eba84044c0552d7b683ec4fdaef`.

### 2026-08-20 HIP-matmul SuperCollider clean refresh

A current-tip direct physical-gfx950 run of corpus revision
`aa54cc86c9ebff3eb840743b36ff8d9b3b2d43c4` executes
`hip_matmul_matmul -m 128 -n 128 -k 128` with `FIXED_ITERATIONS=1`.
The 0.118-second baseline-before, 0.372-second SuperCollider row, and
0.129-second baseline-after each pass all three exact numerical checks.
SuperCollider patches all 739/739 supported LDS accesses, reports complete
static and dynamic coverage with zero mismatches, and leaves the physical GPU
healthy.  This supersedes the July prefix-729 numerical failure; the current
implementation no longer reproduces it.  This clean refresh promotes the
SuperCollider cell to yellow; paired overhead and a reviewed fault still remain
before green acceptance.  The source revision is `9254dfa26842` and the loaded
hook SHA-256 is
`0fe9aab79ab7a2d8b589ee232a263cde69c8fe6d336d87ba487324e02fc980a7`.

### 2026-08-20 Jakub all-profile clean refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-jakub-all-d26f2fa`
records one clean-provenance physical-gfx950 baseline and all four strict
profiles at source revision `d26f2fab95`.  Baseline, SuperCollider,
Record/Replay, Sampled, and Inline Shadow pass all four exact host-reference
oracles in 0.15, 0.34, 189.08, 0.47, and 13.07 seconds respectively.  Every
engine has complete 338/338 access coverage.  Record/Replay, Sampled, and
Inline Shadow also cover 35/35 barriers and have complete static, analysis,
and dynamic verdicts; Record/Replay reports zero diagnostics.  The cells
remain yellow only because paired-overhead and reviewed-fault acceptance are
not part of this clean refresh.  The loaded hook SHA-256 is
`0fe9aab79ab7a2d8b589ee232a263cde69c8fe6d336d87ba487324e02fc980a7`.

### 2026-08-20 Jakub Record/Replay clean refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-jakub-rr-stride1-e589ad6`
records a physical-gfx950 baseline and Record/Replay row at source revision
`e589ad6bbc`.  Both pass all four exact host-reference oracles in 0.13 and
189.27 seconds.  Record/Replay has complete 338/338 access and 35/35 barrier
coverage, zero diagnostics, and complete static, analysis, and dynamic
verdicts.  The target/workload-specific validation cadence is stride 1; it
does not change the production operating point or any other target.  The
preceding diagnostic artifact
`rebase-20260820-gfx950-jakub-rr-e589ad6` proves that stride 256 still selects
no workgroup for this compact four-oracle schedule despite complete static
instrumentation.  The clean cell is now yellow; paired-overhead and
reviewed-fault acceptance remain.  The loaded hook SHA-256 is
`0fe9aab79ab7a2d8b589ee232a263cde69c8fe6d336d87ba487324e02fc980a7`.

### 2026-08-20 Jakub Sampled clean refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-jakub-sampled-d476dd3`
records a clean-tree physical-gfx950 baseline and Sampled row at source
revision `d476dd383d`.  Both pass all four exact host-reference oracles in 0.12
and 0.47 seconds.  Sampled has complete 338/338 access and 35/35 applicable
barrier coverage, zero diagnostics, and a complete dynamic verdict.  This
supersedes the stale entry-island planner rejection and promotes the clean
cell to yellow; paired-overhead and reviewed-fault acceptance are still
required for green.  The loaded hook SHA-256 is
`1d59d7f12eae10ff42d2d5e9eb0b9d1db5f547a1f333cf72ef2973d5f0de4a22`.

### 2026-08-20 CLIP BF16 clean refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-clip-all-039bfbd`
records a clean-tree baseline and all four strict physical-gfx950 profiles at
source revision `039bfbdf75`.  The baseline and every instrumented profile pass
the BF16 cosine oracle.  Baseline, SuperCollider, Record/Replay, Sampled, and
Inline Shadow complete in 1.63, 1.65, 161.03, 1.80, and 9.99 seconds
respectively.  All engines have complete 45/45 access coverage;
Record/Replay, Sampled, and Inline Shadow also cover all 24/24 applicable
barriers, and every dynamic gate is complete.  Record/Replay reports zero
diagnostics.

Commit `c504d1be05` lets the idempotent SuperCollider marker reuse a wide LDS
readback tuple, closing the one-site B128 resource gap at the compact
ordinary-VGPR/AccVGPR boundary with a focused gfx950 host regression.  Commit
`039bfbdf75` gives compact physical CLIP replay a 256-stride validation cadence
and 300-second bound, also under target-resolution tests; this replaces the
standard-stride zero-evidence result without changing the workload.  The
successful current Sampled row supersedes the stale planner rejection.  The
loaded hook SHA-256 is
`0478e8837d6c7c516c6d3d25ab7eea4db2b6cf95fed1e5c360f2a01297bbe47b`.

### 2026-08-20 MFMA attention inventory refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-mfma-all-7c82ed3`
re-audits the rebuilt physical-gfx950 workload at source revision
`7c82ed3b2a`.  The 0.10-second baseline passes both exact host-reference
oracles.  SuperCollider and Sampled also pass both oracles, but the larger
current object invalidates the old 12-access acceptance inventory.

SuperCollider patches all 56 access forms it supports but inventories 58,
leaving two forms unsupported.  Record/Replay patches 33/58 accesses and 5/14
barriers; Sampled patches 33/58 accesses and all 4/4 applicable barriers.  The
25 missing access sites and nine replay barrier sites are resource failures,
not expert-limit omissions.  Record/Replay reports zero diagnostics but exits
under the strict incomplete-coverage contract.  Inline Shadow fails closed
before an oracle because its scalar owner/epoch prologue has no entry-local
VGPR scratch in the exact-context kernel.  This evidence immediately demotes
the four stale cells while the unsupported forms, shared resource gap, and
Inline prologue placement are repaired under focused regressions.  The loaded
hook SHA-256 is
`e90e75343df20766f5b5b6eca2c936965df1eddc17087cd96a150bb8f9f3806c`.

Candidate-tree follow-up artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-mfma-sc-b16-fix`
closes the SuperCollider analysis gap: both `ds_write_b16` sites now use a
CDNA4 `ds_read_u16` readback plus an explicit 16-bit value mask.  Both exact
oracles pass in 0.23 seconds with complete 58/58 access coverage.  A focused
host regression pins the decoded support decision and the emitted readback and
mask sequence.  The candidate hook SHA-256 is
`1d59d7f12eae10ff42d2d5e9eb0b9d1db5f547a1f333cf72ef2973d5f0de4a22`;
a clean-revision all-engine consolidation remains required after the other
three cells are repaired.

A current-tip Record/Replay diagnostic at `2c78f806cf0a`, with dumps under
`/home/ossci/xx/consan-diagnostic-dumps/gfx950-mfma-rr-current`, narrows the
shared resource gap to the ExactContext component.  The fully admitted
FastContext kernel accounts for all 33 patched accesses and 5 patched
barriers.  ExactContext accounts for every omitted site: 25 accesses and 9
barriers in a kernel that declares the full 128-VGPR bank, 106 SGPRs, 612 VGPR
spills, and 18 SGPR spills.  With the runtime stride reduced to one, the
admitted component publishes 2,945 access records and 18 barrier records with
zero diagnostics while both host-reference oracles pass.  The next repair is
therefore a spill-backed transient-scalar route for a full-pressure fixed-stack
CDNA owner, not an access decoder, report ABI, or workload-oracle change.  The
loaded hook SHA-256 is
`cd9529d6c6536208691b03d949ad66a44c78a6fded5646e0ce1c509dc00add69`.

Candidate-tree artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-mfma-rr-full-pressure-fix`
closes that Record/Replay resource gap.  The physical row passes both exact
host-reference oracles in 150.78 seconds with complete 58/58 access and 14/14
barrier coverage, zero diagnostics, and complete static, analysis, and dynamic
verdicts.  This includes all 25 access and nine barrier sites in the
full-pressure ExactContext component.

Fixed-stack CDNA owners can now preserve a borrowed transient scalar window in
their existing private frame and use register-free direct branch routes when
no liveness-dead PC pair or SCC slot exists.  Barriers use a two-branch local
round trip; ordinary indirect barrier relays now use the descriptor-local
assignment rather than the unrelated root fallback, which previously
clobbered live `s[0:1]` in ExactContext.  Focused host regressions cover the
no-dead-register access and barrier routes and the descriptor-local relay
encoding.  The validation manifest also carries a regression-tested
300-second bound for this workload: complete replay scans take about 151
seconds even though the physical kernel oracles finish immediately.  The
candidate hook SHA-256 is
`ae1c5b63c70dafdf433586bfb9ae61c67158b1b2f99e7e2124847e4a941e9008`.

Clean-revision Inline Shadow artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-mfma-inline-frontier-e68fa44`
supersedes the older whole-object rejection.  At source revision `e68fa44965`,
both exact host-reference oracles pass in 9.25 seconds and the admitted
FastContext component publishes 27,776 visible exact-shadow events with zero
dynamic incomplete state.  Static coverage remains incomplete at 33/58
accesses plus 5/14 barriers: all 25 access and nine barrier omissions are still
in the full-pressure ExactContext component.  The cell is therefore orange,
and component-local persistent-state placement is the next repair boundary.

A current candidate-tree physical follow-up closes that Inline Shadow
coverage gap.  Both exact host-reference oracles pass in 9.38 seconds with
all 58/58 accesses and 14/14 barriers patched.  The run publishes 8,704
visible exact-shadow cells, reports zero diagnostics, undercoverage,
overflow, or malformed state, and reaches complete static, analysis, and
dynamic verdicts.  ExactContext's 25 accesses and nine barriers now execute
under the same behavioral contract as FastContext.

The full-pressure Exact owner cannot retain the code-object-wide hardware
dispatch-ID SGPR pair.  Its kernel-entry prologue now captures that identity
in private state, and each exact-shadow access reloads it into the
owner-local indirect-PC pair after routing has made the pair available.  A
focused host regression verifies the private layout, entry capture, access
reload, and final-validation bounds; 1,236 ConSan host tests pass and one
benchmark is skipped.  The loaded candidate hook SHA-256 is
`13204ce2e649619be7a4cb5edb108694378866ae05a759b99d386b5f5c175ffa`.
The cell is yellow pending paired-overhead and reviewed-fault acceptance.

### 2026-08-20 TP2 Sampled clean refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-tp2-sampled-0ed9a29`
records a clean-provenance physical-gfx950 baseline and Sampled row at source
revision `0ed9a29790`.  The 5.62-second baseline and 9.97-second Sampled run
both pass all three exact TP2 prefill, decode, and combined oracles.  Sampled
has complete 936/936 access and 120/120 applicable-barrier coverage, zero
forbidden diagnostics, and complete static, analysis, and dynamic verdicts.
This supersedes the stale ordinary-VGPR/AccVGPR placement rejection and moves
the cell from red to yellow.  Paired-overhead and reviewed-fault acceptance
remain.  The loaded hook SHA-256 is
`13204ce2e649619be7a4cb5edb108694378866ae05a759b99d386b5f5c175ffa`.

### 2026-08-20 TP1 prefill clean refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-tp1-prefill-all-b552df9`
records a clean-tree baseline and all four strict physical-gfx950 profiles at
source revision `b552df9306`.  The baseline, SuperCollider, Sampled, and Inline
Shadow pass the exact prefill oracle in 3.55, 3.62, 3.92, and 16.77 seconds.
Their static coverage is complete: 120/120 accesses for all three engines,
24/24 applicable barriers for Sampled, and 31/31 barriers for Inline Shadow.
The successful Sampled row supersedes the older ordinary-VGPR/AccVGPR planner
rejection.

The same artifact showed that Record/Replay's production 65,536 stride selected
no runtime evidence despite complete 120/120 access and 31/31 barrier
instrumentation.  Commit `43661cd604` gives this compact physical validation
schedule the same 256-stride cadence already required by its gfx1250 form and
a 300-second bound for host replay analysis of the conservative 2M-slot table.
Clean artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-tp1-prefill-rr-43661cd`
passes the exact oracle in 171.29 seconds with 9,216 visible events, zero
diagnostics, complete 120/120 plus 31/31 static coverage, and a complete dynamic
verdict.  All rows loaded hook SHA-256
`59ae90f075525cb84925717b7322d8f0d98b59c7a41b9d3e81e989a5ce0615c4`.

### 2026-08-20 TP1 decode/combined clean refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-tp1-decode-all-40af1c1`
records a clean-tree baseline and all four strict physical-gfx950 profiles at
source revision `40af1c1a2f`.  The baseline, SuperCollider, Sampled, and Inline
Shadow pass both exact decode/combined oracles in 3.67, 3.35, 4.44, and 29.67
seconds.  Their static coverage is complete: 240/240 accesses for every
engine, 48/48 applicable barriers for Sampled, and 62/62 barriers for Inline
Shadow.  The successful Sampled row supersedes the older
ordinary-VGPR/AccVGPR planner rejection.

The initial Record/Replay row likewise had complete 240/240 plus 62/62 static
coverage, but the production stride selected no runtime evidence.  Commit
`14a186ffb0` extends the regression-tested gfx950 TP1 validation cadence and
300-second host-analysis bound to decode/combined.  Clean artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-tp1-decode-rr-14a186f`
passes both exact oracles in 166.03 seconds with 11,520 visible events, zero
diagnostics, and a complete dynamic verdict.  All rows loaded hook SHA-256
`59ae90f075525cb84925717b7322d8f0d98b59c7a41b9d3e81e989a5ce0615c4`.

### 2026-08-20 rebuilt D128-pressure regression

A clean current-tip physical-gfx950 Sampled refresh at source revision
`409fe514da` is retained in
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-d128-pressure-sampled-409fe51`.
It supersedes the earlier entry-island rejection: all four exact clean oracles
pass in 1.36 seconds, with zero diagnostics and dynamically complete execution.
The static result remains incomplete, however.  Only 73/252 accesses and all
4/4 applicable barriers patch; the other 179 access sites fail resource
planning.  This promotes Sampled from red to orange while owner-level resource
isolation continues.

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-d128-pressure-all-3d1b708`
records a clean-tree physical-gfx950 run at source revision `3d1b708b08` with
the rebuilt hip-moi executable hash
`a5054ec6b783274f3613637ee91193e9359b67cc95a53ea3459d713e2355e33c`.
The workload now contains four kernels and 252 discovered accesses, so its
older 12-site accepted bundles do not describe the current executable.

SuperCollider preserves all four numeric oracles but is incomplete at 112/248
supported accesses; all 136 omissions are placement-or-lowering failures.
Record/Replay and Inline Shadow both preserve the first three oracles and
instrument 73/252 accesses plus 9/36 barriers.  Their fourth kernel has a
640-byte runtime private segment and faults in the unrestricted runs.  Sampled
fails closed before execution with `ConSan MOI sampled probe could not encode
its entry island`.

A same-day bounded physical follow-up found that `s_set_gpr_idx_on` made the
kernel's apparently dead scratch VGPRs unsafe: ordinary VOP operands can name
M0-indexed registers that point liveness cannot see.  ConSan now reuses the
liveness analysis's whole-scope indirect-VGPR completeness proof and selects a
fresh or spill-preserved window instead.  The focused host regression and all
671 `ConSanMoi.*` tests passed at that revision; on physical gfx950 the formerly
corrupting 28-access prefix passed the fourth exact numeric oracle.  Adding the
29th access, `ds_read_b32 v50, v50 offset:132` at text offset `0x4d470`, exposed
a separate interaction with an adjacent pair of full barriers.  Consecutive
full barriers with no intervening instruction impose the same ordering edge,
so recording both is redundant.  ConSan now classifies the second member as
semantically not applicable and leaves that guest barrier in place rather than
relocating it into a second record trampoline.  A focused host regression
checks the disposition, the single record probe, and the preserved second
barrier.  The bounded 29-access physical run now passes all four numeric
oracles with zero diagnostics and a clean exit; its fourth dispatch reports
29/118 accesses and 5/10 nonredundant barriers patched, dynamic completion, and
no replay saturation or invalid-site tokens.  All 672 current `ConSanMoi.*`
host tests pass.  Static coverage is still
incomplete, and Inline Shadow's corresponding behavior remains to be isolated,
so neither row is promoted beyond orange.

A further same-day Inline isolation found that its private-state entry
prologue restored a newly inserted dispatch-ID preload into the guest ABI and
then reused the overlapping guest `s0:s4` as workgroup-key temporaries.  The
prologue now composes a fixed-private scalar spill with its existing temporary
VGPR spill, preserving exactly the live entry-ABI overlap after preload
remapping and restoring it before guest entry.  Host regressions cover the
shared spill allocator and the complete emitted gfx950 prologue.  On physical
gfx950, the previously nil-faulting fourth oracle passes with its first access
probe enabled.  The unrestricted Inline run still faults only on that fourth
oracle, now at nonzero address `0xc00e2000`, which isolates a separate
access-probe defect without promoting the matrix cell.  A tighter physical
bisect with barrier and atomic instrumentation disabled passes prefixes of 8,
10, and 11 access probes and first faults when a twelfth probe is added.  The
twelfth access (`ds_write_b32 v97, v32` at text offset `0x4c214`) passes by
itself with its complete 30-SGPR fixed-private spill, and the eleventh and
twelfth probes pass as a pair.  Conversely, replacing that access with the
next candidate while retaining a 12-site prefix still faults.  The remaining
defect is therefore aggregate cross-site Inline state or its conflict path,
not the individual instruction or scalar-spill encoding.  The passing
11-site prefix emits 8,704 conflict diagnostics (704 retained and 8,000
capacity-dropped) when synchronization tracking is deliberately disabled,
giving the next investigation a bounded high-conflict reproducer.

The 12-site boundary was the first entry prologue too far from guest entry for
a direct SOPP return.  Its absolute return reused `s0:s1` after restoring the
guest ABI, corrupting the incoming kernarg pointer.  Long entry-prologue
returns now use VCC, whose value is not an incoming kernel ABI input, matching
the existing local-entry-island contract.  A host regression constructs a
private-epoch prologue beyond the SOPP range with `s0:s1` live at entry and
checks that the generated return uses VCC rather than the restored ordinary
SGPR pair.  On physical gfx950, the formerly faulting 12-access fourth oracle
now passes in 1.262 seconds with 12/252 accesses patched.  Deliberately
disabling synchronization tracking still overflows the small report buffer,
so this bounded diagnostic-pressure run is not an acceptance artifact and the
cell remains orange.  The first gfx950 RocJitsu simulator rerun reached the
oracle without the former invalid-address fault but produced different
numerical output because the VM's pre-execution null-target guard treated the
architectural VCC selector as ordinary SGPR storage and halted the wave.  The
guard now reads the decoded scalar operand, and a focused VM regression keeps
the raw pseudo-SGPR slots zero while transferring through a nonzero VCC target.
With that fix, the bounded 12-access fourth oracle also passes in the gfx950
simulator.  The remaining orange qualification is therefore the unrestricted
coverage and acceptance work, not a known physical-versus-simulator semantic
disagreement.

Fresh unrestricted physical artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-d128-pressure-inline-full-f733128`
then passed all four clean workload oracles in 20.26 seconds with zero
diagnostics.  Static analysis remains incomplete at 73/252 accesses plus 5/28
nonredundant barriers.  Debug inventory partitions that result exactly by
independent owner component: `DoubleBuffered32KeyPressure/FastContext` supplies
all 73 access and 5 barrier probes, while `FullKvDoubleBuffered/FastContext`,
`FullKvDoubleBuffered/ExactContext`, and
`DoubleBuffered32KeyPressure/ExactContext` account respectively for 45/5,
55/9, and 79/9 access/barrier resource failures.  Every omission has the typed
reason `forbidden_overlap`; the three excluded owners allocate 104 SGPRs and
cannot retain the current code-object-wide hardware dispatch-ID pair, whereas
the admitted owner allocates 96.  This refresh closes the outstanding physical
correctness question and supersedes the older 9/36 raw-barrier denominator,
but it does not promote the cell: cross-dispatch identity for full-pressure
CDNA owners needs a higher-level state design rather than weakening the
existing fail-closed owner exclusion.

### 2026-08-21 D128-pressure SuperCollider full-coverage refresh

Candidate-tree physical-gfx950 artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-d128-pressure-supercollider-current`
passes all four exact host-reference oracles under the strict SuperCollider
profile in 1.40 seconds.  All 252/252 supported accesses patch, and the run
reports complete static, analysis, and dynamic verdicts.  This supersedes the
112/248 coverage frontier: the intervening shared-function ownership and dense
relay repairs are now guarded by focused host tests and the checked-in
correct/incorrect CDNA dense-routing device workload.  Paired overhead,
reviewed-fault, and clean-provenance acceptance remain before green promotion.

### 2026-08-21 D128-pressure Record/Replay full-coverage refresh

Candidate-tree physical-gfx950 artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-d128-pressure-record-replay-scc-fixed`
supersedes the incomplete-barrier artifact at
`rebase-20260821-gfx950-d128-pressure-record-replay-timeout240`.  Under a
diagnostic 240-second process bound, strict Record/Replay passes all four exact
host-reference oracles in 42.27 seconds, emits zero diagnostics, patches all
252/252 supported accesses and all 28/28 nonredundant barriers, and reports
complete static, analysis, and dynamic verdicts.  The remaining clean-row gap
is the ordinary 30-second latency contract; paired overhead and reviewed-fault
acceptance also remain before green promotion.

Two independent dense-routing defects caused the prior result.  The barrier
router was enabled only above a compact site-count threshold even when early
barriers could not reach its appended relay, leaving 22 barriers stranded.
After repairing that reachability decision, a late `ds_write_b16` exposed that
the access dispatcher entered a direct body with the SCC produced by its route
comparison rather than the guest SCC saved at the entry island.  The barrier
threshold fix has a host regression below the former count limit.  The SCC fix
has both a host emission regression and a checked-in correct/incorrect CDNA
B16 device pair, exercised under Record/Replay on simulated gfx942/gfx950 and
physical gfx950.

The clean-source refresh at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-d128-pressure-rr-current-881-v1`
closes the remaining gates. Baseline-before and baseline-after take 66 and 85
ms, while the exact four-oracle Record/Replay row takes 1,463 ms, for a 75.5 ms
paired baseline and 19.38x slowdown. Coverage is complete at 252/252 accesses
and 28/28 barriers with zero diagnostics or overflow, and all static,
analysis, and dynamic verdicts are complete. The prospectively frozen fault
described above is accepted with exact-one accounting, bounded report memory,
cleanup, healthy pre/post probes, an unchanged hook hash, and clean source
revision `881fffe487778df6452db64ce487f324aa885bb7`. The cell is green. Its
redundant adjacent-barrier idiom is also transported into the architecture-
general checked-in double-buffered pipeline correct/incorrect contract rather
than remaining physical-gfx950-only coverage.

### 2026-08-21 D128-pressure Inline full-coverage refresh

Candidate-tree physical-gfx950 artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-d128-pressure-inline-dense-scc-fixed`
supersedes the component-exclusion result above.  Strict Inline Shadow passes
all four exact host-reference oracles in 22.32 seconds, emits zero forbidden
diagnostics, patches all 252/252 supported accesses and all 28/28
nonredundant barriers, and reports complete static, analysis, and dynamic
verdicts.

Two scalar-pressure routing defects were exposed in sequence.  An explicit
access route stored its route number and incoming SCC in one SGPR but left the
tagged route for the common return to interpret as a Boolean SCC snapshot.
The call-return-derived barrier router discarded SCC altogether when its
derived key shared that SGPR.  Dense access routes now normalize their tagged
key to zero or one before entering the body; dense call routes carry SCC in
the aligned return PC's low bit, decode the route for dispatch, and likewise
normalize SCC before entering the body.  The host regression covers an SCC-
dependent continuation after both a late access and a dense barrier under the
aliased scalar ABI.  Spill-backed private-epoch barriers additionally drain
guest VMEM before saving a temporary, guarding the adjacent
`scratch_load; barrier; waitcnt; use` idiom present in this workload.  Paired
overhead and a reviewed fault campaign remain before green acceptance.

### 2026-08-20 Jakub physical clean refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-jakub-clean-all-27c0851`
records one physical-gfx950 baseline and all four current clean profiles.  The
baseline passes all four parameterized exact oracles.  SuperCollider preserves
those results with complete 338/338 access coverage.  Inline Shadow also
preserves all four results with complete 338/338 accesses plus 35/35 barriers,
zero diagnostics, and a complete dynamic verdict.  Their paired-overhead and
reviewed-fault bundles remain to be collected.

Record/Replay preserves all four numeric results and statically patches every
one of the same 338 accesses and 35 barriers, but the standard runtime stride
of 65,536 observes no records across the four one-workgroup dispatches.  It
therefore exits through the strict missing-evidence gate rather than qualifying
the row.  Exploratory physical runs with denser strides retained all four
oracles and committed 107,520 records at stride 1 or 33,792 records from one
selected dispatch at stride 4, but both exceeded 90 seconds in host replay.
A replay fix now excludes unpublished fixed-table slots from event scheduling
and state sizing, with a 262,144-slot sparse-table regression; processing the
actual published records remains too slow, so the row stays orange.  Sampled
fails earlier: its planner cannot encode the first entry island and rejects the
code object before an oracle.  These are distinct engine issues; neither is
reported as a workload failure.

### 2026-08-20 D128-block Sampled fast-gate fallback

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-d128-block-sampled-fast-gate-fix`
records a physical-gfx950 baseline and Sampled run from the exact candidate
tree that introduces the fallback, based on source revision `da87610c6d`.  The
instrumentation hook has SHA-256
`832d51d66cf355f9ee3412e1939f41689345b46647b9b1e9683056c20a8d1f4f`.
The baseline passes both exact clean oracles in 0.129 seconds.  Sampled also
passes both exact clean oracles and exits successfully in 1.224 seconds, with
59/128 access sites, all 49/49 barriers, and a complete dynamic verdict.

The prior rejection was caused by the workgroup/dispatch gate.  At the gfx950
ordinary-VGPR/AccVGPR boundary, entry-captured workgroup state can live in
private memory and cannot be read before the spill-backed body saves guest
VGPRs.  A later gfx1250 clean-workload run proved that simply omitting this
gate is behaviorally unsound because unrelated workgroups can collide in the
finite causal-window banks.  The planner now emits the same workgroup
selection inside the spill-safe body before its independent LDS-cell
selection.  A focused host regression constructs this exact private-state
boundary and requires a successful transform, retained workgroup state, the
typed body-gate warning, and final validation.  Static coverage is still
incomplete because 69 access sites fail resource planning, so this cell is
orange rather than green; the physical row must also be refreshed after the
body-gate correction.

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-d128-block-sampled-body-gate-fix`
performs that physical refresh with candidate-tree hook SHA-256
`18d108f47cf3972487eb96cac0ee0bebe0c04212359cbb1293954bc703823c01`.
Both exact host-reference tests pass, the process exits cleanly in 0.819
seconds, and the report contains zero diagnostics or sampled conflicts.  The
static result is unchanged at 59/128 accesses and 49/49 barriers, with a
complete dynamic verdict; the row therefore remains orange solely because of
the 69 resource-planning omissions.

### Current-matrix executable audit

The exact validation IDs below are the workload definitions used by
`consan_validation.py --target gfx950`.  The original workload-scoped `doctor`
audit ran on 2026-07-22; the Jakub row and complete six-workload hip-moi
campaign audit were refreshed on 2026-07-26. “Runnable” means that all required
sources, assets, target-native executables, and workspace tools resolve in this
workspace now; it is not an instrumentation acceptance claim or a claim that
an unpublished companion repository commit is remotely reachable.

| Validation ID | Current gfx950 availability | Exact definition |
|---|---|---|
| `qwen-prefill` | **Runnable and baseline-passed** | Input, expected output, parameters, and the locally compiled gfx950 VMFB exist.  The direct physical-device oracle passes; a bounded patch-count reproducer exposed and verified the SuperCollider wave64-VCC preservation fix. |
| `tp1-prefill` | **Runnable** | Sharktank `toy_llama.mlir` + `toy_llama.irpa`, prefill mode; workload-scoped doctor passes. |
| `tp1-decode-combined` | **Runnable** | The same TP1 assets, decode and combined modes; workload-scoped doctor passes. |
| `tp2-family` | **Runnable** | Sharktank `toy_llama_tp2.mlir` with common, rank-0, and rank-1 parameters; workload-scoped doctor passes. |
| `clip-bf16` | **Runnable** | Sharktank toy CLIP BF16 MLIR, parameters, input, and expected result; workload-scoped doctor passes. |
| `d128-block` | **Runnable and smoke-passed** | `hip_moi_instrumented_cdna4_d128_attention_block_test`, `HipMoiCdna4D128AttentionBlock.SampledFastContextMatchesHostReference`; physical gfx950 oracle passes in 120 ms. |
| `d128-pressure` | **Runnable and smoke-passed** | `hip_moi_instrumented_cdna4_d128_attention_pressure_test`, `HipMoiCdna4D128AttentionPressure.FullKvDoubleBufferedExactContextMatchesHostReference`; physical gfx950 oracle passes in 188 ms. |
| `wmma-attention` | **Runnable and smoke-passed** | `hip_moi_instrumented_cdna4_mfma_attention_block_test`, `HipMoiCdna4MfmaAttentionBlock.ExactContextMatchesHostReference`; physical gfx950 oracle passes in 142 ms.  The historical validation ID is retained, but the native operation is MFMA rather than WMMA. |
| `streamk-arrival` | **Runnable and smoke-passed** | `hip_moi_instrumented_cdna4_mfma_streamk_arrival_counter_test`, `HipMoiCdna4MfmaStreamKArrivalCounter.AcqRelFetchAddOrdersMfmaPartials`; physical gfx950 oracle passes in 92 ms. |
| `tree-atomic-or` | **Runnable and smoke-passed** | `hip_moi_instrumented_cdna4_mfma_streamk_tree_atomic_or_test`, `HipMoiCdna4MfmaStreamKTreeAtomicOr.AcqRelBitmaskOrdersMfmaPartials`; physical gfx950 oracle passes in 101 ms. |
| `jakub-attention` | **Locally runnable and simulator-smoke-passed** | hip-moi `29a1c212183b65f1ec9200b24a445862532e4dd8` builds `hip_moi_reference_cdna4_jakub_matmul`; `SafeFp16Packed/JakubCdna4MatmulReference.MatchesHostReference/*` passes both parameterized cases through the gfx950 RocJITsu simulator. That companion commit is not yet contained by a fetched remote ref. |

The five previously available smoke commands use the physical device through
the workspace TheRock runtime, with software-model environment variables
unset.  The new Jakub row has target-native simulator evidence only; it is not
yet counted as physical gfx950 or instrumentation acceptance evidence.

## RocJitsu test-corpus expansion

The source-built kernel survey below began at historical corpus revision
`aa54cc86c9eb`.  Revision `61b5af0b5ee9` first enabled the bounded gfx950 HIP
Stream-K cases, and `46a4c58a7be8` packaged the runtime-generated gfx950
Tensile row described below.  The current workspace corpus at descendant
revision `0db836e7bd8c` retains those cases and adds the runner provenance and
isolation coverage used for the fresh qualification.  The older pre-generated
Tensile artifact tree remains gfx1250-only and may not be relabeled as gfx950
evidence; gfx950 code objects are generated from the checked-in gfx950 YAML.
A source-built cell is promoted only after the case has an independent oracle,
a retained target-native inventory, and a standard-profile assessment for that
flavor.  A profile that runs the oracle without finding an applicable code
object is compatibility evidence rather than instrumentation acceptance.

The gfx950 corpus configuration enables HIP matmul, HipKittens, HIP Stream-K,
and rocBLAS. It has no run-time skip list. The portable configuration still
skips the large 4096³ HIP matmul case and both four-wave FP8 HipKittens cases,
but the clean targeted gfx950 build described below explicitly enables and
qualifies the two bounded FP8 cases.

### Corpus executable audit

| Tracking unit | Current gfx950 availability | Exact source/build contract |
|---|---|---|
| `hip_matmul_matmul::m128_n128_k128` | **Runnable and smoke-passed** | `corpus/kernels/cases/hip-matmul/matmul/case.json`; executable `rocjitsu-test-corpus-build/kernels-gfx950-hip-matmul/cases/hip-matmul/hip_matmul_matmul`; `-m 128 -n 128 -k 128`, `FIXED_ITERATIONS=1`.  All three selected MFMA/shared-memory kernels pass correctness on the physical gfx950. |
| `hipkittens_gemm_bf16fp32_16x32::m256_n256_k256` | **Runnable and baseline-passed** | A clean gfx950-only build uses GCC 13's OpenMP host runtime with TheRock's `amdclang++` HIP compiler. The physical-gfx950 exact oracle passes with zero maximum absolute error. |
| `hipkittens_gemm_mxfp8_4wave::m256_n256_k256` | **Runnable and baseline-passed** | The same clean build compiles the four-wave MXFP8 case despite the portable corpus skip and its physical-gfx950 exact oracle passes all 65,536 outputs. |
| `hipkittens_gemm_fp8fp32_4wave::m256_n256_k256` | **Runnable and baseline-passed** | Corpus revision `14180b3` keeps the rotating-buffer count runtime configurable without incorrectly requiring it to be a constant expression. The checked-in `--rotating-buffer-count 4` regression now compiles and its physical-gfx950 optimized/reference oracle passes. |
| `hip_streamk_simple::m256_n256_k256` and `hip_streamk_two_tile::m256_n256_k256` | **Runnable and physical-baseline-passed** | At corpus revision `aa54cc8`, a gfx950-only build uses the current workspace TheRock SDK plus its staged rocThrust and rocPRIM packages.  The simple exact oracle uses one run, `--grid 4`, and `--validate`; the two-tile oracle uses one run and `--validate`.  Both pass with zero errors on the physical gfx950 under a 120-second per-case bound. |
| `rocblas_sgemm` exact cases | **Runnable and baseline-passed** | A dedicated gfx950 build now provides `rocjitsu-test-corpus-build/kernels-gfx950-rocblas/cases/rocblas/rocblas_sgemm`.  `RocblasGemmTest.Square_64x64` passes its physical-device baseline in 183 ms; the bounded strict Record/Replay assessment below records the current instrumentation frontier. |

HIP-matmul, HipKittens, HIP Stream-K, and rocBLAS are runnable today. The
source-matched physical assessments below distinguish fully qualified green
cells from clean-only yellow cells; a clean oracle alone is never promoted as
detector acceptance.

| Priority | Tracking unit | SuperCollider | Record/Replay | Sampled | Inline Shadow | Why it matters and next proof |
|---|---|---|---|---|---|---|
| P0 | `hip_matmul_matmul::m128_n128_k128` | 🟨 Current physical clean run passes all three exact numerical checks with complete 739/739 supported LDS-access coverage, complete static and dynamic verdicts, and zero mismatches; paired overhead and reviewed-fault acceptance remain | 🟩 Canonical physical paired run passes all three exact numerical oracles with a 279.37-ms process baseline and 787.89-ms Record/Replay process time (2.82×), complete 739/739 access and 109/109 barrier coverage, and zero clean diagnostics. A prospectively reviewed exact-one removal of the selected i8 pipeline's initial LDS tile-publication barrier produces 1,020 attributable replay diagnostics with complete 739/739 access and 108/108 remaining-barrier coverage, no overflow/incomplete state, and healthy pre/post GPU probes | 🟩 Current canonical paired run passes all three exact numerical oracles with a 276.976-ms mean process baseline and 571.145-ms Sampled process time (2.06×), complete 739/739 access and 109/109 barrier coverage, and zero clean diagnostics. A prospectively reviewed exact-one removal of the selected i8 pipeline's initial LDS tile-publication barrier emits 24 Sampled conflicts with complete 739/739 access and 108/108 surviving-barrier coverage, 392 losslessly claimed windows, bounded memory and cleanup, exact mutation accounting, and healthy pre/post GPU probes | 🟨 Final candidate-tree physical run passes all three exact numerical oracles in 18.57 seconds with 188,416 visible evidence events, zero diagnostics, overflow, or dynamic-incomplete state, complete 739/739 access and 109/109 barrier coverage, and complete static and dynamic verdicts; paired overhead and reviewed-fault acceptance remain | Native gfx950 MFMA kernels use shared-memory tiling, repeated workgroup barriers, and a double-buffered LDS path. Record/Replay and Sampled are fully qualified. Their reviewed faults are the same observable publication edge already protected by `CdnaMfmaPipeline`, `MultiWaveGemmTile`, and `CdnaMixedPersistentState`; SuperCollider and Inline Shadow still need paired-overhead and reviewed-fault evidence. |
| P0 | `hipkittens_gemm_bf16fp32_16x32::m256_n256_k256` | 🟨 Physical strict clean run preserves the exact zero-error oracle with complete 96/96 access coverage and complete static/dynamic verdicts; paired overhead and reviewed-fault acceptance remain | 🟩 Clean-source paired qualification preserves the exact zero-error oracle with complete 128/128 supported accesses--including all 32 direct-global-to-LDS writers--and 32/32 barriers, zero diagnostics, and a 134.08-ms paired baseline versus 606.10-ms Record/Replay time (4.52×). The prospectively frozen exact-one phase-reuse barrier removal is reached and accepted through a detector-owned runtime diagnostic with complete 128/128 access and 31/31 remaining-barrier coverage, no overflow/incomplete state, and complete containment, cleanup, and health evidence | 🟩 Current canonical paired run preserves the exact zero-error oracle with a 133.037-ms mean process baseline and 195.963-ms Sampled process time (1.47×), complete 128/128 access and 32/32 barrier coverage, and zero clean diagnostics. A prospectively reviewed exact-one removal of the reached phase-reuse barrier emits 24 Sampled conflicts with complete 128/128 access and 31/31 surviving-barrier coverage, all 1,024 selected windows claimed, bounded memory and cleanup, exact mutation accounting, and healthy pre/post GPU probes | 🟨 Current physical strict run preserves the exact zero-error oracle in 22 seconds, patches all 96/96 accesses and 32/32 barriers, publishes 229,376 clean evidence events, and has complete static/dynamic verdicts; paired-overhead and reviewed-fault acceptance remain | Explicit gfx950 case with dynamic LDS, wide DS reads, direct global-to-LDS traffic, a deep 32-barrier schedule, and MFMA. Record/Replay and Sampled qualify both the DS readers and the implicit physical-lane/M0 destinations of direct-to-LDS producers; `CdnaDirectToLdsPublication` and `CdnaMfmaPipeline` transport those contracts to gfx942/gfx950 simulation and physical gfx950. |
| P1 | `hipkittens_gemm_fp8fp32_4wave::m256_n256_k256` | 🟨 Current physical strict run preserves the exact optimized/reference oracle, patches all 64/64 supported LDS accesses, reports zero mismatches, and has complete analysis, static, and dynamic verdicts; paired overhead and reviewed-fault acceptance remain | 🟩 Clean-source paired qualification preserves the exact optimized/reference oracle with complete 96/96 accesses--including all 32 direct-global-to-LDS writers--and 5/5 barriers, zero diagnostics, and a 435.70-ms paired baseline versus 718.95-ms Record/Replay time (1.65×). A prospectively frozen exact-one drop of the reached initial tile-publication barrier produces the required detector-owned diagnostic with complete 96/96 access and 4/4 remaining-barrier coverage, no overflow/incomplete state, and complete containment, cleanup, and health evidence | 🟩 Current canonical paired run preserves the exact optimized/reference oracle with a 287.859-ms mean process baseline and 542.311-ms Sampled process time (1.88×), complete 96/96 access and 5/5 barrier coverage, and zero clean diagnostics. A prospectively reviewed exact-one initial direct-to-LDS tile-publication removal emits 24 Sampled conflicts with complete 96/96 access and 4/4 surviving-barrier coverage, all 384 visible windows claimed, bounded cleanup, exact containment, and healthy pre/post GPU probes | 🟨 Current physical strict run preserves the exact oracle in 16.17 seconds with complete 64/64 access and 5/5 barrier coverage, 196,608 visible evidence events, zero diagnostics or malformed/incomplete state, and complete verdicts; paired-overhead and reviewed-fault evidence remain | Explicit gfx950 four-wave FP8 case. Corpus revision `14180b3` repairs the compile blocker; Record/Replay and Sampled qualify the implicit direct-to-LDS producer addresses as well as the wide DS consumers. |
| P1 | `hipkittens_gemm_mxfp8_4wave::m256_n256_k256` | 🟨 Current physical strict run passes all 65,536 exact output checks with complete 64/64 supported LDS-access coverage, zero check mismatches, and complete static/dynamic verdicts; paired overhead and reviewed-fault acceptance remain | 🟩 Clean-source paired qualification passes all 65,536 exact checks with complete 96/96 accesses--including all 32 direct-global-to-LDS writers--and 5/5 barriers, zero diagnostics, and a 226.24-ms paired baseline versus 725.83-ms Record/Replay time (3.21×). A prospectively frozen exact-one drop of the reached initial tile-publication barrier produces the required detector-owned diagnostic with complete 96/96 access and 4/4 remaining-barrier coverage, no overflow/incomplete state, and complete containment, cleanup, and health evidence | 🟩 Current canonical paired run passes all 65,536 exact output checks with a 229.447-ms mean process baseline and 267.764-ms Sampled process time (1.17×), complete 96/96 access and 5/5 barrier coverage, and zero clean diagnostics. A prospectively reviewed exact-one initial direct-to-LDS tile-publication removal emits 24 Sampled conflicts with complete 96/96 access and 4/4 surviving-barrier coverage, all 384 visible windows claimed, bounded cleanup, exact containment, and healthy pre/post GPU probes | 🟨 Current physical strict run passes all 65,536 exact checks in 17.11 seconds with complete 64/64 access and 5/5 barrier coverage, 294,912 visible evidence events, zero diagnostics or malformed/incomplete state, and complete static/dynamic verdicts; paired overhead and reviewed-fault acceptance remain | Explicit gfx950 four-wave microscaling GEMM. Record/Replay and Sampled qualify the implicit direct-to-LDS producer addresses as well as the wide DS consumers while retaining all 65,536 output checks. |
| P1 | `hip_streamk_simple::m256_n256_k256` gfx950 port | 🟨 Current physical strict run preserves the exact zero-error oracle, patches all 32/32 supported LDS accesses, and reports complete static and dynamic verdicts; paired overhead and reviewed-fault acceptance remain | 🟩 Clean-source paired qualification preserves the exact `Errors: 0` oracle with complete 32/32 access, 3/3 barrier, and 2/2 fence coverage, zero clean diagnostics, and a 155.74-ms paired baseline versus 1,091.97-ms Record/Replay time (7.01×). A prospectively frozen exact-one drop of the reached loop phase-publication barrier produces 36,864 attributable replay diagnostics with complete 32/32 access, 2/2 remaining-barrier, and 2/2 fence coverage, no overflow/incomplete state, bounded report memory and cleanup, and healthy pre/post GPU probes | 🟩 Current-tip paired qualification preserves the exact `Errors: 0` oracle with complete 32/32 access, 3/3 barrier, and 2/2 atomic coverage and zero clean diagnostics. Sampled takes 258.367 ms against a 141.029-ms bracketing mean baseline (1.83×). A prospectively reviewed exact-one phase-publication removal produces the required Sampled conflict with complete 32/32 access, 2/2 surviving-barrier, and 2/2 atomic coverage, all 192 selected windows claimed, bounded report memory and cleanup, exact containment, and healthy pre/post GPU probes | 🟨 Current physical standard-profile run preserves the exact oracle in 18.53 seconds, patches all 32/32 accesses plus 3/3 barriers, publishes 4,800,512 visible evidence events with zero diagnostics or incomplete state, and reports complete static/dynamic verdicts; paired-overhead and reviewed-fault acceptance remain | The current manifest pins the exact m=n=k=256, grid=4, one-run oracle and the checked-in architecture-general Stream-K pairs own the corresponding publication/conflict behavior. Record/Replay and Sampled are fully qualified; paired-overhead and reviewed-fault evidence remain for SuperCollider and Inline Shadow. |
| P1 | `hip_streamk_two_tile::m256_n256_k256` gfx950 port | 🟨 Current physical strict run preserves the exact zero-error oracle, patches all 80/80 supported LDS accesses, and reports complete static and dynamic verdicts; paired overhead and reviewed-fault acceptance remain | 🟩 Clean-source paired qualification preserves the exact `Errors: 0` oracle with complete 80/80 access, 5/5 barrier, and 2/2 fence coverage, zero clean diagnostics or dropped records, and a 305.30-ms paired baseline versus 9,479.56-ms Record/Replay time (31.05×). Replay processes all 2,359,296 committed accesses, 2,208 barriers, and 20,525 fences while retaining only one live release-metadata component. A prospectively frozen exact-one drop of the reached loop publication barrier produces 281,397 attributable replay diagnostics with complete 80/80 access, 4/4 surviving-barrier, and 2/2 fence coverage, no overflow/incomplete state, bounded report memory and cleanup, and healthy pre/post GPU probes | 🟩 Current-tip paired qualification preserves the exact `Errors: 0` oracle with complete 80/80 access, 5/5 barrier, and 2/2 atomic coverage and zero clean diagnostics. Sampled takes 566.342 ms against a 292.727-ms bracketing mean baseline (1.94×). A prospectively reviewed exact-one phase-publication removal emits six required Sampled conflicts with complete 80/80 access, 4/4 surviving-barrier, and 2/2 atomic coverage, all 384 selected windows claimed, bounded report memory and cleanup, exact containment, and healthy pre/post GPU probes | 🟨 Current physical standard-profile run preserves the exact oracle in 17.72 seconds, patches all 80/80 accesses plus 5/5 barriers, publishes 10,485,760 visible evidence events with zero diagnostics or incomplete state, and reports complete static/dynamic verdicts; paired-overhead and reviewed-fault acceptance remain | The checked-in access-heavy `CdnaStreamkTwoTile` pair owns the target-native relay and publication shape, while the architecture-general Stream-K pair owns the portable last-arriver contract. Record/Replay and Sampled are fully qualified; paired-overhead and reviewed-fault evidence remain for SuperCollider and Inline Shadow. |
| P2 | `rocblas_sgemm` compact exact cases | 🟨 The repaired physical strict run passes the exact `Square_64x64` oracle in 28.963 seconds, patches all 49,435/49,435 supported LDS accesses, reports no check mismatches, and has complete static/analysis/dynamic verdicts; paired overhead and reviewed-fault evidence remain | 🟩 The registered exact `Square_64x64` row passes 292.12- and 282.59-ms bracketing baselines and the 17,125.36-ms Record/Replay run (59.60×), with complete 49,435/49,435 access and 4,997/4,997 barrier coverage and zero clean diagnostics. A prospectively frozen exact-one drop of the sole observed dispatch's reviewed tile-publication barrier is reached and produces 4,086 detector-owned replay diagnostics while the exact oracle still passes; all 49,435 accesses and 4,996 surviving barriers remain covered, mutation/reservation evidence is complete, and pre/post GPU health is clean | 🟨 Current physical stride-1 diagnostic passes the exact oracle in 14.698 seconds with complete 49,435/49,435 access and 4,995/4,995 barrier coverage, 96 visible samples, zero diagnostics, and complete static/analysis/dynamic verdicts; the standard stride-256 cadence still selects no workgroup, and paired-overhead/reviewed-fault evidence remain | 🟨 Current physical strict Inline Shadow passes the exact oracle in 23.742 seconds with complete 49,435/49,435 access and 4,997/4,997 barrier coverage, zero forbidden diagnostics, and complete static/analysis/dynamic verdicts; paired-overhead and reviewed-fault evidence remain | The exact command is now a registered validation descriptor with a pinned host regression. The all-target `MultiWaveGemmTile` correct/incorrect pair distills the 256-thread, 16-KiB cooperative tile-publication behavior into 60 green baseline/all-engine rows; `CdnaMfmaPipeline` retains the target-native MFMA/AccVGPR part. Current qualification artifacts are detailed below. |

### 2026-08-22 HIP Stream-K simple Record/Replay qualification

The exact source/build contract is now a registered validation workload rather
than an external manual command. The host manifest regression pins the
`hip_streamk_simple` executable, m=n=k=256, grid 4, one measured run,
`--validate`, gfx950 target, barrier fault family, stride 4, 120-second bound,
and all validation phases.

The clean-source paired artifact at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-streamk-simple-rr-paired-clean-9488-v1`
passes `Errors: 0` in both 157.94- and 153.55-ms bracketing baselines and in the
1,091.97-ms Record/Replay row, a 7.01x paired slowdown. Record/Replay covers all
32/32 access sites, 3/3 barriers, and 2/2 fences with complete static, analysis,
and dynamic verdicts and zero diagnostics.

The retained inventory at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-streamk-simple-rr-inventory-clean-9488-v1`
prospectively freezes occurrence 1 at PC `0x12a4`. Final-ISA review maps that
site to virtual address `0x47a4`: the reached loop phase completes its scalar
state update and wait, executes the unconditional workgroup barrier, switches
the LDS-buffer state, and begins peer `ds_read2_b32` consumers at `0x47c4`.
The exact-one fault artifact at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-streamk-simple-rr-fault-clean-9488-v1`
requests, reserves, installs, and applies precisely that barrier removal. It
produces 36,864 detector-owned Record/Replay diagnostics with complete 32/32
access, 2/2 surviving-barrier, and 2/2 fence coverage, no overflow or
incomplete state, bounded report allocation and complete cleanup, and healthy
pre/post device probes. This promotes the Record/Replay cell to green.

The checked-in `CdnaStreamkTwoTile` and Stream-K arrival behavioral pairs own
the architecture-general device semantics, so this qualification adds a
source-matched manifest regression and E2E evidence rather than a duplicate
prototype-shaped device workload.

### 2026-08-22 HIP Stream-K two-tile Record/Replay qualification

The first fully covered clean attempt exposed two independent post-oracle
complexity defects rather than a device hang. Its 19,064 compiler-fence events
were each logged individually, and replay eagerly allocated two
event-count-squared causal tables before scanning those mostly empty tables for
every fence. The final implementation retains the existing fail-closed
theoretical capacity, but grows each causal table by at most one owner-component
tranche when an operation needs it and trims unused tail entries. Auto-report
planning now budgets barriers in wave-scaled units and atomics/fences in
lane-scaled units, and detail logging retains the first four events plus an
omission count. The access pass also reserves independent relay slots for the
later barrier and fence passes, preventing the five barriers from consuming the
two compiler-fence routes.

Focused host regressions pin all four contracts: access-heavy CDNA placement
must lower 80 accesses, five barriers, and two fences; a two-static-fence report
must reserve lane-scaled dynamic headroom; a 4,096-fence replay must retain only
one release and one acquired component without becoming metadata-full; and
detail logging must remain bounded independently of trace size. All 752
`ConSanMoi.*` tests and all 192 HSA-hook tests pass. The full simulator
Record/Replay device tier also passes all 314 correct/incorrect rows across
`gfx942`, `gfx950`, `gfx1250`, `gfx1100`, and `gfx1201` in one CTest run.

The clean-source paired artifact at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-streamk-two-tile-rr-paired-demand-grown-v4`
passes `Errors: 0` in both 311.05- and 299.56-ms bracketing baselines and in the
9,479.56-ms Record/Replay row, a 31.05× paired slowdown. Record/Replay patches
80/80 accesses, 5/5 barriers, and 2/2 fences and losslessly processes 2,359,296
committed accesses, 2,208 barriers, and 20,525 fences. It emits no clean
diagnostic, overflow, unsupported event, or incomplete verdict, and the causal
model retains only one live release component.

The fresh inventory at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-streamk-two-tile-rr-inventory-demand-grown-v4`
exposes five barriers. Before mutation, final-ISA review selected occurrence 1
at PC `0x1810`: an unconditional loop-phase barrier immediately followed by
the peer `ds_read2_b64` consumer sequence. The reviewed policy froze
`detected/any`, requiring a detector-owned diagnostic without pretending that
the schedule-dependent numerical race must manifest. The exact-one artifact at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-streamk-two-tile-rr-fault-demand-grown-v4`
requests, reserves, installs, and applies precisely that deletion. It emits
281,397 replay diagnostics with complete 80/80 access, 4/4 surviving-barrier,
and 2/2 fence coverage, bounded memory with complete cleanup, and healthy
pre/post device probes. The numerical oracle does not manifest, exactly as the
prospective `any` policy allowed. This promotes the Record/Replay cell to green.

The existing adjacent `CdnaStreamkTwoTile` correct/incorrect device pair owns
the access-heavy CDNA relay and publication behavior on `gfx942` and `gfx950`;
the portable Stream-K last-arriver pair carries the synchronization idea across
all five targets. The new defects outside those device-observable contracts are
covered by the focused host tests above rather than by prototype-layout device
assertions.

### 2026-08-22 rocBLAS SGEMM Record/Replay qualification

`rocblas-sgemm-square-64` is now a registered validation workload rather than
an external manual command. Its host manifest regression pins the exact
`rocblas_sgemm` executable, `RocblasGemmTest.Square_64x64` filter, physical
gfx950 target, barrier fault family, and 120-second bound for every phase.

The paired clean artifact at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-rocblas-sgemm-rr-paired-v1`
passes the exact GoogleTest oracle in both 292.12- and 282.59-ms bracketing
baselines and in the 17,125.36-ms Record/Replay row, a 59.60× paired slowdown.
The 7,240,976-byte generated code object contains 285 kernels; the sole
observed dispatch is instrumented with complete 49,435/49,435 access and
4,997/4,997 barrier coverage, 24,576 committed accesses and 20 dynamic barrier
records, no drops, no unsupported event, zero diagnostics, and complete static,
analysis, dynamic, and replay verdicts.

The inventory at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-rocblas-sgemm-rr-inventory-v1`
was reviewed before mutation. The selected exact site is occurrence 0 at
original `.text+0x56f18` in the dispatched
`Cijk_Ailk_Bljk_SB_MT64x64x32...` specialization. In the unbundled final code
object, the branch at VMA `0x170090` and its fallthrough reconverge at
`0x170114`; `s_waitcnt` is followed by the selected `s_barrier` at `0x170118`
and peer LDS reads. The clean trace independently observes all four waves at
that barrier. The reviewed policy therefore froze `detected/any` without using
the later mutation result to choose the site or expected outcome.

The exact-one fault artifact at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-rocblas-sgemm-rr-fault-v1`
requests, reserves, installs, and applies precisely that barrier deletion. It
emits 4,086 detector-owned replay diagnostics while the exact numerical oracle
still passes, covers all 49,435 accesses and 4,996 surviving barriers, reports
no overflow or incomplete evidence, preserves the hook hash, and passes both
pre- and post-run GPU health probes. This promotes the Record/Replay cell to
green.

The new checked-in `MultiWaveGemmTile` adjacent pair preserves the E2E row's
portable behavioral idea without freezing rocBLAS code generation: 256 threads
cooperatively publish a 16-KiB tile, then each wave consumes a peer wave's
stripe after a reconverged publication edge; the incorrect member removes only
that edge. Its 50 RocJitsu rows pass on all five targets in 2.01 seconds, and
its ten physical-gfx950 rows pass in 6.30 seconds. The existing CDNA MFMA pair
continues to own the target-native live-MFMA/AccVGPR aspect.

### 2026-08-21 Record/Replay fine-grained report snapshot

GDB interruption after the exact two-tile Stream-K oracle localized the former
post-oracle timeout to `AutoMoiReportBufferRegistry::summarize`: it was reading
the 474-MB fine-grained HSA allocation field by field during several parser
and replay passes. A single sequential snapshot into cacheable process-local
memory now precedes every pass, just as coarse-grained reports already required
an HSA copy. The simple and two-tile cases consequently fall from a 120-second
timeout to 10.83 and 28.44 seconds, and the 526-MB rocBLAS report completes in
roughly 36 seconds instead of timing out at 45 seconds. All three preserve their
exact device oracles and finish with complete static and dynamic verdicts.

The hook summary reports fine- and coarse-grained snapshot bytes separately.
`AutoReplayProducerLogPinsCoverageAndFineGrainedSnapshotContracts` seeds the
real fine-grained hook path and requires the full registered report to be
snapshotted before replay. All 186 hook tests pass. The existing adjacent
correct/incorrect `CdnaStreamkTwoTile` behavioral pair already owns the device
semantics; all 30 of its baseline/four-engine rows pass on gfx942 and gfx950
RocJitsu plus physical gfx950 in 13.02 seconds. Retained E2E artifacts are
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-streamk-record-replay-snapshot-v1`
and
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-rocblas-record-replay-snapshot-v1`.

### 2026-08-21 rocBLAS SGEMM current assessment

Artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-rocblas-sgemm-current`
records a physical-gfx950 baseline and strict SuperCollider run of corpus
revision `aa54cc86c9ebff3eb840743b36ff8d9b3b2d43c4` against the workspace's
freshly composed TheRock rocBLAS artifact.  The baseline and instrumented
`RocblasGemmTest.Square_64x64` exact oracles pass in 0.315 and 25.264 seconds.
The 7,240,872-byte rocBLAS code object contains 49,435 supported LDS accesses,
of which SuperCollider patches 2,027; the other 47,408 fail placement or
lowering.  The workload result and dynamic report are complete, but static and
analysis verdicts are not.  That historical run exposed an architecture gate:
SuperCollider's dense explicit-key dispatcher was enabled for RDNA4 and
gfx1250, but not CDNA4, so the planner spent 1,767,007,794 SOPP-relay work
units before covering only the sites with private local islands.

The focused
`Cdna4DenseCheckTrapCoversRocblasShapedLargeKernel` host regression captures
the same large, dense LDS shape and fails against the old gate with zero of
1,025 check bodies emitted.  Enabling the existing explicit-key route for
CDNA4 makes that regression cover all 1,025 sites in 0.18 seconds while the
neighboring RDNA4 and gfx1250 dense contracts continue to pass.  The retained
physical-gfx950 artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-rocblas-supercollider-dense-final`
then passes the exact `Square_64x64` oracle in 28.963 seconds, patches all
49,435 accesses into a valid 15,445,264-byte replacement, reports no check
mismatches, and has complete static, analysis, and dynamic verdicts.  Its
SOPP-relay work count is 10.  The hook SHA-256 is
`c9ce9a6ec03e09620e7face72187300bc1e380a4defc417880caba7d16444e57`.
Only paired-overhead and reviewed-fault evidence keep the SuperCollider cell
yellow.

The current Record/Replay implementation also clears the historical red
state-planning failure.  Under the ordinary stride-65,536 standard profile it
builds a valid 57,097-patch transform for the same heterogeneous object: the
resource plan uses 37,233 dead windows, 8,146 descriptor-growth windows, and
9,053 spill-backed windows rather than requiring one impossible object-wide
owner/epoch tuple.  A quiet, 45-second-bounded physical diagnostic passes the
exact `Square_64x64` oracle in 16.669 seconds, then spends the rest of the
budget in replay/teardown.  The normal logged diagnostic likewise reaches a
valid transform, but not its replacement, final coverage, or dynamic verdict
before the bound.  This advances the cell from red to orange while leaving the
post-oracle latency as the explicit residual issue; repeated long waits are
not useful until replay/teardown itself is profiled or bounded more tightly.

The strict standard-v1 Sampled run passes the exact oracle in 14.445 seconds
and patches all 49,435 accesses plus all 4,995 applicable barriers.  Its
stride-256 cadence selects no workgroup, however, leaving zero visible evidence
and causing the required-records gate to exit 86.  The prescribed stride-1
diagnostic initially rejected before execution because the sampled probe's
dense dispatcher reserved only its indirect-jump words, not the comparison and
conditional branch needed by a far explicit-key target.  The focused
`Cdna4LargeSampledDenseDispatchReservesFarTargetRoutes` host regression
reproduces the 1,024-site failure.  After giving large explicit-key Sampled
objects the existing conservative relay envelope used by the other MOI
engines, that same physical diagnostic passes the exact oracle in 14.698
seconds with 49,435/49,435 accesses, 4,995/4,995 barriers, 96 visible samples,
zero diagnostics, and complete static, analysis, and dynamic verdicts.  The
cell is yellow because the standard cadence, paired overhead, and reviewed
fault remain.  The candidate is based on source revision `1b93fbe9f75e`; the
loaded fixed hook SHA-256 is
`5643e546e24fb2cf50c304980fe1fd037d76455115190c15ec5b50b42e6f4f40`.

The first strict standard-v1 Inline Shadow run rejected the rocBLAS code object
before the exact oracle.  Three access reachability partitions had selected
overlapping nine-word relocatable hosts.  Later host emission overwrote an
earlier literal-bearing indirect jump, leaving `s_add_u32` as the final word of
one inventoried range and causing `Invalid instruction opcode: truncated
instruction encoding`.  The existing
`Cdna4FarInlineShadowAccessesShareExplicitKeyRelay` host regression now checks
that every relocated host range is disjoint; it fails against the old selector
and passes after sharing the caller's host reservations across partitions.

With that fix, two consecutive physical runs pass the exact
`RocblasGemmTest.Square_64x64` oracle in 23.312 and 23.194 seconds.  Final ELF
validation succeeds, all 4,997 barriers patch, and access coverage improves to
49,345/49,435.  The remaining 90 access placement/lowering failures keep the
analysis verdict incomplete, so the cell advances from red to orange rather
than being accepted prematurely.

A component-local dispatch-identity follow-up closes that remaining gap.  The
full-pressure Tensile owner has no legal transient dispatch SGPR pair, so its
otherwise-legal persistent owner/epoch/workgroup VGPR tuple now also captures
the hardware dispatch preload.  The retained `inline-shadow/component-dispatch-fix.log`
run passes the exact oracle in 23.742 seconds, patches all 49,435 accesses and
all 4,997 barriers, emits zero forbidden diagnostics, and reports complete
static, analysis, and dynamic verdicts.  The cell advances to yellow pending
paired-overhead and reviewed-fault evidence.

### 2026-08-22 HipKittens BF16 Record/Replay direct-to-LDS qualification

The first prospectively frozen phase-reuse fault exposed a real instrumentation
gap rather than a weak oracle. Removing exactly the reviewed reached barrier
corrupted the exact BF16 result (`max_abs_diff=1.34375`), but Record/Replay
emitted no diagnostic: the inventory contained the 96 DS readers but omitted 32
`buffer_load_dwordx4 ... lds` producers whose LDS destination is implicit. That
trial remains rejected under its original policy; neither the oracle nor the
expected diagnostic was relabeled.

The adjacent checked-in `CdnaDirectToLdsPublication` correct/incorrect device
pair reduces this behavior without depending on the prototype layout. Both
members use a direct global-to-LDS producer and a peer DS reader; the incorrect
member removes only their publication barrier. Baseline and all four engines
now contribute 20 green RocJitsu rows on gfx942/gfx950 and 10 green physical
gfx950 rows. The correct member requires exact results and no diagnostic, while
the incorrect member requires the publication conflict. The focused
`ConSan.InventoriesCdnaDirectGlobalToLdsAsAnLdsWrite` host regression pins the
architectural operand contract: documented CDNA3/CDNA4 dword, dwordx3, and
dwordx4 forms with zero raw VDATA are LDS writes, while undocumented dwordx2
and nonzero-VDATA forms are rejected. The implementation materializes the
implicit destination as `M0 + physical_lane_id * stride` under all-lane EXEC,
then restores the guest EXEC state.

Clean-source artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-hipkittens-bf16-rr-paired-direct-lds-clean-731c-v3`
records source revision `731c20e8f0`, hook SHA-256
`84dbe66bdcdbadff95d24144af00770de632d1331dbf2cb7c874a6f214feb09a`,
an exact zero-error oracle, complete 128/128 access and 32/32 barrier coverage,
zero diagnostics, and complete static, analysis, and dynamic verdicts. Its two
baseline legs average 134.08 ms and the Record/Replay leg takes 606.10 ms, for
4.52x paired overhead.

The same pre-existing reviewed exact-one phase-reuse specification is accepted
after the instrumentation repair in
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-hipkittens-bf16-rr-fault-direct-lds-clean-731c-v3`.
The selected barrier occurrence 4 at PC `0x8d0` is reserved, installed, reached,
and applied exactly once. The detector terminates with its required owned
runtime diagnostic, as allowed by the prospectively committed `oracle=any`
policy, with complete 128/128 access and 31/31 remaining-barrier coverage,
4,194,304 access events, 240 barrier events, no overflow, and complete static,
analysis, dynamic, memory, cleanup, containment, and GPU-health evidence. This
promotes the Record/Replay cell to green while preserving the rejected original
trial as historical evidence of the gap.

### 2026-08-23 HipKittens BF16 Sampled direct-to-LDS qualification

Current paired artifact
`/home/ossci/xx/consan-validation/prep-20260823-gfx950-hipkittens-bf16-sampled-overhead-v1`
preserves the exact zero-error oracle before, during, and after
instrumentation. The bracketing baselines take 121.446 and 144.628 ms, versus
195.963 ms for Sampled, for 1.473x overhead against their 133.037-ms mean.
Sampled patches all 128/128 supported accesses--including all 32 implicit
direct-global-to-LDS destinations--and all 32/32 barriers, emits no clean
diagnostic, and reports complete static, analysis, and dynamic verdicts.

Inventory artifact
`/home/ossci/xx/consan-validation/prep-20260823-gfx950-hipkittens-bf16-sampled-inventory-v1`
reconfirms code-object fingerprint `fnv1a64:9f8a9a9a3d8766cb` and the existing
reviewed phase-reuse barrier at occurrence 4, PC `0x8d0`. Final-ISA review
places it after the eight-wave kernel consumes one staged LDS tile through
`ds_read_b128` and BF16 MFMA, and immediately before the next LDS-read and
direct-global-to-LDS phase. The m=n=k=256 launch reaches this reuse edge.
Because the ordinary one-workgroup cadence selects no workgroup, audited
stride one is prospectively confined to this destructive sensitivity trial.

Contained artifact
`/home/ossci/xx/consan-validation/prep-20260823-gfx950-hipkittens-bf16-sampled-fault-v1`
applies the reviewed removal exactly once and is accepted through a
detector-owned runtime diagnosis. It emits 24 Sampled conflicts, retains
complete 128/128 access and 31/31 surviving-barrier coverage, and claims all
1,024 selected windows without drops, saturation, stale snapshots, or
malformed/incomplete state. Required and allocated report memory are 167,368
bytes, peak live report memory is 167,096 bytes, live memory returns to zero,
and pre/post discovery and dispatch health probes pass.

The adjacent `CdnaDirectToLdsPublication` and `CdnaMfmaPipeline` pairs own the
same implementation-independent producer/publication and MFMA-pressure
contracts. Their focused 12-row Sampled gate passes on gfx942/gfx950
simulation and physical gfx950 in 1.02 seconds. No duplicate device fixture is
warranted; the Sampled E2E cell is green.

### 2026-08-22 HipKittens FP8/MXFP8 Record/Replay direct-to-LDS qualification

The BF16 direct-to-LDS repair generalizes to both low-precision four-wave
kernels without workload-specific implementation changes. Clean-source paired
artifacts
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-hipkittens-fp8-rr-paired-direct-lds-clean-731c-v1`
and
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-hipkittens-mxfp8-rr-paired-direct-lds-clean-731c-v1`
retain exact optimized/reference FP8 output and all 65,536 MXFP8 checks,
respectively. Both Record/Replay rows cover 96/96 accesses--the original 64 DS
readers plus 32 direct-global-to-LDS writers--and 5/5 barriers with zero clean
diagnostics and complete static, analysis, and dynamic verdicts. FP8's 564.67
and 306.73 ms bracketing baselines average 435.70 ms versus 718.95 ms for
Record/Replay (1.65x); MXFP8's 194.77 and 257.71 ms baselines average 226.24 ms
versus 725.83 ms (3.21x). Both use clean source revision `731c20e8f0`, corpus
revision `14180b3`, and hook SHA-256
`84dbe66bdcdbadff95d24144af00770de632d1331dbf2cb7c874a6f214feb09a`.

Fresh inventories freeze separate reached initial-publication sites from each
final code object. FP8 occurrence 0 at inventory PC `0x528` maps to
unconditional barrier `0x2028`, between its initial direct-to-LDS loads and
peer B128 readers. MXFP8 occurrence 0 at inventory PC `0x1768` maps to
unconditional barrier `0x4668` with the same producer/consumer role. The
prospectively reviewed exact-one campaigns in
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-hipkittens-fp8-rr-fault-direct-lds-clean-731c-v1`
and
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-hipkittens-mxfp8-rr-fault-direct-lds-clean-731c-v1`
are both accepted through detector-owned runtime diagnostics under their
precommitted `oracle=any` policies. Each requests, plans, reserves, installs,
and applies exactly one mutation; retains complete 96/96 access and 4/4
remaining-barrier coverage plus 4,194,304 access events; reports no overflow or
incomplete evidence; cleans report memory to zero; and passes pre/post device
discovery and dispatch health. FP8 retains 32 barrier events and MXFP8 retains
48, reflecting their different dispatch counts. The existing adjacent CDNA
direct-to-LDS device pair and host encoding regression own the quick behavioral
contract, so these separate E2E qualifications do not require duplicate
implementation-shaped fixtures. Both Record/Replay cells advance to green.

### 2026-08-23 HipKittens FP8/MXFP8 Sampled direct-to-LDS qualification

Current paired artifacts
`/home/ossci/xx/consan-validation/prep-20260823-gfx950-hipkittens-fp8-sampled-overhead-v1`
and
`/home/ossci/xx/consan-validation/prep-20260823-gfx950-hipkittens-mxfp8-sampled-overhead-v1`
pass their independent exact oracles before, during, and after
instrumentation. FP8's 312.748- and 262.970-ms bracketing baselines average
287.859 ms versus 542.311 ms for Sampled (1.884x). MXFP8's 219.488- and
239.405-ms baselines average 229.447 ms versus 267.764 ms for Sampled (1.167x).
Both instrumented rows cover all 96 accesses--64 wide DS readers plus 32
implicit direct-global-to-LDS writers--and all 5 barriers with zero clean
diagnostics and complete static, analysis, and dynamic verdicts.

Fresh inventories
`/home/ossci/xx/consan-validation/prep-20260823-gfx950-hipkittens-fp8-sampled-inventory-v1`
and
`/home/ossci/xx/consan-validation/prep-20260823-gfx950-hipkittens-mxfp8-sampled-inventory-v1`
reconfirm the separate current code-object fingerprints and the previously
reviewed reached initial-publication barriers: FP8 occurrence 0 at PC `0x528`
and MXFP8 occurrence 0 at PC `0x1768`. In each final ISA, direct-to-LDS
producers and their wait precede the unconditional barrier, while peer
`ds_read_b128` consumers follow it. Audited Sampled stride one is frozen only
for these destructive one-workgroup sensitivity trials.

Contained artifacts
`/home/ossci/xx/consan-validation/prep-20260823-gfx950-hipkittens-fp8-sampled-fault-v1`
and
`/home/ossci/xx/consan-validation/prep-20260823-gfx950-hipkittens-mxfp8-sampled-fault-v1`
each apply exactly one reviewed removal and are accepted through a
detector-owned runtime diagnosis. Each emits 24 Sampled conflicts, retains
complete 96/96 access and 4/4 surviving-barrier coverage, and claims all 384
visible windows without drops, saturation, stale snapshots, or
malformed/incomplete state. Required and allocated report memory are 125,640
bytes per run, peak live report memory is 125,368 bytes, live memory returns
to zero, and all pre/post discovery and dispatch health probes pass.

Datatype and scale encoding change the E2E oracle, but not the sanitizer's
observable publication contract. `CdnaDirectToLdsPublication` owns that
adjacent correct/incorrect behavior on gfx942/gfx950 simulation and physical
gfx950; `CdnaMfmaPipeline` independently retains native accumulator pressure.
Their focused 12-row Sampled gate already passes. Adding FP8- and
MXFP8-shaped copies would duplicate the prototype-independent contract, so
both Sampled cells advance to green without new fixtures.

### 2026-08-21 HipKittens FP8 four-wave source repair and profile assessment

Corpus revision `14180b3` repairs the FP8 four-wave compile blocker by keeping
the local rotating-buffer count runtime configurable instead of incorrectly
declaring it `constexpr`.  The existing checked-in case supplies
`--rotating-buffer-count 4`, so its successful clean build and physical-gfx950
optimized/reference correctness pass directly guard the formerly failing path.
The current profile runs use hook SHA-256
`14363d31ea87fc629dc501687d30d484c01d8a9b75f419d1dc906199d6db790e`.

Artifact
`consan-validation/rebase-20260821-gfx950-hipkittens-fp8-supercollider-6PR2L6`
records a strict SuperCollider pass with complete 64/64 supported LDS-access
coverage, zero mismatches, and complete analysis, static, and dynamic verdicts.
Artifacts
`consan-validation/rebase-20260821-gfx950-hipkittens-fp8-record-replay-standard-LY8Yir`
and
`consan-validation/rebase-20260821-gfx950-hipkittens-fp8-record-replay-stride1-38Qxr3`
separate the standard stride-65,536 cadence, which selects no workgroup, from
the deterministic dense path.  The stride-1 run passes the exact oracle with
32,768 committed access records, 40 barrier records, complete 64/64 access and
5/5 barrier coverage, and zero diagnostics, conflicts, or drops.

Artifacts
`consan-validation/rebase-20260821-gfx950-hipkittens-fp8-sampled-standard-B9b5wv`
and
`consan-validation/rebase-20260821-gfx950-hipkittens-fp8-sampled-stride1-2wgV3t`
likewise distinguish standard-cadence selection from instrumentation.  Stride
1 passes the exact oracle with 256 visible samples, four synchronization
records, complete 64/64 access and 4/4 applicable-barrier coverage, and zero
diagnostics or malformed/incomplete state.  Artifact
`consan-validation/rebase-20260821-gfx950-hipkittens-fp8-inline-TJPC2v`
records a 16.17-second Inline Shadow pass with 196,608 visible evidence events,
65,536 exact-shadow entries, complete 64/64 access and 5/5 barrier coverage,
and zero diagnostics or malformed/incomplete state.  These four cells advance
from red to yellow; paired overhead and reviewed-fault evidence still remain
for green acceptance.

### 2026-08-20 current HipKittens compile and baseline assessment

Clean build directory
`rocjitsu-test-corpus-build/kernels-gfx950-hipkittens-gcc-20260820`
uses `/usr/bin/gcc` and `/usr/bin/g++` for OpenMP host code and the workspace
TheRock `amdclang++` for gfx950 device code. This removes the prior missing-Clang-
OpenMP environment blocker without changing the corpus or the checked-in build
contract.

`hipkittens_gemm_bf16fp32_16x32` compiles and its physical-gfx950 exact oracle
passes in 0.13 seconds with `max_abs_diff=0`. `hipkittens_gemm_mxfp8_4wave`
also compiles and passes all 65,536 results in 0.24 seconds. The FP8 companion
does not compile: `4_wave.cu:861` declares `block_count` as `constexpr` while
initializing it from the non-constant `ROTATING_BUFFER_COUNT`. The FP8 cells are
therefore red source/toolchain failures. The BF16 and MXFP8 baselines now have
clean physical assessments for all four profiles; their yellow cells retain
the paired-overhead and reviewed-fault work required for green acceptance.

Artifact
`consan-validation/rebase-20260820-gfx950-hipkittens-mxfp8-supercollider-QzGFlj`
records the first strict physical MXFP8 SuperCollider assessment. It preserves
the exact oracle for all 65,536 outputs, patches all 64/64 supported LDS reads,
reports zero check mismatches, and has complete analysis, static, and dynamic
verdicts. The loaded hook SHA-256 is
`587f327b6c269414c61a79c8915867fdb68fc2a434cc8586ffe41e19e11cfbbc`.
The cell is yellow pending paired overhead and a reviewed fault run.

Artifacts
`consan-validation/rebase-20260820-gfx950-hipkittens-mxfp8-sampled-standard-gKinYm`
and
`consan-validation/rebase-20260820-gfx950-hipkittens-mxfp8-sampled-stride1-wz5j9I`
separate Sampled's standard cadence from the instrumented path. The standard
stride-256/offset-0 run patches all 64/64 accesses and 4/4 applicable barriers
but correctly reports a dynamic-incomplete verdict because the one-workgroup
dispatch is not sampled. At stride 1, all 65,536 exact checks pass in 2.99
seconds wall time, with 256 visible samples, 4 synchronization records, zero
diagnostics or malformed/incomplete state, and complete static and dynamic
verdicts. The loaded hook SHA-256 is
`587f327b6c269414c61a79c8915867fdb68fc2a434cc8586ffe41e19e11cfbbc`.

Artifact
`consan-validation/rebase-20260820-gfx950-hipkittens-mxfp8-inline-wMbLGf`
records the strict physical MXFP8 Inline Shadow assessment. It passes all
65,536 exact checks in 17.11 seconds wall time, patches all 64/64 accesses and
5/5 barriers, publishes 294,912 visible evidence events, and reports zero
diagnostics, malformed/incomplete state, or overflow, with complete analysis,
static, and dynamic verdicts. The loaded hook SHA-256 is
`587f327b6c269414c61a79c8915867fdb68fc2a434cc8586ffe41e19e11cfbbc`.

Artifacts
`consan-validation/rebase-20260820-gfx950-hipkittens-mxfp8-record-replay-standard-eWvyK1`
and
`consan-validation/rebase-20260820-gfx950-hipkittens-mxfp8-record-replay-stride1-w7q1tc`
separate Record/Replay's standard cadence from its dense instrumented path. The
standard stride-65,536/offset-0 run patches all 64/64 accesses and 5/5 barriers
but correctly reports a dynamic-incomplete verdict because the one-workgroup
dispatch is not sampled. At stride 1, all 65,536 exact checks pass in 165.38
seconds wall time. Replay processes 49,152 committed access records and 60
barrier records without drops, diagnostics, conflicts, invalid tokens, or
metadata exhaustion; analysis, static, and dynamic verdicts are complete. The
loaded hook SHA-256 is
`587f327b6c269414c61a79c8915867fdb68fc2a434cc8586ffe41e19e11cfbbc`.

Artifact
`consan-validation/rebase-20260820-gfx950-hipkittens-bf16-supercollider-cMKIkA`
records the first strict BF16 SuperCollider run with hook SHA-256
`dc7d952995dfbcb0cf8601c10b0b546fefd95f69f055bc08d62e4e2ec5664b2a`.
The physical run passes the exact zero-error oracle in 0.17 seconds, patches
all 96/96 supported LDS reads, reports no check mismatch, and has complete
static and dynamic verdicts. The cell is yellow pending paired overhead and a
reviewed fault run.

Artifact
`consan-validation/rebase-20260820-gfx950-hipkittens-bf16-record-replay-u02Uq3`
records a strict stride-1 Record/Replay clean run. It preserves the exact
zero-error result in 340.17 seconds, patches all 96/96 accesses and 32/32
barriers, processes 49,152 access plus 248 barrier records without drops,
diagnostics, conflict, saturation, invalid tokens, or metadata exhaustion, and
reports complete static and dynamic verdicts. This intentionally dense run is
structural and correctness evidence; a standard-cadence run, paired overhead,
and reviewed-fault evidence still remain for green acceptance.

Artifacts
`consan-validation/rebase-20260820-gfx950-hipkittens-bf16-sampled-0cQbvf`
and
`consan-validation/rebase-20260820-gfx950-hipkittens-bf16-sampled-stride1-18FHRB`
separate Sampled's cadence policy from its instrumentation path. The standard
stride-256/offset-0 run patches all 96/96 accesses and 29/29 applicable
barriers but correctly exits 86 because its one dispatch selects no workgroup.
At stride 1, the physical exact oracle passes in 12.61 seconds with 768 visible
samples, 13 synchronization records, zero diagnostics or malformed/incomplete
state, and complete static and dynamic verdicts. A workload-qualified standard
residue is not claimed because this kernel lacks a stable ABI dispatch-ID
source; stride 1 is the deterministic bounded clean contract.

Artifact
`consan-validation/rebase-20260820-gfx950-hipkittens-bf16-inline-JPQw8z`
records the first physical Inline Shadow assessment. Static instrumentation is
complete at 96/96 accesses and 32/32 barriers, and the report reaches 196,608
events with zero diagnostics, malformed state, or overflow. The workload does
not complete, however: after more than five minutes its main thread remains
blocked in `kfd_wait_on_events` and the run is interrupted. The red cell is a
runtime hang, not a static-coverage rejection.

Current artifact
`consan-validation/rebase-20260820-gfx950-hipkittens-bf16-inline-entry-prefix-fix-5CvZuY`
supersedes that interrupted assessment. The physical workload exits normally
in 22 seconds and preserves the exact zero-error oracle. Inline Shadow patches
all 96/96 accesses and 32/32 barriers, publishes 229,376 visible evidence
events, and reports zero diagnostics, malformed/incomplete state, or overflow,
with complete analysis, static, and dynamic verdicts. The loaded hook SHA-256
is `587f327b6c269414c61a79c8915867fdb68fc2a434cc8586ffe41e19e11cfbbc`.

The same change set adds a checked-in gfx950 large-LDS pipeline distilled from
the HipKittens shape: 160 KiB LDS, eight waves, wide DS stores, repeated scalar
reads, 32 publication stages, and a correct/incorrect missing-barrier pair. Its
20 baseline/profile cases pass in gfx950 RocJitsu simulation and on the
physical gfx950 in 131.17 seconds wall time. A focused host regression covers
the independently exposed planner case where a seven-word far entry relay
reaches into an already instrumented LDS access and must chain into that access
body rather than overlap or bypass it. Paired overhead and reviewed-fault
acceptance remain before the E2E cell can become green.

### 2026-08-20 HIP matmul Inline Shadow mixed-owner closure

Final candidate-tree hook SHA-256
`dc7d952995dfbcb0cf8601c10b0b546fefd95f69f055bc08d62e4e2ec5664b2a`
closes the packaged `HybridStreamKTree` gap on a physical MI355X/gfx950.
Artifact root
`consan-validation/rebase-20260820-gfx950-hip-matmul-owner-dispatch-inline-final-qEKCLI`
records the strict Inline Shadow run. All three selected numerical oracles pass
in 18.57 seconds, with 739/739 accesses, 109/109 barriers, 188,416 visible
evidence events, complete static and dynamic verdicts, and zero diagnostics,
overflow, or dynamic-incomplete events.

The linked object requires mixed persistent representations: fixed-stack
owners with live AccVGPR banks use entry-captured private state, while the
dynamic-stack empty-AccVGPR-boundary owner uses a component-local VGPR tuple.
When that owner cannot use the otherwise valid code-object dispatch SGPR pair,
its tuple now also retains dispatch identity. A checked-in two-kernel host
regression reproduces that partial-owner dispatch case and verifies both
accesses, both barriers, both entry prologues, and final exact-shadow semantic
validation.

### 2026-08-20 HIP matmul Sampled and Inline Shadow clean refresh

The earlier candidate-tree hook SHA-256 was
`9b01bf1ff91b811edc70e53720f02f8122a23540eabca0cd9d82fcf6ed5170c5`.
The physical-gfx950 baseline passes all three selected m128-n128-k128 exact
oracles in 0.124 seconds. Sampled at its standard runtime stride 256 and
workload-qualified residue 4 passes the same three oracles in 18.19 seconds,
publishes 112 visible samples without a diagnostic, and has a complete dynamic
verdict. It patches 709/739 accesses and all 104/104 applicable barriers.
Inline Shadow passes all three exact oracles in 17.59 seconds with 188,416
visible evidence events, zero diagnostics or dynamic-incomplete state, and
709/739 access plus 104/109 barrier coverage.

Both former red cells therefore advance to orange: neither profile still
rejects before execution. The remaining misses are static sites in the
packaged but unexecuted dynamic-stack `HybridStreamKTree` kernel: 30 accesses
for Sampled, and the same 30 accesses plus 5 barriers for Inline Shadow. Those
gaps must be closed or explicitly qualified before either cell can advance to
yellow; paired-overhead and reviewed-fault evidence remain after that.

### 2026-08-20 HIP matmul Sampled mixed-owner closure

Final candidate-tree hook SHA-256
`dc7d952995dfbcb0cf8601c10b0b546fefd95f69f055bc08d62e4e2ec5664b2a`
closes Sampled's packaged `HybridStreamKTree` gap on a physical MI355X/gfx950.
Artifact root
`consan-validation/rebase-20260820-gfx950-hip-matmul-owner-recovery-sampled-final-hQZn6f`
records the strict standard-stride run at workload-qualified residue 244. All
three selected numerical oracles pass in 22.53 seconds. The run patches all
739/739 accesses and 109/109 barriers, publishes 112 sampled accesses and 2
synchronization records, reports complete static and dynamic verdicts, and
has zero diagnostics, conflicts, malformed records, dropped windows, or
dynamic-incomplete events.

The linked object needs Sampled's owner placement to reach a fixed point.
Fixed-stack owners with live AccVGPR banks use entry-snapshotted private
state; the full-pressure dynamic-stack owner uses a component-local
persistent VGPR tuple and owner-local transient scalar window. Resource-failed
plans remain in owner discovery because that smaller final ABI makes their
739-access/109-barrier resource-plan rebuild succeed. Checked-in host tests
cover mixed private/persistent state and the initially failed-owner recovery.
The repeated-dispatch identity correct/incorrect device pair now runs on every
CDNA3/4/5 and RDNA3/4 configuration, including physical gfx950.

### 2026-08-22 HIP matmul Record/Replay mixed-owner closure

Artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-hip-matmul-rr-mixed-state-v2`
records a current physical-gfx950 strict stride-1 run of
`hip_matmul_matmul -m 128 -n 128 -k 128` with `FIXED_ITERATIONS=1`. All three
exact numerical checks pass. Record/Replay patches all 739/739 supported LDS
accesses and 109/109 barriers, commits 50,176 access records and 168 barrier
records without a drop, and replays them with zero diagnostics or conflicts.
Static, dynamic, and overall analysis are complete. This supersedes the stale
709/739 partial-coverage result; the cell remains yellow only because paired
overhead and a prospectively reviewed fault result are still required.

The fix carries the same disconnected-owner placement to Record/Replay that
the Sampled closure established: fixed-stack owners with live accumulator
banks use entry-snapshotted private state, while the dynamic-stack owner uses
an owner-local persistent VGPR tuple and scalar window. Focused host tests
cover access, barrier, ordered-atomic, and fence lowering plus recovery of an
initially resource-failed dynamic owner. The checked-in
`CdnaMixedPersistentState` device pair retains the observable contract without
asserting those representation choices: two disconnected CDNA owners preserve
live MFMA/AccVGPR, scalar, full-VGPR, and dynamic-private state; the correct
member has an ordered cross-wave publication and the incorrect member removes
only that edge. All 20 baseline/four-engine gfx942/gfx950 simulator rows and
all 10 physical-gfx950 rows pass. In Record/Replay, both members have complete
4/4 access and 2/2 barrier coverage; clean replay has zero diagnostics and no
conflict, while the incorrect member reports the required conflict.

### 2026-08-22 HIP matmul Record/Replay qualification

Commit `8923302443` registers the external native executable as canonical
validation ID `hip-matmul-m128-n128-k128`, including its exact m128-n128-k128
argv, `FIXED_ITERATIONS=1` environment, source and executable provenance,
process-latency timing, and barrier-fault policy. Host regressions pin that
manifest/command contract and forbid native workload metadata from injecting
ConSan or HSA-tool settings. The full 353-test ConSan Python suite produced
one unrelated process-group wall-clock flake at 5.083 seconds against a strict
five-second assertion; the failed test and the new focused tests all pass in
the immediate isolated rerun.

Clean artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-hip-matmul-rr-paired-892330-v1`
records a 279.37-ms paired process baseline and 787.89-ms Record/Replay process
time, or 2.82×. All three exact numerical checks pass, all 739/739 supported
LDS accesses and 109/109 barriers are patched, replay is lossless and
conflict-free, and all static/dynamic verdicts are complete. The validator and
all recorded rocm-systems sources have clean `8923302443` provenance.

Inventory artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-hip-matmul-rr-inventory-892330-v1`
contains 109 exact barrier identities. Before execution, final-ISA review
selected occurrence zero at code-object PC `0x1b58` in the dispatched i8 MFMA
pipeline-v3 kernel. Every workitem publishes four 128-bit LDS tiles, waits,
and crosses this barrier immediately before peer `ds_read_b128` operations.
The reviewed policy therefore precommitted `detected` with oracle `any`.
Contained artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-hip-matmul-rr-fault-tile-publication-892330-v1`
applies exactly one removal and is accepted: Record/Replay emits 1,020
attributable diagnostics, retains complete 739/739 access and 108/108
remaining-barrier coverage without overflow or incomplete evidence, and both
pre/post discovery and dispatch health probes pass. The existing
`CdnaMixedPersistentState` correct/incorrect device pair already owns this
observable publication contract on gfx942/gfx950 simulation and physical
gfx950, so the E2E fault adds qualification evidence rather than a duplicate
prototype-shaped fixture.

### 2026-08-23 HIP matmul Sampled qualification

The canonical physical paired artifact
`/home/ossci/xx/consan-validation/prep-20260823-gfx950-hip-matmul-sampled-overhead-v1`
passes all three exact m128-n128-k128 numerical oracles before, during, and
after instrumentation. The bracketing baselines take 262.411 and 291.541 ms,
versus 571.145 ms for Sampled, for a 2.062× slowdown against their 276.976-ms
mean. Sampled patches all 739/739 supported LDS accesses and all 109/109
barriers, emits no clean diagnostic, and reports complete static, analysis,
and dynamic verdicts.

Inventory artifact
`/home/ossci/xx/consan-validation/prep-20260823-gfx950-hip-matmul-sampled-inventory-v1`
prospectively freezes occurrence zero at code-object PC `0x1b58` in the
dispatched i8 MFMA pipeline-v3 kernel. Final-ISA review confirms that every
workitem stores four 128-bit LDS tiles, waits, and crosses this unconditional
barrier immediately before peer `ds_read_b128` operations. The reviewed
fault therefore precommits `detected` with oracle `any` and uses audited
Sampled stride one only for the destructive qualification run.

Contained artifact
`/home/ossci/xx/consan-validation/prep-20260823-gfx950-hip-matmul-sampled-fault-v2`
applies exactly that one removal. Sampled emits 24 detector-owned conflicts,
retains complete 739/739 access and 108/108 surviving-barrier coverage, and
claims all 392 selected windows without drops, saturation, stale snapshots,
or malformed/incomplete state. Required and allocated report memory are
1,219,696 bytes, peak live memory is 1,219,424 bytes, live memory returns to
zero after cleanup, and both pre/post discovery and dispatch health probes
pass. The hook is unchanged across the trial.

This is the same observable cross-wave publication edge already owned by the
adjacent `CdnaMfmaPipeline`, `MultiWaveGemmTile`, and
`CdnaMixedPersistentState` device pairs, not a new prototype-specific
behavior. Their focused 24-row Sampled gate passes on all five RocJitsu
targets plus physical gfx950 in 1.52 seconds. The E2E evidence therefore
promotes the Sampled cell to green without adding a duplicate fixture.

### gfx950 Tensile follow-on

Corpus revision `46a4c58a7be8` first packaged one bounded, target-native gfx950
TensileLite Stream-K row.  The selected row is intentionally a runtime
validation case rather than a tuning sweep; fresh qualification uses
descendant revision `0db836e7bd8c`:

- one assembly FP32 GEMM solution with `StreamK: 3`;
- exact problem size `[129, 129, 1, 129]` with beta;
- full output validation, one enqueue, no warmup, and a 16 MiB workspace cap;
- source provenance at ROCm rocm-libraries revision
  `a8f0845f87ab50adc3dc8d0edd86693cb31065b1`; and
- three generated code objects whose ELF flags all name gfx950.

The checked-in runner executes `tensilelite-client` through
`gfx950_mi355x.json`, parses every numeric CSV result row, requires every
validation field to be `PASSED`, and rejects missing or non-gfx950 code
objects.  At corpus revision `0db836e7bd8c`, fresh default-tool runs complete
in about 5.0 seconds.  The runner and negative-oracle suite pass 21/21.  The
maintained corpus gate `python -m pytest -q tests` passes 92/92 with the
required IREE tools on `PATH`; bare repository-root collection is not the
gate.

| Priority | Tracking unit | SuperCollider | Record/Replay | Sampled | Inline Shadow | Current evidence |
|---|---|---|---|---|---|---|
| P0 | `gfx950_sk_sgemm_streamk` | 🟩 Current wait-fixed clean and exact-one qualified-miss bundle: exact oracle, complete 82/82 access coverage, zero mismatches, clean containment, and 1.10x paired slowdown | 🟩 Current clean and exact-one qualified-miss bundle: 82/82 accesses plus every present barrier, complete replay, zero diagnostics, clean containment, and 1.13x paired slowdown | 🟩 Current clean and exact-one qualified-miss bundle: 82/82 accesses plus every present barrier, complete sampled report, zero diagnostics, clean containment, and 1.35x paired slowdown | 🟩 Current clean and exact-one qualified-miss bundle: 82/82 accesses plus every present barrier, complete inline report, zero diagnostics, clean containment, and 5.98x paired slowdown | All current runs use rocm-libraries `0a323b7493` and a 120-second bound; instrumented legs use the standard profiles.  Every retained log records the driver working directory, exact shell-quoted invocation, and timeout; instrumented logs additionally record the hook and `RJ_*` profile settings. |

This row is the executable denominator selected from the larger gfx950 YAML
survey.  Static YAML features alone do not promote or expand that denominator.
The current baseline-before, SuperCollider, and baseline-after runs all pass
the exact numeric oracle and target-native gfx950 ELF checks.  Their
host-dominated end-to-end elapsed times are 5.107, 5.651, and 5.125 seconds,
respectively.  The mean of the two baseline legs is 5.116 seconds, giving a
1.10x paired ratio; this is not a kernel-overhead measurement.  SuperCollider
discovers, selects, and patches all 82 LDS
accesses, reports no unsupported or resource-failed sites, and finishes with
marker zero, no mismatch, and complete report cleanup.

The current fault bundle selects occurrence zero of the target-native full
`s_barrier` form from a fresh dry-run inventory.  It is the first of 11 such
sites sharing code-object identity `fnv1a64:89754d9658cf27ec`, at text PC
`0x10a4`.  Two independent runs require exactly one barrier drop and both record
`requested=1`, `planned=1`, and `applied=1` while retaining 82/82 access
patches.  Both exact numerical oracles pass and SuperCollider reports zero
mismatches, so this is a qualified miss rather than a detector hit.  The
trials finish in 5.649 and 5.670 seconds under the 120-second bound, with zero
allocation, read, or cleanup failures and a complete report.  A clean
post-fault SuperCollider health run then passes the same exact oracle and
82/82 coverage in 5.597 seconds with complete cleanup.  These are
host-dominated end-to-end timings that include Tensile generation and
simulator startup; they are not kernel-overhead measurements.

The registered `tensile-gfx950-lds-positive` row is the separate detector-hit
control. It uses a generated `[64,64,1,129]` BF16 GEMM from rocm-libraries
`0a323b74932c57d6d1a94af4a009dd7676b8f695`, selects the first target-native
`ds_write_b128` at kernel-relative PC `0x1344`, and rewrites its address from v2
to v54. The generated kernel declares 55 ordinary VGPRs and initializes v54
from v0 before the selected access, so the mutation uses an allocated runtime
value. The frozen profile policy requires an exact-one installed mutation and a
failing numerical oracle for every engine; SuperCollider additionally requires
a detector diagnostic. The checked-in JSON records the regeneration command,
toolchain version, selector identity, and register basis so a rebuilt kernel
fails loudly instead of silently reusing stale coordinates.

Artifact paths in this section are relative to the workspace root.  The
paired clean artifacts and retained fault bundle are:

- `rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-wait-fix-review-20260725`;
- `rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-wait-fixed-sc-clean-20260725`;
- `rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-wait-fixed-sc-paired-20260725`;
- `rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-wait-fixed-moi-paired-20260725`;
- `rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-wait-fixed-sc-inventory-20260725`;
- `rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-wait-fixed-sc-fault-trial1-20260725`;
- `rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-wait-fixed-sc-fault-trial2-20260725`;
- `rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-wait-fixed-sc-post-fault-health-20260725`.

The checked-in ledger pins rocJITsu
`0f79cfc7f7ea3bd1f149c0be4fc662eeb548f97d`, corpus
`0db836e7bd8c6400b7ffd187d749225899875d7c`, and rocm-libraries
`0a323b74932c57d6d1a94af4a009dd7676b8f695`.  SHA-256 also pins the Tensile
config (`3d40a61d238f82aaa6bdd6b9e8fb4d417d8fe94de507bad35ea2da848d581d52`),
gfx950 simulator config
(`2f36f532932e5960424b31d9909ead67936168fbd551a125658170d066e0fd49`),
corpus runner
(`717bb3f1d639d3a3452be2ac146cc4e56eb5b9c69d307d051ac3f36dd867f666`),
client wrapper
(`cf6a90c93cdaadfca898a62ea88edbff737aaf27e6cb9631f35f7dc61fb86ec0`),
and loaded hook
`0be89aec2512038d31c389523796e9b755e4d9c0e7422b22b72a3e3cdea8744e`;
the SuperCollider and MOI paired-overhead summaries are
`f29aa2cb8488c2f0a2d5744af750649cdac6f5cae5f1cb486eb71025a84b126e`
and
`099ab4439e5993beaba0d096cb5790a746c5153f59490150766fb39bf2f37079`;
the MOI fault summary is
`6d86da84430c4338e84fd2ab8169f56897ac48d1158fe4bc33489d5488707842`;
the retained runner logs record the exact shell invocation, paths, runtime
environment, target, and generated gfx950 ELF checks, while only instrumented
logs contain hook and `RJ_*` settings.  All four profile cells satisfy the
declared bundle contract; the accepted qualified misses are not claims of
positive detector sensitivity.  The MOI fault qualification was completed
under `bd-1w9.42`, a Tensile-specific positive detector control is tracked by
`bd-1w9.43`, and the independent scalar-load wait hazard tracked by
`bd-1w9.9.8` is closed.

A separate generator cleanup for `bd-1w9.9.5` is pinned at rocm-libraries
`a4d052933951130bb6776f6ea39d4b7b87c4cad3`.  It keeps the
`GlobalReadPerMfma` default on the registry's float wire type and centralizes
the derived `DirectToLdsMetadata` integer flag without weakening strict type
validation.  The regression executes the production sparse-metadata
finalization branch for both enabled and disabled outcomes, while unrelated
matrix-conversion tests copy rather than mutate the process-global defaults.
At corpus `0db836e7bd8c6400b7ffd187d749225899875d7c`, the retained
`rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-msgpack-fixed-a4d05293-verified-20260725`
run passes the exact numerical oracle and gfx950 ELF checks in 5.084 seconds
with neither schema-mismatch nor `std::bad_cast` output.  The logged failure
for the unqualified helper filename is an expected lazy-loading probe before
the client tries the available `xnack-` variant; the debug-loaded evidence
below makes that sequence explicit.  Its `results.csv` and runner log hashes
are
`3aa62283aa84cd32fdee46564ccb19d3cbc451411ca39c61fad4311b9db042be`
and
`08ab09a24a5ae86aada3bb611e3a3775e69a65b5326176c0fbc50285c4d6ccef`.
The retained
`rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-msgpack-unit-fix-20260725`
logs record 63/63 focused type tests and the complete TensileLite Python unit
suite at 1,177 passes, 202 skips, and one expected failure; their SHA-256
hashes are
`6b6da24cabbdadaa866f2628b826afe29a7c8a71e690fd4451de2f10e18a0106`
and
`8e82a54b2b0342c7e9d179778ab58f3a601e6fba20cfd7cd4d1c22d9127823d5`.
The `bd-1w9.9.11` follow-up is pinned at rocm-libraries
`ccf6befac45a48e0e309ba940b7733ccb8d8e4a4`.  Its production assignment
boundary converts only plain integer `GlobalReadPerMfma` values whose float
equivalents belong to the canonical registry; booleans, strings, and
out-of-range integers remain unchanged for strict diagnostics.  The opt-in
table does not infer coercions from registry shape; it is the shared extension
point for another parameter only after that parameter's wire contract is
established.  The bead's original “roughly 200 files” was a scoping estimate.
A retained exhaustive offline sweep at
`rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-explicit-normalization-20260725`
passes the measured complete population of 4,522 explicit values across 97
upstream YAML files through that production boundary with zero mismatches.
Its README and executable sweep command pin the rocm-libraries and corpus
revisions, exact invocation, and all 98 grep-selected candidates before the
YAML parser identifies the 97 files containing values; their hashes are
`eb9139a5ac1fb41fec2333be91518b085d5003c560b7ffda3e2983077104feae`
and
`842a588c0a8153c33abffba7a877b33ff6aad236e25ff63dc04b693bbb85bd22`.
The same artifact root records
70/70 focused tests and the complete unit gate at 1,184 passes, 202 skips, and
one expected failure.  The verbose focused log names both accepted
MessagePack round trips and rejected boolean, string, and out-of-range cases.
The focused, full, and sweep log SHA-256 hashes are
`c30e5e87559a4108430017de02c32a46d792c107fc95987146cd05d11367fb4e`,
`35e7b95605f4179b319e3ac3dba11fb06a1965723992137d8efd4e4de7e85b28`,
and
`78a866f49d9c058c285edf8f95720e1a8960b9d025b29848c46635386ac25ba5`.
The committed-revision
`rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-explicit-normalization-ccf6befa-20260725`
run passes the exact numerical oracle and gfx950 ELF checks in 5.065 seconds
without a type warning or `std::bad_cast`; its `results.csv` and runner log
hashes are
`d0d5957424d574c10a88fee8e951428cb03b54f6a35197da03a03144ddbe2048`
and
`723d71b01c4f3209b676c6242407ba0c030f71879699f83f19a38259e9a20036`.
The `bd-1w9.9.12` follow-up is pinned at rocm-libraries
`88c30745d152f23b6955edc0fec1d30b7f7e7c7d`.  It models
`LocalWritePerMfma` as float scheduling values plus integer sentinels derived
from the registry.  It reuses the explicit opt-in int-to-float table established
for `GlobalReadPerMfma`, rather than adding a bespoke conversion path; a future
sentinel added to the registry joins the derived sentinel set.  The production
assignment path preserves `-1` as an integer, canonicalizes the equivalent
`-1.0` spelling to that integer sentinel, and converts valid non-sentinel
integer shorthands to floats.  Booleans, strings, the undocumented and
unsupported `-2`, and out-of-range integers remain strict diagnostics.  This
is a generator-state contract: the final runtime `TensileLibrary.yaml` does
not serialize either per-MFMA scheduling parameter.

The retained sweep at
`rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-local-write-contract-20260725`
is exhaustive over the observed corpus population: all 4,598 explicit values
across 106 upstream YAML files pass through the production assignment,
validation, and MessagePack round-trip paths with zero mismatches.  The source
and wire populations are identical: 4,596 integer `-1` sentinels and two float
`0.5` values.  The focused suite, rather than this two-value corpus
population, covers `-1.0` canonicalization, integer-shorthand conversion, and
strict rejection branches.  The README, executable sweep, sweep log,
focused-test log, and full-unit log hashes are
`46f46b3657fdc7457de8da139e6de20c1e58c42423b6789593960edaf843562e`,
`6e7776928318f11337cdbb5733ac9ae385100fa0d6b16e55cede1c05001dfb8c`,
`770a2899fde1cc29bba5a8262f47489562d78da10402461ae2415eb02c6a31c7`,
`ba899ca7b2af1d608ddf8db3df8ccc6070777556b0534c176de1991c343f61bd`,
and
`4bb01585606aff084922753c993d1cf934cbe6f2b41c526d83872e70ef163939`.
The logs record 82/82 focused tests and the complete unit gate at 1,196
passes, 202 skips, and one expected failure.  The committed-revision
`rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-local-write-contract-88c30745-20260725`
run passes the exact numerical oracle and gfx950 ELF checks in 5.167 seconds
without a type warning or `std::bad_cast`; its `results.csv` and runner log
hashes are
`cf0109a1198dc0b57219476a363dcee2c8cf45745f11d140eb309d2fbae395aa`
and
`202a76d50dcd06bc32d33282adcd0958aff77f5478bd65652d0c2c6e368cc619`.
The debug-loaded rerun at
`rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-local-write-contract-debug-88c30745-20260725`
shows `TensileLibrary_gfx950.co` loading, the expected miss on the unqualified
helper name, successful fallback to
`Kernels.so-000-gfx950-xnack-.hsaco`, the retained Stream-K kernel invocation,
and `validation=PASSED`.  Its non-finite timer fields reflect the simulator's
unavailable GPU event timer rather than a skipped kernel.  The debug README,
results, and runner log hashes are
`98e10d082e46ad5f28dc47862f6c23388aa2d3ca96a5162a8501564fa7fb3e98`,
`2f1028f0fda675ae1ab5f2e60a98c59fde596431141e535e697891fd8073f5d5`,
and
`2525969cc6b114b49d0508cf335e34dea1fb69deacabdddb87f43bc13e843db2`.
Exactly one local review round covered the original producer head
`76ad46c3b837`; all findings were amended into final head `88c30745d152` and
resolved without a reviewer rerun or Curator pass.  The final fixes therefore
have focused/full test and audit coverage, but not an independent rebuttal
review pass.

The `bd-1w9.9.8` producer fix is pinned at rocm-libraries
`0a323b74932c57d6d1a94af4a009dd7676b8f695`.  The shared grouped-GEMM
user-argument loader now builds supplemental fixed-slot reloads into their own
module and derives one `s_waitcnt lgkmcnt(0)` dependency boundary from whether
that module and an actual non-preloaded prefix load are both present.  The
rule is therefore tied to emitted production loads rather than a target,
register number, test selector, or hand-maintained list of current argument
types.  A focused construction-level regression covers Beta-only, ScaleA-only,
ScaleB-only, combined Beta-plus-scales, no-supplemental-load, and no-in-flight-
prefix cases plus the fixed-slot offset contract.  It passes 10/10; the
complete TensileLite unit gate passes 1,206 tests with 202 skips and one
expected failure.

The regenerated `TensileLibrary_gfx950.co` now contains
`s_waitcnt lgkmcnt(0)` between the overlapping
`s_load_dwordx8 s[40:47]` prefix and `s_load_dword s45` Beta reload.
Waitcheck reports zero diagnostics for that library and both xnack helper
HSACOs, and the exact numerical oracle passes in 5.741 seconds.  The retained
artifact root is
`rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-wait-fix-review-20260725`.
Its `results.csv`, runner log, waitcheck log, dependency disassembly, and full
unit log hashes are
`05db297d454686c8c062ae9c1be97355cc4f7d3b2ac7b8efa4024508ad2aa938`,
`3dd09ec194176595310bf410b678975d3cfcee903706cdab0eb58011a0c021fc`,
`805f6874a3dd5d6ad7ae4bf68367311fa2a3660b0b166bedf9bc365e8fa3c6d9`,
`440bdb3d2207a4302dd5f7d30b543dd83acc5b65c9d7ee992c57e8cba3619f0d`,
and
`708575d2fe14d621ea36315b4e10030e65f1440f2c3ac800863240bac4aeef00`.
Exactly one four-reviewer local round covered original head `c4dbaf1a70b4`;
all seven comments were addressed in amended head `0a323b74932c`, resolved,
and approved without a reviewer rerun or Curator pass.

The current producer also passes a paired campaign over every strict MOI
profile.  Its execution order is baseline-before, Record/Replay, Sampled,
Inline Shadow, then baseline-after.  The 5.697- and 5.401-second baseline legs
give a 5.549-second paired baseline.  Its raw files are under the explicit
profile directories
`rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-wait-fixed-moi-paired-20260725/record-replay`,
`rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-wait-fixed-moi-paired-20260725/sampled`,
and
`rocjitsu-test-corpus/.pytest-artifacts/consan-gfx950-tensile-wait-fixed-moi-paired-20260725/inline-shadow`:

- Record/Replay emits 187 total patches, covering all 82 accesses and 11
  barriers.  Its 462,544-byte primary report contains 82/82 visible access
  records with none dropped;
  replay processes all 82, reports no conflict, overflow, unsupported record,
  or diagnostic, and returns report memory to zero.  The exact numerical
  oracle passes in 6.291 seconds, or 1.13x.  Its `results.csv` and runner log
  hashes are
  `257b2674264672c410effd99c359b8589c3e950019b324f4926f2ae5150a89b2`
  and
  `6381da3c8475a0148e60929b17a8488029ea0f0cd57fa525388af07954eaa6d8`.
- Sampled emits 187 total patches, covering all 82 accesses and 11 barriers.
  Its 106,448-byte primary report provisions 656 banks/watchpoints, records
  eight visible sampled windows, and reports no dropped window,
  stale/incomplete/changed/malformed snapshot, conflict, unsupported
  synchronization, or diagnostic.  Cleanup returns report memory to zero.
  The exact numerical oracle passes in 7.467 seconds, or 1.35x.  Its
  `results.csv` and runner log hashes are
  `f9da87d699742980d963656deaa359c284182c866fdb0c1a8cd1dae671844116`
  and
  `1ae2b8673a9cadcc434bb3f93f79cead76559cc2b82e0deecc03a461721af09c`.
- Inline Shadow emits 176 total patches, covering all 82 accesses and 11
  barriers.  Its 12,598,640-byte primary report includes 8,192 inline-LDS
  bytes, 524,288 exact-shadow entries, and 64 release/snapshot/token records.
  Runtime records 465,408 events and 18,432 visible exact-shadow entries with
  no incomplete, changed, malformed, unsupported, overflow, or diagnostic
  outcome.  Cleanup returns report memory to zero.  The exact numerical oracle
  passes in 33.200 seconds, or 5.98x.  Its `results.csv` and runner log hashes
  are
  `ea641830bc5e9fe5e0fd7b6de43c31ed9867866c47006fc004aedd554d455f8a`
  and
  `0d32c1989c4889c8aeb834d05ef43aaf4b3ceed7a41b738c1c4102e842f0df28`.

The SuperCollider and MOI ratios come from separate paired campaigns with
different baseline legs.  Both are host-dominated end-to-end measurements;
they must not be compared with each other as relative engine overhead.

The same current-producer inventory and selector also qualify each MOI fault
cell.  Fault discovery is profile-independent and follows the validator
contract: one SuperCollider dry run inventories 11 target-native barriers,
then each requested profile applies the prospectively frozen occurrence-zero
selector at code-object identity `fnv1a64:89754d9658cf27ec` and PC `0x10a4`.
The precommitted outcome is an exact-oracle pass with no MOI diagnostic, so
each accepted result is a qualified miss rather than a detector hit.  Green
for this row means that every declared bundle gate passes with that
prospectively frozen outcome; it is not a positive detector-sensitivity claim.
Positive gfx950 detector-hit rows exist elsewhere in this matrix, including
Record/Replay D128, while a Tensile-specific positive control is tracked by
`bd-1w9.43`.

Each profile has two independent fault trials.  Record/Replay, Sampled, and
Inline Shadow finish their first trials in 6.400, 8.008, and 33.245 seconds
and their second trials in 6.202, 7.012, and 32.970 seconds.  Every trial
records one requested, planned, and applied deletion, patches all 82/82
accesses and all 10 surviving barriers, passes the exact oracle with zero
diagnostics, and returns report memory to zero.  Clean pre-fault health runs
immediately before the second trials pass in 6.208, 6.976, and 33.181 seconds;
clean post-fault runs immediately afterward pass in 6.164, 6.724, and 33.350
seconds.  Each health run restores 82/82 access and 11/11 barrier coverage,
keeps the exact oracle and no-diagnostic contract, and completes report
cleanup.

The paired root above contains `fault-summary.json`, which pins both trials,
the earlier post-fault checks, the directly containing before/after health
runs, and every raw-file SHA-256.  The literal trial roots under
`rocjitsu-test-corpus/.pytest-artifacts` are:

- `consan-gfx950-tensile-wait-fixed-rr-fault-trial1-20260725`;
- `consan-gfx950-tensile-wait-fixed-sampled-fault-trial1-20260725`;
- `consan-gfx950-tensile-wait-fixed-inline-fault-trial1-20260725`;
- `consan-gfx950-tensile-wait-fixed-rr-pre-fault-health-20260725`;
- `consan-gfx950-tensile-wait-fixed-rr-fault-trial2-20260725`;
- `consan-gfx950-tensile-wait-fixed-rr-post-fault-health-trial2-20260725`;
- `consan-gfx950-tensile-wait-fixed-sampled-pre-fault-health-20260725`;
- `consan-gfx950-tensile-wait-fixed-sampled-fault-trial2-20260725`;
- `consan-gfx950-tensile-wait-fixed-sampled-post-fault-health-trial2-20260725`;
- `consan-gfx950-tensile-wait-fixed-inline-pre-fault-health-20260725`;
- `consan-gfx950-tensile-wait-fixed-inline-fault-trial2-20260725`;
- `consan-gfx950-tensile-wait-fixed-inline-post-fault-health-trial2-20260725`.

The cumulative placement work is closed under `bd-1w9.9.6` and
`bd-1w9.9.7`.  The producer fix for `bd-1w9.9.8` makes all three generated
code objects waitcheck-clean at this same revision; none of the four current
instrumented runs reports the historical scalar-load dependency hazard.  The
current MOI fault parity work was completed under `bd-1w9.42`.

## PyTorch expansion

The portable exact-oracle operators are now runnable on the physical gfx950.
The earlier `hipErrorInvalidImage` was an environment error, not a PyTorch
operator or ConSan defect: the selected interpreter contained a thin gfx1250
wheel whose external kernel-pack directory had only `torch_gfx1250.kpack`,
even though `torch.cuda.get_arch_list()` also advertised gfx950.  It remains
untouched for software-target validation.

A separate official ROCm 7.1 nightly environment at
`$CONSAN_VALIDATION_WORKSPACE_DIR/consan-pytorch-gfx950-venv` contains
`torch==2.14.0.dev20260722+rocm7.1`.  Select it explicitly with
`CONSAN_VALIDATION_PYTORCH_PYTHON`; do not reuse the gfx1250 interpreter.
With workspace TheRock discovery tools and all software-model variables unset,
the validator doctor reports `gfx950:sramecc+:xnack-`, passes its numeric
dispatch, and confirms that the exact workspace ConSan hook is mapped.  The
official wheel supplies its own HIP/HSA runtime libraries, as documented in
`VALIDATION.md`; this is not evidence that PyTorch maps TheRock's runtime.
One uninstrumented repetition of all six rows below passes its exact oracle.
The validator now scrubs those software-model variables automatically for
`--target gfx950` while preserving them unchanged for gfx1250.

```sh
export CONSAN_VALIDATION_PYTORCH_PYTHON="$CONSAN_VALIDATION_WORKSPACE_DIR/consan-pytorch-gfx950-venv/bin/python"
unset HSA_MODEL_LIB HSAKMT_SIM_LIB HSA_MODEL_TOPOLOGY HSA_MODEL_NUM_THREADS
unset HSA_ENABLE_SDMA HSA_ENABLE_SCRATCH_ASYNC_RECLAIM HSA_ENABLE_INTERRUPT
```

The tensor-descriptor-add and cluster-synchronization rows from the gfx1250
ledger are intentionally not copied.  Their runner implementations call
target-specific descriptor or cluster APIs.  A gfx950 row must instead come
from a native gfx950 lowering (for example the Tensile cluster-local-read rows
above), not from renaming architecture-specific source.

| Priority | Tracking unit | SuperCollider | Record/Replay | Sampled | Inline Shadow | Why it matters and next proof |
|---|---|---|---|---|---|---|
| P0 | `torch.mode`, large rows | 🟧 Exact oracle and dynamic execution pass, but static analysis is incomplete at 199/23,298 supported accesses | 🟩 The current physical-gfx950 run passes the exact value/index oracle and ordinary 30-second contract in 27.61 seconds with complete static, analysis, and dynamic verdicts, all 25,523 accesses and 3,920 barriers patched, lossless replay of 13,017 accesses plus 49 barriers, and zero diagnostics. Record/Replay now prefers coarse-grained auto-report storage when available, eliminating the fine-grained sparse-copy teardown floor without changing Sampled or Inline Shadow allocation policy | 🟨 The current physical-gfx950 run passes the exact oracle and ordinary 30-second contract in 29.47 seconds, with complete static and dynamic evidence, all 25,523 accesses and 3,920 barriers patched, and zero forbidden diagnostics. Paired overhead and reviewed-fault acceptance remain | 🟨 The repaired 60-second diagnostic run is fully accepted in 49.62 seconds with the exact value/index oracle, zero forbidden diagnostics, complete static/analysis/dynamic evidence, and all 25,523 accesses plus 3,920 barriers patched. The ordinary 30-second latency contract, paired overhead, and reviewed-fault acceptance remain | Exact values/indices. Current Inline artifact `rebase-20260821-gfx950-pytorch-mode-inline-coplan-final-v3` closes the entry-prologue composition and private-dispatch validation regressions. Current Sampled artifact `rebase-20260821-gfx950-pytorch-mode-sampled-linear-final` records the accepted clean frontier. Current Record/Replay artifact `rebase-20260822-gfx950-pytorch-mode-rr-coarse-v1` closes the remaining latency gate and promotes the cell from red to green; the other named artifacts retain the per-profile frontiers. |
| P0 | `torch.topk`, FP64 spill and BF16 coverage cases | 🟧 Exact oracle and dynamic execution pass, but static analysis is incomplete at 3,056/230,438 supported accesses | 🟧 The current 180-second diagnostic exits cleanly in 116.12 seconds and passes both exact BF16 and FP64 value/index oracles with zero diagnostics and complete dynamic replay. Reachable full-pressure sites use shared branch-only relay routing, and reservoir planning now counts later barrier demand even when access selection omits its donor; the large object nevertheless has no safe owner-local donor corridor from the affected sites to appended text. Static analysis remains incomplete at 232,814/239,722 accesses and 6,743/11,423 barriers: 6,908 accesses still have no legal first SOPP hop and consequently 4,680 barriers remain omitted | 🟧 The current 180-second diagnostic exits cleanly in 92.06 seconds and passes both exact BF16 and FP64 value/index oracles with zero diagnostics and complete dynamic evidence. Branch-only scalar-spill routing and the empty-EXEC guard patch 232,814/239,722 accesses and all 6,743/6,743 applicable barriers. The remaining 6,908 access sites have no legal first SOPP hop; ordinary 30-second latency and static completeness remain | 🟧 The current physical-gfx950 clean run clears the former growth-cap rejection, passes the exact oracle in 171.37 seconds with zero diagnostics and complete dynamic evidence, and patches all 11,423 barriers. Static coverage remains incomplete at 122,420/239,722 supported accesses: the 20,792,320-byte large object now produces a valid 363,435,008-byte replacement, but patches 65,536/182,838 supported accesses while the smaller object patches all 56,884 | Exact values/indices across FP64 register pressure and BF16 coverage. Current Record/Replay artifact `prep-20260822-gfx950-topk-rr-barrier-frontier-candidate-v6` proves the exact barrier-demand rule and directly bounds the residual donor-free no-first-hop limitation; the earlier growth artifact records the independent envelope repair. Current Inline Shadow artifact `rebase-20260821-gfx950-pytorch-topk-inline-diagaddr-drRcFh` records the first valid large-object replacement. Current Sampled artifact `rebase-20260822-gfx950-pytorch-topk-sampled-empty-exec-fix-v2` closes the physical fault and converges on the same bounded no-first-hop frontier as Record/Replay. |
| P1 | `torch.sort` over segmented rows | 🟧 The repaired dense route completes on physical gfx950 with exact sorted values and indices, zero mismatches, and complete 45,340/45,340 supported-access coverage. Device execution takes 31.52 seconds and the full row takes 36.25 seconds. The ordinary 30-second latency and static classification of 11,544 additional unsupported accesses remain | 🟩 At clean revision `ef787fd7fe`, the target-resolved 60-second bundle accepts both bracketing baselines and Record/Replay. The exact value/index oracle passes with complete 56,884/56,884 access plus 6,032/6,032 barrier coverage, zero clean diagnostics, and complete static/analysis/dynamic evidence. The process takes 32.00 seconds and the device median is 27.52 seconds. A prospectively frozen exact-one drop of the actually dispatched FP32 radix kernel's cooperative-LDS-initialization barrier is reached and accepted: it produces 23 replay diagnostics while preserving the exact oracle, with complete mutation, parser, containment, and health evidence | 🟨 The current physical clean row passes the exact value/index oracle in 32.92 seconds with complete 56,884/56,884 access plus 6,032/6,032 barrier coverage, zero forbidden diagnostics, and complete static/analysis/dynamic verdicts. Paired overhead and reviewed-fault acceptance remain | 🟨 The repaired owner-local scalar/dispatch plan completes in 56.74 seconds with exact sorted values and indices, zero forbidden diagnostics, complete 56,884/56,884 access plus 6,032/6,032 barrier coverage, and complete static/analysis/dynamic verdicts. The ordinary 30-second latency, paired overhead, and reviewed-fault acceptance remain | Current clean-source Record/Replay artifacts `prep-20260822-gfx950-pytorch-sort-rr-clean-ef787-v1` and `prep-20260822-gfx950-pytorch-sort-rr-fault-lds-init-ef787-v1` close the target envelope, reviewed fault, parser completeness, containment, health, and provenance gates and promote the cell from yellow to green. The inventory and occurrence 7/8/9/10 artifacts retain four rejected prospective contracts without relabeling. Earlier occurrence-6 artifacts expose and close silent overlimit replay skipping while retaining their rejected prospective oracle contract. Earlier `rebase-20260821-gfx950-pytorch-sort-record-replay-stride1-v2` closes the no-evidence standard-cadence rejection. Current SuperCollider artifact `rebase-20260821-gfx950-pytorch-sort-supercollider-cdna-tail-fix-diagnostic90-v6` closes the physical-VCC collision and retains the remaining unsupported-access frontier. Current Inline artifact `rebase-20260821-gfx950-pytorch-sort-inline-owner-dispatch-coplan-final` closes the residual index corruption and lifts that cell from red to yellow. Current Sampled artifact `rebase-20260821-gfx950-pytorch-sort-sampled-slot-budget-v1` closes the mixed-unit report-slot budget and lifts that cell from orange to yellow. Current all-profile artifact `rebase-20260821-gfx950-pytorch-sort-all-current` retains the other frontiers. Earlier Record/Replay artifact `consan-gfx950-pytorch-sort-rr-dense-host-fix-20260722-195514` remains historical evidence. |
| P1 | `torch.histc` with a shared-memory-sized bin count | 🟧 Current exact oracle and dynamic execution pass with 102/102 supported accesses patched, but the aggregate analysis/static verdict is incomplete | 🟩 The expanded clean-source pair passes exact FP32 and FP64 bin-count oracles in 7.54 seconds with complete 179/179 access plus 84/84 barrier coverage, zero diagnostics, complete analysis/static/dynamic verdicts, and 127.22x maximum device slowdown. A distinct prospectively frozen exact-one drop of the FP64 specialization's final `ds_add_f64`-to-copyout publication barrier is reached and accepted: Record/Replay diagnoses the conflict and the unconstrained numerical manifestation passes. The two earlier FP32 trials remain rejected under their original policies | 🟩 Current exact FP32/FP64 bundle has complete 179/179 access plus 84/84 barrier coverage, zero diagnostics, complete static/analysis/dynamic verdicts, and 209.80x maximum paired device slowdown. A distinct prospectively frozen exact-one FP32 copyout-barrier drop at audited stride one is reached and accepted as `not_detected/pass`, with 179/179 accesses, 83/83 surviving barriers, 24 sampled windows, bounded teardown, and healthy pre/post probes. The stronger FP64-copyout and FP32-initialization detection hypotheses remain rejected without relabeling | 🟨 Current exact oracle passes in 28.95 seconds with complete 179/179 access plus 84/84 barrier coverage and complete analysis/static/dynamic verdicts; paired overhead and reviewed-fault acceptance remain | Clean artifact `prep-20260822-gfx950-pytorch-histc-dual-precision-rr-v1` and fault artifact `prep-20260822-gfx950-pytorch-histc-fp64-rr-fault-v1` promote Record/Replay from yellow to green. Current Sampled artifact `prep-20260822-gfx950-pytorch-histc-sampled-fp64-v1` records the accepted paired and qualified-miss bundle plus both rejected stronger hypotheses and promotes Sampled from yellow to green. The all-profile artifact retains the other engine frontiers. No site, expectation, or observed outcome is revised or relabeled. The expanded checked-in `HistogramScatter` pair owns the architecture-general integer/FP32 LDS-atomic collision and missing-publication behavior on all five targets plus physical gfx950. |
| P2 | Collision-heavy `torch.scatter_reduce` (`sum`, BF16 and FP32) | 🟧 Exact collision-count oracles and dynamic execution pass in 7.05 seconds, but analysis/static coverage is incomplete with no applicable site | 🟩 Current clean-source paired bundle passes both exact collision-count oracles with complete 27/27 ordinary-access coverage, zero diagnostics, complete static/analysis/dynamic verdicts, and 257.55x maximum slowdown; ordered-atomic fault modes are typed N/A for these relaxed singleton updates | 🟩 Current clean-source paired bundle passes both exact collision-count oracles with complete 27/27 ordinary-access coverage, zero diagnostics, complete static/analysis/dynamic verdicts, and 144.92x maximum slowdown. Current target-native inventory confirms real BF16/FP32 relaxed singleton atomics, and the ordered-atomic fault family is reviewed and accepted as typed N/A because these updates establish no release/acquire publication edge | 🟨 Exact oracles pass in 10.49 seconds with complete 27/27 ordinary-access coverage and no diagnostics; paired overhead remains | The relaxed collision behavior is independently protected by the checked-in adjacent `HistogramScatter` correct/incorrect device pair. SuperCollider remains inapplicable to this object; the other engines transform ordinary accesses around the atomics. Current Sampled artifact `prep-20260822-gfx950-pytorch-scatter-sampled-v1` promotes that cell from yellow to green without inventing a fault contract. |
| P2 | `torch.linalg.vector_norm` and large-row `torch.softmax` | 🟧 The repaired longer diagnostic passes the exact 3-4-5 norm and CPU-softmax oracle, is dynamically complete, and patches all 4,436/4,436 supported accesses. Static analysis remains incomplete because a separate library contains 384 unsupported accesses, and the 60.87-second device execution plus transformation still exceeds the ordinary 30-second contract | 🟩 At clean revision `a99ba905b8`, the target-resolved 60-second paired bundle passes both exact oracles with complete 4,820/4,820 access plus 2,096/2,096 barrier coverage, zero diagnostics, and complete static/analysis/dynamic verdicts. Record/Replay device time is 27.65 seconds versus 51.30 and 54.25 ms bracketing baselines; the complete process takes 32.06 seconds. A prospectively frozen exact-one drop of the first of two adjacent barriers in the actually dispatched FP32 norm specialization is reached and accepted as a qualified miss with the exact oracle passing, no diagnostic, exact mutation accounting, clean containment and health, and clean provenance | 🟨 The current physical-gfx950 row passes the exact 3-4-5 norm and CPU-softmax oracle and the ordinary 30-second contract in 28.98 seconds. It has complete 4,820/4,820 access plus 2,030/2,030 barrier coverage, zero forbidden diagnostics, and complete static, analysis, and dynamic verdicts; paired overhead, reviewed-fault, and clean-provenance acceptance remain | 🟥 The current 90-second diagnostic passes the exact oracle in 69.95 seconds and patches all 2,096 barriers, but only 4,779/4,820 accesses. Compact clobbered-address spill recovery is fixed; the remaining 41 full-pressure sites need a safe branch-only relay roughly 1.5 MiB away and fail the static verdict | Current clean-source Record/Replay artifacts `prep-20260822-gfx950-pytorch-norm-softmax-rr-clean-a99-v1` and `prep-20260822-gfx950-pytorch-norm-softmax-rr-fault-clean-a99-v1` close the process-bound, reviewed-fault, containment, health, and provenance gates and promote the cell from yellow to green. Current Inline artifact `rebase-20260821-gfx950-pytorch-norm-softmax-inline-compact-clobber-v2` isolates the residual long-range route after the compact-load repair. Current Sampled artifact `rebase-20260821-gfx950-pytorch-norm-softmax-sampled-sparse-semantic-final` proves bounded sparse-report teardown and lifts the cell from red to yellow. Current SuperCollider diagnostic artifact `rebase-20260821-gfx950-pytorch-norm-softmax-sc-wave64-text-gate` proves the wave64 relay-reservoir and executable-text gate repairs. Current all-profile artifact `rebase-20260821-gfx950-pytorch-norm-softmax-all-current` retains the other current-tip frontiers. |

### 2026-08-22 PyTorch histogram Sampled qualification

Current artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-histc-sampled-fp64-v1`
passes the exact 64-bin FP32 and FP64 count oracles with complete 179/179
access and 84/84 applicable-barrier coverage, zero forbidden diagnostics, and
complete static, analysis, and dynamic verdicts. The bracketing FP32 device
baselines take 15.508 and 10.065 ms versus 2,682.408 ms for Sampled, for a
12.786-ms mean baseline and 209.80x slowdown. The much smaller FP64 row takes
0.042/0.102/0.037 ms in baseline-before/Sampled/baseline-after order.

The first prospectively frozen trial dropped the executed FP64
accumulation-to-copyout barrier and required a Sampled diagnosis at audited
stride one. The mutation was reached and applied exactly once, but both exact
oracles passed and Sampled emitted no diagnosis. The second stronger trial
dropped the executed FP32 initialization-to-accumulation barrier under the
same frozen detector policy. An initial identity with the wrong template
variant was correctly not admitted; the current-inventory identity was then
re-frozen before execution, reached, and applied exactly once, but again both
oracles passed without a diagnosis. Both admitted hypotheses remain rejected;
neither expectation is changed after observation.

A distinct third trial prospectively selected the FP32
accumulation-to-copyout barrier at PC `0x3a8a0`. In light of the two independent
schedule-masked results, its policy was frozen before execution as
`not_detected/pass`, still at audited stride one. The mutation is requested,
planned, reserved, installed, and applied exactly once. It preserves both
exact oracles and complete 179/179 access plus 83/83 surviving-barrier
coverage, commits 24 sampled windows, and emits no diagnosis. The command
takes 7.628 seconds, peaks at 252,040 live report bytes, releases all report
memory without allocation, capacity, or cleanup failure, and passes both
pre/post physical-GPU health probes.

The checked-in `HistogramScatter` pair retains the stronger observable
contract instead of copying the physical scheduler's qualified miss: its
correct member requires exact collision-heavy integer/FP32 LDS-atomic results
without a diagnostic, while its adjacent missing-publication member requires
the conflict. All 60 baseline/all-engine rows pass on the five simulator
targets and physical gfx950. The Sampled cell is green.

### 2026-08-22 PyTorch histogram and norm Record/Replay refresh

The expanded clean-source histogram artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-histc-dual-precision-rr-v1`
passes exact 64-bin count oracles for both the FP32 and FP64 shared-memory
specializations with complete 179/179 access and 84/84 barrier coverage, zero
diagnostics, and complete static, analysis, and dynamic verdicts. The
instrumented process completes in 7.54 seconds. Its maximum paired device
slowdown is 127.22x: the FP32 kernel takes 2,880.11 ms versus the bracketing
baseline median, while the newly covered FP64 kernel takes 0.341 ms and has an
8.66x device slowdown.

Before execution, a new fault policy was frozen against the actually
dispatched FP64 specialization. The selected unconditional barrier follows
collision-heavy `ds_add_f64` accumulation and precedes final global copyout.
Artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-histc-fp64-rr-fault-v1`
accepts the exact-one mutation: Record/Replay reaches the site and emits its
own conflict diagnostic, while the deliberately unconstrained numerical
manifestation passes. This distinct positive contract promotes the
Record/Replay cell from yellow to green.

The two earlier FP32 trials remain historical rejected evidence. The
prospectively frozen drop of the final accumulation-to-copyout barrier produced
the expected Record/Replay diagnosis, but physical scheduling preserved the
exact counts instead of the precommitted failure. A separate distinct
initialization-barrier trial at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-histc-rr-init-fault-0a4-v2`
also reaches and produces the expected diagnosis, but this time the exact
counts fail instead of the precommitted pass. It too is correctly rejected.
Neither site nor expectation is reused, relabeled, or revised. The expanded
checked-in `HistogramScatter` correct/incorrect pair now executes exact FP32
LDS atomic-add payloads alongside the original integer histogram and global
scatter idioms; it owns the architecture-general collision and
missing-publication behavior across all five RocJitsu targets and physical
gfx950.

Clean-source norm/softmax artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-norm-softmax-rr-current-0a4-v1`
passes the exact 3-4-5 norm and CPU-softmax oracles with complete 4,820/4,820
access and 2,096/2,096 barrier coverage, zero diagnostics, and complete
verdicts. The paired device baseline is 58.28 ms versus 27,486.21 ms for
Record/Replay, a 471.64x slowdown. A bounded target override now gives this
physical row 60 seconds of whole-process time because a baseline process spends
about 4.7 seconds in fixed Python/framework startup and teardown even though
its device interval is only about 51 ms. The confirming paired artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-norm-softmax-rr-timeout60-UPO6DG`
accepts both baselines and Record/Replay: the instrumented device interval is
27,347.70 ms and the process completes in 32.18 seconds with the same full
coverage and zero diagnostics. A focused validator regression pins the
target-resolved 60-second contract without changing the production
Record/Replay cadence.

The first prospectively frozen fault trial targeted the loaded-but-unused
`NormTwoOps<float,...,Lb0>` specialization. It applied exactly one mutation
and passed its frozen `not_detected/pass` policy, but was correctly rejected
because mutation installation was not execution reach. Descriptor-to-dispatch
mapping then proved that the workload actually launches the sibling `Lb1`
specialization. Final-ISA review shows all paths reconverge before its adjacent
barriers at text offsets `0x281644` and `0x281648`. A new prospective trial,
retained under
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-norm-softmax-rr-fault-dispatched-redundant-v1`,
drops the first barrier exactly once and is accepted as a reached qualified
miss: the exact oracle passes, no diagnostic is emitted, coverage remains
complete, containment and pre/post health pass, and no expectation is revised
after execution. The architecture-general `DoubleBufferedPipeline` and
`TwoStageSoftmax` correct/incorrect pairs already own the adjacent-barrier and
reduction/publication behaviors across all five targets plus physical gfx950;
adding an E2E-layout-specific device fixture would duplicate those contracts.
Clean-source reruns at revision `a99ba905b8` are retained under
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-norm-softmax-rr-clean-a99-v1`
and
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-norm-softmax-rr-fault-clean-a99-v1`.
Both bracketing baselines and Record/Replay are accepted; the instrumented
device interval is 27,647.42 ms, its process completes in 32.06 seconds, and
the sources record the exact clean revision. The same frozen fault specification
again applies exactly once and is accepted with the exact oracle, no diagnostic,
reviewed reach, healthy pre/post probes, and clean provenance. This completes
the bundle and promotes Record/Replay from yellow to green.

### 2026-08-22 PyTorch TopK Record/Replay no-first-hop inventory

Bounded physical diagnostics under
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-topk-rr-owner-debug-v1`
and
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-topk-rr-nop-inventory-v1`
preserve both exact TopK oracles and reproduce the unchanged
232,814/239,722 access plus 6,743/11,423 barrier frontier. The 6,908 omitted
accesses are all eight-byte instructions in 72 full-pressure owners; none of
those owners contains an ordinary access with a dense router. The large object
has zero pristine NOP relay words even before dense planning, and none of the
6,908 sources can reach the generated appended relay bank as its first SOPP
hop. Reducing direct-reservoir discovery to two words found no legal owner-local
first-hop reservoir, and neither dense-host tails, seven-word dense-entry
compaction, nor owner-affine NOP allocation changed the physical coverage. The
unsafe or ineffective rediscovery experiments were removed; in particular,
live dense-entry instructions are not mislabeled as relay storage. Selected
multiword access tails are now published only from exact paid ownership and
are useful in the generic cross-pass contract described below, but this TopK
object has no spare selected-tail corridor at the affected sites.

A focused positive host regression now proves the independent useful rule:
once a multiword Record/Replay access is selected, its dead replaced tail is
exactly recorded and may extend the routing frontier for later selected
accesses. The existing negative regression still proves that an intrinsically
unreachable site is rejected before reserving its large body. These contracts
do not claim that TopK is fixed: the remaining orange cell requires a design
that creates owner-local routing capacity without incidental caves, most
likely by explicitly relocating additional host instructions or by changing
the full-pressure routing/code-layout ABI. Repeating long TopK runs without
such a design change is deferred under the plan's tactical-latency rule.

Two current physical follow-ups preserve that boundary. Artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-topk-rr-access-tail-frontier-candidate-v5`
passes both exact TopK oracles with zero diagnostics and complete dynamic
replay in 116.98 seconds, but leaves coverage unchanged at
232,814/239,722 accesses and 6,743/11,423 barriers. Artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-topk-rr-barrier-frontier-candidate-v6`
also passes both exact oracles with zero diagnostics and complete dynamic
replay in 116.12 seconds, with the same static frontier. The latter uses hook
SHA-256 `f3851af8835f8118b12d699928ace41b6585daddd79c8ee2a4df26ebe828b8ea`.

The follow-up adds two durable host contracts even though it does not promote
the E2E cell. `Cdna4RecordReplayBarrierRoutesThroughSelectedAccessAnchorTails`
proves that a later barrier may consume exact NOP tails published by already
selected multiword accesses, including independent entry and return claims.
`Cdna4RecordReplayBarrierConsumesReservoirOmittedByAccessSelection` now places
the barrier more than one SOPP hop before an omitted full-pressure access and
requires reservoir planning to count the barrier itself as downstream demand.
An explicit A/B run against the former access-only rule rejects the barrier
and leaves every reservoir unused; the current rule passes final validation.

A debugger stop at the large TopK planning call confirms that the current rule
sees all 6,908 access demands and 4,680 barrier demands, with the earliest
source at `.text` offset `0x23fcc`. A complete inventory finds 57,155 legal
reservoir candidates throughout `.text`, correcting the earlier inference that
the 400 adopted reservoirs were the complete donor population. The limiting
fact is instead a mandatory terminal branch-range cut: every affected access
and barrier needs independent entry and return paths across that cut, for
`2 * (6,908 + 4,680) = 23,176` relay words. The 400 reservoirs reachable while
the recursive frontier advances to approximately `.text+0xe66204` provide only
13,688 relay words, a deficit of 9,488. The final bounded scan examines 204
candidates in that cut and rejects none for overlap, placement, routing, or
dependency; all 204 have already been adopted. Capacity earlier in the object
cannot compensate because every appended entry and return route must first
cross this bottleneck.

The paired host regression
`DirectReservoirDemandCapacityMustHoldAcrossEveryCut` retains that
architecture-independent planning invariant. Its sufficient member provides
two 16-word donors in each of three recursive cuts and requires all six
reservoirs, 90 relay words, and four recursively routed donors to reach the
earliest cut. Its adjacent insufficient member removes half the terminal-cut
capacity while retaining surplus donors in both earlier cuts; planning must
stop at one local reservoir rather than falsely treating earlier capacity as a
route. The focused test passes in 24 ms. The existing checked-in
`CdnaRecordReplayLongRangeFullPressure` correct/incorrect device pair remains
the observable TopK-derived contract for exact execution, no clean diagnostic,
and the missing-publication conflict. Inflating a device fixture until it needs
23,176 prototype relay paths would freeze an incidental implementation layout,
so the distinct cut-capacity rule belongs in the host planner suite.

A subsequent owner-local-entry experiment halves the relay demand of an
*eligible* fixed-stack site by relocating one single-entry instruction window,
backing up its otherwise live PC/SCC tuple in a liveness-dead VGPR, jumping
indirectly to the appended probe, and routing only the return. The router has a
focused preplaced-entry contract, and access/barrier host tests pin emission,
metadata, final validation, and rejection across a reconvergence leader or
another synchronization site. This work also strengthened the existing
TopK-derived `CdnaRecordReplayLongRangeFullPressure` device contract: it caught
both the initial use of a live spill-window VGPR and an entry window that
crossed a reconvergence target. The corrected pair passes on RocJitsu gfx942
and gfx950 and on physical gfx950; the complete focused host suite passes
848/848 tests and the five-target simulator device suite passes 1,658/1,658.

The authoritative physical follow-up at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-topk-rr-borrowed-single-entry-v1`
uses hook SHA-256
`5b3fdcf7a336642ac93467f23c5f7d042288bf0840067ada8ff8264fba1c1759`.
It exits cleanly in 117.59 seconds, passes both exact BF16 and FP64 TopK
value/index oracles, emits zero diagnostics, and reports complete dynamic
replay. Its static frontier is nevertheless unchanged at 232,814/239,722
accesses and 6,743/11,423 barriers: none of the limiting sites satisfies the
complete relocation, single-entry CFG, protected-range, and dead-backup-VGPR
proof. The optimization is therefore retained as a safely tested primitive,
not represented as a TopK repair or color promotion.

This is direct evidence for a deeper routing/code-layout ABI frontier rather
than missing candidate discovery, cross-pass ownership, demand accounting, or
search effort. A production repair must either reduce the number of independent
far-cut routes--for example through owner-local gateways or shared dispatch--or
provide at least 9,488 additional legal relay words in the limiting cut. The
cell remains orange, and another long physical run is deferred until such a
design changes the capacity equation.

### 2026-08-21 PyTorch norm/softmax Record/Replay barrier-only validation

The current physical-gfx950 run completes in 54.56 seconds, passes the exact
3-4-5 norm and CPU-softmax oracles, and has complete 4,820/4,820 access plus
2,096/2,096 barrier coverage. Record/Replay emits no diagnostics and reports
complete static, analysis, and dynamic verdicts. Its only apparent failure was
in the validation harness: the replayed reader had barriers but no visible
access records, so its compact replay input and scratch diagnostic capacity
were both zero while its full report diagnostic capacity remained 1,474,560.
The validator incorrectly expected the full capacity to be clamped to the
zero-access replay input.

The parser now validates the full report capacity and the compact replay
scratch capacity as separate producer contracts. Focused host tests cover both
the barrier-only zero-input case and a nonempty replay whose scratch capacity
is clamped below its report capacity. All 212 validation tests pass. The
existing two-stage-softmax correct/incorrect device pair already owns the
reduction, global-intermediate publication, and missing-barrier behavior, so
this host-only validator repair does not duplicate that device semantic.

### 2026-08-21 PyTorch top-k Record/Replay growth envelope

The first current-tip diagnostic reached strict load rejection in 52.16
seconds. Waitcheck and inventory completed in 22.50 and 11.42 seconds, but the
20,792,320-byte generated object required 403,542,016 bytes of total patched
image growth, 888,832 bytes beyond the former 384-MiB default. This was the
same default policy used for every ordinary workload, not a process limit or a
top-k-specific override.

Qualified generated workloads now set the bounded default at 400 MiB. The
post-fix artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-pytorch-topk-record-replay-growth-fix-v2`
exits zero in 118.16 seconds and passes the exact BF16 and FP64 value/index
oracles. Replay is dynamically complete with zero diagnostics. The remaining
orange frontier is honest static coverage: the large object patches
175,930/182,838 accesses and 711/5,391 barriers, while the smaller object
patches all 56,884 accesses and 6,032 barriers. The FP64 device execution itself
takes 93.19 seconds, so the ordinary 30-second contract cannot be recovered by
host teardown changes alone.

`PatchedImageGrowthDefaultCoversQualifiedGeneratedOperator` pins admission of
the measured alignment-inclusive envelope. All 587 selected ConSan/hook host
tests and all 244 checked-in Record/Replay simulator rows across gfx942,
gfx950, gfx1100, gfx1201, and gfx1250 pass. The existing segmented-top-k
correct/incorrect device pair owns the behavior; the growth envelope is a host
resource-policy regression rather than a new device semantic.

### 2026-08-21 PyTorch top-k Record/Replay branch-only routing

The large TopK object assigns 6,908 supported LDS accesses to full-pressure
scalar-spill plans with no dead PC/SCC registers. The former implementation
could only use a direct SOPP branch from each access to its appended body, so
all 6,908 were omitted before execution. Record/Replay now uses the shared
branch-only relay router when a legal monotonic corridor exists. It fixes the
general large-kernel case without adding a Record/Replay-specific relay
algorithm, and validation accepts the emitted access body as the same
well-formed branch-only continuation already used by other engines.

Two focused host regressions bracket the contract. A 33-access, 160-KiB CDNA4
kernel with abundant NOP relay capacity routes every full-pressure access in
one shared planning transaction and passes final validation. Its paired
no-first-hop fixture uses only one-word branch blocks, supplies neither NOPs
nor relocation donors, and proves that an unreachable body is rejected before
reservation, with zero router calls and no partial patch or transform error.
The latter also guards the image-growth and stale-cave failures exposed while
studying TopK.

Current physical artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-pytorch-topk-record-replay-far-route-v5`
uses hook SHA-256
`43cbfbb38a38a232b2a38fff47a7d29c9209857fe5c24e639dc162754f5c3d0c`.
Both exact BF16 and FP64 value/index oracles pass, the process exits zero in
135.86 seconds, replay is dynamically complete over 37,504 visible accesses,
and no diagnostics are emitted. Static coverage is honestly incomplete at
232,814/239,722 accesses and 6,743/11,423 barriers. All 6,908 remaining access
omissions have no legal first SOPP hop in their `sbtopk::gatherTopK` kernels;
the associated 4,680 barrier omissions therefore remain a production-design
frontier rather than a transform crash. The cell stays orange.

### 2026-08-22 PyTorch top-k Sampled empty-wave spill repair

The pre-fix physical Sampled row faulted at a nil address in the first
full-pressure BF16 TopK dispatch. A compact reduction of
`sbtopk::gatherTopK` retained eight LDS sites, live scalar pressure across
every transient window, and an empty `EXEC`. It exposed two coupled planner
requirements: Sampled must route a branch-only scalar-spill body when no
indirect PC/SCC state is available, and that body must test empty `EXEC` before
performing any per-lane private save.

The focused host regression
`ConSanMoi.Cdna4SampledBranchOnlyScalarSpillGuardsEmptyExecBeforePerLaneSave`
requires the branch-only assignment, forbids indirect PC/SCC state, verifies
final validation, and proves that the body's first instruction jumps directly
to its return continuation. The adjacent correct/incorrect device pair now
runs Sampled as well as baseline and Inline Shadow on RocJitsu `gfx942`,
`gfx950`, `gfx1100`, `gfx1201`, and `gfx1250`, plus physical `gfx950`. The
cross-target run also exposed an RDNA3 entry-prologue bug: scalar-spill setup
clobbered `v0` before private owner capture, collapsing two wave owners into
one. The focused host regression
`ConSanMoi.Rdna3SampledPrivateOwnerCapturePrecedesEntryScalarSpillScratchClobber`
pins capture-before-clobber ordering. All ten new simulator rows now pass in
1.15 seconds and both new physical rows pass in 0.55 seconds. The complete
150-test Sampled host slice and the complete 731-test ConSan MOI host suite
also pass; all 1,336 simulator device rows pass in 56.12 seconds.

Current physical artifact
`/home/ossci/xx/consan-validation/rebase-20260822-gfx950-pytorch-topk-sampled-empty-exec-fix-v2`
no longer faults: the instrumented process exits zero in 92.06 seconds and
passes both exact BF16/FP64 value and index oracles with zero diagnostics and
complete dynamic evidence. Static coverage rises from 107,882 to
232,814/239,722 accesses and reaches all 6,743/6,743 applicable barriers. The
remaining 6,908 accesses have the same no-first-SOPP-hop geometry already
isolated by Record/Replay, so this evidence lifts the technical floor to a
bounded orange limitation rather than promoting the cell to green.

### 2026-08-22 PyTorch mode Record/Replay coarse-report allocation

The sparse-snapshot repair left the fully accepted `torch.mode` row at 40.79
seconds, and a current-tip confirmation reproduced the same frontier in 41.17
seconds. The device oracle itself completed in 22.68 seconds. The remaining
latency was host teardown: automatic report allocation always selected the
first fine-grained HSA region, so Record/Replay copied its roughly 335 MiB
semantically visible sparse snapshot with CPU loads before replay.

Record/Replay auto reports now prefer a coarse-grained region when one is
available, retaining the former fine-grained fallback. Sampled, Inline Shadow,
retain their fine-grained preference; SuperCollider uses its separate report
allocator and is unaffected. Coarse Record/Replay teardown uses the existing
HSA bulk-copy path, so the full 533,183,672-byte report is transferred without
the fine-grained CPU-copy floor. The focused host regression offers fine and
coarse regions in that order, requires Record/Replay to choose coarse and to
fall back to fine when needed, and requires Sampled and Inline Shadow to remain
fine; all 166 HSA hook tests pass. This is a host allocation-policy contract
rather than a new device synchronization semantic, so the existing adjacent
Record/Replay correct/incorrect pairs remain the appropriate device coverage.

Artifact
`/home/ossci/xx/consan-validation/rebase-20260822-gfx950-pytorch-mode-rr-coarse-v1`
uses hook SHA-256
`2cb6b9e9bb5fb180ac4f74797b3ec1a3ad1792d085f50bb9afd98f4d6759053e`.
The physical run exits zero in 27.61 seconds, passes the exact value/index
oracle, and reports complete static, analysis, and dynamic verdicts. All
25,523 accesses and 3,920 barriers patch; replay consumes all 13,017 access
records and 49 barrier records with zero diagnostics, drops, unsupported
records, or incomplete evidence. This clears the ordinary 30-second contract
and promotes the Record/Replay cell from red to green. As a checkpoint, all
250 Record/Replay simulator device rows pass across the five target
architectures, all 58 physical-gfx950 Record/Replay rows pass, and the
post-instrumentation GPU health check passes.

### 2026-08-21 PyTorch `sort` SuperCollider physical-VCC tail repair

The pre-fix dense transform patched all 45,340 supported radix-sort accesses
and emitted a valid replacement, but the first physical dispatch faulted at a
nil address. A focused checked-in device reproducer reduced the same failure
to 1,040 static `ds_write_b32` sites and a single active lane. RocJitsu
simulation passed while physical gfx950 faulted, establishing a hardware-state
modeling gap rather than an LDS behavioral mismatch.

CDNA's descriptor SGPR count includes a six-register allocation tail for
VCC, XNACK, and FLAT_SCRATCH. In the failing 16-SGPR kernel, physical VCC
occupied `s[10:11]`; dense routing independently selected that pair for its
indirect return PC because descriptor growth treated all 16 slots as ordinary
SGPR capacity. Descriptor growth now reserves the complete CDNA tail above
new ordinary temporaries. The affected route consequently grows from 16 to 24
allocated SGPRs, moving physical VCC to `s[18:19]` while retaining the proven
`s[10:11]` router assignment.

Artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-pytorch-sort-supercollider-cdna-tail-fix-diagnostic90-v6`
exits successfully in 36.25 seconds. The exact segmented-row value/index
oracle passes after 31.52 seconds of device execution, dynamic evidence is
complete, and all 45,340 supported accesses are patched. The row remains
orange because 11,544 additional accesses are classified unsupported, making
the aggregate static verdict incomplete, and because it exceeds the ordinary
30-second contract. Post-run physical health passes.

The host regression pins the exact `s[10:11]` route and 16-to-24 descriptor
growth. The new correct/incorrect device contract crosses the production
1,024-site dense threshold and passes all eight baseline/SuperCollider cases
on gfx950 simulation and physical gfx950. The existing generalized CDNA
dense-SCC fixture remains unchanged and passes all 24 gfx942/gfx950
simulation plus physical-gfx950 cases across the applicable engines. All
1,147 ConSan host tests pass.

### 2026-08-21 PyTorch norm/softmax Sampled sparse-report repair

The retained pre-fix artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-pytorch-norm-softmax-all-current`
passes the exact oracle after 24.23 seconds of device execution, then reaches
the ordinary 30-second bound before emitting an analysis verdict. Its last
complete report has 46,080 allocated Sampled slots but zero claimed windows,
zero visible entries, and zero pending acquires. The host reader nevertheless
walked the entire capacity. This isolates teardown, rather than device
execution or transformation, as the live latency failure.

The reader now treats the committed `sampled_causal_window_count` as the
number of `Ready` publications it must observe. An empty sparse report reads
zero slots, and a nonempty sparse report stops after finding its committed
windows. Deferred acquire-release reconstruction is linearized at the same
time: visible release slots are indexed once, and each relevant owner bank is
scanned once instead of once per visible entry. Direct-before-release-before-
acquire sync composition and malformed-duplicate handling remain unchanged.

Final-code artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-pytorch-norm-softmax-sampled-sparse-semantic-final`
uses hook SHA-256
`08143831681f0c6b57a057a2138e3c32d80d8502509c8abc57ec7ddda04e59cd`.
It exits successfully in 28.98 seconds; the exact 3-4-5 norm and CPU-softmax
oracle passes after 24.17 seconds of device execution. Both large reports
record zero examined watchpoint slots despite capacities of 46,080 and 14,368.
The final verdict is complete for all 4,820 accesses and 2,030 barriers, with
zero diagnostics or dynamic-incomplete evidence.

Focused host regressions pin both complexity contracts: an empty allocated
report examines zero capacity slots, while two visible entries sharing a
pending owner bank examine one causal-window capacity rather than two. All 186
hook tests pass. The checked-in two-stage-softmax pair already owns the
device-level reduction/global-intermediate behavior, and the Sampled Stream-K
pairs own acquire-release publication. Their 18 selected correct/incorrect
rows pass on all five RocJitsu targets plus physical gfx950 in 0.94 seconds, so
this host-teardown defect does not justify a duplicate device workload.

### 2026-08-21 PyTorch norm/softmax Inline compact-load repair

Artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-pytorch-norm-softmax-inline-compact-clobber-v2`
completes the exact 3-4-5 norm and CPU-softmax oracle in 69.95 seconds. It
patches 4,779/4,820 accesses and all 2,096 barriers, then exits 86 because 41
accesses remain uninstrumented. All 41 reside in
`cunn_SoftMaxBackward<2,double,double,double,LogSoftMaxBackwardEpilogue>` and
use compiler-generated LDS loads whose result overwrites their address VGPR.

The first resource cliff was valid and tractable: ordinary- and AccVGPR-bank
liveness leave a compact spill window that is disjoint from the clobbered
address. Inline Shadow can consume that address in place, restore the compact
window, and only then execute the displaced load. It no longer reserves an
unproven snapshot just beyond the spill window. Three focused host regressions
cover overlapping and disjoint addresses on CDNA3 and CDNA4 and verify that no
live out-of-window VGPR is borrowed.

The checked-in `cdna_clobbering_lds_reduction` correct/incorrect pair distills
the PyTorch max/sum reduction, tied address/result load, ordinary-VGPR and
AccVGPR liveness, and missing-publication fault. Baseline plus all four engines
pass on simulated gfx942, simulated gfx950, and physical gfx950: 30/30 rows in
6.39 seconds. Inline rows additionally require the selected 16-VGPR compact
spill-backed recovery plan.

This does not promote the E2E cell. Once the compact plan is available, the 41
full-pressure sites still need branch-only scalar state. Their first safe relay
after the owning kernel is roughly 1.5 MiB away, beyond a direct SOPP branch.
A bounded experimental admission also regressed four barriers and was not
retained. The remaining red therefore requires recursive code relocation or a
higher-level long-control-flow abstraction; it is documented and deferred
while work moves to Record/Replay.

### 2026-08-21 PyTorch `sort` Inline empty-wave scalar-spill diagnosis

Artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-pytorch-sort-inline-empty-exec-fix-long`
records a 900-second physical-gfx950 diagnostic after adding a leading
empty-EXEC guard to scalar-spill Inline bodies.  The hook transforms the
4,763,376-byte radix-sort object in 16.77 seconds, emits a valid 364,101,360-byte
replacement, and reports complete 56,884/56,884 access plus 6,032/6,032 barrier
coverage.  It enters the first radix dispatch and grows its private segment
from zero to 128 bytes without the former immediate GPU fault, but the dispatch
does not complete before the bound and therefore emits no exact oracle or
dynamic verdict.

Inspection of that retained object exposed a second edge of the same bug: the
guard jumped directly to the inert guest LDS operation, bypassing the indirect
long-return PC construction that a production-scale body needs.  The current
candidate instead targets the start of that return setup.  Artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-pytorch-sort-inline-empty-exec-long-return-fix`
confirms that the repaired dispatch completes in 58.05 seconds with complete
static and dynamic evidence and no ConSan diagnostic.  Values are exact, but
the ordinary index oracle finds 744 mismatches: rows 0, 2, and 3 contain mostly
identity indices, while row 1 is exact.  This is now the isolated remaining
failure rather than a device hang.

Bounded kernel-filter diagnostics identify the dispatched specialization as
`radixSortKVInPlace<2, -1, 128, 8, float, long, unsigned int>`.  Instrumenting
only its 242 accesses and 40 barriers uses the local VGPR-backed `68/68`
EXEC/owner plan and passes the exact oracle with eight attributed dispatches.
Forcing the full object's scalar `36/36` ABI on that same specialization fails
in 26.58 seconds, after lowering 84/242 accesses, with a physical GPU memory
fault.  The remaining defect is therefore tied to the object-wide scalar ABI,
not the radix algorithm itself.  These diagnostics used a 45-second hard cap;
further work proceeds from checked-in reductions rather than repeated PyTorch
runs.

A focused host test
forces eight accesses, full scalar pressure, and a body beyond direct-return
range, while an adjacent correct/incorrect device pair generalizes the
empty-EXEC scalar-spill contract across RocJitsu `gfx942`, `gfx950`, `gfx1100`,
`gfx1201`, and `gfx1250` plus physical `gfx950`.  All 24 baseline and Inline
rows pass together in 1.13 seconds.  Each member has eight LDS access sites,
selecting the scalable appended-body path rather than merely exercising a
one-site compact route.  This stronger reduction also exposed a separate
gfx1201/gfx1250 fixed-stack failure in the quick suite: those targets chose an
indirect router whose call/PC state could only be saved per lane, which is not
available under empty `EXEC`.  Fixed-stack RDNA4-family owners now prefer their
branch-only scalar-spill route, and target-specific host planner tests retain
that contract.  The complete 171-test Inline host slice passes in 1.65 seconds.

### 2026-08-21 PyTorch `sort` Inline owner-local dispatch closure

The final physical-gfx950 clean artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-pytorch-sort-inline-owner-dispatch-coplan-final`
uses hook SHA-256
`a123f4045133b947cf6bf25f9bab007078238325be7607678637263b2fb44ecf`.
It exits successfully in 56.74 seconds and passes the exact sorted-value and
index oracle with zero forbidden diagnostics. Static, analysis, and dynamic
verdicts are complete for all 56,884 accesses and 6,032 barriers. This closes
the 744-index corruption above; the ordinary 30-second latency, paired
overhead, reviewed-fault, and clean-provenance gates remain open.

The remaining fault was in object-wide scalar-state planning. The generated
radix library combines owners with different complete SGPR tails. Planning
could choose a low site-liveness hole for the whole object, or let the
automatic persistent dispatch pair occupy the stronger high owner-wide EXEC
window. The repaired planner prefers the complete owner-wide proof, relocates
the automatic dispatch pair only when doing so recovers another owner, and
gives an incompatible owner its own scalar-spill ABI plus entry-captured
dispatch VGPR pair. Owners that admit the object-wide pair keep the smaller
representation.

`Cdna4InlinePrefersOwnerWideFreshWindowToObjectWideSiteDeadWindow` is the
focused host regression. It synthesizes disconnected 64- and 96-SGPR owners,
requires both LDS sites to be emitted, rejects the tempting liveness-only
window, and verifies that only the high-pressure owner receives persistent
dispatch VGPRs. The existing heterogeneous-owner and dynamic-stack dispatch
tests retain their prior representations, and all 722 `ConSanMoi.*` host tests
pass together. The adjacent eight-site scalar-spill device pair continues to
cover the implementation-independent empty/nonempty-wave behavior across all
five RocJitsu targets plus physical gfx950; the new regression is intentionally
host-side because its additional invariant is code-object resource planning,
not a distinct synchronization behavior. The complete checked-in device
matrix also passes all 1,541 rows in 553.71 seconds at `-j64`, including all
293 serialized physical-gfx950 rows and the final GPU health check.

The immediate neighboring `torch.mode` regression is also closed by the two
small host contracts added with this repair. A dense access router may relocate
an ordinary entry instruction into an indirect-island host whose first word is
the final word of the seven-word owner/epoch entry prefix; prologue composition
now chains through that host instead of overwriting it. Exact-shadow final
validation also derives a private dispatch reload from either the selected
owner-local non-spill EXEC window or the code-object-wide fallback, even when
unrelated owners have component-local assignments. The physical-gfx950 result
at
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-pytorch-mode-inline-coplan-final-v3`
is fully accepted in 49.62 seconds with the exact value/index oracle, zero
forbidden diagnostics, complete static/analysis/dynamic evidence, and all
25,523 accesses plus 3,920 barriers patched. Its hook SHA-256 is
`509c61c39bb7344105c30d85cbdc2e7bee639356a32a3948a03482e55017e442`.
Only the ordinary 30-second latency contract, paired overhead, and reviewed-
fault acceptance keep the Inline cell yellow.

### 2026-08-21 PyTorch `sort` Sampled report-slot budget repair

Artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-pytorch-sort-sampled-slot-budget-v1`
is fully accepted in 32.92 seconds with the exact sorted-values-and-indices
oracle, zero forbidden diagnostics, complete static/analysis/dynamic evidence,
and all 56,884 accesses plus 6,032 barriers patched.  It supersedes the prior
52,904/56,884-access diagnostic frontier and lifts the Sampled cell from orange
to yellow; paired overhead and a reviewed fault remain.

The 3,980 omissions were not placement failures.  PyTorch radix-sort uses
dual-address LDS forms, so 56,884 access sites expand to 70,420 logical ranges.
Sampled incorrectly compared those banked report slots with `max_patches`, a
site-count limit, and stopped when the unlike units crossed.  Lowering now has
independent site and report-slot budgets.  A focused host regression exercises
two dual-address sites on CDNA3/4/5 and RDNA3/4, while a checked-in adjacent
correct/incorrect device contract runs baseline and Sampled on all five
RocJitsu targets plus physical gfx950.  All 24 focused rows pass in 1.11
seconds.

### 2026-08-21 PyTorch `sort` Record/Replay cadence repair

The standard stride-65,536 run transforms all 56,884 accesses and 6,032
barriers and passes the exact segmented-sort oracle in 32.42 seconds, but its
compact four-row schedule selects no workgroup. It therefore exits 86 with no
dynamic evidence. This is the same validation-cadence failure already modeled
for compact gfx950 histogram and application schedules, not a missing patch or
workload-correctness failure.

The gfx950 workload override selects stride 1; the production profile remains
unchanged. The initial qualification used a conservative 300-second contract.
The artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-pytorch-sort-record-replay-stride1-v2`
is fully accepted in 110.21 seconds. Device execution takes 27.54 seconds,
the exact value/index oracle passes, all 56,884 accesses and 6,032 barriers are
patched, static/analysis/dynamic evidence is complete, and replay reports zero
diagnostics. The later clean qualification below tightens this target-specific
bound to 60 seconds. The validator's target-resolution unit matrix pins stride
1 and that bound only for gfx950; gfx1250 and the other targets retain their
ordinary defaults. The existing checked-in `RepeatedDispatchIdentity`
correct/incorrect device pair already guards the corresponding Record/Replay
dispatch-identity behavior across applicable targets.

### 2026-08-22 PyTorch `sort` Record/Replay bounded replay

Current paired artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-sort-rr-current-cd4-v1`
passes the exact sorted-value/index oracle with zero diagnostics and complete
56,884/56,884 access plus 6,032/6,032 barrier coverage. The baseline-before
and baseline-after device medians are 41.63 and 35.16 milliseconds; the
27,497.49-millisecond Record/Replay median is a current paired 716.19x
slowdown. The instrumented process takes 32.55 seconds, so the ordinary
30-second latency contract remains open.

Fault inventory artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-sort-rr-inventory-cd4-v2`
collects all 6,032 exact singleton barrier identities. The first prospectively
frozen selector targeted an undispatched Half specialization and was correctly
rejected without a reach witness. A second prospective selector targeted
occurrence 6 in the dispatched float radix specialization, with detector
`detected` and oracle `pass` frozen before execution. Pre-fix artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-sort-rr-fault-float-occurrence6-ce09-v1`
applies exactly one barrier drop and makes the exact sort oracle fail, but
Record/Replay emits no diagnostic. Its teardown explains the miss precisely:
corrupted LDS offsets raise the dense scratch estimate to 1,073,741,824
entries, above the one-million-entry host limit, so replay is skipped.

Auto replay now caps its scratch at 1,048,576 entries and still executes the
sparse fail-closed model. Focused host regression
`AutoReplayBoundsSparseShadowAndFailsClosedForOverlimitRange` synthesizes an
overlimit record and requires bounded replay, one metadata-full diagnostic,
an incomplete dynamic verdict, and no silent skip. Post-fix artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-sort-rr-fault-float-occurrence6-bounded-v1`
uses hook SHA-256
`978cb83c4a7269902c236534cc7a0e338791fb85752b690ea54fdb4126d11b30`,
applies the same exact mutation once, processes all 95,633 committed access
records and 368 barrier records, and emits four access-conflict plus six
metadata-full diagnostics without exhausting diagnostic capacity. The
detector contract now passes. The trial remains formally rejected, without
relabeling, because its exact oracle still fails against the precommitted
`pass` policy. The existing cross-target `SegmentedTopK` correct/incorrect
device pair owns the user-visible radix-phase publication and missing-edge
contract; bounded host replay is the distinct regression fixed here. All 168
HSA-hook tests and all 292 simulator Record/Replay device rows across gfx942,
gfx950, gfx1100, gfx1201, and gfx1250 pass together.

A current inventory using hook SHA-256
`3eaac597ecd8ad73bd2de95b102005651048ab6250cdaf0e5d86c3326e0cecf6`
is retained at
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-sort-rr-inventory-own-rocm-9RPOVp`.
It again collects all 6,032 singleton barriers in 22.35 seconds. Instruction
review then froze three additional selectors before execution, each in the
dispatched 128-thread float radix specialization:

- occurrence 7 publishes eight 32-bit LDS lanes before strided reads. Artifact
  `prep-20260822-gfx950-pytorch-sort-rr-fault-publication-occ7-kZMaB0`
  applies it exactly once, preserves the exact oracle, and emits no diagnostic,
  so its prospective `detected/fail` contract is rejected;
- occurrence 8 separates those completed 32-bit reads from reuse of the same
  LDS storage for 64-bit value/index pairs. Artifact
  `prep-20260822-gfx950-pytorch-sort-rr-fault-phase-transition-occ8-FwBoYa`
  applies it exactly once and emits 1,016 replay diagnostics, but the exact
  output passes against the frozen `fail` policy, so it is rejected; and
- occurrence 9 publishes the 64-bit stage before its strided consumers.
  Artifact
  `prep-20260822-gfx950-pytorch-sort-rr-fault-wide-publication-occ9-xSOSsC`
  applies it exactly once and preserves the exact output without a diagnostic,
  so its prospective `detected/pass` contract is rejected; and
- occurrence 10 is the loop handoff after those 64-bit reads. Clean-source
  artifact
  `prep-20260822-gfx950-pytorch-sort-rr-fault-loop-handoff-occ10-clean-a99-v1`
  applies it exactly once but preserves the exact output without a diagnostic,
  so its prospective `detected/any` contract is rejected.

No policy was revised after observing an outcome. These trials show why
continuing to sample nearby phase barriers is not productive: physical
scheduling can mask either the oracle, the diagnostic, or both.

Final-ISA review then selected occurrence 11 in the actually dispatched FP32
radix kernel. All waves first initialize two LDS regions, wait, and synchronize
before active lanes publish bin counts and peer lanes begin LDS reads. Its
`detected/any` policy was frozen before execution. Clean-source artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-sort-rr-fault-lds-init-ef787-v1`
applies the barrier drop exactly once, preserves the exact sorted-value/index
oracle, and emits 23 Record/Replay diagnostics. The row passes in 50.14 seconds
under its 60-second target bound with complete static, dynamic, mutation,
parser, containment, and health evidence.

The first run exposed a validator vocabulary omission rather than a device
failure: the hook's legitimate `empty_accumulator_descriptor_growth` resource
alternative made evidence appear incomplete. The parser now recognizes that
event, and a focused host regression pins the exact descriptor-growth record
while retaining fail-closed rejection of unknown future vocabulary. Clean
paired artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-sort-rr-clean-ef787-v1`
accepts both bracketing baselines and Record/Replay at clean revision
`ef787fd7fe`. The instrumented row completes in 32.00 seconds with a
27,515.50-millisecond device median versus 37.83 and 38.01 milliseconds for
the baselines, complete 56,884/56,884 access plus 6,032/6,032 barrier coverage,
zero diagnostics, and complete static/analysis/dynamic evidence. This closes
the process-envelope and reviewed-fault gates and promotes Record/Replay to
green.

The architecture-general `CooperativeLdsInitialization` adjacent pair now
owns the newly extracted initialization-publication behavior; all 50
RocJitsu rows across CDNA3/4/5 and RDNA3/4 and all 10 physical-gfx950 rows
pass. `DoubleBufferedPipeline` continues to own LDS storage reuse and wide
publication, while `SegmentedTopK` owns radix selection. The complete expanded
1,995-row device matrix also passes in one `ctest -j64` invocation, including
all 1,618 simulator rows, all 377 serialized physical rows, and the final
physical health check.

### 2026-08-20 PyTorch `histc` all-profile refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-pytorch-histc-all-X38XO3`
records a current-tip physical-gfx950 baseline and all four profiles at source
revision `6ad008b654` and hook SHA-256
`587f327b6c269414c61a79c8915867fdb68fc2a434cc8586ffe41e19e11cfbbc`.
The exact baseline passes in 4.61 seconds. Sampled passes in 7.71 seconds and
Inline Shadow in 28.95 seconds; both cover all 179/179 supported accesses and
84/84 barriers with complete analysis, static, and dynamic verdicts.

SuperCollider preserves the exact oracle and patches 102/102 supported
accesses in 6.49 seconds, but reports an incomplete aggregate analysis/static
verdict. Record/Replay preserves the exact oracle and statically patches all
179/179 accesses plus 84/84 barriers with zero diagnostics, but its standard
cadence publishes no visible record. It exits 86 after 7.35 seconds with
`analysis_complete=false` and `dynamic_complete=false`. This current-tip
result downgrades the former green Record/Replay claim until the runtime
selection regression is repaired and covered.

Repair artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-pytorch-histc-rr-stride-fix-ZJHioX`
uses the target/workload-resolved stride 1 and a 300-second bound. The exact
baseline passes in 4.56 seconds and Record/Replay passes in 146.47 seconds,
publishing 320 access and 32 barrier records without drops or diagnostics.
Coverage is complete at 179/179 accesses plus 84/84 barriers, and the final
analysis, static, and dynamic verdicts are all complete. The conventional
validator suite's 209 tests pin both the gfx950-only cadence and timeout while
proving that gfx1250 and the other targets retain their standard defaults.

### 2026-08-20 PyTorch `scatter_reduce` all-profile refresh

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-pytorch-scatter-reduce-all-76lDCB`
records an exact physical-gfx950 baseline and all four clean profiles at source
revision `672fab534c` and hook SHA-256
`587f327b6c269414c61a79c8915867fdb68fc2a434cc8586ffe41e19e11cfbbc`.
Both FP32 and BF16 collision-count oracles pass under every profile. Sampled
and Inline Shadow report complete 27/27 ordinary-access coverage in 8.48 and
10.49 seconds; Record/Replay reports the same in 8.41 seconds with zero
diagnostics. SuperCollider finishes in 7.05 seconds but has no applicable
site and an incomplete analysis/static verdict.

The retained fault inventory proves that the reduction contains real atomic
instructions, but they are relaxed singleton updates rather than qualified
acquire/release sequences. They therefore are not causal synchronization
events and the ordered-atomic fault families are typed N/A, matching the
accepted gfx1250 contract. Their behavior is still covered: the exact E2E
collision sums and the checked-in `histogram_scatter_test.hip` correct/incorrect
pair both execute collision-heavy relaxed global scatter atomics and forbid a
false diagnostic. Sampled and Inline Shadow advance from gray to yellow on
current clean evidence; SuperCollider remains orange because the CDNA4 object
presents no applicable non-atomic LDS site.

Current Record/Replay paired artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-scatter-rr-current-622-v2`
closes that engine's remaining gate at clean source revision
`622039cb430e94614b7b96272f0e5dd324aba151`. Baseline-before and
baseline-after preserve both exact FP32/BF16 collision sums, as does the
Record/Replay row. The instrumented row covers all 27/27 ordinary accesses,
emits no diagnostic, and has complete static, analysis, and dynamic verdicts.
The BF16 device timing is 1,332.18 ms against a 5.17 ms paired baseline, for
the cell's maximum 257.55x slowdown; the FP32 device path is 1.60x. The real
atomics remain relaxed singleton updates, so ordered acquire/release mutation
is not a truthful fault contract. With that fault family typed N/A and the
adjacent checked-in collision-heavy correct/incorrect device contract already
green across architectures, the Record/Replay cell is green.

### 2026-08-22 PyTorch `scatter_reduce` Sampled qualification

Current artifact
`/home/ossci/xx/consan-validation/prep-20260822-gfx950-pytorch-scatter-sampled-v1`
passes the exact BF16 and FP32 collision-count oracles in both bracketing
baselines and Sampled. Sampled has complete 27/27 ordinary-access coverage,
zero diagnostics, and complete static, analysis, and dynamic verdicts. The
BF16 device row takes 1,269.311 ms against an 8.759-ms mean paired baseline,
for the cell's maximum 144.92x slowdown; the FP32 device row is 1.43x.

The current target-native inventory completes in 7.42 seconds and contains
real `global_atomic_pk_add_bf16` and FP32 singleton reduction sites. They are
relaxed collision updates, not release/acquire publication edges: weakening
an ordering suffix cannot remove a cross-agent payload edge that does not
exist. A reviewed external fault specification therefore records
`atomic-weaken-order` as typed N/A for Sampled, and the validator accepts that
disposition without executing a destructive mutation. This is not a detector
miss and no expectation was revised after observation. The checked-in
`HistogramScatter` correct/incorrect pair remains the adjacent portable
contract for collision-heavy updates plus a real missing-publication fault.
With the exact E2E collision behavior and that causal device diagnostic both
covered, the Sampled cell is green.

### 2026-08-20 PyTorch `mode` Record/Replay decoder repair

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-pytorch-mode-rr-MzeBQ8`
records an exact 4.92-second physical-gfx950 baseline followed by the current
strict Record/Replay rejection in 9.59 seconds. Diagnostic replay retained the
exact 49,633,376-byte original object under
`pytorch-torch-mode/diagnostic-code-objects/`. Both waitcheck preflight and the
ConSan inventory fail while building the CFG because the decoder reports an
invalid scalar-register selector. The rejected instructions occur in three
rocPRIM block-reduction trampoline kernels at text offsets 2,808,464,
3,911,008, and 4,212,624. No workload oracle or coverage verdict is reached.
The minimized failure was legal compiler output:
`v_readlane_b32 vcc_lo, v253, 0` and its matching `vcc_hi` form. LLVM accepts
the same VCC destination contract on gfx908, gfx90a, gfx942, and gfx950, but
their MR ISA descriptions use the restricted `OPR_SREG_NOVCC` destination.
The generator now widens only `v_readlane_b32` and `v_readfirstlane_b32` to the
general scalar selector on every affected CDNA target. Checked-in generator
tests pin the semantic exception and its negative boundary; exact-encoding C++
tests decode both VCC halves for both operations on CDNA1 through CDNA4 and
verify their architectural register identity.

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-pytorch-mode-rr-vcc-dst`
uses rebuilt hook SHA-256
`d1587d3a2290edc228d27df12070cb5d4c5689a05df685062511c682516af414`.
Its exact physical-gfx950 baseline passes in 4.88 seconds. Record/Replay no
longer emits a decode or load rejection: analysis reaches all 27,942 access
ranges and 1,003,520 barriers, completes the 533,071,032-byte report plan and
allocation, and enters patching. The contained process then reaches the
ordinary 30-second bound before patch completion, so no workload oracle or
coverage verdict is available. The remaining red frontier is therefore
large-object patch latency, not decoding or resource placement.

An explicit 120-second diagnostic run in artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-pytorch-mode-rr-120s`
then reached substantially farther. The 6.67-second waitcheck and 5.76-second
inventory were followed by a successful 19.12-second transform with 32,019
patches. The exact value/index oracle passed after 32.31 seconds of device
execution, and the report contained 13,017 committed access records plus 49
barriers with zero diagnostics. Teardown nevertheless reached the 120-second
bound because automatic replay iterated the fixed table's full 4,194,304-slot
capacity rather than its committed records.

Automatic replay now compacts only entirely zero, never-published slots before
model replay and provenance repair. Partially initialized Empty records remain
visible and fail closed. The producer log separately preserves the report ABI's
1,788,288-record diagnostic capacity and exposes the 13,017-record replay
scratch capacity. Conventional hook/model tests pin sparse first/last-slot
replay, the all-zero versus partial-record boundary, and parser treatment of
the two capacities.

Final artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-pytorch-mode-rr-replay-contract`
uses rebuilt hook SHA-256
`fe709b00fa9445c74c87a5f2ecd071b9a03d4d9e20e67de123f47293ad7c9781`.
The diagnostic run exits normally in 100.60 seconds and passes the exact
oracle. It patches all 25,503 supported accesses and all 3,920 physical barrier
sites; the report-plan value 1,003,520 is barrier-record capacity, not a static
site count. Replay consumes all 13,017 committed access records and 49 barrier
records with no drops, unsupported records, conflict, metadata exhaustion, or
diagnostics. Dynamic evidence is complete. At this checkpoint static analysis
remained incomplete only because 20 of 25,523 discovered accesses were
semantically unsupported; all 20 were `ds_read_b96` operations in rocPRIM
reduce-by-key kernels.

Artifact
`/home/ossci/xx/consan-validation/rebase-20260820-gfx950-pytorch-mode-rr-b96`
records the CDNA3/CDNA4 native-B96 repair under rebuilt hook SHA-256
`a0ae782c984366599fbf9ebbb40095af3a5f5a55002bcc755b6808cbc08156d3`.
The explicit 120-second diagnostic run is accepted in 100.16 seconds and
passes the exact value/index oracle. Analysis, static instrumentation, and
dynamic evidence are all complete: 25,523/25,523 accesses and 3,920/3,920
barriers patch, replay consumes all 13,017 access and 49 barrier records, and
there are no drops, unsupported replay records, diagnostics, or incomplete
code objects. The checked-in behavioral reduction pairs synchronized and racy
native 96-bit tuple publication, including an address/destination alias, on
RocJitsu-emulated gfx942 and gfx950 and on the physical gfx950 under every
ConSan engine. The sole remaining Record/Replay gate is now the ordinary
30-second latency contract.

Current-tip artifact
`/home/ossci/xx/consan-validation/rebase-20260821-gfx950-pytorch-mode-rr-sparse-snapshot-v1`
uses hook SHA-256
`53f025a91b1aa0d38464fffbd2813f076e47efec79d7233ba632b11ac7da1385`.
Record/Replay teardown now snapshots the full open-addressed access table but
only the published prefixes of append-only barrier, atomic, fence, and
diagnostic regions. Its one mandatory table scan simultaneously compacts the
13,017 replay inputs; pressure telemetry consumes that compact population
instead of rescanning and allocating for all 4,194,304 slots. Snapshot traffic
falls from 533,183,672 to 335,546,464 bytes. The physical run is fully accepted
in 40.79 seconds, down from 100.16 seconds, with the same exact value/index
oracle, complete 25,523/25,523 access and 3,920/3,920 barrier coverage, 49
dynamic barrier records, and zero diagnostics, drops, unsupported replay
records, or incomplete evidence. The ordinary 30-second contract remains the
only red criterion; the retained phases already include 6.36 seconds of
waitcheck, 5.74 seconds of inventory, 9.77 seconds of patching, and 22.58
seconds of device execution, so further work should target a measured phase or
revisit the row budget rather than repeatedly wait on this cell.

The portable rows are now registered for gfx950.  The next step is an
all-profile clean inventory, followed by focused repair of the first typed
failure in each column.  Do not carry gfx1250 site counts, fault selectors,
timing ratios, or applicability decisions into this table.

## Evidence baseline

The following establishes machine access and corpus availability.  It is not
instrumentation acceptance evidence.

| Item | Current evidence |
|---|---|
| Working branch | `shared/rocjitsu/sanitizers` |
| Survey base | `4495672ad45f3b90d0e367916e3231420a5be579`; this refresh records candidates above that committed implementation state |
| Retained first-campaign provenance | The detailed native evidence below was produced on the former `users/bjacob/consan-gfx950-take2` line.  It remains useful compatibility evidence, but is not current-tip acceptance evidence. |
| Device | AMD Instinct MI355X, `gfx950`, wave64 |
| ISA | `amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-` |
| Driver/runtime | ROCk 6.14.14; workspace TheRock HSA runtime 1.21 |
| ROCm distribution | `$WORKSPACE_ROOT/TheRock/build/dist/rocm` |
| hip-moi source | Local commit `29a1c212183b65f1ec9200b24a445862532e4dd8` supplies the shared CDNA source set and distinct gfx942/gfx950 targets. At validation time it is eight commits ahead of `origin/main` and is not contained by a fetched remote ref, so clean remote reproduction requires publishing or otherwise transferring that exact companion commit. |
| Physical dispatch smoke | On 2026-07-22, workspace TheRock `rocminfo` reports MI355X / `gfx950:sramecc+:xnack-`. Five native CDNA4 hip-moi host-reference tests pass in 92--188 ms, and corpus `hip_matmul` m128³ passes correctness for all three selected MFMA/shared-memory kernels. Separately, all 13 target-native hip-moi binaries pass 33/33 tests through the gfx950 RocJITsu simulator. |
| Validation corpus | `iree-test-suites` `49f46d6d4370e5aa0a6367751474e20c6c4e95c0`; required Sharktank assets present; LFS fsck clean |
| Validation doctor | The target-aware registry and workload-scoped doctor resolve all six native hip-moi roles through the explicit `hip-moi-build-gfx950-tests` build tree, including the Jakub counterpart. |
| RocJitsu test corpus | Local commit `0db836e7bd8c6400b7ffd187d749225899875d7c`; gfx950 enables source-built HIP matmul, HipKittens, HIP Stream-K, and rocBLAS cases and packages the bounded `gfx950_sk_sgemm_streamk` runtime row.  Commits `61b5af0b5ee9ef9221391f5f81550b5e295e7e59` and `46a4c58a7be89b4118c2e3f94081591783d5391a` introduced the Stream-K and Tensile rows, respectively; commits after the earlier `f88d4583022d438ea72fb82c0e89143ccbf61843` snapshot harden Tensile runner provenance and isolation.  No fetched remote ref contains the current commit, so clean remote reproduction requires publishing or transferring it.  The historical pre-generated Tensile artifact tree remains gfx1250-only. |
| gfx950 Tensile source pool | The 36-YAML pool was surveyed and the bounded Stream-K candidate selected at `rocm-libraries` `c2fafc16393d0ce47a0a5801d827d43f0d3714a4`; the packaged `gfx950_sk_sgemm_streamk` row was reduced from `a8f0845f87ab50adc3dc8d0edd86693cb31065b1`.  The remaining source pool is not part of the validation denominator. |
| PyTorch discovery | The gfx1250-only thin-wheel mismatch is diagnosed and isolated.  The separate official nightly environment passes `torch.arange` plus all six portable one-repetition exact oracles on gfx950.  Workload-scoped doctor confirms gfx950 numeric dispatch and exact-hook mapping. |
| Registry boundary | The six portable PyTorch rows, six hip-moi campaign roles, and `tensile-gfx950-lds-positive` are registered validation IDs for gfx950. Seven additional hip-moi binaries are offline simulator prerequisites, not campaign workload IDs. The earlier bounded Tensile Stream-K and HIP Stream-K rows remain external corpus executable denominators. Remaining source-built rows stay planned expansion until they are built and registered. |

## Implementation evidence

Every row below is gray because it is an inventory of earlier implementation
evidence, not a current-tip acceptance result.  Commit IDs and artifacts are
retained to guide focused revalidation; labels such as “host-proven” and
“native-proven” describe what the earlier campaign established only.

This table records prerequisite implementation gates.  They do not promote a
workload/profile cell without the end-to-end evidence above.

| Area | Status | Evidence |
|---|---|---|
| CDNA4 fixed scratch encoding | 🩶 Host-proven | `319950aacd`: exact two-word `scratch_store_dword` / `scratch_load_dword` encodings decode as 8-byte CDNA4 instructions; aligned offsets through 4092 accepted and 4096 rejected. |
| CDNA4 private wait | 🩶 Host-proven | `319950aacd`: `s_waitcnt vmcnt(0)` exact word `0xbf8c0f70`; wrong-architecture requests rejected. |
| CDNA4 scalar control builders | 🩶 Host-proven | `38cf90d7c2`: 13 exact LLVM gfx950 words and decoder round trips cover EXEC/VCC/SCC preservation, compares, and conditional branches in the separate CDNA4 backend. |
| CDNA4 vector builders | 🩶 Host-proven | `113cd36228`: exact LLVM bytes and decoder round trips cover arithmetic, comparisons, lane identity, and address updates. Literal add/multiply use materialization because gfx950 rejects VOP3 literal operands; in-place multiply exposes its extra-VGPR requirement. |
| CDNA4 SMEM/FLAT builders | 🩶 Host-proven | `4c51920eb2`: exact LLVM bytes and decoder round trips cover dispatch loads and B32 publication load/store, including CDNA4 pair alignment and immediate boundaries. |
| CDNA4 DS/atomic builders | 🩶 Host-proven | `73241ff07f`: exact LLVM bytes and decoder round trips cover LDS store/exchange and B32/B64 FLAT atomics; tuple alignment, return forms, and device scope fail closed. |
| CDNA4 synchronization/cache builders | 🩶 Host-proven | `f34be1ae72`: exact LLVM bytes and decoder round trips cover VM/LGKM drains, pre-drained workgroup barrier, trap, SALU dependency NOP, and scalar I/D-cache controls. |
| Architecture-neutral builder dispatch | 🩶 Host-proven | `b21df67176` and `a0b96f50ce`: scalar control, EXEC/VCC/SCC, vector arithmetic, literal recipes, address arithmetic, SMEM/FLAT/DS publication, atomics, private scratch, semantic waits, and workgroup barriers select separate CDNA4/RDNA4 backends. Five dispatch tests cover fixed and variable instruction lengths plus unsupported-architecture rejection. Engine-level gfx950 emission remains the separate active node `B5B`. |
| Native-shape CDNA4 analysis corpus | 🩶 Initial slice | `0cb03769ea`: a synthetic gfx950 ELF carries exact LLVM CDNA4 `ds_write_b32`/`ds_read_b32` encodings. ConSan decodes both eight-byte instructions, extracts the expected VGPR operands, classifies both as supported LDS sites, and marks the kernel as a preflight candidate. Wider DS, FLAT, atomic, barrier, and exclusion coverage remains in `A0-A4`. |
| CDNA4 FLAT atomic inventory/address plan | 🩶 Initial slice | `1806f7c90d` and `089fdf0585`: a native two-word gfx950 `flat_atomic_add` retains exact address/data/result VGPRs, offset, return mode, normalized B32 width, and device scope. Its flat guest-pair address plan is supported without materialization, and the exact compiler `buffer_wbl2`/wait/returning-atomic/wait/`buffer_inv` shape associates as one conservative acquire-release event. Wider operations, global forms, fences, and fault mutation remain open in `A4`. |
| CDNA4 group-FLAT access inventory/emission | 🩶 Native vertical | `6e997a6a39`, `31552da206`, `ef04607118`, and `bd-1w9.34`: CDNA4 raw FLAT decoding retains op, segment, address/data/result VGPRs, offset, and cache fields; explicit `SRC_SHARED_BASE` provenance admits native two-word D16/D32/D64/D128 group accesses without weakening strict provenance. The complete subword load and store families are normalized by exact memory width and destination/source half: unsigned/signed byte loads, byte stores, and B16 forms, including their high-half variants, share one target-neutral semantic contract across all five supported targets. Target-native runtime fixtures cover high-half B16 and signed-byte loads plus low-byte, high-byte, and high-half B16 stores on physical gfx1201 and simulated gfx1250, gfx942, and gfx950. The retained `d128-block` Record/Replay run at `consan-validation/gfx950-take2-d128-block-record-007` executes successfully with all 12/12 admitted accesses and all 4/4 shared-function barriers patched and dynamically complete. The newly exercised many-candidate path uses target-aware CDNA4 `s_getpc_b64`/`s_setpc_b64` indirect islands. Broader FLAT forms and campaign-level fault/performance evidence remain open, so `A2` is active and no workload cell is promoted. |
| CDNA4 far shared-function ownership | 🩶 Native clean vertical | `b58799fda0` retains execution-owner descriptors directly on ordinary FLAT sites, allowing SuperCollider to construct far local indirect relays independently of MOI resource plans. A CDNA4 host regression exercises a recovered local call and liveness-proven dead-SGPR fallback; all 591 ConSan host tests pass. Retained artifact `consan-validation/gfx950-take2-d128-pressure-clean-005` accepts all four profiles with 12/12 accesses under each engine, 4/4 barriers under Record/Replay and Inline Shadow, passing workload oracles, and zero dynamic-incomplete encounters. `V5` is active pending the remaining application matrix, faults, overhead, and frozen-tip rerun. |
| CDNA4 SuperCollider native LDS checks | 🩶 Native-proven | `afbca95de0` and `3e7c5bdf02` port native LDS check/trap emission without routing gfx950 through RDNA4 encodings. Exact host coverage includes B32/B64/B128 reads and writes, transpose B16, read2/read2st64, non-contiguous write2st64 operands, and even-aligned automatic-report address tuples. All 601 ConSan host tests pass. Retained CLIP, TP1 prefill, TP1 decode/combined, and TP2 artifacts accept 45/45, 120/120, 240/240, and 936/936 accesses respectively with passing workload oracles, analysis/static/dynamic completeness, and zero mismatches. `SC0` is green/DONE; campaign cells remain orange pending faults, overhead, and frozen-tip evidence. |
| CDNA4 SuperCollider group-FLAT checks | 🩶 Native clean vertical | `c7ca36d375`, `ef04607118`, `bd-1w9.34`, `bd-1w9.35`, and `bd-1w9.36`: SuperCollider handles gfx950's two-word group-FLAT D16/D32/D64/D128 loads and stores. Subword readback comparisons use the exact memory width and select the architecturally consumed source or produced destination half before comparison. The lowering emits the target's complete group-FLAT wait, preserves the full guest VCC value around every redundant check on every supported target, preserves original load destinations and store sources, and even-aligns the CDNA report address tuple. The ordinary VCC-save pair is selected from site liveness and grows every owning descriptor to the selected register; no preserved-SGPR range is hard-coded. If a gfx1250 site has no dead pair, the same resource plan dynamically borrows an even pair already allocated by every owner, spills both live words into fixed lanes of one additional scratch VGPR, and restores them after VCC independently of EXEC. Wave32/wave64, ordinary/high-half load, zero-EXEC runtime, and heterogeneous shared-owner descriptor tests cover that fallback. Target-native fixtures confirm real `inline-flat-{load,store}-check-trap` patches and exact conditional-control output on physical gfx1201 plus simulated gfx1250, gfx942, and gfx950. A deterministic oracle now applies every production high-half byte/B16 group-FLAT load and store patch on every supported target, executes the exact emitted half selection, byte mask where applicable, U16 comparison, conditional branch, and marker action in the corresponding target simulator, and proves that ignored bits do not report while a difference in the consumed subword does. Retained artifact `consan-validation/gfx950-take2-d128-block-supercollider-006` is analysis-, static-, and dynamic-complete with 12/12 supported accesses patched and zero report mismatches. The coverage denominator excludes global, private, and provenance-unknown FLAT operations outside this shared-memory detector's semantics. Broader racy/fault campaigns and trap breadth remain open in `SC1`. |
| CDNA4 singleton-barrier fault mutation | 🩶 First contained campaign | `8c8480e68f` makes the reviewed selector target-aware: gfx950's single full `s_barrier` is selected by exact site identity, while gfx1201 retains exact site-plus-sequence selection for its associated signal/wait pair. Retained artifact `consan-validation/gfx950-take2-d128-block-fault-003` is accepted for all four profiles with exact-one `requested=1`, `planned=1`, `applied=1` mutation accounting, the precommitted `not_detected` qualified miss and workload-oracle failure, bounded termination, and healthy before/after probes. This activates `A5` and the D128 slice of `V6`; other mutation families and workload campaigns remain open. |
| D128 paired overhead | 🩶 First accepted timing row | `ecd529c017` selects the gfx950 `SampledFastContext` specialization whose native execution produces all admitted group-FLAT evidence; the superseded Exact-only attempt produced no Record/Replay records and was rejected. In retained artifact `consan-validation/gfx950-take2-d128-block-overhead-002`, baseline-before, all four three-process profile rows, and baseline-after are accepted. The paired baseline is 130.5 ms; process slowdowns are 6.61x SuperCollider, 8.35x Record/Replay, 8.31x Sampled, and 9.29x Inline Shadow. `V7` remains active pending peak-memory evidence and the broader matrix. |
| CDNA4 Record/Replay access emission | 🩶 Native vertical | `4fdaee5298`, `44e7767b9b`, and `3a52aaee88`: a gfx950 DS-store candidate emits a `ModifiedValid` first-light access-record patch and passes final re-decode. The body retains the guest DS access and contains CDNA4 FLAT publication load/atomics, VM/LGKM completion waits, and two-word lane-rank recipes. A production-hook native dispatch now writes and validates the access record; replay breadth remains open in `RR0`. |
| CDNA4 Sampled access emission | 🩶 Native vertical | `ec84cd7af7` and `2f9cc947b0`: a gfx950 DS-store candidate emits a `ModifiedValid` Sampled patch and passes final re-decode. The body contains CDNA4 FLAT swap-x2 publication, compare-swap claim, counter atomics, VM/LGKM completion waits, and the displaced guest DS access. A production-hook forced-spill dispatch preserves all live values and publishes a valid write entry plus ready causal window. Runtime selection and immediate/host-scan agreement remain open in `SA0-SA1`. |
| CDNA4 Sampled barrier lowering | 🩶 Native-proven vertical | The report-buffer target gate now admits both supported native architectures, with a CDNA4 host fixture proving a selected LDS causal window followed by singleton `s_barrier` metadata publication and final validation. Retained native artifacts `gfx950-take2-tp1-prefill-sampled-clean-004`, `gfx950-take2-tp1-decode-sampled-clean-002`, and `gfx950-take2-clip-bf16-sampled-clean-002` accept 104/104 + 7/7, 208/208 + 14/14, and 39/39 + 20/20 access/qualified-barrier coverage respectively, with passing workload oracles and zero dynamic-incomplete state. TP2 admission remains open in `SA0`. |
| CDNA4 sampled scalar-state boundary | 🩶 Native-proven | `2da1350b8e`: gfx950 dump inspection proved that encoded `s102:s103` is architectural `flat_scratch`, not ordinary scalar storage. Automatic owner, dispatch, and temporary-state placement now stops below CDNA4's special aliases; explicit overlap fails closed. Removing the obsolete runtime-sample counter and unused sampled-atomic tail reservation fits the high-pressure Stream-K helper in `s92:s101`. All 594 ConSan host tests pass, and retained artifact `consan-validation/gfx950-take2-streamk-arrival-clean-005` accepts native Sampled execution with 4/4 accesses and 10/10 atomics instead of the prior `0x1000` GPU fault. |
| CDNA4 Sampled forced spill | 🩶 Native-proven | `2f9cc947b0`: the production HSA hook forces a three-VGPR Sampled spill around native `ds_write_b32`, grows dispatch-private storage, and completes native AQL dispatch. Eight live values survive for all 64 lanes; the sampled entry and causal window agree on kind, epoch, generation, range, and 1D workgroup identity. |
| CDNA4 InlineShadow access/barrier vertical | 🩶 Native-proven TP1 lifecycle | A gfx950 DS-store candidate emits `ModifiedValid` and passes final re-decode plus exact-shadow semantic validation; workgroup-local forced-spill publication and global versioned transactions both have host byte proof. The global table hashes dispatch plus stable workgroup identity across 32 banks, and metadata-distinct lanes sharing one address are serialized by the pending-mask loop. Sequential decode+combined stress localized the residual loss to stable-version snapshots of stale nonempty payload. CDNA4 retries now issue `buffer_inv sc1`, consume a coherent opening version, and leave the ordinary fast path unchanged. Artifacts `gfx950-take2-tp1-decode-combined-inline-invalidate-retry-026` through `-031` provide six consecutive accepted runs at 208/208 accesses, 62/62 barriers, passing oracles, and zero dynamic-incomplete state. Prefill artifact `gfx950-take2-tp1-prefill-inline-serialized-lanes-037` remains accepted at 104/104 + 31/31, and `gfx950-take2-d128-block-inline-004` at 12/12 + 4/4. Atomic semantics, faults, and broader campaign evidence remain open in `IS0-IS1` and validation nodes. |
| Mixed CDNA4 AccVGPR/VCC persistent identity | 🩶 Native-proven | Moving compiler `ACCUM_OFFSET` remains prohibited because it changes existing MFMA operand meaning. Per-owner/component tuples avoid AccVGPR banks; private-state entry relays preserve kernarg preloads; persistent epoch/owner/workgroup-key slots remain disjoint from ephemeral spill storage; and appended probes snapshot displaced DS addresses. Automatic scalar state also excludes the original physical VCC pair at the top of each CDNA4 kernel's allocated user-SGPR bank. Full TP1 artifact `gfx950-take2-tp1-prefill-inline-vcc-safe-029` moves the attention kernel's Inline state from overlapping s70:s71 to s72 and changes a 533,757-incomplete oracle failure into a passing oracle with 27 broader undercoverage publications. A direct native run filtered to `prefill_bs1$async_dispatch_12_attention_4x2xDx32x32xD` accepts 4/4 accesses and 6/6 barriers with `dynamic_complete=true`, zero incomplete publications, and a passing oracle. Together with retained CLIP artifact `gfx950-take2-clip-bf16-inline-clean-021` at 39/39 + 24/24 and zero incomplete state, this closes `IS0A` and `I4`; the formerly separate full-workload publication residue is closed by the TP1 access/barrier row. |
| CDNA4 InlineShadow forced spill | 🩶 Native-proven | `54673f205e`: a full-VGPR-file fixture forces the 16-register Inline Shadow window into private spill storage. The local exact-shadow path uses even CDNA4 `ds_wrxchg_rtn_b64` and diagnostic CAS tuples, passes final semantic validation, and completes native AQL dispatch with all live values intact and zero diagnostic, overflow, unsupported, or malformed counts. |
| CDNA4 InlineShadow atomic ordering | 🩶 Native Stream-K vertical, AcqRel token import open | `089fdf0585` established compiler-shaped gfx950 acquire-release emission and structural validation. Checkpoint `255749430d` extends the CDNA4 reservation to 28 VGPRs so token producer state, scalar persistent owner/epoch materialization, and the even final address pair are disjoint; all 622 ConSan host tests pass there. Artifact `gfx950-take2-streamk-scalar-state-011` executes 4/4 accesses, 4/4 barriers, and 10/10 atomics but exposes the empty-release-slot race. Artifacts `gfx950-take2-streamk-claimed-acqrel-013` through `-016` localize lost VGPR and SGPR journals. In `-017`, scratch+8 seeding plus retention of the outer claim mask in the dead retry pair lets wave 1 stage and commit version 2; wave 2 then leaves a fully staged version-3 reservation. The remaining rejection is now the nonempty-predecessor token transaction: one malformed token, one overflow, no visible acquired token, and the old false diagnostic. Successor commit, repeated clean acceptance, fault detection, and full host semantic regression remain open in `IS1`. |
| Transactional static VGPR spill plan | 🩶 Host-proven | `319950aacd`: stable four-byte slots, pre-save drain, 16-byte required-private alignment, 4 KiB bound, failure rollback, and unchanged RDNA4 behavior. |
| CDNA4 MOI resource alignment/growth | 🩶 Host-proven | `44e7767b9b`: gfx950 descriptors decode and grow in eight-VGPR granules throughout planning and emission. Automatic first-light planning grows 8 to 16 VGPRs, all CDNA4 scratch windows use an even FLAT-address base, and odd explicit bases fail closed as `ExplicitMisaligned`. |
| CDNA4 Record/Replay forced spill | 🩶 Native-proven | `44e7767b9b` and `3a52aaee88`: an integrated first-light patch spills three VGPRs around the emitted probe using exact native CDNA4 scratch save/restore words and VM_CNT waits, preserves the displaced guest DS instruction, and passes final validation. The production hook adds the CDNA4 owner/epoch entry prologue, grows the live kernel object's private segment from 0 to 16 bytes, and completes native AQL dispatch with all live values intact and a valid LDS-write record. |
| Shared/private layout policy | 🩶 Native-proven | `97f978941d` and `c4b0c3a1a4`: architecture-typed 8 MiB/4 KiB limits and gfx950 16-byte final normalization feed one shared-helper layout. A native noinline LDS helper has two full-VGPR-file owners; the production hook selects one three-VGPR spill window, grows both live kernel objects from 0 to 16 private bytes, and both AQL dispatches preserve output while publishing a valid write record. |
| CDNA4 Record/Replay barrier forced spill | 🩶 Native-proven | `33075e7aac`: CDNA4 barrier-record lowering uses the target-neutral builder dispatch and passes final validation. A full-VGPR-file native kernel emits separate three-VGPR access and six-VGPR barrier spill trampolines, grows live dispatch-private storage from 0 to 32 bytes, preserves eight values for all 64 lanes, and publishes both access and barrier records. |
| CDNA4 Record/Replay atomic forced spill | 🩶 Native-proven | `6f42088f3d`: qualified CDNA4 cache-release/atomic/cache-acquire ordering no longer requires an RDNA-only `TH` field. A native full-VGPR-file kernel emits and final-validates a three-VGPR atomic-record spill, grows live private storage from 0 to 16 bytes, preserves eight values for all 64 lanes, executes exactly 64 increments, and publishes the expected address, scope, and raw-zero CDNA semantics. Companion-fence parity is recorded separately below. |
| CDNA4 Record/Replay fence spill parity | 🩶 Typed fail-closed | `af03dfa4c4`: CDNA4 reaches the same deliberate second-text-growth limitation as RDNA4. Both qualified companion fences retain spill plans but emit no guessed patch; each final ledger entry is `placement_or_lowering_failed` / `instrumentation_patch_missing`, accompanied by the explicit second-growth spill warning. |
| CDNA4 dynamic-stack spill policy | 🩶 Native-proven / typed fail-closed | `8068d7d018` and `4dcba398f9`: Inline Shadow uses a site-local, SCC-preserving frame around the compiler `s32:s33` stack convention, with exact explicit-`saddr` CDNA4 scratch words and VM_CNT waits. Host integration final-validates a 16-VGPR spill and 64-byte private requirement. A production-hook gfx950 dynamic-allocation kernel forces the same recipe and preserves all 64 private-stack values without diagnostics. Record/Replay deliberately remains unpatched with typed `DynamicStack` / `ResourceFailed` / `UnsupportedResourcePlan` evidence. |
| Instrumented group-segment dispatch growth | 🩶 Native-proven | The production HSA hook now propagates each instrumented kernel's required group bytes through cached symbol metadata and copied AQL dispatch packets, independently of the existing private-segment transaction. Retained TP1 artifact `consan-validation/gfx950-take2-tp1-prefill-inline-clean-027` proves the affected attention dispatch grows from 768 to 2304 group bytes. Its unchanged oracle failure and 533,757 dynamic-incomplete publications ruled out underallocated LDS as the identity root cause; the subsequent physical-VCC repair is recorded in the mixed-identity row. |
| Focused host gate | 🩶 730/730 focused | The complete current `ConSan*.*` host suite passes in the canonical `/home/ossci/xx/rocjitsu-build`, including the CDNA4 old/grown physical-VCC allocation regression. Exact encoding/decoder proof pins `buffer_inv sc1`; earlier private epoch/owner/workgroup-key, entry-save, displaced-address snapshot, and cache-control regressions remain covered. |
| Live CDNA4 scratch round trip | 🩶 Native-proven | `0ad99bd3ac`: full-wave64 and divergent even-lane native MI355X round trips pass. Code-object inspection retains the exact scratch store/load pairs, `0xbf8c0f70` waits, partial-`EXEC` save/restore, wave64 metadata, and 32 private bytes. |

## Frozen campaign provenance

The final whole-matrix frozen campaign is not established.  Individual green
cells above retain same-revision bundles, but the final campaign must name one
committed executable revision, freshly rebuilt hook SHA-256, target-aware
manifest and audit hashes, workload source revisions, exact artifact roots,
and the controlled runtime/toolchain environment, then rerun every accepted
row at that one tip.

## Promotion requirements

A workload/profile cell remains non-green until its retained evidence proves:

1. the standard-v1 profile is used, with no forbidden coverage-limiting or
   manually selected register controls;
2. the independent numerical or semantic oracle passes on the clean workload;
3. ConSan sees an applicable gfx950 code object, records the complete typed
   exclusion ledger, and patches every admitted supported access, barrier,
   atomic, and fence site;
4. the selected engine executes dynamically without incomplete state,
   forbidden overflow, or unexpected clean diagnostics;
5. each admitted fault has a freshly inventoried and reviewed gfx950 selector,
   exact-one final-byte mutation proof, and a precommitted detector/oracle
   outcome;
6. every fault terminates within its bound and retains before/after device
   health evidence; and
7. one paired baseline/profile repetition, peak memory, commands, hashes,
   timeout state, and retained artifact paths are recorded at a frozen
   campaign revision.

Clean execution alone is compatibility evidence, not a green cell.  A timeout,
trap, crash, output mismatch, or GPU reset is an execution outcome rather than
a ConSan detection.

## Progress log

- 2026-07-31: Requalified the selected large Qwen object at `0bf1d17c7d`
  after the shared planning and routing fixes.  All four standard profiles pass
  the exact oracle with complete static and dynamic coverage: SuperCollider
  628/628 accesses, Record/Replay 628/628 accesses plus 52/52 barriers,
  Sampled 628/628 accesses plus 51/51 barrier members, and Inline Shadow
  628/628 accesses plus 52/52 barriers.  The retained clean artifacts are
  `consan-validation-large-objects-gfx950-qwen-sim-sc-selfpin-20260731`,
  `consan-validation-large-objects-gfx950-qwen-sim-rr-gated-v2-20260731`,
  `consan-validation-large-objects-gfx950-qwen-sim-sampled-standard256-20260731`,
  and
  `consan-validation-large-objects-gfx950-qwen-sim-inline-0bf1d17-2400s-20260731`.
  The single-shot paired campaign
  `consan-validation-large-objects-gfx950-qwen-sim-overhead-all-final-v2-0bf1d17-20260731`
  passes baseline-before and baseline-after and records 1.34x, 1.25x, 1.70x,
  and 21.75x slowdowns respectively. Its Sampled row loaded hook SHA-256
  `2113c773bdcac837e58cbd0cc0e784d7d99d971de3a8e08c377743cd245f8e6e`,
  which contains the guest-SCC return fix and repeats the exact 628/628 plus
  51/51 gate. The standalone Sampled clean artifact above used the earlier
  hook SHA-256
  `bfd03857e262d7b02da55a1a72ea0207737b25cb3a40acd6cf90ea547ff1dfdf`;
  artifact names containing `0bf1d17` identify the checked-out source ref, not
  the loaded hook binary. The cells remain yellow until their current
  reviewed-fault and containment bundles are refreshed.

- 2026-07-25: Completed the current-producer `bd-1w9.42` qualification for
  Record/Replay, Sampled, and Inline Shadow.  One shared target-native
  inventory supplies the exact occurrence-zero barrier selector; two
  independent trials per standard strict profile record a 1/1/1 mutation,
  complete 82/82 access plus 10/10 surviving-barrier coverage, a passing exact
  oracle, zero diagnostics, and complete cleanup.  Clean health runs directly
  before and after the second trial restore 11/11 barriers.  These are
  prospectively frozen qualified misses, not detector hits; the separate
  positive Tensile control is tracked by `bd-1w9.43`.  A
  baseline-before/all-profiles/baseline-after campaign replaces the earlier
  single-baseline quotients with true paired ratios of 1.13x, 1.35x, and
  5.98x.  Every individual row stays below one minute.

- 2026-07-31: Added the registered `tensile-gfx950-lds-positive` detector-hit
  control for `bd-1w9.43`. A production-derived LDS address mutation makes the
  exact numerical oracle fail in all four profiles; SuperCollider reports the
  required diagnostic. The control records exact-one installation and clean
  pre/post containment, while the reusable implementation validates target,
  instruction form, execution-owner allocation, and final relocated bytes.

- 2026-07-25: Completed `bd-1w9.9.8` at rocm-libraries `0a323b7493`.
  The general grouped-user-argument loader now retires real non-preloaded
  prefix loads before any emitted fixed-slot reload; it does not hard-code the
  observed SGPR range, target, PC, or test.  The regenerated gfx950 library
  emits the required `lgkmcnt(0)` boundary and all three final code objects are
  waitcheck-clean.  The focused regression, complete unit gate, and exact
  numerical oracle pass.  Fresh standard-profile runs at the same producer
  revision also pass all four clean profiles with complete 82/82 access plus
  11/11 barrier coverage for every MOI engine.  SuperCollider's exact-one
  selector and containment bundle were re-frozen after the machine-code
  change.

- 2026-07-25: Closed `bd-1w9.9.13` after a source audit and debug-loaded rerun
  disproved the suspected fail-open numeric runner.  TensileLite intentionally
  probes the unqualified helper filename, clears file-not-found, and then loads
  the matching `xnack-` variant.  The trace records that successful load, the
  Stream-K invocation, and `validation=PASSED`; non-finite simulator timing is
  not the output-comparison oracle.  No runner exception or filename workaround
  was added.

- 2026-07-25: Completed `bd-1w9.9.12` at rocm-libraries `88c30745d1`.
  `LocalWritePerMfma` now reuses the explicit per-parameter normalization table
  while preserving its registry-derived integer `-1` sentinel.  The focused
  suite covers integer and float sentinel spellings, valid integer shorthands,
  and strict invalid inputs; the complete unit gate and all 4,598 observed
  upstream YAML values pass.  The retained gfx950 runner and debug trace prove
  the warning-free assignment path, target-native loads, kernel invocation,
  and exact numerical oracle.

- 2026-07-25: Completed `bd-1w9.9.11` at rocm-libraries `ccf6befac4`.
  The parameter-aware production boundary now converts every in-range plain
  integer `GlobalReadPerMfma` shorthand to the float MessagePack wire type,
  while incompatible values retain strict diagnostics.  MessagePack
  round-trip tests, the full unit gate, and a 4,522-value upstream YAML sweep
  all pass.  The bounded gfx950 numerical/ELF row also passes without type
  warnings.  The adjacent `LocalWritePerMfma` sentinel contract was
  subsequently completed under `bd-1w9.9.12`.

- 2026-07-25: Completed the `bd-1w9.9.5` gfx950 Tensile msgpack producer fix
  at rocm-libraries `a4d0529339`.  The canonical scheduling default is now a
  float, every derived `DirectToLdsMetadata` write uses the integer wire
  representation, and hermetic whole-default plus production-derivation
  regressions preserve the strict registry contract.  The full upstream unit
  gate passes in one session, and a fresh bounded gfx950 Stream-K run preserves
  the exact numerical oracle and target checks while eliminating both
  field-type warnings.  Broader explicit integer YAML overrides were isolated
  under `bd-1w9.9.11` rather than hidden by generic coercion.

- 2026-07-25: Closed the `bd-1w9.9.9` gfx950 Tensile SuperCollider
  fault-bundle gap with a reviewed exact target-native selector rather than a
  broad barrier index.  A fresh inventory selects occurrence zero of the full
  `s_barrier` form at PC `0x10a0`, the first of 11 sites sharing the exact
  code-object identity.  Two bounded trials each prove one requested, planned,
  and applied mutation, retain all 82/82 access patches, and pass the exact
  numerical oracle with zero SuperCollider mismatches; the outcome is therefore
  a qualified miss.  Both trials terminate in under six seconds with complete
  cleanup, and the same clean profile passes immediately afterward as a health
  check.  The checked-in ledger pins the rocJITsu, corpus, rocm-libraries, and
  hook identities; retained logs pin commands, runtime settings, and target
  checks while distinguishing host-dominated end-to-end time from kernel
  overhead.  The row stays yellow solely for the separately tracked
  generated-code wait hazard.

- 2026-07-25: Closed the `bd-1w9.35` deterministic high-half mismatch-oracle
  gap without adding a hook-only fault switch or changing production
  lowering.  For each of gfx1201, gfx1250, gfx942, and gfx950, the host test
  applies all five real high-half byte/B16 group-FLAT load and store patches
  and extracts their exact emitted half selection, byte mask where applicable,
  U16 comparison, conditional branch, and report-marker action.  The matching
  target simulator executes those production words from controlled original
  and duplicate result registers.  Differences confined to ignored bits leave
  the report marker zero; changing the consumed subword writes the configured
  marker.  The action boundary and execution budget are derived from the
  emitted branch and VCC restore rather than duplicated per-target word
  counts.  The full matrix runs in milliseconds and complements, rather than
  replaces, the existing target-native clean load/store fixtures.

- 2026-07-25: Closed the `bd-1w9.36` gfx1250 group-FLAT scalar-pressure
  gap without reserving a fixed SGPR range.  When site liveness cannot supply
  the full VCC-save pair, the production planner chooses an even pair already
  allocated by every execution owner, excludes the configured runtime-delay
  source, saves both live scalar words in fixed lanes of one additional
  scratch VGPR, and restores them after the check without depending on EXEC.
  Shared-function planning uses the owners' common minimum allocation and
  leaves their descriptors unchanged.  Exact host coverage drives all
  ordinary SGPRs live in both gfx1250 wave modes, covers ordinary and
  high-half loads, and separately gives two shared owners heterogeneous SGPR
  allocations.  A production-hook gfx1250 simulator case enters the patched
  FLAT store with EXEC zero and verifies the borrowed live scalar pair remains
  exact.  Other targets retain their liveness-proven, fail-closed allocation
  path.

- 2026-07-25: Closed the `bd-1w9.34` group-FLAT subword-store gap across
  gfx1201, gfx1250, gfx942, and gfx950.  One target-neutral classifier now
  carries exact byte/B16 width and low/high source placement into all three
  MOI engines and SuperCollider.  SuperCollider performs exact-width readback
  and preserves guest VCC around the original store, runtime group gate,
  comparison, and mismatch action.  Its save location comes from site
  liveness, uses an ordinary pair on every target so wave32 and wave64 retain
  the full architectural value, and grows every owning descriptor to the
  selected location.  Conditional-control runtime fixtures pass on physical
  gfx1201 and all three simulators; they consume a VCC predicate after the
  instrumented store and specifically guard against the clobber that an
  unconditional fixture would miss.  Low-byte execution now joins the
  high-byte and high-half B16 store cases.  The clean exact-output result is
  not mismatch-detection evidence; `bd-1w9.35` retains that separate oracle.

- 2026-07-24: Reassessed both bounded HIP Stream-K rows after the workspace
  TheRock SDK gained rocThrust.  The two exact gfx950 simulator baselines pass
  under a 120-second per-case bound.  SuperCollider preserves both oracles but
  reports zero applicable code objects and zero supported LDS sites;
  Record/Replay, Sampled, and Inline Shadow fail closed with exit 92 on
  function-resource or barrier-placement gaps.  The five retained roots are
  `consan-gfx950-streamk-{baseline,supercollider,record-replay,sampled,inline-shadow}-provenance-20260724`
  under `rocjitsu-test-corpus/.pytest-artifacts`; `bd-1w9.9.10` tracks the
  provenance-unknown FLAT and function synchronization work.

- 2026-07-26: hip-moi `29a1c212183b` makes all 12 instrumented CDNA sources
  common to gfx942 and gfx950. The target-specific offline gate now registers
  13 binaries per family and passes 33/33 tests through each RocJITsu
  simulator. The seven additional functional suites remain simulator
  prerequisites rather than being mislabeled as campaign workload rows.

- 2026-07-24: Advanced the gfx950 corpus evidence baseline to local commit
  `0db836e7bd8c6400b7ffd187d749225899875d7c`, regenerated the bounded Tensile
  baseline and four standard-profile artifacts with retained 120-second
  invocation provenance, and qualified the result honestly: baseline and
  SuperCollider pass, while Record/Replay, Sampled, and Inline Shadow reject
  before the oracle on their tracked placement gaps.  The corpus commit is not
  present on a fetched remote ref and must be published or transferred for a
  clean remote reproduction.

- 2026-07-23: Added the target-native gfx950 Jakub reference executable and
  resolved all six hip-moi roles through the explicit
  `hip-moi-build-gfx950-tests` tree at hip-moi commit
  `288b3c17a7bfd9e28966a754f453fa69cb9616c1`.  The shared CDNA simulator gate
  passes all six binaries and 14/14 tests, including both parameterized Jakub
  cases.  This is simulator baseline evidence only; the companion commit is
  not yet present on a fetched remote ref, and the Jakub profile cells remain
  unassessed on physical gfx950 and under ConSan.

- 2026-07-22: Commits `7fbd3b708d`, `cb82107577`, and `cd8230c019`
  move the hip-moi Record/Replay frontier past its typed dynamic-stack limit.
  Unassociated cache operations are classified as not applicable; transient
  allocation excludes the compiler SGPR that saves the caller's dynamic frame
  base; and access, atomic, and fence records use the persistent entry-captured
  owner and epoch.  Tree artifact
  `consan-validation/gfx950-rr-tree-persistent-owner-tip-20260722` is accepted
  with both workload tests passing, zero clean diagnostics, and complete 4/4
  access, 4/4 barrier, 10/10 atomic, and 16/16 fence coverage.  Stream-K
  artifact `consan-validation/gfx950-rr-streamk-persistent-owner-tip-20260722`
  reaches the same complete coverage and passes its workload oracle, but is
  not accepted because replay emits a false race.  Focused tracing localizes
  that result to inconsistent workgroup identity: ordinary access records use
  `(0,0,0)`, while synchronization emitted inside the shared helper rereads
  compiler-reusable entry SGPRs and records `(131072,0,4294967295)`.  The next
  narrow fix is to use the existing entry-captured persistent workgroup key
  consistently for Record/Replay records; no diagnostic-only logging remains
  in the source.

- 2026-07-22: Record/Replay HIP-matmul now executes cleanly without an expert
  patch bound.  The first compact scalar-epoch barrier had reused its reserved
  SCC snapshot as an epoch-overflow temporary and also touched one unreserved
  SGPR.  The barrier is followed by a guest SCC-dependent branch, so the bad
  restore selected the wrong control path and faulted the GPU.  The replacement
  compares the epoch against a literal maximum, skips the increment when
  saturated, and restores SCC without any temporary SGPR.  The focused host
  regression, all 400 `ConSanMoi.*` tests, the one-access/one-barrier physical
  reproducer, and the unrestricted physical run pass.  The unrestricted run
  patches 709/739 accesses and 109/109 barriers; all 30 resource failures are
  in the packaged, unexecuted dynamic-stack HybridStreamKTree kernel.  The
  three selected numerical oracles pass, promoting Record/Replay from red to
  yellow while static completeness remains open.

- 2026-07-22: SuperCollider now lowers all 739/739 HIP-matmul LDS accesses,
  including the 20 B128 reads whose destinations are AccVGPR tuples, without
  changing `ACCUM_OFFSET`.  Focused host validation and the full 740-test
  ConSan suite pass.  Physical artifact
  `consan-gfx950-hip-matmul-supercollider-accvgpr-20260722-263` nevertheless
  changes the first numerical result from 44 to 42.  Patch-prefix bisection
  proves that 728 patches pass and 729 fail; artifact
  `consan-gfx950-hip-matmul-supercollider-site729-20260722-264` retains that
  exact boundary.  Patch 729 is the sixth ordinary `ds_read_b128` in its
  double-buffered owner, not an accumulator read.  Its original 206-VGPR
  metadata allocates 208 VGPRs; passing prefix 728 already uses v206:v212 and
  already grows the descriptor to 216.  The original and duplicate load words
  plus branch arithmetic are exact, localizing the next fix to the final-load
  interaction rather than descriptor growth.  Static completeness therefore
  advances, but the cell stays orange.

- 2026-07-22: The earlier private-segment hypothesis is disproved.  A native
  forced-spill test passes both 16-byte and 32-byte zero-to-nonzero private
  growth, and HIP-matmul passes with its first access probe both with and
  without persistent-state initialization when barrier tracking is disabled.
  Enabling the first compact barrier alone reproduced the fault and led to the
  SCC-preservation fix above.

- 2026-07-22: The owner-qualified CDNA4 scalar-planning change moves the P0
  HIP-matmul Record/Replay frontier beyond its earlier heterogeneous-object
  rejection.  Artifact
  `consan-gfx950-hip-matmul-record-replay-ownerqualified-20260722-261`
  emits and installs a 2,291,168-byte patched object, selects all 739 accesses
  and 109 barriers, and patches 709 accesses plus every barrier.  The first
  instrumented dispatch then causes a physical-GPU memory fault, so the cell
  remains red and the next bounded problem is emitted-code/runtime
  correctness.  This is progress in implementation reach, not workload
  acceptance.

- 2026-07-22: The first P0 HIP-matmul Inline Shadow assessment advances from
  gray to red.  Artifact `consan-gfx950-hip-matmul-inline-clean-20260722-258`
  analyzes all 46 kernels, then rejects before execution at the identical
  code-object-wide dispatch-ID and EXEC-save SGPR placement boundary as
  Record/Replay and Sampled.  The owner-qualified CDNA4 planner task therefore
  covers all three MOI engines; no numerical failure is inferred.

- 2026-07-22: The first P0 HIP-matmul Sampled assessment advances from gray to
  red with precise evidence rather than a timeout.  Artifact
  `consan-gfx950-hip-matmul-sampled-clean-20260722-257` analyzes all 46 kernels
  but rejects the heterogeneous 1.84 MiB object before execution because a
  single code-object-wide persistent dispatch-ID pair and EXEC-save window
  cannot satisfy every owner.  This is the same early planning boundary as
  Record/Replay; both profiles now share one owner-qualified fallback task.
  The P0 HipKittens source is present, but its local build is explicitly gray
  because a CPU-reference build dependency is absent; no instrumentation
  failure is inferred.

- 2026-07-22: Qwen SuperCollider advances from yellow to green.  Artifact
  `consan-validation-gfx950-qwen-sc-final-output-fault-20260722-254` applies
  the reviewed final-output barrier mutation exactly once.  Its single trial
  preserves the exact expected-output oracle without a SuperCollider
  diagnosis, as prospectively required for a barrier after the final store,
  and passes the marker-contained before/after health checks.  The accepted
  clean and 3.33x one-repetition paired evidence complete the green bundle.

- 2026-07-22: Qwen Record/Replay advances from yellow to green.  The reviewed
  final-output matmul site at `main$async_dispatch_562_batch_matmul_1x5x151936x1024_f32`
  applies its exact third-occurrence barrier mutation once in artifact
  `consan-validation-gfx950-qwen-rr-final-barrier-fault-recursionfix-20260722-253`.
  The single trial satisfies its prospective non-detection policy, preserves
  the workload oracle, and passes the marker-contained before/after health
  checks.  This complements the already accepted clean and paired evidence.

- 2026-07-22: CLIP BF16 Inline Shadow advances from red to green at commit
  `c24431f77e`.  Its full ordinary VGPR bank selects entry-snapshotted private
  owner/epoch state.  The CDNA4 generation-tagged local-shadow path had omitted
  the matching private workgroup key and later reread compiler-reusable entry
  SGPRs, honestly accounting all 177,152 encounters as unsupported.  Private
  state now retains and reloads that key before dispatch qualification.  The
  forced-spill regression and all 701 ConSan host tests pass.  Committed-tip
  paired artifact `consan-validation-gfx950-clip-inline-private-key-committed-20260722-235`
  passes the cosine oracle with complete 45/45 access plus 24/24 barrier
  coverage and zero incomplete encounters; 0.601093 ms versus the 0.398951-ms
  mean control is 1.51x.  Reviewed fault artifact
  `consan-validation-gfx950-clip-inline-private-key-fault-20260722-236`
  accepts an exact-one final-barrier qualified miss with passing oracle,
  surviving evidence, cleanup, health, and clean provenance.

- 2026-07-22: CLIP BF16 Record/Replay advances from yellow to green at commit
  `8075a15390`.  The former 2/10 stress diagnostics paired access PCs from
  different kernel dispatches that shared report generation and workgroup
  coordinates; they were false cross-dispatch replay aliases, not workload
  races.  CDNA4 now records the hardware dispatch identity in the existing
  generation field, matching the established replay partition without an ABI
  change and without changing RDNA4.  The focused dispatch-isolation and VCC
  tests plus all 701 ConSan host tests pass.  Committed-tip one-repetition
  artifact `consan-validation-gfx950-clip-rr-dispatch-committed-20260722-230`
  accepts both cosine controls and Record/Replay with complete 45/45 access
  plus 24/24 barrier coverage and no diagnostics; 0.512484 ms versus the
  0.396996-ms mean control is 1.29x.  Reviewed fault artifact
  `consan-validation-gfx950-clip-rr-dispatch-fault-committed-20260722-231`
  accepts an exact-one final-barrier qualified miss with a passing oracle,
  complete surviving evidence, bounded cleanup, physical health before and
  after, and clean source provenance.

- 2026-07-22: D128-pressure Inline Shadow advances from yellow to green at
  committed tip `b4978f4250`.  Fresh inventory
  `consan-validation-gfx950-d128-pressure-inline-generation-inventory-20260722-098`
  freezes four singleton barriers.  The earlier first-barrier trial remains
  useful negative evidence: it produced 12 valid diagnostics but overflowed
  the deliberately small diagnostic buffer.  The separate, prospectively
  reviewed fourth-barrier artifact `...-fault-fourth-20260722-099` accepts
  exactly one mutation, its schedule-masked qualified miss and passing oracle,
  complete surviving 12/12 access plus 3/3 barrier coverage, bounded report
  memory with full cleanup, and healthy physical-device probes.  Current-tip
  clean artifact `...-clean-20260722-100` passes all four exact oracles with
  zero diagnostics and complete 12/12 access plus 4/4 barrier coverage.
  Paired artifact `...-paired-20260722-101` measures 4,013 ms against a
  178.5-ms mean control, or 22.48x.  The accepted qualified-miss campaign does
  not erase the causal first-barrier overflow result; it closes the workload
  bundle while leaving diagnostic-capacity stress visible for future work.

- 2026-07-22: D128-block and MFMA-attention Inline Shadow advance from yellow
  to green at committed tip `c86ecb77bc`.  MFMA clean artifact
  `consan-validation-gfx950-mfma-inline-generation-clean-20260722-088`
  accepts both exact oracles with zero diagnostics and complete 12/12 access
  plus 4/4 barrier coverage.  Paired artifact `...-paired-20260722-090`
  measures 1,813 ms against a 116.5-ms paired control (15.56x), and fresh
  inventory `...-inventory-20260722-091` freezes four singleton barriers.
  The old first-barrier policy correctly rejects artifact `-092` because
  Inline now detects the induced race rather than producing its frozen
  qualified miss.  Two independently reviewed intermediate barriers are
  schedule-masked and retain their rejected precommitted policies in `-093`
  and `-094`.  Prospectively reviewed fourth-barrier artifact `-095` accepts
  exactly one mutation, its qualified miss and passing oracle, complete
  surviving 12/12 access plus 3/3 barrier coverage, a 12,600,096-byte peak
  with full cleanup, bounded completion, and healthy physical-device probes.
- The corresponding D128-block inventory
  `consan-validation-gfx950-d128-block-inline-generation-inventory-20260722-096`
  freezes the same four structural barriers.  Prospectively reviewed artifact
  `...-fault-fourth-20260722-097` accepts exactly one fourth-barrier mutation,
  its schedule-masked qualified miss, complete surviving 12/12 access plus
  3/3 barrier coverage, bounded memory and cleanup, and healthy probes.  With
  clean artifact `-085` and paired artifact `-086`, the D128-block Inline cell
  is also green.

- 2026-07-22: Commit `4d654e573e` fixes the clean D128-block Inline false
  diagnostic.  Physical-gfx950 discriminators isolated the failure to reused
  local LDS shadow state rather than the exact-shadow algorithm or incomplete
  clearing.  CDNA4 local cells now carry a nonzero 20-bit generation derived
  from workgroup identity and both native dispatch-ID dwords, so an LDS image
  reused by a later dispatch cannot be mistaken for current metadata.  All
  734 ConSan host tests pass.  Clean artifact
  `consan-validation-gfx950-d128-block-inline-generation-clean-20260722-085`
  accepts both exact oracles with zero diagnostics, complete 12/12 access and
  4/4 barrier coverage, and clean committed provenance.  Paired artifact
  `consan-validation-gfx950-d128-block-inline-generation-overhead-20260722-086`
  accepts one process per leg and measures 1,986 ms against a 126-ms paired
  baseline (15.76x).  The cell remains yellow while its reviewed-fault,
  containment, health, and peak-memory bundle is open.
  The related MFMA-attention discriminator also passes both exact oracles with
  zero diagnostics and complete 12/12 access plus 4/4 barrier coverage in
  artifact `consan-validation-gfx950-mfma-inline-generation-clean-20260722-087`;
  because that run overlapped this documentation edit, a clean-provenance
  committed-tip repeat remains before it can serve as final evidence.

- 2026-07-22: D128-pressure Inline Shadow advances from gray to yellow at
  commit `a6714e31f0`.  A disconnected full-VGPR dynamic-stack component had
  selected code-object-wide scalar persistence while planning entry scratch
  only for its own owners.  The planner now transactionally proves scratch
  for every emitted owner before committing scalar mode.  Clean artifact
  `consan-validation-gfx950-d128-pressure-inline-multicomponent-clean-20260722-074`
  passes all four exact oracles in one repetition with complete 12/12 access
  plus 4/4 barrier coverage.  One-process paired artifact
  `consan-validation-gfx950-d128-pressure-inline-single-overhead-20260722-078`
  measures 4,028 ms against 178-ms and 160-ms controls (22.6x against the
  slower paired control).  Fresh inventory `...-inventory-20260722-076`
  freezes four singleton barriers.  Exact-one fault artifact
  `...-fault-20260722-077` removes the first barrier, fails the exact oracle,
  and produces 12 valid Inline diagnostics, but 18,036 further diagnostics
  overflow the statically sized 12-record buffer.  That is useful detection
  evidence, not a green bundle: bounded diagnostic-capacity planning remains
  the fault gate.

- 2026-07-22: The same scalar-owner fix advances D128-block and MFMA-attention
  Inline from planner rejection to yellow.  Current one-repetition artifacts
  `consan-validation-gfx950-d128-block-inline-multicomponent-clean-20260722-080`
  and `consan-validation-gfx950-mfma-inline-multicomponent-clean-20260722-079`
  pass both exact oracles with complete 12/12 access plus 4/4 barrier coverage
  and complete analysis, but each emits one clean access-conflict diagnostic;
  they therefore are not accepted clean runs.  Current Stream-K Sampled
  artifact `...-streamk-sampled-current-clean-20260722-082` replaces the stale
  historical claim with a red current result: its selected state requires a
  spill in a dynamic-stack owner, which remains deliberately unsupported.

- 2026-07-22: TP2 Inline Shadow is green and closes `IS0`.  Fresh inventory
  artifact `consan-validation-gfx950-tp2-inline-spill-inventory-20260722-071`
  retains the current target identities.  Prospectively reviewed artifact
  `consan-validation-gfx950-tp2-inline-spill-fault-20260722-072` applies
  exactly one final-byte mutation to the same prefill-matmul barrier whose
  independent earlier trial established a failing external oracle.  The
  current run matches its frozen fail-oracle/no-diagnosis contract, retains
  complete 936/936 access and 167/167 surviving-barrier coverage, reclaims all
  report memory, and passes physical-device health before and after.  Together
  with the clean and 167.0x maximum paired evidence, all three spill-recovered
  Sharktank Inline profiles now have complete accepted bundles.

- 2026-07-22: TP1 decode/combined Inline Shadow is green.  Fresh inventory
  artifact `consan-validation-gfx950-tp1-decode-inline-spill-inventory-20260722-069`
  freezes 31 singleton barrier sequences.  Prospectively reviewed artifact
  `consan-validation-gfx950-tp1-decode-inline-spill-fault-20260722-070`
  applies exactly one final-byte mutation to the independently characterized
  late decode-attention barrier, matches its frozen pass-oracle/no-diagnosis
  contract, retains complete 240/240 access and 61/61 surviving-barrier
  coverage across decode and prefill, reclaims all report memory, and passes
  physical-device health before and after.  Together with the current clean
  and 31.2x maximum paired evidence, this closes the profile bundle.

- 2026-07-22: TP1 prefill Inline Shadow is green at clean commit
  `5d196e32f4` with hook SHA-256 `2495cd05...2f065f`.  Fresh inventory artifact
  `consan-validation-gfx950-tp1-prefill-inventory-cdna4-sgpr-spill-20260722-067`
  freezes 31 singleton barrier sequences.  Prospectively reviewed artifact
  `consan-validation-gfx950-tp1-prefill-inline-cdna4-sgpr-spill-fault-20260722-068`
  applies exactly one final-byte barrier-to-NOP mutation in the attention
  kernel, matches its precommitted pass-oracle/no-diagnosis contract, retains
  complete 120/120 access and 30/30 surviving-barrier coverage, reclaims all
  4,747,424 report bytes, and passes target health before and after.  Together
  with the current clean and paired artifacts recorded below, this completes
  that profile's clean, overhead, fault, containment, and provenance bundle.

- 2026-07-22: Commit `85c831f0c5` converts the corrected CDNA4 VCC boundary
  from a fail-closed Inline regression into safe component-local scalar spill.
  The planner protects initialized/kernarg SGPRs and validates the complete
  spill/router tuple against both original and allocation-grown physical VCC;
  partial assignment is transactional and rebuilds fail closed.  Focused
  safety tests pass 4/4 and all 733 ConSan host tests pass.  Clean physical
  artifacts `...tp1-prefill...-063`, `...tp1-decode...-061`, and
  `...tp2...-062` pass every exact oracle with complete 120/120 + 31/31,
  240/240 + 62/62, and 936/936 + 168/168 access/barrier coverage.  Paired
  artifacts `...-066`, `...-064`, and `...-065` are also accepted at 243.9x,
  31.2x maximum, and 167.0x maximum slowdown.  All three Inline cells move
  red to yellow; `IS0` remains ACTIVE for reviewed-fault closure.

- 2026-07-22: Current-tip physical diagnostics at clean commit `935acc5e01`
  retire stale gray and pre-VCC-safety evidence.  TP1 decode/combined Sampled,
  TP2 Sampled, and MFMA-attention Sampled all reject persistent state at the
  ordinary-VGPR/AccVGPR boundary.  TP1 decode/combined and TP2 Inline find no
  legal 30-SGPR transient window after excluding original and grown physical
  VCC; the old successful windows were unsafe and are invalidated.  MFMA
  Inline lacks entry-local prologue scratch, and tree atomic-OR Inline rejects
  dynamic-stack scalar placement.  Every run used one repetition.  `IS0` is
  ACTIVE on VCC- and kernarg-safe spill-backed Inline scalar state; the harder
  Sampled AccVGPR frontier rotates in `SA0`.

- 2026-07-22: The CDNA4 physical-VCC fix also recovers TP1
  decode/combined Record/Replay, promoting that cell from red to yellow.
  Clean artifact
  `consan-validation-gfx950-tp1-decode-rr-vcc-fix-20260722-048` passes both
  exact oracles with complete 240/240 access plus 62/62 barrier coverage.
  One-repetition paired artifact
  `consan-validation-gfx950-tp1-decode-rr-vcc-fix-paired-20260722-050`
  accepts both controls and the instrumented leg; its maximum slowdown is
  1.41x.  Current inventory `...-inventory-vcc-fix-20260722-051` freezes 31
  singleton barriers.  Prospectively reviewed exact-one artifact
  `...-rr-fault-vcc-fix-20260722-052` drops the late decode-attention barrier
  exactly once, preserves both oracles, and emits no Record/Replay diagnosis,
  contradicting its frozen detected/pass policy.  It is rejected rather than
  post-hoc relabeled, so the remaining gate is a causal or diagnostically
  visible fault.

- 2026-07-22: Commit `4ad984b1c9` fixes the TP2 Record/Replay oracle corruption
  by modeling CDNA4's six-register descriptor-allocation tail correctly.  At
  the formerly corrupting partial-`EXEC` `ds_write_b32`, the old five-register
  transient window s72:s76 overlapped the grown physical VCC pair s74:s75.
  The corrected allocator preserves both the original and post-growth VCC
  locations; its focused physical discriminator changes from `13.0608` to the
  exact `0.5778079629` oracle, and all 730 ConSan host tests pass.  Frozen clean
  artifact `consan-validation-gfx950-tp2-family-rr-physical-vcc-fix-20260722-043`
  accepts prefill, decode, and combined with complete 936/936 access plus
  168/168 barrier coverage.  Paired artifact `...-overhead-20260722-044`
  records a 1.57x combined slowdown with one repetition per leg.  Fresh
  inventory `...-inventory-vcc-fix-20260722-045` retains 28 exact singleton
  barriers.  Two independently precommitted exact-one faults (`...-046` and
  `...-047`) were not diagnosed—one broke the external oracle and one was
  schedule-masked—so Record/Replay is yellow rather than green.

- 2026-07-22: The fast physical-gfx950 lane assessed four previously gray
  profile cells at clean hook SHA-256 `0832ad97...d594bc`.  TP1 prefill
  Inline Shadow passes its exact clean and paired oracles with complete
  120/120 access plus 31/31 barrier coverage; the paired slowdown is 246.6x.
  Its prospectively frozen exact-one barrier mutation is applied 1/1 with
  complete evidence and healthy containment, but is schedule-masked: the
  oracle passes and Inline emits no diagnostic, so the cell is yellow rather
  than green.  TP1 prefill Sampled and CLIP Sampled both fail closed at the
  connected ordinary-VGPR/AccVGPR boundary and are red.  CLIP Inline lowers
  all 45 accesses and 24 barriers but corrupts the warmup cosine oracle to
  NaN, while TP2 Record/Replay lowers all 624 currently supported accesses
  and 112 barriers but fails the prefill warmup oracle at `13.0608`; those
  cells are red.  Retained artifacts end in `-015` through `-021` under
  `/home/ossci/xx/consan-validation-gfx950-*`.

- 2026-07-22: TP1 decode/combined Inline Shadow passes both exact workload
  modes with complete 240/240 access plus 62/62 barrier coverage in artifact
  `consan-validation-gfx950-tp1-decode-combined-inline-20260722-023`.  The
  source revision is docs-dirty, so this promotes the cell only to orange;
  clean-tip paired and reviewed-fault evidence remain.

- 2026-07-22: Commit `92678db569` closes the CDNA4 MOI normalization gap for
  `ds_read_b64_tr_b16`, `ds_read2*`, and `ds_write2*`, retaining exact
  single- and two-range byte footprints.  All 728 ConSan host tests pass.
  TP1-prefill Record/Replay is green: clean artifact
  `consan-validation-gfx950-tp1-prefill-rr-lds-range-20260722-008` passes the
  exact oracle with complete 120/120 access and 31/31 barrier coverage;
  paired artifact `...-overhead-20260722-009` measures 1.29x; and reviewed
  exact-one artifact `...-fault-20260722-011` detects the removed causal
  barrier while the external oracle passes, with containment and health
  accepted.  CLIP clean artifact `...clip-bf16-rr-lds-range-20260722-012`
  likewise reaches complete 45/45 plus 24/24 coverage and passes its cosine
  oracle.  Its separate ten-process stress artifact `...-overhead-20260722-013`
  reproduces a replay conflict in two processes despite complete coverage, so
  CLIP remains yellow and the stress result is not accepted as paired
  evidence.  TP1 decode/combined rotates out after artifact
  `...decode-combined-rr-lds-range-20260722-014` fails the decode warmup oracle
  with complete first-object lowering, exposing a distinct mode-sensitive
  replay defect rather than reopening the opcode gap.

- 2026-07-22: Record/Replay now preserves the compiler's CDNA4
  ordinary/accumulator VGPR boundary by selecting fresh wave-uniform scalar
  owner/epoch state when a persistent VGPR pair cannot fit.  Commit
  `15e84c6d6c` retains `ACCUM_OFFSET`, passes its focused gfx950 and gfx1250
  regressions, all 727 ConSan host tests, and 112 validator tests.  A current
  gfx1250 D128-block regression remains accepted at 18/18 accesses and 4/4
  barriers.  On physical gfx950, TP1 prefill now executes with 104/104
  supported accesses and 31/31 barriers instead of failing during planning.
  D128 block, D128 pressure, and MFMA attention execute their exact oracles
  with 6/12 accesses and 4/4 barriers.  Stream-K arrival and tree atomic-OR
  execute with 4/4 accesses, 4/4 barriers, and 16/16 fences.  Their remaining
  gaps are independently typed: unsupported DS forms in TP1/CLIP, and the
  existing Record/Replay dynamic-stack resource limit in shared hip-moi
  access/atomic helpers.  CLIP additionally exposes one clean replay false
  conflict, so none of these partial rows is promoted to green.

- 2026-07-22: Tree atomic-OR SuperCollider is green from exact clean tip
  `b536af8f67`.  Paired artifact
  `consan-validation-gfx950-tree-atomic-or-supercollider-overhead-tip-20260722-002`
  accepts both exact tests and complete 4/4 coverage, measuring a 16,695-ms
  profile median against the 90-ms paired baseline (185.5x).  Fresh inventory
  `consan-validation-gfx950-tree-atomic-or-inventory-tip-20260722-002` exposes
  23 order sites and correctly omits scope.  Semantic review proves that the
  selected `flat_atomic_or` publishes producer MFMA partials.  Exact-one fault
  artifact
  `consan-validation-gfx950-tree-atomic-or-supercollider-atomic-order-fault-tip-20260722-001`
  replaces only its eight-byte release `buffer_wbl2` with two native NOPs,
  preserves the atomic bytes, matches the prospectively frozen
  pass-oracle/qualified-miss policy, and passes coverage, containment, cleanup,
  and physical-device health gates.

- 2026-07-22: Stream-K-arrival SuperCollider is green at clean
  `0942528a56`.  Commit `0942528a56` admits CDNA4 atomic-order faults while
  keeping instruction-scope and address rewrites fail-closed.  Same-tip paired
  artifact
  `consan-validation-gfx950-streamk-arrival-supercollider-overhead-tip-20260722-002`
  accepts at 143.70x with complete 4/4 coverage.  Inventory artifact
  `consan-validation-gfx950-streamk-arrival-inventory-tip-20260722-002`
  accepts 23 order sites and correctly omits the inapplicable scope family.
  Reviewed artifact
  `consan-validation-gfx950-streamk-arrival-supercollider-atomic-order-fault-tip-20260722-001`
  replaces exactly one eight-byte release `buffer_wbl2` with two target-native
  NOPs while preserving the atomic, waits, and acquire cache operation
  byte-for-byte.  Its precommitted pass-oracle/qualified-miss policy matches;
  coverage, cleanup, containment, and physical-device health all pass.

- 2026-07-22: TP1 decode/combined reviewed fault artifact
  `consan-validation-gfx950-tp1-decode-combined-supercollider-fault-tip-20260722-001`
  removes exactly one semantically reviewed barrier between waited LDS
  producers and immediate LDS consumers.  Coverage remains complete at
  240/240 and all containment checks pass, but one execution is schedule-masked:
  the external oracle passes and SuperCollider emits no diagnostic.  That
  contradicts the prospectively frozen detected/pass policy, so the validator
  correctly rejects the trial and the cell remains yellow without a post-hoc
  policy change.

- 2026-07-22: TP1 decode/combined SuperCollider paired artifact
  `consan-validation-gfx950-tp1-decode-combined-supercollider-overhead-tip-20260722-001`
  accepts at clean `1a16f718ac`, with exactly one repetition per leg.  Both
  exact workload oracles pass, coverage is complete at 240/240 accesses, and
  paired slowdowns are 1.18x for decode and 1.33x for combined.  Fresh
  inventory artifact
  `consan-validation-gfx950-tp1-decode-combined-inventory-tip-20260722-001`
  accepts 31 selectable singleton barriers; semantic review and one contained
  fault remain before green.

- 2026-07-22: TP1-prefill SuperCollider paired artifact
  `consan-validation-gfx950-tp1-prefill-supercollider-overhead-tip-20260722-002`
  accepts at clean `c3a15aabea`, with exactly one repetition per leg.  The
  paired baseline is 0.811866 ms, SuperCollider is 0.920680 ms (1.13x), and
  coverage is complete at 120/120 accesses.  Fresh inventory artifact
  `consan-validation-gfx950-tp1-prefill-inventory-tip-20260722-002` accepts 31
  exact-one barrier sites; semantic review and contained fault execution are
  active before the cell can become green.

- 2026-07-22: TP1-prefill reviewed fault artifact
  `consan-validation-gfx950-tp1-prefill-supercollider-fault-tip-20260722-001`
  removes exactly one semantically reviewed barrier at PC `0x7ff8`, between
  waited LDS publication and immediate LDS consumption in the executed
  prefill-attention kernel.  Coverage remains complete at 120/120 and
  SuperCollider reports exactly one instability diagnostic; health and the
  independent dispatch smoke pass before and after.  The external scalar
  oracle nevertheless passes, contradicting the precommitted
  fail-oracle/no-diagnostic policy, so the validator correctly rejects the
  trial and the cell remains yellow.  The policy was not changed after
  observation.

- 2026-07-22: D128-pressure and MFMA-attention SuperCollider are green at
  current clean tip `499ddfe89d`.  Their paired artifacts report 186.82x and
  88.25x respectively; reviewed fault artifacts each apply exactly one
  `s_barrier`-to-NOP mutation at PC `0x874c`, produce the precommitted failing
  workload oracle and qualified detector miss, retain complete 12/12 coverage,
  terminate within policy, and leave the physical gfx950 healthy.

- 2026-07-22: Earlier Stream-K-arrival and tree atomic-OR inventories exposed
  the missing CDNA4 atomic-order mutation path after their clean and paired
  SuperCollider runs.  Commit `0942528a56` closes that implementation gap;
  Stream-K is now green above, while tree retains its earlier paired evidence
  and awaits a same-tip inventory and reviewed fault.

- 2026-07-22: Completed the first current-tip green gfx950 cell.  D128-block
  SuperCollider combines the clean artifact, accepted one-repetition paired
  artifact `consan-validation-gfx950-d128-block-supercollider-overhead-tip-20260722-001`,
  fresh inventory, and reviewed fault artifact
  `consan-validation-gfx950-d128-block-supercollider-fault-tip-20260722-001`.
  The paired baseline is 128.5 ms and SuperCollider is 112.16x slower.  The
  exact-one mutation rewrites the reviewed `s_barrier` at PC `0x874c` to
  `s_nop 0`; requested/planned/applied are 1/1/1.  Its precommitted qualified
  detector miss and failing workload oracle occur exactly as expected, with
  12/12 complete coverage, bounded termination, and healthy physical-device
  probes before and after.

- 2026-07-22: Current-tip SuperCollider clean revalidation now also accepts
  TP1 prefill (120/120), TP1 decode/combined (240/240), TP2 family (936/936),
  CLIP BF16 (45/45), MFMA attention (12/12), Stream-K arrival (4/4), and tree
  atomic-OR (4/4).  Every workload oracle passes and every admitted object is
  statically and dynamically complete.  These cells are yellow because clean
  evidence alone does not satisfy the fault/resource/frozen promotion gates.

- 2026-07-22: Physical D128-pressure artifact
  `consan-validation-gfx950-d128-pressure-supercollider-tip-20260722-001`
  accepts baseline and SuperCollider at clean current tip `aec5d5a3a9`.  All
  four host-reference oracles pass in each row.  SuperCollider covers 12/12
  accesses with complete static/dynamic analysis, zero incomplete state, and
  a complete mismatch-free report; the physical device remains healthy.
  Record/Replay, Sampled, and Inline Shadow were deliberately not run in this
  artifact and remain gray.

- 2026-07-22: Began current-tip physical revalidation with D128 block artifact
  `consan-validation-gfx950-d128-block-clean-tip-20260722-001`.  The baseline
  passes both exact host-reference oracles.  SuperCollider passes both oracles
  with complete 12/12 access coverage and static/dynamic analysis.  Record/
  Replay and Sampled safely reject persistent VGPR state at the kernels'
  ordinary-VGPR/AccVGPR boundary; Inline Shadow separately selects scalar
  state for one connected component but fails to assign entry-local scratch
  for another.  Provenance is clean at `2deee22c30`, and postflight physical
  gfx950 discovery remains healthy.  These are current placement regressions,
  so the three failing cells are red rather than inheriting old acceptance.

- 2026-07-22: Rechecked the actual physical device rather than relying on the
  retained campaign.  Workspace TheRock `rocminfo` reports MI355X / gfx950,
  five exact native CDNA4 hip-moi tests pass, and the source-built corpus
  `hip_matmul` m128³ case passes correctness for all three selected kernels at
  one iteration.  Workload-scoped doctor checks define every current-matrix
  row: TP1, TP2, CLIP, and the five native hip-moi roles are runnable; Qwen is
  blocked on its gfx950 VMFB and Jakub on its target-native executable.
- 2026-07-22: Resolved the PyTorch `hipErrorInvalidImage` setup blocker.  The
  failing interpreter was a thin gfx1250 installation with no gfx950 kernel
  pack despite its broad architecture-list metadata.  A separate official
  ROCm 7.1 nightly environment now passes the validator doctor and all six
  portable one-repetition exact baselines on the physical gfx950 without
  modifying the gfx1250 environment.  Those rows are registered for gfx950;
  the first strict Record/Replay `torch.mode` run now reaches ConSan and
  records a pre-execution transform rejection rather than a runtime setup
  failure.
- 2026-07-22: The first physical gfx950 TopK SuperCollider assessment exposed
  stale appended-cave mappings before execution.  The generic planner was
  reserving bodies for a gfx1250-only branch-continuation route on CDNA4, then
  correctly declining to emit that route; all later body offsets therefore
  drifted from the append cursor.  Restricting both reservation and routing to
  gfx1250 preserves that architecture's existing fallback while letting CDNA4
  skip the inapplicable candidates.  Artifact
  `consan-gfx950-pytorch-topk-sc-branchguard-20260722-009` passes every TopK
  exact oracle in one repetition with zero dynamic-incomplete encounters;
  3,056/230,438 supported accesses patch, so the cell is orange rather than
  green.
- 2026-07-22: Refreshed this ledger to use the same four-profile matrix and
  red/orange/yellow/green/gray maturity scale as the expanded gfx1250 ledger.
  Every current status is gray because the substantially changed current tip
  requires complete revalidation; old campaign results remain only as
  historical annotations.
- 2026-07-22: Surveyed `rocjitsu-test-corpus` at `aa54cc86c9eb`.  Native
  gfx950 HIP matmul and HipKittens BF16 cases are the first packaged corpus
  candidates.  The two four-wave low-precision cases are compile-skipped,
  hip-stream-k metadata is gfx942-only, and the packaged Tensile artifacts
  are gfx1250-only.  These distinctions are now explicit rather than silently
  treating source availability as runnable validation support.
- 2026-07-22: Surveyed the 36 gfx950 GEMM YAMLs in `rocm-libraries` at
  `c2fafc16393d` and selected a bounded Stream-K, LDS-transpose, 160-KiB LDS,
  low-precision, and split-reduction follow-on set.  All remain gray until
  reduced, generated, numerically checked, packaged, inventoried, and added to
  the validation registry.
- 2026-07-22: Mapped only architecture-neutral PyTorch operator rows to the
  gfx950 expansion.  Target-specific descriptor/cluster source remains
  deliberately excluded; fresh gfx950 code-object inventories precede profile
  promotion.

- 2026-07-18: Stream-K Inline Shadow on the current safe planner is rejected
  before installation, not accepted as the older matrix wording implied.
  Both dynamic-stack kernel owners reach the same full CDNA4 ordinary-VGPR
  boundary, and their connected helper needs owner/epoch/workgroup identity;
  allocating the old v112:v114 tuple would move `ACCUM_OFFSET` semantics and
  corrupt MFMA operands.  Artifact
  `gfx950-take2-streamk-inline-after-retry-001` records the typed planning
  rejection.  `IS1` remains orange/ACTIVE while a scalar per-wave persistent
  representation is implemented; no unsafe VGPR growth is being restored.
- 2026-07-18: The TP1 Inline Shadow lifecycle gap is closed while `IS0` and
  `IS1` remain orange/ACTIVE for atomic breadth and campaign evidence.  Split
  telemetry proved every residual sequential decode+combined miss was a
  stable-version rejection of stale nonempty payload; no claim, CAS, commit,
  changed-version, odd-version, or terminal-version failure was involved.
  CDNA4 retry paths now execute `buffer_inv sc1`, retain the coherent atomic
  opening version, and preserve ordinary-load performance on the uncontended
  fast path.  Six consecutive retained authoritative runs (`-026` through
  `-031`) accept 208/208 accesses and 62/62 barriers with passing oracles and
  `dynamic_incomplete=0`.  The synchronized plan box remains orange, and 606/606
  focused host tests pass.
- 2026-07-17: CDNA4 Sampled barrier lowering is native-proven.  The prior
  architecture gate was the direct cause of TP1/CLIP's missing supported
  barriers; a focused CDNA4 singleton-barrier patch test now passes alongside
  the existing CDNA4 atomic test.  Retained native TP1 prefill, TP1
  decode/combined, and CLIP artifacts accept 104/104 + 7/7, 208/208 + 14/14,
  and 39/39 + 20/20 access/qualified-barrier coverage with passing workload
  oracles and no dynamic-incomplete state.  The TP2 aggregate still rejects
  one 140-access slice because Sampled selects no applicable patch there, so
  `SA0A` is green/DONE while `SA0` and `V5` remain orange/ACTIVE.  The Mermaid
  class assignments are synchronized with those labels.
- 2026-07-17: Commit `3e7c5bdf02` completes the native-LDS application
  breadth exposed after the CLIP vertical.  Automatic B32 reporting now pads
  to an even CDNA4 FLAT address tuple, while `ds_read2_b64` and
  `ds_read2st64_b64` duplicate both 64-bit addresses into a contiguous result
  tuple.  All 601 ConSan host tests pass.  Retained artifacts
  `consan-validation/gfx950-take2-tp1-prefill-sc-clean-005`,
  `consan-validation/gfx950-take2-tp1-decode-combined-sc-clean-003`, and
  `consan-validation/gfx950-take2-tp2-family-sc-clean-003` are accepted at
  120/120, 240/240, and 936/936 native accesses with passing oracles and full
  static/dynamic coverage.  The synchronized `SC0` DAG node is green/DONE;
  workload cells stay orange pending their remaining campaign requirements.
- 2026-07-17: Immediate native Sharktank reruns after `afbca95de0` retain the
  earlier 0/120, 0/240, and 0/936 SuperCollider coverage.  They now expose a
  precise error rather than an architecture gate: a B32 check places the
  automatic report address pair at an odd CDNA4 VGPR.  The whole code object
  correctly rolls back on the unencodable report action.  The CLIP 45/45 row
  remains accepted evidence, while synchronized `SC0` returns to orange/ACTIVE
  until report-scratch parity is fixed and the broader reruns pass.
- 2026-07-17: Commit `afbca95de0` establishes the initial CDNA4 native-LDS
  SuperCollider CLIP vertical.  Exact host tests cover ordinary reads/writes,
  transpose B16 reads, and the non-contiguous data operands of
  `ds_write2st64_b64`; all 599 ConSan tests pass.  Retained artifact
  `consan-validation/gfx950-take2-clip-bf16-sc-clean-005` accepts CLIP BF16
  with its independent oracle passing, all 45/45 supported native LDS accesses
  patched, analysis/static/dynamic completeness, and zero mismatches.  The
  CLIP cell remains orange until its fault, overhead, and frozen-tip evidence is
  complete.  The immediate Sharktank audit above subsequently broadened its
  implementation exit criterion.
- 2026-07-17: Native TP1 decode/combined, TP2 family, and CLIP BF16
  reconnaissance is retained in `gfx950-take2-tp1-decode-combined-clean-001`,
  `gfx950-take2-tp2-family-clean-001`, and `gfx950-take2-clip-bf16-clean-001`.
  Record/Replay accepts all three with 208/208 + 62/62, 840/840 + 168/168,
  and 39/39 + 24/24 access/barrier coverage.  SuperCollider consistently
  patches 0 supported accesses; Sampled consistently omits supported barriers
  and additionally misses one TP2 aggregate slice; Inline reports either
  undercoverage or dynamic-incomplete state.  All cells and synchronized DAG
  nodes remain orange/ACTIVE pending fixes and authoritative reruns.
- 2026-07-17: Retained artifact
  `consan-validation/gfx950-take2-tp1-prefill-clean-002` runs native TP1
  prefill through the workspace's IREE Python venv.  Record/Replay is accepted
  with 104/104 accesses and 31/31 barriers.  SuperCollider passes the workload
  but patches 0/120 supported accesses; Sampled passes and patches 104/104
  accesses but 0/10 supported barriers; Inline patches 104/104 accesses and
  31/31 barriers but records 28 dynamic-incomplete encounters and exits 90.
  These are orange active evidence cells, not green campaign completion; the
  synchronized `V1` and `V5` DAG nodes are orange/ACTIVE.
- 2026-07-17: Retained artifact
  `consan-validation/gfx950-take2-tree-atomic-or-clean-001` accepts tree
  atomic-OR under all four profiles with passing workload oracles and no
  unexpected diagnostics.  All profiles patch 4/4 accesses; Record/Replay
  additionally patches 10/10 atomics, 4/4 barriers, and 16/16 fences, Sampled
  patches 10/10 atomics, and Inline patches 10/10 atomics plus 4/4 barriers.
  The four cells remain orange pending fault, overhead, and frozen-tip evidence,
  and the synchronized `V5` node remains orange/ACTIVE.
- 2026-07-17: Stream-K Inline now lowers and dynamically executes all 4/4
  accesses alongside 10/10 atomics and 4/4 barriers, with zero ConSan
  diagnostics or incomplete state in accepted individual trials.  Sequential
  stress is not yet accepted: hip-moi intermittently reports its own
  `0x1c05`/`0x1c06` access conflict because its RMW release bookkeeping occurs
  after the hardware atomic and permits only four acquire retries; ConSan is
  instrumenting that nested atomic before control returns to the bookkeeping.
  The application output remains correct and ConSan remains clean.  The cell
  and synchronized `V5` DAG node therefore stay orange/ACTIVE, not green.
- 2026-07-17: Commit `2da1350b8e` fixes the native sampled Stream-K fault by
  excluding CDNA4 `flat_scratch`/`xnack_mask` aliases from ordinary scalar
  instrumentation state and removing two stale scalar reservations.  All 594
  ConSan host tests pass.  In retained artifact
  `consan-validation/gfx950-take2-streamk-arrival-clean-005`, SuperCollider,
  Record/Replay, and Sampled are accepted with 4/4 access coverage;
  Record/Replay additionally covers 10/10 atomics, 4/4 barriers, and 16/16
  fences, while Sampled covers 10/10 atomics.  Inline Shadow passes the
  workload oracle and covers 10/10 atomics plus 4/4 barriers but is rejected
  at 0/4 accesses.  All four cells are orange to distinguish three retained
  clean acceptances and one active coverage gap from frozen green campaign
  completion; the orange/ACTIVE `V5` Mermaid box names that same state.
- 2026-07-17: Retained artifact
  `consan-validation/gfx950-take2-mfma-attention-clean-001` accepts native
  MFMA attention under all four profiles.  Every row passes its independent
  workload oracle, covers 12/12 accesses, and records zero dynamic-incomplete
  encounters; Record/Replay and Inline Shadow also cover 4/4 barriers.  The
  four MFMA-attention cells are orange pending faults, overhead, and frozen-tip
  evidence, and the orange/ACTIVE `V5` box text names this added vertical.
- 2026-07-17: Commit `b58799fda0` closes the D128-pressure SuperCollider
  placement gap by retaining execution-owner descriptors on ordinary FLAT
  sites.  The focused CDNA4 far-call regression and all 591 ConSan host tests
  pass.  In retained artifact
  `consan-validation/gfx950-take2-d128-pressure-clean-005`, SuperCollider,
  Record/Replay, Sampled, and Inline Shadow all pass the workload oracle and
  patch 12/12 accesses; Record/Replay and Inline Shadow also patch 4/4
  barriers, and every row has zero dynamic-incomplete encounters.  The four
  pressure cells and `V5` are orange/ACTIVE; faults, overhead, the remaining
  application matrix, and frozen-tip evidence remain open.
- 2026-07-17: Commit `ecd529c017` corrects the gfx950-only D128 overhead
  workload selection.  The initial retained `overhead-001` experiment proved
  that the ExactContext specialization executes no admitted ConSan access
  probes on this target; its Record/Replay rows correctly failed the required
  dynamic-evidence gate.  The replacement
  `consan-validation/gfx950-take2-d128-block-overhead-002` uses the native
  `SampledFastContext` specialization and accepts baseline-before, all four
  profiles, and baseline-after across three processes each.  The paired
  baseline is 130.5 ms, with 6.61x, 8.35x, 8.31x, and 9.29x process slowdown
  for SuperCollider, Record/Replay, Sampled, and Inline Shadow respectively.
  The cells and `V7` remain orange pending peak-memory and frozen-tip evidence.
- 2026-07-17: Commit `8c8480e68f` corrects the gfx950 barrier-drop
  specification contract for CDNA4's single full `s_barrier`; RDNA4's paired
  signal/wait selector is unchanged.  The reviewed D128 selector was fixed
  before rerunning the campaign, retaining its precommitted detector and
  oracle policies.  Artifact
  `consan-validation/gfx950-take2-d128-block-fault-003` is accepted for
  SuperCollider, Record/Replay, Sampled, and Inline Shadow.  Every row applies
  exactly one mutation, produces the expected independent oracle failure and
  qualified detector miss, terminates within its bound, and passes both GPU
  health gates.  The D128 matrix cells remain orange pending overhead and a
  frozen-tip rerun; `A5` and `V6` are now orange/ACTIVE in the plan.
- 2026-07-17: Commit `c7ca36d375` ports SuperCollider's redundant
  group-FLAT check to CDNA4 without changing the established RDNA4 sequence.
  It supports two-word D16/D32/D64/D128 loads and stores, U16 short-value
  comparisons, the CDNA4 VM/LGKM completion wait, and even report-address
  tuples.  Retained artifact
  `consan-validation/gfx950-take2-d128-block-supercollider-006` is accepted
  with analysis/static/dynamic completeness, 12/12 accesses patched, and
  zero mismatches.  A permissive earlier 0/0 result is superseded rather than
  treated as evidence.  The focused, ConSan, and full gates pass 413/413,
  590/590, and 2145/2145.  `SC1` is orange/ACTIVE pending a retained racy/fault
  detection proof and the wider acceptance campaign.
- 2026-07-17: Commit `125419056d` fixes the CDNA4 Inline Shadow identity
  lifetime exposed by group-FLAT D128 attention.  Workgroup identity needed by
  shared helpers is now entry-persistent and allocated above each owning
  descriptor's full VGPR allocation, rather than relying on site-local
  liveness across arbitrary guest code.  Retained artifact
  `consan-validation/gfx950-take2-d128-block-inline-004` is fully accepted:
  12/12 accesses and 4/4 barriers are patched, analysis/static/dynamic
  coverage are complete, and dynamic-incomplete is zero.  The focused,
  ConSan, and full gates pass 413/413, 588/588, and 2143/2143.  This advances
  `IS1` to orange/ACTIVE; it does not yet complete multidimensional identity,
  fault, overhead, or frozen-campaign requirements.
- 2026-07-17: Retained four-profile artifact
  `consan-validation/gfx950-take2-d128-block-all-001` classifies the next
  slices.  Record/Replay repeats its 12/12 access and 4/4 barrier clean
  vertical.  Sampled executes with 12/12 admitted accesses.  Inline Shadow is
  static-complete at 12/12 plus 4/4 but rejects the run after accounting
  47,360 unsupported dynamic lane-site encounters: CDNA4 function-time
  workgroup SGPR sources are not stable without the entry snapshot/restoration
  transaction.  Identity nodes `I0-I3` are therefore active (orange), not TODO.
  SuperCollider still finds no applicable gfx950 site and remains an open
  implementation node; its runner-side zero-site acceptance is not treated as
  evidence.
- 2026-07-17: Commit `31552da206` completed the first native Record/Replay
  group-FLAT slice with full supported-site coverage.  D16 `flat_store_short` and
  `flat_load_ushort` now normalize to two-byte accesses.  Enabling all twelve
  workload sites exposed the previously dormant many-candidate indirect
  island path, whose RDNA-specific raw `s_getpc_b64`/`s_setpc_b64` words are
  now architecture-aware builders.  The same commit enables CDNA4 persistent
  epoch advancement at barriers.  Retained artifact
  `consan-validation/gfx950-take2-d128-block-record-007` runs successfully
  with 12/12 access and 4/4 shared-function barrier sites patched and dynamic
  evidence complete.  Nodes `A3`, `RR1`, and `Q0` are now active (orange) in
  the DAG.  The focused, ConSan, and full gates pass 413/413, 588/588, and
  2143/2143.
- 2026-07-17: Commit `6e997a6a39` started access-normalization node `A2`.
  CDNA4 two-word FLAT decode, raw-field retention, explicit shared-base
  provenance, candidate admission, and Record/Replay/Inline first-light
  emission now have exact host coverage.  In retained native artifact
  `consan-validation/gfx950-take2-d128-block-record-003`, Record/Replay runs
  successfully and patches all 8/8 admitted D32/D64 group-FLAT accesses, with
  all eight dynamically visible.  The workload remains deliberately
  non-green: four D16 short/ushort accesses and four shared-function barriers
  are still missing.  The focused, ConSan, and full rocJITsu gates pass
  410/410, 586/586, and 2140/2140 respectively.
- 2026-07-17: Commit `f5c91c6d1d` started validation-environment nodes `E1`
  and `V0`.  The manifest no longer emits RDNA4 executable or test-suite
  spellings for gfx950 hip-moi rows: D128 block/pressure, MFMA attention,
  Stream-K arrival, and tree atomic-OR all resolve to present CDNA4 binaries.
  Workload-scoped doctor checks let those ready rows proceed independently;
  `d128-block` passes its complete preflight using the current hook and
  workspace TheRock `rocminfo`.  The full doctor honestly retains two gaps:
  Qwen artifacts and a semantically equivalent CDNA4 Jakub build.
- 2026-07-17: Commits `8068d7d018` and `4dcba398f9` completed dynamic-stack
  spill node `S8C`.  Exact host tests prove CDNA4's explicit-`saddr` scratch
  encodings, SCC preservation, stack advance/restore, final-valid Inline
  Shadow integration, and Record/Replay's typed rejection.  The production
  hook also instruments a real compiler-generated gfx950 dynamic-allocation
  kernel: its forced 16-VGPR spill completes native dispatch, preserves every
  lane's private-stack value, and reports no Inline Shadow error counters.
  All nine native spill cases, the 409/409 focused gate, the 584/584 ConSan
  suite, and the full 2138/2138 rocJitsu suite pass.
- 2026-07-17: Commit `af03dfa4c4` completed fence parity node `S8D`.
  CDNA4 no longer stops at an RDNA-only architecture gate: both qualified
  fence sites reach the intentional second-text-growth spill limitation,
  retain spill resource evidence, emit no unsafe patch, and receive typed
  `placement_or_lowering_failed` / `instrumentation_patch_missing` outcomes.
  The native atomic case and the 406/406 plus 582/582 host gates remain clean.
- 2026-07-17: Commit `6f42088f3d` completed atomic spill node `S8B`.
  CDNA4's compiler-shaped acquire-release atomic sequence is accepted without
  fabricating an RDNA `TH` encoding.  The production hook forces its
  three-VGPR record probe into private scratch, grows the live kernel object
  from 0 to 16 bytes, and completes native dispatch with all 512 live values
  intact, the counter at exactly 64, and a valid atomic record.  All eight
  native spill cases, the 406/406 focused host gate, and all 582 ConSan host
  tests pass.  Fence second-growth rejection remains open in `S8D` and is not
  claimed by this row.
- 2026-07-17: Commit `33075e7aac` completed barrier spill node `S8A`.
  The native gfx950 fixture forces both its LDS access and `s_barrier` through
  production-hook private spills.  The access uses three VGPRs, the barrier
  uses six, both trampolines pass final validation, and dispatch-private
  storage grows from 0 to 32 bytes.  All eight live values survive across the
  barrier for every lane and both access and barrier records are visible.  All
  seven native spill cases, the 405/405 focused host gate, and all 581 ConSan
  host tests pass.
- 2026-07-17: Commit `c4b0c3a1a4` completed shared-layout node `S3`.
  One native noinline LDS helper is owned by two full-VGPR-file gfx950
  kernels.  The production hook reports a common three-register spill plan
  with `owners=2`, assigns private-backed persistent epoch state, grows both
  live kernel objects from 0 to 16 private bytes, and successfully dispatches
  both callers.  Their distinct 64-lane outputs survive and the report contains
  a valid LDS-write record.  All six native spill cases, the 404/404 focused
  host gate, and all 580 ConSan host tests pass.
- 2026-07-17: Commit `54673f205e` completed Inline Shadow spill node
  `S7B`.  CDNA4's workgroup-local path now rotates only its B64 DS-exchange
  return and first-diagnostic CAS tuples to even VGPR pairs, leaving the global
  versioned transaction unchanged.  Host final validation recognizes and
  proves the native instruction shapes.  The production-hook native dispatch
  spills 16 VGPRs, preserves all live output, publishes required evidence, and
  leaves every diagnostic/error counter zero.  The focused host gate is
  404/404, all ConSan host tests are 580/580, and all five native spill cases
  pass serially.
- 2026-07-17: Commit `2f9cc947b0` completed Sampled spill node `S7A`.
  The production-hook gfx950 fixture forces three spilled VGPRs, preserves
  eight live values for all 64 lanes through native AQL dispatch, and
  publishes a valid LDS-write sampled entry with a ready, matching causal
  window.  All four native spill cases—standalone full-wave and
  partial-`EXEC`, Record/Replay, and Sampled—pass serially on the MI355X.
- 2026-07-17: Commit `3a52aaee88` completed native spill nodes `S4` and
  `S6B`.  A dedicated gfx950 HIP fixture runs through the production HSA hook,
  forcing a three-VGPR Record/Replay spill around a native `ds_write_b32`.
  The patch includes a newly enabled CDNA4 owner/epoch entry prologue, passes
  final validation, grows the live dispatch-private requirement from 0 to 16
  bytes, preserves eight live values for all 64 lanes, and publishes a valid
  LDS-write access record.  The native full-wave, partial-`EXEC`, and
  hook-driven cases pass serially; the focused host gate is 403/403 and all
  ConSan host tests are 579/579.
- 2026-07-17: Commit `44e7767b9b` completed host-integrated spill node
  `S6A`.  It fixed two resource assumptions exposed only by gfx950 integration:
  descriptor VGPR granularity is eight, and every ConSan scratch window must
  begin at an even VGPR for CDNA4 FLAT addresses.  Automatic 8-to-16 VGPR
  growth, typed odd-base rejection, and an exact three-VGPR Record/Replay
  spill around a retained native DS instruction all pass.  The spill grows a
  preexisting 32-byte private segment to 48 bytes and survives final
  validation.  The focused gate is 402/402 and all ConSan host tests are
  578/578.  Its then-open native dispatch/AQL evidence was completed by
  `3a52aaee88`.
- 2026-07-17: Commit `089fdf0585` completed the first CDNA4 InlineShadow
  atomic-ordering vertical.  The exact gfx950 compiler acquire-release shape
  is associated, admitted, emitted, and structurally revalidated; exact
  instruction assertions retain the guest atomic, two release CAS operations,
  fifteen acquired-token CAS operations, and semantic waits.  CDNA4 scratch
  rotation makes every FLAT address and CAS tuple even-aligned while leaving
  RDNA4 allocation unchanged.  The focused gate is 400/400 and all ConSan
  host tests are 576/576.  This advances `A4`/`IS0`; automatic-resource,
  forced-spill, broader synchronization, and native evidence remain open.
- 2026-07-17: Commit `1806f7c90d` added the first native CDNA4 FLAT-atomic
  inventory and supported address plan.  Exact ordered synchronization and
  inline-atomic emission remain open.  All ConSan host tests are 574/574.
- 2026-07-17: Commit `b0ddc85877` produced a fully validated CDNA4
  InlineShadow access patch.  It also routed scalar subtraction through the
  target-neutral builder and taught exact-shadow final validation to inspect
  both in-place and trampoline bodies.  The focused gate is 399/399 and all
  ConSan host tests are 573/573.  This starts `IS0`; atomic/synchronization,
  automatic-resource, forced-spill, and native evidence remain open.
- 2026-07-17: Commit `ec84cd7af7` produced a fully validated CDNA4 Sampled
  access patch and removed the last embedded gfx12 LDS-wait word from engine
  emission.  The focused gate is 398/398 and all ConSan host tests are
  572/572.  This starts `SA0`; runtime/native and broader synchronization
  evidence remain open.
- 2026-07-17: Commit `6f654e596d` removed the last hard-coded gfx12
  `s_wait_loadcnt 0` words from ConSan engine emission.  Every publication,
  compare-swap, guest-atomic, and snapshot path now obtains its completion
  wait from the target-semantic builder; all 571 ConSan host tests pass.
- 2026-07-17: Commit `4fdaee5298` produced the first fully validated CDNA4
  engine patch: Record/Replay first-light access emission on a native DS
  instruction shape.  It also moved conditional branch fixups and publication
  waits onto target-semantic builders after the new test caught their hidden
  RDNA assumptions.  The focused gate is 397/397 and all ConSan host tests are
  571/571.  This starts `RR0`; native execution and full replay evidence are
  still required.
- 2026-07-17: Commit `0cb03769ea` established the first target-native analysis
  fixture.  Exact CDNA4 DS read/write instruction shapes survive the synthetic
  gfx950 ELF path and reach supported ConSan LDS candidates with correct
  operands.  This is analysis evidence only; no engine or workload cell is
  promoted yet.
- 2026-07-17: Commit `a0b96f50ce` completed neutral dispatch for ConSan's
  variable-length emission recipes: literal materialization, vector/address
  arithmetic, FLAT/DS/private memory, atomics, workgroup barriers, and their
  target-semantic waits.  CDNA4 multiply recipes explicitly receive a
  nonaliasing materialization VGPR, and private scratch uses VM_CNT completion
  rather than the broader general-FLAT drain.  The dispatch gate is 5/5, the
  focused builder/resource/spill/engine gate is 396/396, and the complete
  ConSan host suite is 569/569.  The plan now separates completed neutral
  routing (`B5A`) from active gfx950 engine-emission/shape proof (`B5B`).
- 2026-07-17: Commit `b21df67176` introduced the neutral instrumentation
  facade and routed ConSan's fixed-width scalar/vector control operations
  through it.  The facade has exact CDNA4/RDNA4 dispatch tests and typed
  rejection for unsupported architectures.  The focused gate is 394/394 and
  the complete ConSan host suite is 569/569; variable-length emission remains
  open in `B5`.
- 2026-07-17: Commit `f34be1ae72` completed the CDNA4 synchronization/cache
  primitive slice.  The wait immediates reflect gfx950's actual counter model:
  general FLAT drains VM_CNT and LGKM_CNT, LDS/SMEM drain LGKM_CNT, and scratch
  drains VM_CNT.  `s_barrier` is paired with a conservative memory pre-drain;
  gfx950 uses `s_nop 0`, never gfx12 `s_delay_alu`, for the one-cycle SALU
  separation.  The focused builder/resource/spill gate is 391/391.
- 2026-07-17: Commit `73241ff07f` completed the CDNA4 DS/atomic builder slice.
  Seven assembler-authoritative encodings and decoder mnemonics pass together
  with negative tuple/scope/return tests; the focused builder/spill gate is
  89/89.
- 2026-07-17: Commit `4c51920eb2` completed the non-atomic CDNA4 memory-builder
  slice.  Dispatch SMEM loads and FLAT publication accesses match LLVM and
  decode correctly with target-specific pair and offset rejection.  The
  focused builder/spill gate passes 87/87.
- 2026-07-17: Commit `113cd36228` completed the CDNA4 vector-builder slice and
  documented a new resource-planning requirement: in-place multiply by a
  literal needs a distinct materialization VGPR on gfx950.  The implementation
  uses only assembler-accepted sequences and the focused scalar/vector/spill
  gate passes 85/85.  Engine integration remains pending.
- 2026-07-17: Commit `38cf90d7c2` completed the separate CDNA4 scalar-control
  builder slice.  Exact gfx950 words and decoder mnemonics cover EXEC/VCC/SCC
  save/restore, scalar compares, and all branch predicates needed by probes;
  the RDNA4 builder remains CDNA-free.  This is builder evidence, not yet a
  workload promotion.
- 2026-07-17: Commit `97f978941d` made ConSan private-layout planning
  architecture-typed instead of inheriting RDNA4's 8 MiB range.  CDNA4
  persistent and spill extents are now bounded at 4 KiB and their final
  descriptor/AQL requirement is rounded to 16 bytes without moving dword
  slots.  The expanded focused gate passes 379/379; integrated shared-owner
  CDNA4 evidence is still pending, so this prerequisite remains orange.
- 2026-07-17: Commit `0ad99bd3ac` completed the standalone live CDNA4 scratch
  proof.  Both full-wave64 and partial-`EXEC` round trips pass serially on the
  MI355X using only the workspace TheRock runtime.  Disassembly confirms the
  intended scratch instructions and waits survived compilation, the partial
  kernel narrows and restores `EXEC`, and both kernel descriptors provision 32
  private bytes.  This promotes implementation node `S5`, not an e2e workload
  cell.
- 2026-07-17: Commit `319950aacd` established the separate CDNA4 fixed-scratch
  backend and transactional static VGPR spill plan.  Exact encodings, decoder
  round trips, limits, alignment, waits, rollback, and RDNA4 preservation pass
  the 377-test focused host gate.  This is prerequisite evidence only; all e2e
  cells remain unknown until native workload qualification.
- 2026-07-17: Created the CDNA4 ledger.  Verified native device discovery and
  an actual HIP dispatch using only the workspace TheRock runtime.  Verified
  the new `iree-test-suites` checkout and all directly consumed Sharktank
  assets.  All workload/profile cells intentionally begin unknown.
