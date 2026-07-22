# ConSan gfx1250 status

This is the `gfx1250` workload × instrumentation evidence ledger.  It follows
the acceptance standard of [STATUS_RDNA4.md](STATUS_RDNA4.md), but inherits no
coverage denominator, machine-code identity, fault expectation, timing,
provenance, or green cell from another architecture.

The executable authority is
[`consan_validation.py`](../../tests/dbi/consan/consan_validation.py), with the
experiment contract described by [VALIDATION.md](VALIDATION.md).  Porting work
and dependencies are tracked in [PLAN_GFX1250.md](PLAN_GFX1250.md).

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
| **P4 hip-moi D128 block attention** | 🟩 Current paired 1.56x; 18/18 accesses | 🟩 Current paired 1.11x; 18/18 accesses, 4/4 barriers | 🟩 Current paired 1.13x; 18/18 accesses, 8/8 applicable barriers | 🟩 Current paired 1.09x; 18/18 accesses, 4/4 barriers |
| **P4 hip-moi D128 pressure attention** | 🟩 Current paired 1.84x; 40/40 accesses | 🟩 Current paired 1.13x; 40/40 accesses, 4/4 barriers | 🟩 Current paired 1.12x; 40/40 accesses, 8/8 applicable barriers | 🟩 Current paired 1.29x; 40/40 accesses, 4/4 barriers |
| **P4 hip-moi WMMA attention** | 🟩 Current paired 1.78x; 18/18 accesses | 🟩 Current paired 1.17x; 18/18 accesses, 4/4 barriers | 🟩 Current paired 1.15x; 18/18 accesses, 8/8 applicable barriers | 🟩 Current paired 1.17x; 18/18 accesses, 4/4 barriers |
| **P4 hip-moi Stream-K arrival** | 🟩 Current paired 7.38x; 4/4 accesses | 🟩 Current paired 2.41x; 4/4 accesses, 4/4 barriers, 10/10 atomics, 16/16 fences | 🟩 Current paired 2.72x; 4/4 accesses, 8/8 applicable barriers, 10/10 atomics | 🟩 Current paired 2.62x; 4/4 accesses, 4/4 barriers, 10/10 atomics |
| **P4 hip-moi tree atomic-OR** | 🟩 Current paired 6.55x; 4/4 accesses | 🟩 Current paired 2.04x; 4/4 accesses, 4/4 barriers, 10/10 atomics, 16/16 fences | 🟩 Current paired 2.57x; 4/4 accesses, 8/8 applicable barriers, 10/10 atomics | 🟩 Current paired 2.19x; 4/4 accesses, 4/4 barriers, 10/10 atomics |
| **P4 Jakub attention variants** | 🟩 Current paired 2.52x; 31/31 accesses | 🟩 Current paired 1.44x; 62/62 accesses, 4/4 barriers | 🟩 Current paired 1.56x; 62/62 accesses, 8/8 applicable barriers | 🟩 Current paired 1.55x; 62/62 accesses, 4/4 barriers |

CLIP BF16 is intentionally omitted from the current acceptance matrix.  Its
uninstrumented execution is not presently practical in the software GPU
environment: the default multi-executor configuration can stall before model
inference, and a single-executor baseline reaches inference but remains too
slow for useful iteration.  Existing static gfx1250 qualification evidence is
retained in the progress log, but CLIP is outside the matrix denominator until
baseline execution becomes suitable for end-to-end validation.

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

| Priority | Tracking unit | SuperCollider | Record/Replay | Sampled | Inline Shadow | Evidence and next proof |
|---|---|---|---|---|---|---|
| P0 | `002_sk_mxf8gemm_explicit` | 🟩 70/70 accesses; current paired 1.04x | 🟩 70/70 accesses; 32/32 barriers; 4/4 fences; current paired 1.08x | 🟩 70/70 accesses; 28/28 barriers; current paired 1.06x | 🟩 70/70 accesses; 32/32 barriers; current paired 1.42x | Exact numeric oracle and complete analysis.  Current one-repetition paired measurements: `consan-revalidation-gfx1250-20260720-tensile-mxf8-exp-all-nullfix-011` and `consan-revalidation-gfx1250-20260720-tensile-mxf8-exp-rr-plannerfix-020`.  Prior artifacts: `consan-validation-gfx1250-tensile-mxf8-002`, `consan-validation-gfx1250-tensile-mxf8-rr-014`, `consan-validation-gfx1250-tensile-mxf8-sampled-016`, and `consan-validation-gfx1250-tensile-mxf8-inline-020`. |
| P0 | `003_sk_mxf4gemm_explicit` | 🟩 42/42 accesses; current paired 1.04x | 🟩 42/42 accesses; 32/32 barriers; 4/4 fences; current paired 1.00x | 🟩 42/42 accesses; 28/28 barriers; current paired 1.08x | 🟩 42/42 accesses; 32/32 barriers; current paired 1.24x | Exact numeric oracle; current one-repetition paired measurements: `consan-revalidation-gfx1250-20260720-tensile-mxf4-exp-all-nullfix-012` and `consan-revalidation-gfx1250-20260720-tensile-mxf4-exp-rr-plannerfix-019`.  Prior bundle: `consan-validation-gfx1250-tensile-mxf4-all-021`. |
| P1 | `037_spmm_tdm_f16_transposes` | 🟩 672/672 accesses; current paired 1.10x | 🟩 672/672 accesses; 176/176 barriers; current paired 1.19x | 🟩 672/672 accesses; 160/160 barriers; current paired 1.19x | 🟩 672/672 accesses; 176/176 barriers; current paired 1.79x | Four numeric clients cover tensor waits and 288 transpose LDS reads.  Current one-repetition paired artifact: `consan-revalidation-gfx1250-20260720-tensile-spmm-f16t-all-nullfix-013`.  Prior artifact: `consan-validation-gfx1250-tensile-spmm-transpose-all-027`. |
| P1 | `016_spmm_tdm_all` | 🟩 1610/1610 accesses; current paired 1.15x | 🟩 1610/1610 accesses; 512/512 barriers; current paired 1.24x | 🟩 1610/1610 accesses; 494/494 barriers; current paired 1.19x | 🟩 1610/1610 accesses; 256/256 barriers; strict-capacity current paired 1.61x | Multi-type exact numeric matrix covers both supported transpose widths.  Current one-repetition strict-capacity clean artifact `consan-green-expansion-20260721-spmm-tdm-all-inline-strict-168` selects a legal external-shadow lowering, passes every numeric oracle in 27.98 seconds, and has complete static and dynamic analysis with 1610/1610 accesses and 256/256 barriers.  Paired artifact `consan-green-expansion-20260721-spmm-tdm-all-inline-strict-overhead-169` accepts at 26,642.79 ms versus 16,579.08 and 16,495.80 ms controls, or 1.61x against their mean, with the same complete coverage.  Historical Inline artifacts `consan-revalidation-gfx1250-20260721-tensile-spmm-tdm-all-inline-shadow-densefix-062` and `-063` grew one instrumented workgroup to 175,104 bytes and remain debugging evidence only.  Artifact `consan-revalidation-gfx1250-20260720-tensile-spmm-all-all-nullfix-014` accepts the first three profiles. |
| P1 | `001_sk_mxf8f4gemm_tdm` | 🟩 768/768 accesses; current paired 1.12x | 🟩 768/768 accesses; 204/204 barriers; 24/24 fences; current paired 1.27x | 🟩 768/768 accesses; 180/180 barriers; current paired 1.22x | 🟩 768/768 accesses; 102/102 barriers; current paired 13.38x; reviewed exact-one fault and health accepted | Exact numeric oracle in every profile.  Current one-repetition paired artifacts: `consan-revalidation-gfx1250-20260720-tensile-mxf8f4-tdm-all-nullfix-015` and `consan-revalidation-gfx1250-20260720-tensile-mxf8f4-tdm-rr-plannerfix-021`; current Inline bundle at clean revision `9b9b12fc8c`: paired artifact `consan-green-expansion-20260721-mxf8f4-inline-current-paired-190`, inventory `...-current-inventory-188`, and reviewed fault `...-current-fault-pass-189`; prior artifacts `053`, `045`, `060`, `068`, and `069`. |
| P1 | `004_sk_mxf8gemm_tdm` | 🟩 992/992 accesses; current paired 1.20x | 🟩 992/992 accesses; 204/204 barriers; 24/24 fences; current paired 1.30x | 🟩 992/992 accesses; 180/180 barriers; current paired 1.23x | 🟧 Compute-active through 600, 1200, and 1800 seconds; no verdict | Current one-repetition paired artifacts: `consan-revalidation-gfx1250-20260720-tensile-mxf8-tdm-supercollider-016`, `consan-revalidation-gfx1250-20260720-tensile-mxf8-tdm-sampled-016`, and `consan-revalidation-gfx1250-20260720-tensile-mxf8-tdm-rr-plannerfix-021`.  Preserve the full denominator.  Latest Inline duration artifact: `consan-gfx1250-sk-mxf8-inline-079`. |
| P1 | `007_sk_mxf4gemm_tdm` | 🟩 2448/2448 accesses; current paired 1.35x | 🟩 2448/2448 accesses; 544/544 barriers; 64/64 fences; current paired 1.33x | 🟩 2448/2448 accesses; 480/480 barriers; current paired 1.38x | 🟧 Compute-active through 1800 seconds; no verdict | Current one-repetition paired artifacts: `consan-revalidation-gfx1250-20260720-tensile-mxf4-tdm-supercollider-018`, `consan-revalidation-gfx1250-20260720-tensile-mxf4-tdm-sampled-018`, and `consan-revalidation-gfx1250-20260720-tensile-mxf4-tdm-rr-plannerfix-021`.  The prior Record/Replay crash was a software-runtime call-displacement bug, not a ConSan encoding error.  Accepted artifacts: `073`, `075`, and `076`; Inline duration artifact: `078`. |
| P2 | Reduced `sk_sgemm_runtime_smoke` | 🟩 Exact numeric oracle; 640/640 accesses; current paired 1.07x | 🟩 Exact numeric oracle; 640/640 accesses; 22/22 barriers; 8/8 fences; current paired 1.32x | 🟩 Exact numeric oracle; 640/640 accesses; 40/40 barriers; current paired 1.34x | 🟩 Exact numeric oracle; 640/640 accesses; 22/22 barriers; paired 1.33x; causal fault diagnosed | Current one-repetition paired artifacts `consan-revalidation-gfx1250-20260720-tensile-sgemm-smoke-sc-022`, `consan-revalidation-gfx1250-20260720-tensile-sgemm-smoke-rr-022`, and `consan-revalidation-gfx1250-20260720-tensile-sgemm-smoke-sampled-022` accept.  The shared Tensile wrapper emits the machine-readable oracle required by fault acceptance.  Prior paired artifacts `313`, `315`, and `317` and reviewed exact-one barrier-drop artifacts `314`, `316`, and `326` retain fault and health evidence.  After the gfx1250 Inline barrier-bank fix, clean artifact `consan-green-expansion-20260721-sgemm-smoke-inline-bankfix-115` accepts the exact oracle with complete 640/640 access and 22/22 barrier coverage.  Paired artifact `consan-green-expansion-20260721-sgemm-smoke-inline-overhead-116` accepts at 1.33x against the mean of its two controls with the same complete coverage.  Fresh inventory `consan-green-expansion-20260721-sgemm-smoke-inventory-117` retains all reviewed selectors.  Late and entry fault artifacts `consan-green-expansion-20260721-sgemm-smoke-inline-fault-118` and `...-fault-early-119` apply exactly one logical mutation but preserve the oracle without a diagnostic, proving those barriers noncausal for the executed case.  The separately precommitted causal selector immediately before LDS consumption accepts in artifact `consan-green-expansion-20260721-sgemm-smoke-inline-fault-causal-121`: exactly one logical barrier mutation is applied as two final-byte rewrites, the independent numeric oracle fails, Inline emits one diagnostic, surviving coverage is complete at 640/640 accesses and 21/21 barriers, report evidence has no overflow or malformed state, and target health passes before and after.  The cell is green. |
| P2 | `000_sk_sgemm_quick` | 🟨 First problem: 12/12 exact numeric rows; 640/640 accesses; static/dynamic complete | 🟨 First problem exact and fully covered; aggregate host analysis fixed; full client is intrinsically execution-bound | 🟨 First problem: 12/12 exact numeric rows; 640/640 accesses; 40/40 barrier members | 🟧 First problem: 12/12 exact rows and complete static coverage; interrupted second problem leaves dynamic analysis incomplete | One-repetition Record/Replay artifact `consan-validation-gfx1250-tensile-sk-sgemm-quick-rr-agent-002` accepts the first benchmark problem with its numeric oracle and complete 640/640 accesses, 22/22 barriers, and 8/8 fences.  Commit `49df87def2` replaces quadratic owner annotation over 115,776 synchronization events with an identity index; artifact `consan-validation-gfx1250-tensile-sk-sgemm-quick-rr-indexed-359` then reaches aggregate execution.  Exact uninstrumented 300-second artifacts `consan-validation-gfx1250-tensile-sk-sgemm-quick-baseline-bounded-368` and `consan-validation-gfx1250-tensile-sk-sgemm-quick-rocjitsu-baseline-bounded-375` reach 1,104 and 1,105 passing rows respectively, proving that neither available software-emulation path materially accelerates the 648 solutions across six sizes.  The remaining acceptance route is a substantially longer unrestricted Record/Replay run.  Clean-tree one-repetition SuperCollider artifact `consan-green-expansion-20260721-sgemm-quick-sc-100` completes all 12 exact numeric rows in the first benchmark problem, covers 640/640 accesses with static and dynamic completeness, reports zero mismatches, and completes report cleanup before the client begins its second problem and reaches the unchanged 120-second bound.  That is clean-partial yellow evidence, not full-client acceptance.  Clean-tree one-repetition Sampled artifact `consan-green-expansion-20260721-sgemm-quick-sampled-101` reaches the same boundary: 12/12 exact numeric rows with zero failures, 640/640 accesses, all 40/40 barrier members, static and dynamic completeness, and report cleanup before the second problem reaches 120 seconds.  Inline artifact `consan-green-expansion-20260721-sgemm-quick-inline-102` also completes the first problem's 12/12 exact rows with zero failures and patches 640/640 accesses plus 22/22 barriers, but its interrupted second problem leaves 128,413,696 dynamic checks unresolved; aggregate analysis is therefore incomplete and the cell stays orange.  No filters, caps, manual registers, or extra repetitions were used. |
| P2 | `005_sk_f8gemm_quick` | 🟩 Exact oracle; 1772/1772 accesses; current paired 1.43x; reviewed fault and health accepted | 🟩 Exact oracle; 1772/1772 accesses; 44/44 barriers; 16/16 fences; current paired 8.00x | 🟧 Current clean execution remains compute-active through 900 seconds; no verdict or measured overhead | 🟧 Current tip executes 49 exact rows with zero failures before the fixed 180-second bound | Current one-repetition Record/Replay artifact `consan-revalidation-gfx1250-20260720-tensile-f8-rr-serial-023` accepts its 729.69-second instrumented run between stable 91.45- and 91.02-second controls.  Current Sampled artifact `consan-revalidation-gfx1250-20260720-tensile-sk-f8gemm-quick-sampled-clean-rocjitsu-038` reaches its 900-second bound inside the instrumented client without a verdict.  The focused regression proves that `ds_bpermute_b32` is not raceable LDS traffic.  SuperCollider clean artifact `224` supersedes stale pre-fix artifact `081`; prior paired artifact `352` and reviewed exact-one attempt `353` leave that cell yellow because postflight health fails.  Bounded alternate-site artifact `consan-green-expansion-20260721-tensile-f8-sc-fault-alternate-independent-029` drops the later `0xd8c0`/`0xd8c4` logical pair, visibly corrupts multiple numeric rows without a SuperCollider diagnosis, and again loses postflight health.  This rules out a first-site-only explanation; the precommitted pass-oracle expectation is not changed after observation, and the cell rotates without a third site.  Record/Replay reviewed fault artifact `331` accepts.  Sampled paired artifact `335` and reviewed exact-one barrier-drop artifact `334` retain the prior historical bundle.  Current-tip SuperCollider fault artifact `consan-green-expansion-20260721-tensile-f8-sc-fault-current-125` applies the reviewed whole barrier mutation once, preserves the exact oracle with the precommitted no-diagnosis outcome, retains complete 1,772/1,772 access coverage, and passes health before and after.  Same-revision paired artifact `consan-green-expansion-20260721-tensile-f8-sc-overhead-current-126` accepts 43,458.36/65,509.16/47,846.30 ms, or 1.43x against the mean baseline, with complete coverage.  The SuperCollider cell is green.  Current-tip one-repetition Inline artifact `consan-green-expansion-20260721-tensile-f8-inline-current-123` clears that obsolete launch boundary and executes 49 exact numeric rows with zero failures before its fixed 180-second bound.  The full client has no verdict or accepted overhead, so the cell advances only to orange and rotates. |
| P2 | `006_sk_hgemm_quick` | 🟧 136 exact numeric passes with zero failures; first 143-solution problem remains active at 300 seconds | 🟩 Exact oracle; 8162/8162 accesses; 292/292 barriers; 80/80 fences; current paired 2.02x | 🟩 Exact oracle; 8162/8162 accesses; 544/544 barriers; current paired 2.24x | 🟧 Current tip executes 189 exact rows with zero failures before the fixed 180-second bound | Current-tip Record/Replay dispatch isolation is commit `6270cbbfd2`.  One-repetition paired artifact `consan-green-expansion-20260721-hgemm-rr-dispatchfix-paired-193` accepts at 2.02x with complete coverage; reviewed fault artifact `...-dispatchfix-fault-195` applies exactly one unchanged late-barrier mutation, matches its frozen pass/no-diagnosis contract, retains complete surviving analysis, cleans report memory, and passes health before and after.  Earlier paired artifacts `consan-revalidation-gfx1250-20260720-tensile-hgemm-rr-017` and `consan-revalidation-gfx1250-20260720-tensile-hgemm-sampled-017` accept.  Clean-tree SuperCollider artifact `consan-green-expansion-20260721-tensile-hgemm-sc-clean-independent-053` reaches 136 exact passes with zero failures inside a fixed 300-second bound, but does not finish the first 143-solution problem or supply both applicable-object records.  It supersedes artifact `225`'s 150-second duration evidence and rotates without another timeout increase.  Sampled paired artifact `342` and reviewed fault `341` retain prior acceptance.  Current-tip one-repetition Inline artifact `consan-green-expansion-20260721-tensile-hgemm-inline-current-124` clears that obsolete launch boundary and executes 189 exact numeric rows with zero failures before its fixed 180-second bound.  The full client has no verdict or accepted overhead, so the cell advances only to orange and rotates. |
| P3 | `015_spmm_f8_ml` stress | 🟧 First contraction exact numeric pass; 298/4316 accesses; second orientation active at 120 seconds | 🟨 19 exact passes; two objects fully covered; relay-window fix unit-complete | 🟨 Three clients exact; all objects fully covered; paired/fault acceptance remains | 🟧 Exact failing kernel fixed; standard run has 8 passes and zero failures before its bound | Dense transpose/sub-dword LDS and full-register stress.  Current-tip one-repetition artifact `consan-validation-gfx1250-tensile-spmm-f8-ml-rr-current-367` reaches 19 exact passes with zero failures before its 600-second bound.  Its two completed applicable objects are fully static/dynamic complete at 22,074/22,074 and 19,960/19,960 accesses plus 403/403 barriers each; all prior `missing_owner` and resource failures are gone.  The interrupted third object has zero resource failures but 1,875 access and one barrier `instrumentation_patch_missing` gaps.  The generic relay-window partition fix for its over-wide owner passes the focused 4/4 tests and the rebuilt full 692/692 ConSan suite.  Rebuilt-hook artifact `consan-green-expansion-20260721-spmm-f8-ml-rr-001` fully patches the first object at 22,074/22,074 accesses and 403/403 barriers with no resource or placement failures, but the software GPU remains inside that first client at 600 seconds without a numeric row; the cell therefore remains yellow and rotates without widening the bound.  Current clean Sampled artifact `consan-green-expansion-20260721-spmm-f8-ml-sampled-independent-057` passes two exact rows before its 150-second validation bound.  Reusing its generated client removes generation time.  Two gfx1250 state-preservation fixes close the former assertion.  First, ConSan now distinguishes the instruction's packed previous/current VGPR-bank transition from its persistent low-byte mode.  Second, every gfx1250 Sampled barrier cave explicitly establishes the low bank before touching scratch VGPRs: the patched call can be immediately followed by a guest bank update that takes effect during control transfer, including when the replaced barrier's static mode was zero.  The focused 78-test Sampled/transition slice passes.  The unrestricted one-repetition exact client now returns normally with three numeric passes and complete 19,960/19,960 access plus 806/806 barrier-member coverage; the separately filtered 1,024-block variant also returns normally with three passes.  No patch cap, manual register, or kernel filter is used for the unrestricted result.  The cell is yellow; paired overhead, reviewed fault, containment, and frozen provenance remain before green.  One-repetition paired attempt `consan-green-expansion-20260721-spmm-f8-ml-sampled-overhead-103` reaches its fixed 180-second instrumented bound with only one of two required applicable-object records, so it supplies no accepted overhead and rotates without a wider timeout.  Clean-provenance paired artifact `consan-green-expansion-20260722-spmm-f8-ml-sampled-overhead-199` completes stable 226,952.48/241,620.19-ms one-repetition controls.  Sampled records 15 exact passes before its 300-second bound and fully covers its first two objects, but a newly reached third object patches 9,279/11,154 accesses and 378/380 barrier members.  Of its 1,875 access gaps, 1,870 are inside a relocated prefix, localizing the missing Sampled equivalent of Record/Replay's relay-window partition; no overhead is claimed.  The narrow partition fix now passes the new two-window regression, the 86-test Sampled/relay slice, and all 723 ConSan tests.  Current-hook log `/tmp/consan-spmm-sampled-third-relayfix-20260722.log` then completes the formerly incomplete third client with five exact passes, complete 11,154/11,154 access and 380/380 barrier-member coverage, 99 visible evidence records, and complete static/dynamic analysis.  Standard-profile Inline artifact `consan-green-expansion-20260721-spmm-f8-ml-inline-clean-108` fully patches the first applicable object at 22,074/22,074 accesses and 403/403 barriers, then records eight exact passes and five numeric failures before its fixed 180-second bound.  Patch-frontier diagnostics isolate the first failure to the ninth barrier, immediately before a guest VGPR-bank update; access-only instrumentation remains correct beyond that frontier.  Commit `e1bbd2608a` makes every gfx1250 Inline barrier cave establish its low scratch bank explicitly.  The formerly failing exact kernel then passes with complete 1,639/1,639 access and 25/25 barrier coverage.  Clean committed-tip standard artifact `consan-green-expansion-20260721-spmm-f8-ml-inline-fixed-clean-113` records eight exact passes, zero failures, and complete first-object static coverage before its unchanged 180-second bound.  Because the standard row still lacks a final verdict, the Inline cell improves within orange and rotates rather than claiming clean completion or widening the timeout. |
| P2 | `019_spmm_f16_sb` closure | 🟧 9,546/9,546 accesses patched; first client exceeds 300 seconds without a numeric row | 🟩 Four exact orientations; 31,265/31,265 accesses; current paired 2.48x | 🟧 9,546/9,546 accesses and 646/646 applicable barriers patched; first client exceeds 300 seconds without a numeric row | 🟧 9,546/9,546 accesses and 323/323 barriers patched; first client exceeds 300 seconds without a numeric row | Current one-repetition clean artifacts `consan-green-expansion-20260721-spmm-f16-sb-supercollider-001`, `-sampled-001`, and `-inline-001` prove complete static lowering but each reaches its fixed 300-second bound before the first numeric result, so they rotate without overhead attempts.  Current one-repetition paired Record/Replay artifact `consan-revalidation-gfx1250-20260720-tensile-spmm-f16-sb-rr-017` accepts.  Prior artifact `consan-validation-gfx1250-tensile-spmm-f16-sb-rr-inferred-all-365` accepts all four exact numeric orientations with complete static and dynamic analysis, 31,265/31,265 accesses, 1,141/1,141 barriers, all 35 `ds_store_b16` sites, and zero resource failures.  Frozen reviewed artifact `consan-validation-gfx1250-tensile-spmm-f16-sb-rr-fault-377` retains fault and health evidence. |
| Survey | Remaining Tensile configurations | 🟩 Architecture-level decoded opcode union covered by accepted selected rows | 🟩 Architecture-level decoded opcode union covered by accepted selected rows, including full `019_spmm_f16_sb` bundle | 🟩 Architecture-level decoded opcode union covered by accepted selected rows | 🟩 Architecture-level decoded opcode union covered by accepted selected rows | No decoded atomics, asynchronous waits, or named-barrier forms were found.  The accepted selected rows plus the complete `019_spmm_f16_sb` Record/Replay bundle cover and freeze the architecture-level decoded opcode union.  These green survey cells record corpus selection completeness, not four additional runtime-acceptance cells. |

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

| Priority | Tracking unit | SuperCollider | Record/Replay | Sampled | Inline Shadow | Shared evidence and next proof |
|---|---|---|---|---|---|---|
| P0 | PyTorch/Triton tensor-descriptor add, one-CTA and two-CTA variants | 🟩 Exact `a + b`; 29/29 accesses; current paired 1.19x | 🟩 Exact `a + b`; 29/29 accesses; 12/12 barriers; current paired 1.33x | 🟩 Exact `a + b`; 29/29 accesses; 20/20 applicable barriers; current paired 2.16x | 🟩 Exact `a + b`; 29/29 accesses; 12/12 barriers; current paired 4.16x | Same-tip SuperCollider clean artifact `199` and paired bundle `200` cover 29/29 accesses after adding byte-LDS checks; both baselines pass and the maximum measured slowdown is 1.05x.  Current one-repetition paired measurements are in `consan-revalidation-gfx1250-20260720-tdm-nullfix-009`.  Reviewed wait-drop bundle `201` applies exactly one mutation, observes the specified no-diagnosis/pass-oracle outcome, and passes containment health.  Its aggregate fault analysis still labels unrelated non-target code objects invalid because the exact-one mutation guard is evaluated per loaded object; the clean and paired runs establish complete instrumentation coverage.  MOI paired bundle `192` and reviewed bundle `194` retain the other three green cells.  This proves clustered dispatch, not cluster-memory instructions. |
| P0 | `torch.mode`, large rows | 🟩 Exact values/indices; 28,195/28,195 accesses; current paired 120.67x | 🟩 Exact values/indices; 28,939/28,939 accesses and 4,446/4,446 barriers; current paired 232.20x | 🟩 Exact values/indices; 28,939/28,939 accesses and 8,892/8,892 barrier members; current paired 203.53x | 🟩 Exact values/indices; 28,939/28,939 accesses and 4,446/4,446 barriers; current paired 341.90x | Commit `dacb1d3b05` extends the shared-call dispatcher to large majority-stranded objects.  Frozen clean artifact `252` passes the unfiltered SuperCollider oracle with static/dynamic completeness.  Paired artifact `253` and reviewed fault `260` complete its bundle.  Current one-repetition paired measurements are in `consan-revalidation-gfx1250-20260720-mode-sc-004` and `consan-revalidation-gfx1250-20260720-mode-rr-nullfix-009`.  Current-tip one-repetition Record/Replay clean artifact `348` accepts the unfiltered exact oracle in 34.3 seconds with every supported access and barrier.  Paired artifact `349` accepts both baselines and the complete instrumented run, measuring 30,548 ms versus 104--108 ms baselines.  Reviewed barrier-drop artifact `351` applies exactly one mutation, preserves the exact oracle, matches the precommitted no-diagnosis outcome, covers 28,939/28,939 accesses and all 4,445 surviving barriers, and passes containment health.  The decoder and lowering preserve DS byte offsets and support tagged gfx1250 LDS address tokens; the two executed no-return releases have no compatible same-owner acquire consumer and are typed not applicable as ordering roles while remaining covered as ordinary accesses.  Commit `a6721f8e76` corrects final semantic validation to recognize already-supported atomic access candidates in workgroup-local exact shadows.  Clean-tree one-repetition Inline artifact `consan-green-expansion-20260721-pytorch-mode-inline-atomic-validation-independent-055` passes the exact values/indices oracle in 32.26 seconds with complete 28,939/28,939 access and 4,446/4,446 barrier coverage.  A controlled 32-to-128 exact-bank experiment leaves all 13,342 undercoverage events unchanged, disproving publication contention.  Object and dispatch evidence instead identifies a dynamic-LDS sizing gap: the kernel descriptor declares 2 fixed bytes and the dispatch supplies 1,540 group bytes, while the descriptor-sized local mirror covers only the fixed allocation.  Commit `af3b46a020` inventories the hidden dynamic-LDS argument, selects the external exact-shadow table for those owners, and uses entry-snapshotted private owner/epoch state across guest VGPR-bank transitions.  Clean-tree one-repetition artifact `consan-green-expansion-20260721-pytorch-mode-inline-dynamic-clean-065` passes the exact oracle in 33.80 seconds with complete 28,939/28,939 access and 4,446/4,446 barrier coverage, zero dynamic undercoverage, and no diagnostics or overflow.  Paired artifact `consan-green-expansion-20260721-pytorch-mode-inline-dynamic-overhead-066` accepts 99.12/32,739.50/92.39 ms, or 341.90x against the mean baseline, while repeating complete coverage.  Fresh inventory `...-dynamic-inventory-067` retains the reviewed selector.  Fault artifact `...-dynamic-fault-068` applies exactly one whole barrier mutation, preserves the exact oracle with the precommitted no-diagnosis outcome, covers every supported access and all 4,445 surviving barriers without undercoverage, and passes target health before and after.  Commit `2dea78db37` makes Sampled planning near-linear and admits structurally proven long same-block split barriers.  Clean-tree artifact `consan-green-expansion-20260721-pytorch-mode-sampled-semantic-clean-071` passes the exact oracle in 21.72 seconds with static/dynamic completeness, 28,939/28,939 accesses, and 8,892/8,892 barrier members.  Paired artifact `consan-green-expansion-20260721-pytorch-mode-sampled-overhead-072` accepts 96.82/19,932.30/99.05 ms, or 203.53x against the mean baseline, with complete coverage.  Fresh inventory `...-sampled-inventory-073` retains the reviewed selector.  Fault artifact `...-sampled-fault-074` applies exactly one whole barrier mutation, preserves the exact oracle with the precommitted no-diagnosis outcome, covers every supported access and all 8,890 surviving barrier members, bounds peak report memory at 43.3 MB with complete cleanup, and passes target health before and after.  Filters and smaller inputs remain diagnostic only. |
| P0 | `torch.topk`, FP64 spill and BF16 coverage cases | 🟩 Exact FP64/BF16 values and indices; 160,956/160,956 accesses; current paired 903.20x maximum; reviewed exact-one fault and health accepted | 🟧 Current tip transforms both large objects, then signals before an oracle; older exact run covered 160,345/161,136 accesses and all 11,423 barriers | 🟨 Exact FP64/BF16 values and indices; dynamically complete; 102,598/161,136 accesses and 15,182/15,182 barriers | 🟧 Both large objects finish patching; client signals during execution before an oracle | At clean revision `e60b5f0239`, one-repetition paired SuperCollider artifact `consan-green-expansion-20260721-pytorch-topk-sc-overhead-complete-178` passes both exact FP64/BF16 value-and-index oracles with complete static and dynamic 160,956/160,956 access coverage. It measures FP64 at 154,648.503 ms against a 171.223-ms paired control (903.198x) and BF16 at 96.877 ms against 70.387 ms (1.376x). Fresh inventory `consan-green-expansion-20260721-pytorch-topk-inventory-complete-179` retains 5,391 sequences and 10,782 sites. The first independently precommitted selector in artifact `...-sc-fault-complete-181` applied exactly once but preserved the exact oracle rather than failing it, so that frozen expectation remains correctly rejected. Before a second observation, the independent result justified a fresh pass-oracle/no-diagnosis contract for the next double-gather barrier pair. Artifact `consan-green-expansion-20260721-pytorch-topk-sc-fault-complete-182` accepts it: exactly one logical mutation is applied, the exact oracle passes without a SuperCollider diagnosis as precommitted, surviving analysis is complete at 160,956/160,956 accesses, execution stays within its 300-second command bound, report cleanup is complete, and independent target health passes before and after. The SuperCollider cell is green. Preplanned branch-only fallback, signed 16-bit loads, and 96-bit store readback close the former placement, admission, and unsupported-access gaps. Record/Replay artifact `consan-validation-gfx1250-pytorch-topk-rr-collapsed-364` passes both exact oracles in 308.3 seconds and leaves 791 `forbidden_overlap` resource failures across eight `gatherTopK` containers; exact-object artifact `consan-topk-rr-offline-dumps-047` localizes them. Current-tip exact artifacts `consan-green-expansion-20260721-pytorch-topk-rr-ownerlocal-clean-196` and `consan-green-expansion-20260722-pytorch-topk-rr-spill-dispatch-clean-197` instead signal after both objects transform, before either oracle or a final verdict; the narrower spill-backed dispatch-state experiment did not move the boundary and was reverted. Artifact `...-rr-xcnt-clean-198` showed that its three waitcheck warnings originate in the unmodified PyTorch object, not generated ConSan spill code; the redundant translation-drain experiment was also reverted. Sampled artifact `consan-green-expansion-20260721-pytorch-topk-sampled-scaled-clean-095` passes both exact oracles in 80.30 seconds, covers 102,598/161,136 accesses and all 15,182 barriers, and has no forbidden diagnostics or overflow. Inline artifact `consan-green-expansion-20260721-pytorch-topk-inline-bankfix-114` reproduces the execution signal at 116.7 seconds after both large transformations, independently of the fixed barrier-bank issue. |
| P1 | `torch.sort` over segmented rows | 🟩 Exact values/indices; 48,224/48,224 accesses; current paired 184.68x | 🟩 Exact values/indices; 48,224/48,224 accesses and 6,032/6,032 barriers; current paired 370.29x | 🟩 Exact values/indices; 48,224/48,224 accesses and 12,064/12,064 barrier members; current paired 171.77x; reviewed noncausal fault accepted | 🟩 Exact values/indices; 48,224/48,224 accesses and 6,032/6,032 barriers; current paired 416.22x | SuperCollider commit `b00563cd31` replaces the impossible one-relay-per-site layout with one shared call dispatcher per kernel and covers every native LDS/VDS shape in this object.  Frozen clean artifact `241` passes the exact values/indices oracle with complete 48,224/48,224 access coverage.  Current one-repetition paired measurements are in artifacts `consan-revalidation-gfx1250-20260720-sort-sc-004`, `consan-revalidation-gfx1250-20260720-sort-rr-nullfix-009`, and `consan-revalidation-gfx1250-20260720-sort-inline-nullfix-010`.  Paired artifact `242` accepts both baselines and the profile run, measuring 27,003 ms versus a 121 ms paired baseline.  Reviewed barrier-drop artifact `246` applies exactly one mutation, observes the precommitted no-diagnosis/pass-oracle result, and passes both health gates.  Record/Replay and Inline evidence remains in clean artifacts `204`/`209`, paired artifacts `205`/`210`, and reviewed faults `208`/`211`.  Current-tip Sampled clean artifact `consan-green-expansion-20260721-pytorch-sort-sampled-gated-current-127` accepts in 21.51 seconds with the exact values/indices oracle, complete 48,224/48,224 access coverage, complete 12,064/12,064 barrier-member coverage, and complete analysis.  Same-revision one-repetition paired artifact `consan-green-expansion-20260721-pytorch-sort-sampled-overhead-current-128` accepts 134.44/19,985.76/134.04 ms, or 148.88x against the mean baseline, with the same complete coverage.  Reviewed current-tip barrier-drop artifacts `consan-green-expansion-20260721-pytorch-sort-sampled-fault-current-129` and `consan-green-expansion-20260721-pytorch-sort-sampled-fault-late-130` each apply exactly one distinct logical barrier mutation, preserve the exact oracle without a diagnostic, retain complete surviving coverage, and pass health before and after.  Their frozen fail-oracle contracts remain unchanged historical evidence.  At clean revision `31c3d937c2`, artifacts `consan-green-expansion-20260721-sort-sampled-final-clean-164` and `...-final-overhead-165` reconfirm the exact oracle and complete coverage, measuring 25,253.78 ms versus a 147.02-ms control, or 171.77x.  Fresh inventory `...-final-inventory-166` retains the first selector.  Before execution, the two independent prior pass-oracle results justify a new reviewed pass-oracle/no-diagnosis contract; fault artifact `...-current-fault-163` then applies that selector exactly once, preserves the oracle without a diagnostic, retains complete surviving coverage, and passes target health before and after.  The Sampled cell is green. |
| P1 | Collision-heavy `torch.scatter_reduce` (`sum`, BF16 and FP32) | 🟩 Exact collision sums; 23/23 accesses; current paired 24.37x | 🟩 Exact collision sums; 23/23 accesses; current paired 42.30x | 🟩 Exact collision sums; 23/23 accesses; current paired 41.91x | 🟩 Exact collision sums; 23/23 accesses; current paired 40.17x | Current-tip clean artifact `322` accepts all four profiles after making record visibility conditional for a workload with no applicable executed LDS events.  Current one-repetition paired measurements are in `consan-revalidation-gfx1250-20260720-scatter-004`.  Paired artifacts `305`, `306`, `323`, and `324` accept both baselines and complete instrumentation.  Reviewed artifacts `329` and `330` record atomic-order and atomic-scope weakening as typed N/A in all four profiles: inventory proves the relaxed singleton reduction atomics have no synchronization edge to weaken. |
| P1 | `torch.histc` with a shared-memory-sized bin count | 🟩 Exact counts; 133/133 supported accesses; current paired 60.11x | 🟩 Exact counts; 175/175 accesses and 84/84 barriers; current paired 72.00x | 🟩 Exact counts; 175/175 accesses and 168/168 applicable barriers; current paired 67.37x | 🟩 Exact counts; 175/175 accesses and 84/84 barriers; current paired 85.86x | Current-tip clean artifacts `289`-`291` admit 42 relaxed LDS-atomic accesses while correctly typing their synchronization role as not applicable; every MOI engine is statically and dynamically complete.  Current one-repetition paired measurements are in artifacts `consan-revalidation-gfx1250-20260720-histc-004`, `consan-revalidation-gfx1250-20260720-histc-rr-nullfix-007`, `consan-revalidation-gfx1250-20260720-histc-sampled-nullfix-010`, and `consan-revalidation-gfx1250-20260720-histc-inline-nullfix-010`.  Frozen paired artifacts `296`, `300`, `302`, and `308` accept both baselines and complete instrumentation.  Reviewed barrier-drop artifacts `299`, `301`, `303`, and `309` each apply exactly one mutation, fail the exact oracle, retain complete supported-access plus all surviving-barrier coverage, and pass containment health; Inline emits 60 diagnostics while the other profiles preserve their precommitted no-diagnosis outcomes. |
| P2 | `torch.linalg.vector_norm` and large-row `torch.softmax` | 🟩 Exact norm/softmax; 4,756/4,756 accesses; current paired 315.57x | 🟩 Exact norm/softmax; 4,756/4,756 accesses and 2,352/2,352 barriers; current paired 435.17x | 🟩 Exact norm/softmax; 4,756/4,756 accesses and 4,572/4,572 barriers; current paired 534.97x | 🟩 Exact norm/softmax; 4,756/4,756 accesses and 2,352/2,352 barriers; current paired 317.24x | Current one-repetition artifacts use the corrected software-GPU environment.  Clean-tree paired SuperCollider artifact `consan-green-expansion-20260721-norm-softmax-supercollider-overhead-independent-011` accepts 108.98/35,541.53/116.27 ms and repeats the exact oracle with complete access coverage.  Reviewed fault artifact `...-supercollider-fault-independent-010` applies exactly one whole barrier mutation, observes the precommitted no-diagnosis/pass-oracle result, and passes discovery plus exact target-dispatch health before and after.  Record/Replay clean artifact `consan-green-expansion-20260721-norm-softmax-rr-independent-012` accepts the exact oracle with complete 4,756/4,756 access and 2,352/2,352 barrier coverage in 43.30 seconds.  Clean-tree paired artifact `...-rr-overhead-independent-013` accepts 96.46/42,534.82/99.02 ms, or 435.17x against the mean baseline, and repeats complete coverage.  Reviewed artifact `...-rr-fault-independent-014` applies exactly one whole barrier mutation, observes the precommitted no-diagnosis/pass-oracle result, retains complete surviving-site coverage, and passes both exact target-dispatch health gates.  Sampled clean artifact `consan-green-expansion-20260721-norm-softmax-sampled-literal-dispatch-independent-048` accepts the exact oracle with complete 4,756/4,756 access and 4,572/4,572 barrier coverage in 70.14 seconds.  Clean-tree paired artifact `...-sampled-overhead-independent-050` accepts 131.65/67,498.39/120.69 ms, or 534.97x against the mean baseline, and repeats complete coverage.  Reviewed artifact `...-sampled-fault-independent-051` applies exactly one whole barrier mutation, observes the precommitted no-diagnosis/pass-oracle outcome, retains complete 4,756/4,756 surviving accesses and 4,570/4,570 barriers, bounds report memory at 9,295,520 bytes with complete cleanup, and passes exact target-dispatch health before and after.  Commit `19840819c2` uses the report's literal dispatch identity for gfx1250 Sampled probes when persistent scalar pressure leaves no pair; Inline Shadow and other architectures retain their existing persistent-register paths.  Commit `e2c1e026bc` extends that literal-identity strategy to non-atomic gfx1250 Inline probes.  Clean-tree one-repetition artifact `consan-green-expansion-20260721-norm-softmax-inline-literal-dispatch-independent-052` passes the exact oracle in 35.17 seconds and recovers 28 access plus 18 barrier sites.  The former 474 access and 189 barrier resource failures were scalar `forbidden_overlap` exclusions: Inline lacked component-scoped EXEC-save SGPR spill assignment.  They were not VGPR-allocation failures.  The component-scoped implementation now applies one owner assignment consistently to access, barrier, and entry lowering; local-LDS-shadow owners require spill-backed state, while safe site-dead windows remain available only to external-shadow owners.  Commit `4bfa285247` retains the fix.  Clean-tree one-repetition artifact `consan-green-expansion-20260721-norm-softmax-inline-component-spill-clean-079` passes the exact oracle in 37.19 seconds with static/dynamic completeness, all 4,756 accesses, all 2,352 barriers, and none of the prior 1,022 dynamic-undercoverage events.  Clean-tree paired artifact `consan-green-expansion-20260721-norm-softmax-inline-component-spill-overhead-clean-081` accepts 107.31/36,620.42/123.56 ms, or 317.24x against the mean baseline, while repeating complete coverage. Fresh inventory artifact `...-component-spill-inventory-082` retains the reviewed whole-barrier selector. Fault artifact `...-component-spill-fault-083` applies that selector but reaches its 120-second command bound, so it is not accepted. Reviewed fault artifact `...-component-spill-fault-084` applies exactly one logical barrier mutation as two instruction rewrites, observes the precommitted no-diagnosis/pass-oracle result, retains complete surviving coverage at 4,756/4,756 accesses and 2,351/2,351 barriers, cleans all 25,839,568 report bytes, and passes exact target health before and after. The cell is green.  Commit `8d0366b5a8` closes the separate undersized-descriptor Inline object, which now patches all 1,172 accesses and 624 barriers.  The slower-emulator Record/Replay attempt `consan-revalidation-gfx1250-20260720-pytorch-norm-softmax-record-replay-clean-rocjitsu-038` reaches its 600-second bound. |
| P1 | PyTorch cluster synchronization | 🟩 Exact oracle; 25/25 applicable accesses; current paired 1.02x | 🟩 Exact oracle; 25/25 accesses and 2/2 barriers; current paired 1.03x | 🟩 Exact oracle; 25/25 accesses and 4/4 barrier members; current paired 1.07x | 🟩 Exact oracle; 25/25 accesses and 2/2 barriers; current paired 1.24x | Commit `b9af8082f7` gives each of 512 cluster threads a distinct LDS element, creating a causal LDS store/barrier/load window without intentional duplicate writers.  Clean all-profile artifact `consan-green-expansion-20260721-cluster-causal-all-clean-independent-034` accepts the exact oracle with complete static and dynamic evidence in all four modes.  Inventory `...-cluster-causal-inventory-independent-036` distinguishes the selected `barrier_id=-3`, cluster-scoped pair from a compiler-added workgroup pair.  SuperCollider, Record/Replay, and Inline Shadow retain clean, paired, reviewed-fault, and containment acceptance in artifacts `034`, `035`, and `037`.  Commit `a7d2128db3` prevents Sampled dense relay hosts from overlapping pre-applied fault ranges.  Same-revision clean artifact `...-cluster-causal-sampled-clean-fixed-independent-039` accepts 25/25 accesses and 4/4 barrier members.  One-repetition paired artifact `...-cluster-causal-sampled-paired-fixed-independent-040` accepts 464.07/499.42/468.20 ms, or 1.07x against the mean baseline.  Reviewed artifact `...-cluster-causal-sampled-fault-fixed-independent-038` applies the exact whole cluster pair once, preserves the exact oracle with the precommitted no-diagnosis outcome, and passes containment health before and after. |
| Survey | Cluster-memory and inter-workgroup synchronization from PyTorch | 🟩 Executable cluster-scope synchronization full bundle accepted | 🟩 Executable cluster-scope synchronization full bundle accepted | 🟩 Executable cluster-scope synchronization full bundle accepted | 🟩 Executable cluster-scope synchronization full bundle accepted | Clustered placement was already covered.  The concrete row above now supplies a causal LDS access window around callable cluster-scope synchronization with complete clean, paired, reviewed-fault, containment, and provenance evidence in all four modes.  The lowering still uses ordinary LDS and global instructions, so it is not claimed as evidence for a distinct cluster-memory opcode. |

## Environment baseline

This table establishes that the development environment can execute target
code.  It is not instrumentation acceptance evidence.

| Item | Current evidence |
|---|---|
| Port branch | `users/bjacob/consan` |
| Rebased foundation | `origin/develop` merge-base `8bba8691911ad19ce58bef2ac252ac497331fc5f` |
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
| Generated gfx1250 decode | 🟨 Hook-integrated baseline | RocJitsu contains a generated gfx1250 decoder and instruction builders.  The ConSan hook now links that backend and reaches target code-object analysis; ConSan-specific shape and semantic qualification is active. |
| ConSan instruction emission | 🟨 Active | Target-generated VDS/VFLAT tests prove SuperCollider LDS load and group-flat store readback, including signed and zero-extending byte loads, low/high-half byte and 16-bit stores, stride-64 two-address stores, target waits and compares, marker-report emission, and final validation.  Dense SuperCollider objects now use a structurally validated per-kernel call-return dispatcher and relocated entry host; the real 280-kernel `torch.sort` object reaches complete 48,224/48,224 coverage.  The sub-dword store path reserves one extra temporary without clobbering guest data; full-pressure regressions verify spill-private and descriptor updates.  Record/Replay lowers real VFLAT accesses, shared owner/epoch prologues, and inline barrier epoch updates.  Its gfx1250 dense-access path replaces each adjacent access with one `s_call_i64`, shares a return-PC dispatcher, and can relocate one kernel-entry host when no compiler NOP island exists; the real FP64 top-k kernel improves from 44/106 to 106/106 without changing other architectures.  Sampled and Inline Shadow now lower access publication, compare-and-swap, 64-bit atomics, carry-chain address arithmetic, and lane-read operations.  Inline local shadows initialize once per descriptor-declared workgroup dimensionality, omit external-only workgroup-key state, branch around empty cold paths, and reject read/read traffic before owner/epoch extraction.  A forced far-island gfx1250 test proves that a deferred guest load precedes its relocated continuation.  A packed 1000-site Sampled regression proves demand-sized gate reservations keep the full Qwen access inventory branch-reachable.  Ordinary and AMD extended dispatch packets now both propagate spill-private requirements with compile-time ABI-offset checks.  Remaining synchronization families remain open. |
| Register allocation and spilling | 🟨 Active | Wave32 descriptor allocation uses 16-VGPR granules: field 4 yields an 80-VGPR boundary.  TP1 Inline Shadow executes all 72 spill-backed access patches, including an 18-VGPR live window, after growing a zero-private kernel to 152 bytes per lane; SuperCollider independently recovers its 72 high-pressure sites with seven-VGPR windows and grows private storage through 60 bytes per lane.  Qwen Record/Replay now shares spill-private epoch state across its entry, access, and barrier probes, replacing an unbounded dynamic barrier trace with bounded per-wave epochs while retaining 1000/1000 accesses and 92/92 barriers.  These profiles preserve their independent oracles and pass static/dynamic completeness.  The software GPU required an internal fix to honor descriptor-grown private size when its dispatch packet remained zero.  Partial-EXEC, shared-owner, standalone boundary, and remaining-site-kind evidence remain open. |
| Validation target | 🟨 Green expansion active | The registry resolves every gfx1250 workload artifact and the full doctor passes.  Current-tip revalidation retains 92 accepted runnable cells with one-repetition paired overheads, while one attempted runnable cell remains explicitly non-green, no current runnable cell is typed N/A, and six green aggregate/survey indicators have no runtime denominator.  New bounded work is recorded in the matrix and `/tmp/CONSAN_GFX1250_REVALIDATION_PROGRESS.md`; no blanket four-profile acceptance is claimed. |
| Four focused flavor verticals | 🟨 Four-engine bootstrap | A clean ping-pong cooperative-LDS workload passed its host-reference oracle with 4/4 accesses patched by SuperCollider.  Record/Replay passed with 4/4 accesses and 8/8 barriers patched and visible records.  Sampled passed with 4/4 accesses and two visible records, but 0/8 barriers.  Inline Shadow is statically and dynamically complete with 4/4 accesses, 8/8 barriers, and one visible record.  Fault/diagnostic behavior, replay qualification, and Sampled synchronization remain open. |

## Progress log

- 2026-07-22: Narrowed the current Qwen Sampled exception without another
  unrestricted timeout.  The historical accepted and current runs use the
  same byte-identical VMFB and both execute the 151,936-workgroup output
  initializer.  A current-tip, one-repetition diagnostic restricted to that
  kernel passes the exact output oracle in roughly 45 seconds with complete
  3/3 access and 4/4 barrier-member coverage and 16 visible events.  This is
  diagnostic evidence rather than matrix acceptance because it uses the test
  kernel filter.  It disproves the final initializer as a standalone blocker
  and points to cumulative full-object overhead, particularly linear dense
  return-PC dispatch before Sampled's runtime gate.  A separate top-k
  diagnostic raised the expert patch budget from 65,536 to 200,000; the first
  large object then failed transactionally at executable growth before
  execution, proving that top-k Sampled is not a quick default-limit promotion.

- 2026-07-21: The post-merge `001_sk_mxf8f4gemm_tdm` Inline Shadow
  exception is green at clean revision `9b9b12fc8c`.  Commits `0db1b1954c`
  and `9b9b12fc8c` prevent dense access-entry and barrier-epoch relay hosts
  from overwriting preapplied fault ranges; the discriminating regression
  fails without the barrier-relay check and the full 687-test ConSan slice
  passes with it.  Current inventory `...-current-inventory-188` retains 102
  logical sequences / 204 sites.  The original precommitted fault prediction
  remains correctly rejected in artifact `...-current-fault-fixed-187` after
  a real exact-one `not_detected/pass` observation.  A fresh prediction was
  frozen before observation for the next tensor-pipeline pair; artifact
  `...-current-fault-pass-189` accepts it with exact `1/1/1` mutation
  accounting, complete surviving evidence, and healthy containment.  Paired
  artifact `...-current-paired-190` accepts both one-repetition controls and
  the exact 18-row instrumented oracle with complete 768/768 access and
  102/102 barrier coverage.  It measures 367,064.36 ms versus a 27,426.60-ms
  mean control, or 13.38x.

- 2026-07-21: Promoted `torch.sort` Sampled from yellow to green at clean
  revision `31c3d937c2`.  Artifact
  `consan-green-expansion-20260721-sort-sampled-final-clean-164` accepts the
  exact values/indices oracle with complete 48,224/48,224 access and
  12,064/12,064 barrier-member coverage.  Paired artifact
  `...-final-overhead-165` accepts 25,253.78 versus 147.02 ms, or 171.77x.
  Fresh inventory `...-final-inventory-166` retains the reviewed selector.
  After two prior independently selected barriers both preserved the oracle,
  a pass-oracle/no-diagnosis expectation was precommitted; artifact
  `...-current-fault-163` applies that mutation once and accepts the expected
  qualified miss plus both target-health gates.

- 2026-07-21: Restored `torch.mode` SuperCollider to green under strict policy.
  The same atomic-exclusion admission fix that closed histogram also removes
  mode's apparent post-replacement signal.  Committed-tip clean artifact
  `consan-green-expansion-20260721-mode-sc-post-admission-154` accepts the
  exact values/indices oracle with complete 28,195/28,195 ordinary-access
  coverage in 17.70 seconds.  Paired artifact
  `...-post-admission-overhead-155` accepts 15,429.77 ms against 138.84- and
  126.02-ms controls, or 116.51x against their mean.  Fresh inventory
  `...-post-admission-inventory-156` retains the reviewed barrier identity.
  Fault artifact `...-post-admission-fault-158` applies it exactly once,
  observes the precommitted passing-oracle/no-diagnosis qualified miss, and
  passes target health before and after.  The 60-second attempt `157` ended in
  host-side patch construction before mutation; the accepted 120-second retry
  is retained rather than widening the bound further.

- 2026-07-21: Restored `torch.histc` SuperCollider to green under the current
  strict policy at `502b286cfc`.  SuperCollider now keeps LDS atomics as typed
  exclusions without rejecting co-resident ordinary LDS accesses; replaying
  an atomic as an ordinary load/store would change program semantics.  The
  full 299-test host ConSan suite passes.  One-repetition clean artifact
  `consan-green-expansion-20260721-histc-sc-strict-clean-148` accepts the exact
  histogram oracle with complete 133/133 ordinary-access coverage.  Paired
  artifact `...-strict-overhead-149` accepts 5,907.34 versus 68.95 ms, or
  85.67x.  Fresh inventory `...-strict-inventory-150` retains the reviewed
  causal barrier identity, and fault artifact `...-strict-fault-health-153`
  applies it exactly once, observes the precommitted failing-oracle and
  no-diagnosis outcome, and passes target health before and after.

- 2026-07-21: `torch.mode` Sampled returns to green at committed revision
  `96ecd9024a`.  The access pass retains the relocated guest offset, qualified
  gfx1250 DS ordering events derive their inherent workgroup scope, and the
  Sampled metadata cave composes around the atomic access rather than losing
  either role.  The full MOI host suite passes 383/383.  One-repetition clean
  artifact `consan-green-expansion-20260721-mode-sampled-composed-clean-143`
  passes the exact oracle with complete 28,939/28,939 access, 2/2 atomic, and
  8,892/8,892 barrier-member coverage.  Paired artifact `...-overhead-144`
  accepts 151.61/17,191.82/134.71 ms, or 120.09x against the mean baseline;
  inventory `...-145` is complete.  Reviewed fault artifact
  `...-composed-fault-146` applies the selected whole-barrier mutation exactly
  once, observes the precommitted pass-oracle/no-diagnosis qualified miss, and
  passes both health gates.  Like the Record/Replay fault trial below, this
  uses the documented historical best-effort fault policy while clean and
  paired standard-v1 execution remains strict and fully complete.

- 2026-07-21: `torch.mode` Record/Replay returns to green at committed revision
  `6491647e31`.  ConSan now composes the ordering probe around the ordinary
  access probe's relocated guest instruction, so the two ordered no-return LDS
  operations are each represented as both accesses and atomic-ordering events.
  One-repetition clean artifact
  `consan-green-expansion-20260721-mode-rr-composed-clean-136` passes the exact
  oracle with complete 28,939/28,939 access, 2/2 atomic, and 4,446/4,446
  barrier coverage.  Paired artifact `...-composed-overhead-137` accepts
  140.95/28,677.46/133.76 ms, or 208.78x against the mean baseline.  Current
  inventory `...-composed-inventory-138` retains 4,446 barrier sequences.
  Reviewed fault artifact `...-composed-fault-best-effort-141` applies the
  selected whole-barrier mutation exactly once, observes the precommitted
  pass-oracle/no-diagnosis qualified miss, and passes both health gates.  That
  fault trial deliberately retains the historical best-effort fault policy:
  strict analysis rejects unrelated kernels whose ordered-atomic idioms cease
  to qualify after mutation; the clean and paired standard-v1 runs remain
  strict and fully complete.

- 2026-07-21: P0 `torch.topk` SuperCollider now passes the exact FP64/BF16
  value-and-index oracles with complete static and dynamic analysis at
  160,956/160,956 accesses.  One-repetition artifact
  `consan-green-expansion-20260721-pytorch-topk-sc-all-supported-177` is
  accepted in 158.4 seconds.  Preplanned branch-only fallback closes the four
  final placement gaps; signed 16-bit load admission closes eight more; and
  96-bit store readback admits and patches all 108 formerly unsupported
  accesses.  The cell remains yellow only while paired and reviewed-fault
  evidence are refreshed at the frozen revision.

- 2026-07-21: Advanced P0 `torch.topk` SuperCollider from 160,751/160,848 to
  160,836/160,848 accesses.  One-repetition artifact
  `consan-green-expansion-20260721-pytorch-topk-sc-paired-reservoir-174`
  returns normally in 158.4 seconds, passes both exact FP64/BF16 value-and-index
  oracles, and is dynamically complete.  Front-placing the 85 rare
  spill-backed bodies and routing exact site-paired entry/return paths through
  proven straight-line reservoirs recovers all 85 formerly rejected sites.
  Host-side final validation remains fail-closed and near-linear by locating
  each reservoir's owning basic block through a sorted index instead of
  rescanning the full object.  The remaining 12-site aggregate residual keeps
  the cell yellow.

- 2026-07-21: Bounded the P0 `torch.topk` SuperCollider continuation work.
  Artifacts `consan-green-expansion-20260721-pytorch-topk-sc-branch-only-131`
  and `...-132` return normally and pass both exact FP64/BF16 oracles, but
  retain 160,752/160,848 aggregate coverage.  A full-pressure synthetic proves
  scalar spilling plus disjoint forward/backward branch-only continuations and
  rejects a corrupted return relay in final validation.  The real 84-site
  admission now fails only because ordinary anchor relay capacity is
  insufficient; additional proven relay reservoirs, not weaker liveness, are
  the remaining boundary.  The cell stays yellow.

- 2026-07-21: Quick F8 SuperCollider promotes from yellow to green and raises
  the current-tip roll-up to 92/93.  Same-revision one-repetition paired
  artifact `consan-green-expansion-20260721-tensile-f8-sc-overhead-current-126`
  accepts 43,458.36/65,509.16/47,846.30 ms, or 1.43x against the mean
  baseline, with exact numeric results and complete 1,772/1,772 access
  coverage.  Reviewed artifact
  `consan-green-expansion-20260721-tensile-f8-sc-fault-current-125` applies
  the selected barrier mutation once, preserves the exact oracle with the
  precommitted no-diagnosis policy, and passes health before and after.  This
  supersedes the old device-loss containment result without changing its
  expectation after observation.

- 2026-07-21: Current-tip Inline Shadow clears the obsolete dispatch-resource
  launch boundary on both quick F8 and HGEMM.  One-repetition artifacts
  `consan-green-expansion-20260721-tensile-f8-inline-current-123` and
  `consan-green-expansion-20260721-tensile-hgemm-inline-current-124` execute
  49 and 189 exact numeric rows, respectively, with zero failures before their
  fixed 180-second bounds.  Neither full client returns a final analysis
  verdict or overhead, so both cells improve within orange and
  rotate.  The remaining-corpus survey cells are green in all four columns:
  XT4 already froze the architecture-level decoded opcode union over accepted
  selected rows.  Those survey indicators are not runtime denominator cells.

- 2026-07-21: Qwen Sampled remains orange after two bounded current-tip
  diagnostics.  A runtime residue gate now lets unsampled workgroups execute
  the original guest barrier while bypassing causal-window metadata scans;
  all 81 focused and broad Sampled tests pass, and the full Qwen inventory
  remains 1000/1000 accesses and 90/90 barriers.  The independent software
  GPU still has no workload verdict at 180 seconds.  Disabling Sampled barrier
  tracking entirely also produces no verdict at 120 seconds, so barrier
  metadata scanning is not the sole remaining cost.  No timeout is widened,
  no acceptance is inferred, and the roll-up remains 91/93.

- 2026-07-21: Reduced SGEMM is simultaneously green in all four profiles.
  The fault parser first stopped treating `MOI auto report cleanup` as a
  malformed second state summary; focused tests pin that evidence-integrity
  fix, and prior artifacts `118` and `119` then reparse without false overflow
  while retaining their genuine no-diagnosis outcomes.  A new selector was
  precommitted from static structure before execution: the second signal/wait
  pair immediately precedes LDS consumers, unlike the entry and final pairs
  already shown noncausal.  One-repetition artifact
  `consan-green-expansion-20260721-sgemm-smoke-inline-fault-causal-121` applies
  that whole pair exactly once, fails the independent numeric oracle, emits
  one Inline diagnostic, retains complete 640/640 access and 21/21 surviving
  barrier coverage without overflow, and passes health before and after.  The
  Inline cell promotes from yellow to green; the current-tip roll-up is 91/93.

- 2026-07-21: Current-tip atomic regression evidence remains green in all
  four profiles.  After architecture-specific atomic fixtures were aligned
  with the ISA-defined no-SADDR operand, the complete ConSan suite passes
  711/711.  One-repetition Stream-K artifact
  `consan-green-expansion-20260721-streamk-all-current-120` accepts the exact
  workload oracle in all four profiles with 4/4 accesses; Record/Replay,
  Sampled, and Inline additionally retain 10/10 ordered atomics and complete
  4/4, 8/8, and 4/4 barrier coverage, respectively.  This refreshes existing
  green evidence and does not conceal or promote an unrelated non-green cell.

- 2026-07-21: Reduced SGEMM Inline advances from orange to yellow.  Clean
  committed-tip artifact `consan-green-expansion-20260721-sgemm-smoke-inline-bankfix-115`
  accepts the exact numeric oracle with complete 640/640 access and 22/22
  barrier coverage.  One-repetition paired artifact
  `consan-green-expansion-20260721-sgemm-smoke-inline-overhead-116` accepts at
  1.33x against the mean of its two controls with the same complete coverage.
  Fresh inventory artifact `...-inventory-117` retains both reviewed barrier
  selectors.  Late and early exact-one fault artifacts `...-fault-118` and
  `...-fault-early-119` preserve the exact oracle, complete surviving-site
  coverage, report cleanup, and health, but neither emits the precommitted
  Inline diagnostic.  The cell therefore remains yellow and rotates; its
  expectation is not changed after observation.

- 2026-07-21: The P0 top-k Inline signal is independent of the newly fixed
  barrier-bank defect.  Clean committed-tip artifact
  `consan-green-expansion-20260721-pytorch-topk-inline-bankfix-114` again
  completes transformation and signals during execution at 116.7 seconds,
  essentially matching the prior 118-second boundary.  It reaches no oracle
  or final verdict, so the cell remains orange and rotates; the SPMM fix is not
  weakened or generalized beyond its evidence.

- 2026-07-21: P3 SPMM F8 Inline improves within orange after fixing a
  concrete gfx1250 state-preservation defect.  Patch-frontier probes identify
  the ninth barrier as the first corrupting addition; sixteen access probes
  pass with barriers disabled.  That barrier is immediately followed by a
  guest VGPR-bank update.  Commit `e1bbd2608a` makes every gfx1250 Inline
  barrier cave establish its low scratch bank explicitly, matching the
  already-correct Sampled invariant.  The exact formerly failing kernel then
  passes at 1,639/1,639 accesses and 25/25 barriers.  Clean committed-tip
  standard artifact
  `consan-green-expansion-20260721-spmm-f8-ml-inline-fixed-clean-113` records
  eight exact passes, zero failures, and complete 22,074/22,074 access plus
  403/403 barrier coverage for its first object before the unchanged
  180-second bound.  The missing final verdict prevents yellow, but the former
  correctness contradiction is closed.

- 2026-07-21: An exact-kernel discriminator narrows the P3 SPMM F8 Inline
  corruption without changing its acceptance color.  Artifact
  `consan-green-expansion-20260721-spmm-f8-ml-inline-kernel-filter-111`
  reproduces wrong numeric results with all 1,639 accesses and 25 barriers
  patched.  Capped artifact `...-kernel-cap8-112` passes every numeric row in
  the first orientation with eight accesses and eight barriers admitted; its
  fixed bound expires during the second orientation.  This is not acceptance
  evidence, but it excludes common entry state and the earliest probes as a
  sufficient cause.  The active investigation is locating the first failing
  patch frontier rather than widening the workload timeout.

- 2026-07-21: P3 SPMM F8 Inline is now concretely orange rather than
  partially-working orange.  Standard-profile artifact
  `consan-green-expansion-20260721-spmm-f8-ml-inline-clean-108` fully patches
  22,074/22,074 accesses and 403/403 barriers in its first applicable object,
  but the client records eight exact passes and five wrong-result rows before
  its fixed 180-second bound.  Every observed failure is an MT64x64 solution;
  observed MT16x16 solutions pass.  A barrier-disabled control does not reach
  those solutions in 150 seconds, while a one-patch control is rejected by
  dense-relay reservation before execution.  The clean contradiction is
  retained and the cell rotates without a broader bisection.

- 2026-07-21: Rotated to a bounded, one-repetition P3 SPMM F8 Inline clean
  assessment.  This cell currently has only static stress inventory, so its
  first independent numeric and dynamic-coverage verdict can advance the
  matrix without revisiting the resistant top-k execution path.

- 2026-07-21: The bounded one-site top-k Inline discriminator did not reach
  client execution.  Its sole admitted site failed the dense dispatcher's
  reserved relay-space check, so zero relevant patches were applied and the
  required-patch guard stopped the run before an oracle.  This is useful
  construction evidence but cannot distinguish common Inline execution state
  from a later-probe defect.  It is not acceptance evidence, does not change
  the orange cell, and is not being repeated with a wider bound.

- 2026-07-21: Top-k Inline's execution signal reproduces on an independent
  software path.  Artifact
  `consan-green-expansion-20260721-pytorch-topk-inline-rocjitsu-crosscheck-107`
  completes the same two large transformations and signals at 115 seconds
  before an oracle.  This rules out a backend-specific launch quirk and
  classifies the new boundary as a ConSan execution defect.  A bounded
  one-site discriminator was scheduled next; the newer entry above records
  why it stopped before providing that distinction.  No broad patch cap is
  acceptance evidence.

- 2026-07-21: Advanced P0 `torch.topk` Inline Shadow from a construction
  timeout to an execution-stage signal, without a color promotion.  Live
  stacks found three generic scaling defects: quadratic fault-site/event and
  event/sequence association, one heavyweight decoder construction per dense
  relocation candidate, and a linear owner-scope search per automatic scalar
  placement site.  Commits `0007d051bf`, `d3b3cd0df3`, and
  `a793e5db1a` replace them with indexes and decoder reuse.  The strengthened
  8,192-site regression completes in 227 ms, six focused Inline
  dense/shared-owner tests pass, and the 676-test ConSan/MOI sweep retains
  only its 13 pre-existing failures.  Clean-tree artifact
  `consan-green-expansion-20260721-pytorch-topk-inline-owner-indexed-106`
  finishes both large transformations at 72,766 and 55,482 patches, then
  signals after legal resource growth at 118 seconds before an exact oracle.
  The cell remains orange while a bounded independent execution cross-check
  separates an instrumentation defect from a software-GPU limitation.

- 2026-07-21: Bounded the already-yellow P3 SPMM F8 Sampled paired attempt
  without promotion.  One-repetition artifact
  `consan-green-expansion-20260721-spmm-f8-ml-sampled-overhead-103`
  reaches the fixed 180-second instrumented bound with only one of two
  required applicable-object records.  The paired run therefore has no
  accepted overhead value.  This is an execution-duration boundary rather
  than a new oracle failure, and it rotates without extending the timeout.

- 2026-07-21: Bounded P2 `000_sk_sgemm_quick` Inline Shadow without a
  color promotion.  One-repetition artifact
  `consan-green-expansion-20260721-sgemm-quick-inline-102` completes the
  first problem's 12/12 exact rows with zero failures and complete static
  640/640 access plus 22/22 barrier coverage.  Unlike the other profiles, its
  second problem has begun dynamic reporting when the unchanged 120-second
  bound expires, leaving aggregate analysis dynamically incomplete.  The
  cell remains orange and rotates.

- 2026-07-21: Promoted P2 `000_sk_sgemm_quick` Sampled from orange to
  yellow.  Clean-tree one-repetition artifact
  `consan-green-expansion-20260721-sgemm-quick-sampled-101` completes the
  first benchmark problem's 12/12 exact rows with zero failures, 640/640
  accesses, all 40/40 barrier members, static and dynamic completeness, and
  report cleanup.  It begins the second problem before reaching the unchanged
  120-second bound, matching the SuperCollider boundary without extra
  repetitions, filters, caps, or timeout growth.

- 2026-07-21: Promoted P2 `000_sk_sgemm_quick` SuperCollider from orange
  to yellow.  Clean-tree one-repetition artifact
  `consan-green-expansion-20260721-sgemm-quick-sc-100` completes the first
  benchmark problem's 12/12 exact numeric rows with zero mismatches,
  640/640 accesses, static and dynamic completeness, and complete report
  cleanup.  The client then begins its second problem and reaches the
  unchanged 120-second bound, so this is a clean-partial promotion rather
  than a full-client acceptance claim.

- 2026-07-21: Bounded P0 `torch.topk` Inline Shadow after two generic
  construction-scaling fixes.  Commits `5f73127cfa` and `8cbfcc43b3`
  index owner-local host placement and cache immutable descriptor planning;
  focused regressions pass and the broader Inline slice retains only four
  pre-existing atomic-decoder expectation failures.  Clean-tree artifact
  `consan-green-expansion-20260721-pytorch-topk-inline-cached-098` still
  remains in patch construction at the fixed 120-second bound.  The cell
  stays orange and rotates without widening that timeout.

- 2026-07-21: Promoted P0 `torch.topk` Sampled from orange to yellow.  Commit
  `71a333dccf` replaces the remaining nested linear work in dense access and
  barrier construction plus final byte accounting.  All 80 focused Sampled
  tests and an 8,192-range validation regression pass.  Clean-tree,
  one-repetition artifact
  `consan-green-expansion-20260721-pytorch-topk-sampled-scaled-clean-095`
  returns normally in 80.30 seconds, passes both exact FP64/BF16
  value-and-index oracles, is dynamically complete, covers 102,598/161,136
  accesses and all 15,182 barriers, and emits no forbidden diagnostic or
  overflow.  Static access coverage remains incomplete because the bounded
  executable-growth operating point cannot materialize every dense probe, so
  green is not claimed.

- 2026-07-21: Promoted `torch.mode` Sampled from yellow to green.  Paired
  artifact `consan-green-expansion-20260721-pytorch-mode-sampled-overhead-072`
  accepts 96.82/19,932.30/99.05 ms, or 203.53x against the mean baseline, and
  repeats complete coverage.  Fresh inventory `...-sampled-inventory-073`
  retains the reviewed selector.  Fault artifact `...-sampled-fault-074`
  applies exactly one whole barrier mutation, preserves the exact oracle with
  the precommitted no-diagnosis outcome, covers 28,939/28,939 accesses and all
  8,890 surviving barrier members, completely cleans its 43.3 MB peak report,
  and passes target health before and after.

- 2026-07-21: Promoted `torch.mode` Sampled from orange to yellow at commit
  `2dea78db37`.  Clean-tree one-repetition artifact
  `consan-green-expansion-20260721-pytorch-mode-sampled-semantic-clean-071`
  passes the exact oracle in 21.72 seconds with static/dynamic completeness,
  28,939/28,939 accesses, and all 8,892 barrier members.  Paired overhead and
  reviewed-fault evidence remain before green.

- 2026-07-21: `torch.mode` Sampled dirty-tree artifact
  `consan-green-expansion-20260721-pytorch-mode-sampled-semantic-dirty-070`
  passes the exact oracle in 21.43 seconds with static and dynamic
  completeness, 28,939/28,939 accesses, and 8,892/8,892 barrier members.
  Structurally proven same-block split barriers are no longer bounded by a
  corpus-sized instruction distance, while malformed and mismatched pairs
  remain unsupported.  Two isolated no-return LDS releases with no acquire
  consumer are typed not applicable while their ordinary accesses remain
  covered.  The cell remains orange pending a committed-tree repetition.

- 2026-07-21: Promoted `torch.mode` Inline Shadow from yellow to green.  Paired
  artifact `consan-green-expansion-20260721-pytorch-mode-inline-dynamic-overhead-066`
  accepts 99.12/32,739.50/92.39 ms, or 341.90x against the mean baseline, and
  repeats the complete exact oracle and coverage.  Fresh inventory artifact
  `...-dynamic-inventory-067` retains the reviewed barrier selector.  Fault
  artifact `...-dynamic-fault-068` applies it exactly once, preserves the
  precommitted pass-oracle/no-diagnosis outcome, covers 28,939/28,939 accesses
  and all 4,445 surviving barriers with zero undercoverage, and passes target
  health before and after.

- 2026-07-21: Promoted `torch.mode` Inline Shadow from orange to yellow.  Commit
  `af3b46a020` inventories hidden dynamic LDS, selects the external exact-shadow
  table for affected owners, and preserves owner/epoch state privately across
  guest VGPR-bank transitions.  Clean-tree one-repetition artifact
  `consan-green-expansion-20260721-pytorch-mode-inline-dynamic-clean-065`
  passes exact values and indices in 33.80 seconds with 28,939/28,939 accesses,
  4,446/4,446 barriers, zero dynamic undercoverage, and no diagnostics or
  overflow.  Paired overhead and a reviewed fault remain before green.

- 2026-07-21: Sampled SPMM F8 advances from orange to yellow.  The remaining
  assertion was not a CFG-mode-discovery defect.  A dense barrier call can be
  immediately followed by the guest's next `s_set_vgpr_msb`, whose update can
  take effect as control transfers; mode-zero caves had omitted an explicit
  low-bank selection and could therefore enter with the continuation's mode.
  Every gfx1250 Sampled barrier cave now establishes the low bank before its
  first scratch VGPR access.  The focused 78-test Sampled/transition slice and
  the formerly failing filtered client pass.  The unrestricted one-repetition
  exact client also returns normally with three numeric passes and complete
  19,960/19,960 access plus 806/806 barrier-member coverage.  XT3C is DONE and
  work rotates to SuperCollider quick-GEMM; final Sampled acceptance gates
  remain explicit rather than being inferred from clean completion.

- 2026-07-21: At an intermediate XT3C checkpoint, the Sampled SPMM
  investigation found and fixed one real
  gfx1250 VGPR-bank preservation bug.  The target instruction encodes the new
  mode in its low byte and its previous mode in its high byte; ConSan had
  retained the packed transition as persistent state and then emitted invalid
  probe-entry and restoration transitions.  The transition-aware builder,
  parser correction, and seven focused regressions pass.  The cached exact
  client now accepts diagnostic caps through 1,024 probes, including the
  formerly failing nine-probe case.  Unrestricted execution then still
  asserted in
  a software-GPU vector-source read after complete 19,960-access and
  806-barrier-member patching.  That checkpoint's block-entry mode-discovery
  hypothesis and orange disposition were subsequently disproved and
  superseded by the unrestricted passing result above.

- 2026-07-21: Sampled norm/softmax is green, raising the current-tip roll-up to
  89 accepted runnable cells.  Clean-tree paired artifact
  `consan-green-expansion-20260721-norm-softmax-sampled-overhead-independent-050`
  accepts baseline-before/Sampled/baseline-after medians of
  131.65/67,498.39/120.69 ms at one repetition, or 534.97x against the mean
  baseline, while repeating complete 4,756/4,756 access and 4,572/4,572
  barrier coverage.  Reviewed artifact
  `...-norm-softmax-sampled-fault-independent-051` applies exactly one whole
  barrier mutation, observes the precommitted no-diagnosis/pass-oracle result,
  retains complete surviving-site coverage, bounds report memory at 9,295,520
  bytes with complete cleanup, and passes target-dispatch health before and
  after.  Work rotates to the Inline column's bounded publication-contention
  residual rather than spending more effort on an already-green mode.

- 2026-07-21: Sampled norm/softmax advances from orange to yellow.  Commit
  `19840819c2` removes gfx1250 Sampled's code-object-wide persistent
  dispatch-ID pair when scalar pressure reaches `s106`, using the stable
  literal report identity for bank selection and report metadata instead.
  Inline Shadow and other architectures retain their existing persistent
  register paths.  All 96 focused Sampled/report-planning tests pass, including
  a new full-pressure regression.  One-repetition artifact
  `consan-green-expansion-20260721-norm-softmax-sampled-literal-dispatch-independent-048`
  passes the exact oracle in 70.14 seconds with complete 4,756/4,756 access and
  4,572/4,572 barrier coverage, no diagnostics or overflow, and clean source
  provenance.  Paired overhead and reviewed fault evidence remain before green.

- 2026-07-21: P0 `torch.topk` Record/Replay rotates with an exact-object
  feasibility boundary rather than starting a broad scalar-spill subsystem.
  Dump artifact `consan-topk-rr-offline-dumps-047` retains the current
  13,255,032-byte object.  Bounded offline patching reproduces 112,912
  candidates, all 791 `forbidden_overlap` access plans across eight independent
  `gatherTopK` kernels, and 44 successful owner-local assignments.  Removing
  kernel-entry liveness from the diagnostic allocator leaves those counts
  unchanged, so the residual comes from the real access/synchronization-site
  union.  The diagnostic source edit was removed; no weakened entry policy is
  retained.  The yellow cell remains dynamically complete with exact oracles and
  rotates to a smaller implementation target.

- 2026-07-21: P0 `torch.topk` Sampled advances past its stale analysis-bound
  description without claiming acceptance.  Commit `8074f5745d` retains one
  immutable CFG/liveness state across automatic register-selection iterations;
  artifact `consan-green-expansion-20260721-topk-sampled-cfgreuse-independent-044`
  reaches report planning in 32.865 seconds instead of rebuilding the
  1,933,679-instruction graph on every iteration.  The complete eight-bank
  inventory then requires 175,750,736 bytes, above the fixed 128 MiB ceiling.
  Automatic Sampled inventory now retains all 135,610 logical ranges while
  reducing only redundant independent banks by powers of two.  Rebuilt
  artifact `...-sampled-adaptive-independent-045` selects four banks, allocates
  its complete 93,299,856-byte report, and enters full-object patch
  construction.  That construction exceeds the fixed 300-second bound, so the
  orange cell rotates rather than widening its timeout.  The focused 95-test
  Sampled/report-planning suite and all 43 validation-driver tests pass.

- 2026-07-21: D128-pressure Inline Shadow returns to green on the current tip.
  The superseded 120- and 300-second attempts composed two software-GPU
  launch mechanisms.  Direct clean artifact
  `consan-green-expansion-20260721-d128-pressure-inline-direct-independent-041`
  passes the exact oracle with complete 40/40 access and 4/4 barrier coverage
  in 40.13 seconds.  One-repetition paired artifact
  `...-d128-pressure-inline-paired-independent-042` measures 11,586 ms against
  bracketing 8,965 and 8,957 ms baselines, or 1.29x against their mean.

- 2026-07-21: PyTorch cluster synchronization becomes green in Sampled and
  completes the four-profile row.  Commit `a7d2128db3` prevents its dense
  barrier relay planner from consuming pre-applied fault ranges as host code.
  Same-revision clean artifact
  `consan-green-expansion-20260721-cluster-causal-sampled-clean-fixed-independent-039`
  accepts the exact oracle with complete 25/25 access and 4/4 barrier-member
  coverage.  One-repetition paired artifact `...-paired-fixed-independent-040`
  measures 1.07x against its bracketing baseline, and reviewed artifact
  `...-fault-fixed-independent-038` applies exactly one whole cluster-pair
  mutation, preserves the oracle with the precommitted no-diagnosis outcome,
  and passes containment health before and after.

- 2026-07-21: The causal PyTorch cluster workload replaces its superseded
  barrier-only evidence in all four profiles.  Commits `5b7caa9982` and
  `b9af8082f7` add Sampled cluster-scope metadata and give all 512 cluster
  threads distinct LDS elements.  Clean artifact
  `consan-green-expansion-20260721-cluster-causal-all-clean-independent-034`
  and one-repetition paired artifact `...-all-overhead-independent-035`
  accept all four exact-oracle runs with complete coverage.  Reviewed artifact
  `...-cluster-causal-fault-all-independent-037` applies the exact cluster pair
  once in each profile and passes containment health.  SuperCollider,
  Record/Replay, and Inline Shadow preserve the oracle and remain green;
  Sampled rejects the mutated object during final validation and advances from
  typed N/A to yellow with that fault-composition gap recorded.

- 2026-07-21: The bounded alternate-site SuperCollider F8 GEMM fault trial
  does not promote the yellow cell.  Artifact
  `consan-green-expansion-20260721-tensile-f8-sc-fault-alternate-independent-029`
  applies exactly one later logical signal/wait mutation, but multiple exact
  numeric rows fail without a ConSan diagnosis and postflight device health
  fails again.  This rules out a first-barrier-only explanation; the reviewed
  pass-oracle expectation remains unchanged and the cell rotates without a
  third site.

- 2026-07-21: PyTorch cluster synchronization promotes to green in Inline
  Shadow after commit `61f98b14f4` gives barrier-only objects a minimal
  aggregate execution marker.  Clean artifact
  `consan-green-expansion-20260721-cluster-inline-clean-independent-025`
  accepts the exact oracle with 23/23 accesses, 1/1 barrier, and complete
  dynamic evidence.  Paired artifact `...-inline-paired-independent-027`
  accepts 499.49/470.13/500.53 ms (0.94x against the mean baseline), and
  reviewed artifact `...-inline-fault-independent-028` applies exactly one
  logical barrier mutation, matches the precommitted no-diagnosis/pass-oracle
  outcome, and passes target-dispatch health before and after.

- 2026-07-20: Current-tip Qwen/Record-Replay paired artifact
  `consan-revalidation-gfx1250-20260720-qwen-record-replay-paired-058`
  accepts at 5.33x against a 64,016.01 ms bracketing baseline.  The opening
  and closing controls are 63,800.54 and 64,231.47 ms; the instrumented row
  is 340,994.41 ms and retains the exact oracle, 1000/1000 accesses, 46/46
  barriers, and complete static and dynamic analysis.  The cell returns to
  green.

- 2026-07-20: Current-tip Qwen/SuperCollider paired artifact
  `consan-revalidation-gfx1250-20260720-qwen-supercollider-paired-058`
  accepts at 1.94x against a 63,955.46 ms bracketing baseline.  The opening
  and closing controls are 64,040.32 and 63,870.60 ms; the instrumented row
  is 123,953.85 ms and retains the exact oracle, 1000/1000 accesses, and
  complete static and dynamic analysis.  The cell returns to green.

- 2026-07-20: Current-tip Qwen revalidation replaces the interrupted Sampled
  and Inline Shadow claims with measured contradictions.  Isolated Rocjitsu
  Sampled artifacts `-052` and `-054` reproducibly receive signal 11 near 255
  seconds at the final large-output dispatch; Inline Shadow artifact `-051`
  receives the same signal there after 552 seconds.  A canonical Sampled
  cross-check on an independent gfx1250 software GPU, artifact `-057`, passes
  the Rocjitsu failure point but remains compute-active through the unchanged
  600-second bound without an oracle or ConSan verdict.  This makes the crash
  backend-specific evidence, but neither profile has current acceptance or a
  paired overhead, so both cells are orange rather than inferred green.

- 2026-07-20: Commit `e4df0064b1` closes the shared callable-barrier gap in
  Sampled without weakening coverage.  A physical callable now scans the
  causal-window ranges of all statically proven execution owners, validates
  the active dispatch/workgroup/epoch, publishes at most one transition, and
  advances the epoch once.  Its gfx1250 dense relay is grouped and placed by
  the physical function rather than an arbitrary representative kernel.
  Clean artifacts `consan-revalidation-gfx1250-20260720-*-sampled-callable-047`
  and `-048` accept all six affected workloads.  One-repetition paired
  artifacts `consan-revalidation-gfx1250-20260720-*-sampled-overhead-049` and
  `-050` measure D128 block 1.13x, D128 pressure 1.12x, WMMA attention 1.15x,
  Stream-K arrival 2.72x, tree atomic OR 2.57x, and Jakub attention 1.56x,
  each with its exact oracle and complete access/barrier/atomic coverage.

- 2026-07-20: gfx1250 signed FLAT displacement materialization removes the
  strict-analysis gap from D128-pressure and Jakub-attention without changing
  gfx950 or gfx12 admission.  Current clean artifact wave `043` accepts Jakub
  Record/Replay and Inline Shadow at 62/62 accesses and 4/4 barriers, and
  D128-pressure Record/Replay at 40/40 accesses and 4/4 barriers.  The
  concurrent D128-pressure Inline Shadow run reached complete static lowering
  but remained compute-active through its 120-second bound.  Its isolated
  repeat remained compute-active through 300 seconds as well, so it is not
  accepted and the bound is not extended again.

- 2026-07-20: Current-tip one-repetition paired artifact wave `045` promotes
  Jakub-attention Record/Replay and Inline Shadow to green.  Record/Replay is
  1.44x against a 1.857-second paired baseline and Inline Shadow is 1.55x
  against a 1.824-second paired baseline; both retain the exact oracle,
  62/62 access coverage, and 4/4 barrier coverage.

- 2026-07-20: The same paired wave promotes D128-pressure Record/Replay to
  green at 1.13x against a 12.424-second paired baseline.  The exact oracle,
  all 40/40 access sites, and all 4/4 barrier sites accept after signed FLAT
  displacement lowering.

- 2026-07-20: F16 sparse closure and the dependent Tensile opcode survey
  promote to green in Record/Replay.  Clean artifact `365` and paired artifact
  `366` establish complete 31,265/31,265 access and 1,141/1,141 barrier
  coverage.  Frozen reviewed artifact `377` applies exactly one logical
  barrier-pair mutation, retains every access and all 1,140 surviving
  barriers, preserves the exact oracle, matches the precommitted no-diagnosis
  outcome, and passes health before and after.

- 2026-07-20: The wrap-up reruns narrow both remaining implementation rows.
  Top-k artifact `364` passes both exact oracles and all 11,423 barriers while
  reducing the residual to 791 access-only resource conflicts in eight
  containers.  F8 artifact `367` makes its first two objects fully complete
  and removes every owner/resource failure; the remaining third-object gap is
  relay placement.  A generic signed-branch relay-window partition passes
  692/692 ConSan tests and awaits a rebuilt-hook unrestricted rerun.

- 2026-07-20: PyTorch cluster synchronization and its cluster-scope survey
  promote to green in Record/Replay.  Reviewed artifact `370` applies one
  logical signal/wait mutation as two physical rewrites with accounting
  `1/1/1`, preserves the exact copy-and-sentinel oracle, observes the
  precommitted no-diagnosis outcome, and passes health before and after.  Clean
  artifact `363` and paired artifact `364` retain 23/23 accesses and both
  cluster barriers.

- 2026-07-20: The cluster and F16 sparse-closure Record/Replay rows each
  accept their one-repetition paired-overhead gate.  Cluster artifact `364`
  retains 23/23 accesses and both cluster barriers between two passing
  baselines.  F16 artifact `366` retains 31,265/31,265 accesses and
  1,141/1,141 barriers between two passing baselines.  Both cells remain yellow
  solely while their frozen reviewed barrier-drop specifications run.

- 2026-07-20: Four current Record/Replay fronts advance without premature
  promotion.  Cluster synchronization is clean-complete at 23/23 accesses and
  2/2 cluster barriers after conservative CFG pairing plus the barrier-only
  admission fix; its paired and reviewed-fault evidence is running.  Top-k's
  spill-backed transient scalar state recovers 936 accesses while isolating a
  shared 16-container access/barrier placement gap.  F16 sparse closure removes
  all 2,076 unreachable inferred-tail accesses and 32 barriers while retaining
  31,265/31,265 real accesses, 1,141/1,141 real barriers, and every one of its
  35 `ds_store_b16` sites.  Standard one-repetition reruns remain the
  acceptance evidence.

- 2026-07-20: Top-k Record/Replay owner-local artifact `358` passes both exact
  oracles with one repetition in 308.9 seconds, remains dynamically complete,
  and covers all 11,423 barriers.  Access coverage improves by 583 to
  156,591/161,136.  Every remaining 4,545 site is a `forbidden_overlap`
  failure in a `gatherTopK` owner with no component-common eight-SGPR dead
  window; genuine spill-backed transient scalar state is the active next fix.

- 2026-07-20: Full-SGEMM profiling identifies the 900-second aggregate stall
  as quadratic synchronization-owner annotation over 115,776 events rather
  than target execution.  Commit `49df87def2` indexes event identities once,
  preserving first-match semantics.  Its 8,192-event scalability regression
  passes in about 0.55 seconds and all 683 ConSan tests pass; the standard
  one-repetition full-SGEMM Record/Replay rerun is active.

- 2026-07-20: The first standard full-SGEMM Record/Replay run establishes a
  precise scalability frontier with exactly one repetition.  Artifact
  `consan-validation-gfx1250-tensile-sk-sgemm-quick-rr-agent-002` passes the
  first problem's numeric oracle with 640/640 accesses, 22/22 barriers, and
  8/8 fences.  The second problem enumerates 648 solutions in an 8.6 MB code
  object; ConSan plans a 27.7 MB report, reaches patch construction, and stays
  there through the 900-second bound.  The cell improves within orange
  while aggregate-object patch scalability becomes a separate work item.

- 2026-07-20: Norm+softmax Record/Replay is green.  Paired artifact `354`
  accepts both baselines and the fully covered instrumented row.  Reviewed
  barrier-drop artifact `357` applies exactly one selected pair mutation,
  preserves both exact oracles, matches the precommitted no-diagnosis result,
  retains 4,756/4,756 accesses and 2,351/2,351 surviving barriers, and passes
  preflight and postflight device health.  The workload uses exactly one
  repetition; its 191.8-second fault-row duration includes concurrent software
  execution.  Rejected artifact `356` is retained separately: it made no
  mutation because concurrent-load preflight exceeded the fixed health-smoke
  timeout.

- 2026-07-20: Top-k Record/Replay artifact `356` validates the all-supported
  policy fix with exactly one repetition.  Both FP64 and BF16 exact oracles
  pass, dynamic coverage is complete, all 11,423 barriers are patched, and
  access coverage rises from 113,760 to 156,008 of 161,136.  The prior 42,248
  instrumentation-patch omissions are gone.  All 5,128 remaining accesses are
  the same `forbidden_overlap` resource failure caused by one object-global
  transient scalar window; owner/component-local allocation is now the active
  implementation task.

- 2026-07-20: Current-tip one-repetition norm+softmax Record/Replay artifact
  `352` accepts the exact norm and CPU softmax oracles in 44.3 seconds with
  complete 4,756/4,756 access, 2,352/2,352 coalesced-barrier, static, dynamic,
  and overall analysis.  Artifact `116`'s 331-access and 407-barrier residuals
  were all placement-missing and are closed by subsequent generic dispatcher
  and resource-planning fixes.  The cell remains yellow only for its paired and
  reviewed-fault bundles, now running in parallel with other Record/Replay
  work.

- 2026-07-20: F8 quick-GEMM SuperCollider paired artifact `352` accepts both
  one-repetition baselines and complete 1,772/1,772 instrumentation.  Reviewed
  exact-one fault attempt `353` loses the software device and fails postflight
  health.  The cell stays yellow and rotates; the active campaign refocuses on
  Record/Replay.

- 2026-07-20: Promoted `torch.mode` Record/Replay from yellow to green at commit
  `dbf7e289fd`.  One-repetition paired artifact `349` accepts 104--108 ms
  baselines and the 30,548 ms instrumented run with complete coverage.
  Fresh inventory artifact `350` confirms the reviewed barrier identity.
  Fault artifact `351` applies exactly one barrier-pair mutation, preserves the
  exact value/index oracle, matches the precommitted no-diagnosis outcome,
  covers 28,939/28,939 accesses and all 4,445 surviving barriers, and passes
  pre/post containment health.

- 2026-07-20: `torch.mode` Record/Replay clean artifact `348` is accepted with
  one repetition, the exact value/index oracle, 28,939/28,939 accesses,
  4,446/4,446 barriers, and complete static, dynamic, and overall analysis.
  ConSan now preserves the DS byte offset and can materialize a tagged gfx1250
  LDS address token.  The two executed ordered `ds_add_u32` sites are isolated
  no-return releases with no compatible acquire in the same execution-owner
  domain; their synchronization role is typed not applicable while their
  ordinary access role remains covered.  The hook coverage fallback no longer
  resurrects pre-pruning resource plans into the applicable denominator.

- 2026-07-20: Promoted histogram Record/Replay from yellow to green.  Frozen
  paired artifact `300` accepts both baselines and complete 175/175-access,
  84/84-barrier instrumentation.  Reviewed artifact `301` applies exactly one
  barrier drop, fails the exact oracle with the precommitted no-diagnosis
  outcome, retains all 175 accesses and 83 surviving barriers, and passes
  pre/post health.

- 2026-07-20: Promoted histogram Inline from yellow to green at the current
  denominator.  Frozen paired artifact `296` accepts both baselines and the
  instrumented exact-count run with complete 175/175 access and 84/84 barrier
  coverage.  Audited artifact `297` showed that the old no-diagnosis fault
  expectation had become stale: the exact-one barrier drop now produces 60
  race diagnostics.  After reviewing that stronger outcome, artifact `299`
  accepts the precommitted detection plus failed numeric oracle, complete
  175/175 access and 83/83 surviving-barrier coverage, and pre/post health.

- 2026-07-20: Current-tip `torch.mode` Record/Replay artifact `294` preserves
  the exact value/index oracle and dynamic completeness in 30 seconds while
  increasing complete access coverage to 28,939/28,939; all 4,446 barriers
  remain covered.  Standalone cache operations are now typed not applicable
  instead of unsupported on gfx1250.  The only remaining static gap is two
  acquire/release-capable `ds_add_u32` synchronization sites.  They remain
  honestly unsupported until the atomic record model can represent an LDS
  communication token; their ordinary LDS accesses are already shadowed.

- 2026-07-20: Current-tip histogram artifacts `289`--`291` promote relaxed
  gfx1250 LDS atomics to their correct dual treatment: ordinary shadowed
  memory accesses, but not synchronization edges.  Record/Replay, Sampled,
  and Inline all preserve exact counts with complete clean analysis at
  175/175 accesses and every applicable barrier.  Scatter artifact `292`
  preserves exact BF16/FP32 collision counts in every profile.  Its
  Record/Replay and Inline runs are now narrowed to the generic visible-record
  gate: the executed global-atomic kernels contain no applicable LDS sites,
  so they correctly emit no LDS evidence.

- 2026-07-20: Promoted P0 `torch.topk` Record/Replay from orange to yellow.
  Dense entry-host reservations are made before per-site resource filtering;
  a wholly filtered owner group previously emitted an unreachable 60-byte
  relay body with no patch-inventory record, causing final validation to
  reject the unrestricted object.  Record/Replay now retains that reservation
  as inert NOP padding, matching the other MOI engines.  Unfiltered artifact
  `288` passes the exact FP64 and BF16 values/indices oracles in 264 seconds,
  is dynamically complete, instruments 113,760/160,848 supported accesses,
  and covers all 11,423 barriers.  The full 2,245-test suite passes.

- 2026-07-20: Advanced P0 `torch.topk` SuperCollider to 160,752/160,848.
  The scalar-limit dense fallback uses one-word inline keys in each site's
  already-dead VCC-save SGPR rather than clobbering a low guest pair.  A
  1,025-site focused regression forces this path and passes final-byte
  validation.  Unfiltered artifact `282` passes both exact FP64/BF16
  value/index oracles and remains dynamically complete.  All dense groups
  now place; the yellow cell's last 96 accesses are led by 84 sites lacking the
  four scalar temporaries required by the far absolute-jump path.

- 2026-07-20: Advanced P0 `torch.topk` SuperCollider from 159,514/160,848 to
  160,220/160,848 accesses.  Dense groups sharing one kernel now reserve
  distinct entry hosts, and each relocated eight-word window stays within one
  basic block so no alternate control-flow edge can enter its rewritten
  island.  Unfiltered artifact `280` passes both exact FP64/BF16 value/index
  oracles and remains dynamically complete.  A diagnostic low-SGPR fallback
  reached 160,730/160,848 but produced zero BF16 values; it was removed rather
  than retained as apparent coverage.  The cell remains yellow with 628
  supported accesses still unplaced.

- 2026-07-20: Advanced P0 `torch.topk` SuperCollider within yellow.  Unfiltered
  artifact `261` passes both exact FP64 and BF16 values/indices oracles and is
  dynamically complete while improving static coverage from 2,991/135,384
  to 159,514/160,848 accesses.  The remaining 1,334 supported accesses are
  placement failures in the large loaded object.  A follow-up changed only
  the dispatcher eligibility for a hypothesized stranded local-cave tail;
  artifact `262` returned the identical denominator, so that diagnostic
  change was reverted.  Work moves to another P0/P1 cell rather than spending
  another long software-runtime run on the unchanged residual.

- 2026-07-20: Promoted P0 `torch.mode` SuperCollider from orange to green.
  Commit `dacb1d3b05` enables the per-kernel shared-call dispatcher when at
  least 1,024 sites form a reach-stranded majority and adds native
  `ds_load_b96` check lowering.  Frozen artifact `252` accepts the unfiltered
  exact values/indices oracle with static and dynamic completeness at
  28,195/28,195 accesses.  Paired artifact `253` accepts both runs and
  measures 14,301 ms versus a 92 ms baseline.  The 2,242-test RocJITsu suite
  passes.  Fault validation then exposed a process-scope selector bug:
  unrelated runtime code objects were incorrectly required to contain the
  reviewed site, and a transform-spanning mutex could block concurrent code
  object loads.  Commit `81bbc381fb` uses a dry-run match followed by an
  atomic first-match reservation.  Reviewed artifact `260` applies exactly
  one mutation, observes the precommitted no-diagnosis/pass-oracle result,
  and passes both containment-health gates.

- 2026-07-20: Promoted `torch.sort` SuperCollider from orange to green on
  frozen commit `b00563cd31`.  A per-kernel shared-call dispatcher replaces
  the mathematically impossible one-relay-per-site layout and its structural
  validator proves every call, dispatch edge, relocated host, and return.
  Signed-byte loads, low/high-half byte and 16-bit stores, and stride-64
  two-address stores close the complete discovered denominator.  Clean
  artifact `241` accepts the exact values/indices oracle with static and
  dynamic completeness at 48,224/48,224 accesses.  Paired artifact `242`
  accepts both baselines and the instrumented run at 27,003 ms versus 121 ms.
  Reviewed barrier-drop artifact `246` applies exactly one mutation, observes
  its precommitted no-diagnosis/pass-oracle result, and passes pre/post health.
  The complete 2,241-test RocJITsu suite also passes.

- 2026-07-20: The first frozen fault attempt retained in artifact `244`
  exposed a stale cross-profile oracle contract rather than a runtime fault:
  it expected SuperCollider to fail even though the already-reviewed
  Record/Replay and Inline campaigns for the identical mutation both pass.
  The corrected reviewed contract derives from that prior evidence; artifact
  `246` then accepts the exact-one campaign.  An earlier attempt `237` never
  launched the workload because the default full-model health smoke exceeded
  its 30-second bound; the accepted campaign uses a 1.4-second exact
  `torch.sort` target-dispatch smoke for both health gates.

- 2026-07-20: Reused byte-LDS SuperCollider support on `torch.histc`.
  Current-tip artifact `202` preserves the exact histogram oracle and improves
  supported access coverage from 100/100 to 131/131.  The static verdict stays
  yellow rather than green because two LDS atomic compare-store sites remain
  typed exclusions; duplicating them as ordinary readbacks would change
  program semantics.

- 2026-07-20: Promoted the P0 PyTorch/Triton tensor-descriptor SuperCollider
  cell from orange to green.  The 23 missing support-object sites were
  zero-extending `ds_load_u8` and truncating `ds_store_b8` operations, not new
  synchronization forms.  SuperCollider now duplicates byte loads and masks
  byte-store guest values before comparing readback.  Focused load, store, and
  two-VGPR spill regressions pass, as do all 285 ConSan unit tests.  Clean
  artifact `199` and paired bundle `200` are complete at 29/29 accesses with
  exact one-CTA and two-CTA oracles; bundle `201` accepts the reviewed exact
  mutation and both containment-health checks.  The descriptor row is now
  green in all four columns while `XP3` remains yellow for the other
  PyTorch P0/P1 cells.

- 2026-07-20: Promoted the P0 PyTorch/Triton tensor-descriptor workload to
  green in Record/Replay, Sampled, and Inline Shadow.  Same-tip paired bundle
  `192` accepts exact one-CTA and two-CTA results with 29/29 accesses in each
  engine, 12/12 barriers in Record/Replay and Inline, and all 20 Sampled
  barriers.  Reviewed bundle `194` drops one wait from a complete descriptor
  kernel barrier pair in each profile, observes the specified no-diagnosis
  and passing-oracle result, and passes containment health.  The broader run
  corrects SuperCollider from yellow to orange: its six admitted descriptor
  accesses pass, but a loaded PyTorch support object contributes 23 typed
  unsupported accesses.

- 2026-07-20: Histogram validation now retains an accepted four-profile
  barrier-drop fault bundle (`182`) with exactly one mutation, the reviewed
  `not_detected` sanitizer outcome, the expected exact-oracle failure, and a
  passing post-run health check in every profile.  Paired overhead bundle
  `183` preserves exact clean results and timings but confirms that the MOI
  profiles still place only 112/133 accesses.  The 21 misses are sparse VDS
  stores whose default appended entries exceed short-branch reach.  The
  existing gfx1250 call/dispatcher path now also activates for such
  reach-stranded groups, with a nine-site non-adjacent regression passing in
  Record/Replay, Sampled, and Inline Shadow.  The cells remain yellow until the
  committed-tip clean rerun proves 133/133 coverage and the other acceptance
  gates close.

- 2026-07-20: Committed-tip histogram artifacts `186`-`188` confirm that the
  sparse-entry routing change closes all 21 MOI placement misses.
  Record/Replay reaches 133/133 accesses and 84/84 applicable barriers;
  Sampled reaches 133/133 and 168/168.  Both preserve the exact oracle and
  dynamic completeness but remain yellow because 126 unqualified atomic
  synchronization sequences are still typed unsupported.  Inline Shadow is
  now clean-accepted with the exact oracle, 133/133 accesses, 84/84 barriers,
  and complete analysis.  Its cell stays yellow only until fault, resource, and
  same-tip freeze evidence are refreshed.

- 2026-07-20: Promoted histogram Inline Shadow from yellow to green.  Same-tip
  paired overhead bundle `189` accepts both uninstrumented baselines and the
  instrumented exact-oracle run with complete 133/133 access and 84/84
  barrier analysis.  Reviewed barrier-drop bundle `191` applies exactly one
  mutation, produces the specified oracle failure without a sanitizer
  diagnosis, retains complete 110/110 access and 83/83 surviving-barrier
  coverage, and passes both health checks.  Record/Replay and Sampled remain
  yellow on their separately typed atomic-qualification blocker.

- 2026-07-20: Isolated the remaining `torch.sort` SuperCollider denominator.
  The exact values and indices still pass, but the large radix-sort code
  object has no usable local NOP caves.  Of its 38,020 supported LDS sites,
  36,094 cannot directly branch to appended instrumentation after the first
  1,926 bodies consume the short-branch window; relay placement adds three
  more sites for 1,929 total.  This is a distributed entry-island capacity
  blocker, so the cell stays orange while `XP3` advances another P0/P1 cell.

- 2026-07-19: The committed-tip `004_sk_mxf8gemm_tdm` Inline Shadow attempt
  `consan-gfx1250-sk-mxf8-inline-072` remained compute-active through its
  1800-second ceiling without a sanitizer diagnostic, crash, oracle result, or
  analysis verdict.  Together with the earlier 600-second observation, this
  establishes a software-execution duration limitation rather than a current
  correctness result.  The cell remains yellow at 3/4 accepted profiles; work
  advances to `007` while `004` Inline is retried on the faster backend with
  the full 992-site denominator intact.

- 2026-07-19: Advanced the second large Stream-K kernel to 3/4 accepted clean
  profiles.  `004_sk_mxf8gemm_tdm` passes SuperCollider, Record/Replay, and
  Sampled with complete 992/992 access coverage; Record/Replay also retains
  204/204 barriers and 24/24 fences, while Sampled retains all 180 applicable
  barriers.  SuperCollider artifact `consan-gfx1250-sk-mxf8-sc-071` fixes an
  overlapping-address `ds_load_b128` whose low-bank address had previously
  been saved under the guest's high destination-bank mode.  The uncapped run
  now passes every numeric case.  The first Inline attempt in
  `consan-gfx1250-sk-mxf8-all-069` reached 600 seconds before producing an
  analysis verdict, so the next action is a longer unmodified Inline run.

- 2026-07-19: Completed the first large Stream-K kernel under all four clean
  profiles.  Sampled artifact `consan-gfx1250-sk-mxf8f4-sampled-068` passes the
  independent numeric oracle with one repetition and complete static, dynamic,
  and analysis verdicts, covering 768/768 accesses and all 180 applicable
  barriers.  Its dense access and barrier relays preserve the guest VGPR bank
  across spill-backed instrumentation.  The grouped P1 cell remains yellow only
  because sibling kernels `004` and `007` have not yet completed their four
  profiles.

- 2026-07-19: Advanced the first large Stream-K kernel to 3/4 clean profiles.
  Inline Shadow artifact `consan-gfx1250-sk-mxf8f4-inline-060` passes the
  independent numeric oracle in one repetition with complete 768/768 access
  and 204/204 barrier coverage, complete static and dynamic analysis, and
  return code zero in 306 seconds.  Dense adjacent accesses now share a
  one-word call relay, while full-register barrier sites use an independently
  spill-backed return-key dispatcher.  The implementation also resolves
  active kernel descriptors after prior text growth, preventing a stale
  metadata offset from writing into the reserved relay text.  Sampled is the
  sole remaining clean blocker for `001`; the aggregate row is now yellow.

- 2026-07-19: Advanced the large Stream-K P1 row from unseen to a retained
  2/4-profile vertical.  `001_sk_mxf8f4gemm_tdm` now passes the independent
  numeric oracle under SuperCollider with complete 768/768 access coverage in
  `consan-gfx1250-sk-mxf8f4-supercollider-053`; the earlier accepted
  Record/Replay run covers 768/768 accesses, 204/204 barriers, and 24/24
  fences.  Seven formerly omitted `ds_load_b128` sites legally span the
  v255/v256 boundary.  SuperCollider now emits role-specific compare-bank
  transitions for their upper dwords while restoring the low instrumentation
  bank before report actions.  Sampled placement and Inline Shadow execution
  remain the active clean blockers, so the aggregate row is orange rather
  than yellow.

- 2026-07-19: Promoted `003_sk_mxf4gemm_explicit` to green in all four
  profiles and completed the compact P0 pair.  Artifact
  `consan-validation-gfx1250-tensile-mxf4-all-021` retains one baseline and
  one run per profile: 42/42 accesses in every profile, 32/32 barriers in
  Record/Replay and Inline, 28/28 applicable barriers in Sampled, and 4/4
  Record/Replay fences.  Every run passes its numeric oracle and every
  completeness predicate.

- 2026-07-19: Promoted `002_sk_mxf8gemm_explicit` to green across all four
  profiles.  Inline artifact
  `consan-validation-gfx1250-tensile-mxf8-inline-020` passes the numeric oracle
  in one repetition, patches 70/70 accesses and 32/32 barriers, and reports
  complete static, dynamic, and aggregate analysis.  Its gfx1250-only fallback
  preserves a 28-SGPR transient probe window in private memory while keeping
  cross-site evidence and indirect-jump state in independently safe SGPRs.

- 2026-07-19: Promoted the `002_sk_mxf8gemm_explicit` Sampled cell to clean
  accepted.  Sampled now associates a barrier with a selected causal window by
  CFG reachability rather than requiring the barrier to postdominate kernel
  entry; Ready-window owner/epoch validation remains the runtime fail-closed
  guard.  Exact complete workgroup-barrier pairs also remain valid in cyclic
  CFG components, where the private epoch advances on each executed
  iteration.  Artifact `consan-validation-gfx1250-tensile-mxf8-sampled-016`
  preserves the numeric oracle, patches 70/70 accesses and 28/28 barrier
  events, and reports complete static, dynamic, and aggregate analysis.

- 2026-07-19: Promoted `torch.topk` from unseen to clean-partial and completed
  the dense Record/Replay implementation slice.  The checked-in one-repetition
  runner has independent exact sorted-value/index oracles for FP64 and BF16.
  On the dispatched FP64 specialization, gfx1250 one-word call anchors plus a
  shared return-PC dispatcher raise static coverage from 44/106 to 106/106;
  execution preserves the oracle and publishes 134 records.  A focused
  synthetic kernel proves nine adjacent accesses with no compiler NOP island,
  and 659 focused ConSan tests pass.  The unrestricted clean attempt
  `consan-validation-gfx1250-pytorch-topk-rr-033` is intentionally not
  accepted: the monolithic object exposes 112,552 supported accesses whose
  kernels do not share one globally fresh eight-SGPR state window.  The next
  frontier is per-owner ordinary scalar-state selection or an equally
  fail-closed workload surface, not a hidden kernel filter.

- 2026-07-19: Promoted `torch.mode` from unseen to clean-partial.  The new
  deterministic one-repetition workload checks both the returned mode and that
  its returned index selects that value.  Artifact
  `consan-validation-gfx1250-pytorch-mode-sc-025` reaches that oracle under
  SuperCollider after replacing per-instruction linear range scans and the
  quadratic large relay graph.  Full-library access coverage remains
  317/16,104.  Record/Replay artifact
  `consan-validation-gfx1250-pytorch-mode-rr-027` proves that a demand-sized
  95,751,536-byte report can now be planned and allocated, then exposes a
  later instrumentation failure.  The row remains orange; no filter or
  denominator reduction is accepted as a shortcut.

- 2026-07-19: Promoted the PyTorch/Triton tensor-descriptor row from gray to
  yellow.  Artifact `consan-validation-gfx1250-pytorch-tdm-all-008` runs the
  one-CTA and two-CTA clustered variants with one repetition and passes their
  exact numeric oracle under all four profiles.  Every profile is statically
  and dynamically complete: access coverage is 6/6 throughout, Record/Replay
  and Inline Shadow cover 24/24 barriers, and Sampled covers its 20/20
  applicable post-access barriers.  The run exposed and fixed clustered peers
  aliasing in Sampled causal metadata by adding the launch-provided
  workgroup-within-cluster identity.  Fault, resource, and frozen-provenance
  gates remain before this aggregate cell becomes green.

- 2026-07-19: Completed the final simultaneous-green audit at `9acc4dd9b0`.
  Artifact `consan-validation-gfx1250-final-audit-149` reruns all ten
  non-omitted workloads under all four standard profiles after an exact-tip
  hook rebuild.  All 40 rows are accepted: every independent oracle passes,
  every static/dynamic coverage contract retains its previously accepted
  denominator, and there are no diagnostics, overflow, timeout, or process
  failures.  Every result names source tip
  `9acc4dd9b070e1c44008c909ea7c97de1f5831f2` and hook SHA-256
  `81a11216335f44745412f3e2d5b7c134d76c17783194c99003e2d33fd07208a8`.
  Combined with each row's retained frozen fault, containment, and resource
  bundle, this proves the active matrix is simultaneously green.

- 2026-07-19: Promoted all four TP2-family cells from yellow to green with a
  frozen bundle at `837b9f73f5`.  Inventory artifact
  `consan-validation-gfx1250-tp2-freeze-inventory-145`, clean artifact
  `consan-validation-gfx1250-tp2-freeze-clean-146`, contained barrier-drop
  artifact `consan-validation-gfx1250-tp2-freeze-fault-147`, and paired
  resource artifact `consan-validation-gfx1250-tp2-freeze-overhead-148` name
  the same clean revision and hook identity.  Prefill, decode, and combined
  oracles pass in every clean profile, with 2760/2760 accesses throughout;
  Record/Replay and Inline Shadow cover 288/288 barriers, while Sampled covers
  its 48/48 admitted barriers with typed exclusions.  The exact reviewed
  barrier pair is dropped once and fails the independent oracle in every
  profile; the three trace engines produce their precommitted qualified
  misses and Inline produces one required diagnostic.  All real containment
  health and dispatch-smoke gates pass.  One-sample paired slowdowns are
  1.02x, 1.08x, 1.02x, and 2.15x, with report peaks of 4, 3,738,400,
  7,093,024, and 29,733,520 bytes respectively and no allocation, capacity,
  cleanup, or timeout failure.

- 2026-07-19: Promoted all four TP1 decode/combined cells from yellow to green
  with a frozen bundle at `a0c48d4acf`.  Inventory artifact
  `consan-validation-gfx1250-tp1-decode-freeze-inventory-139`, clean artifact
  `consan-validation-gfx1250-tp1-decode-freeze-clean-140`, contained
  barrier-move artifact
  `consan-validation-gfx1250-tp1-decode-freeze-fault-141`, and paired resource
  artifact `consan-validation-gfx1250-tp1-decode-freeze-overhead-142` name the
  same clean revision and hook identity.  Every clean profile passes the
  standalone-decode and prefill/decode oracles and covers 704/704 accesses;
  Record/Replay and Inline Shadow cover 148/148 barriers, while Sampled covers
  its 48/48 admitted barriers with typed exclusions.  The reviewed
  dispatch-30 matmul barrier move is applied exactly once and fails the model
  oracle in every profile; SuperCollider, Record/Replay, and Sampled produce
  their precommitted qualified misses, while Inline produces one required
  diagnostic.  All real containment health and dispatch-smoke gates pass.
  One-sample paired slowdowns are 1.05x, 1.03x, 1.01x, and 1.58x, with report
  peaks of 4, 1,716,576, 1,804,640, and 9,878,608 bytes respectively and no
  allocation, capacity, cleanup, or timeout failure.

- 2026-07-19: Promoted all four TP1-prefill cells from yellow to green with a
  frozen bundle at `8931f54bd2`.  Inventory artifact
  `consan-validation-gfx1250-tp1-prefill-freeze-inventory-122`, clean artifact
  `consan-validation-gfx1250-tp1-prefill-freeze-clean-121`, contained
  barrier-move artifact
  `consan-validation-gfx1250-tp1-prefill-freeze-fault-123`, and paired resource
  artifact `consan-validation-gfx1250-tp1-prefill-freeze-overhead-124` name the
  same clean revision and hook identity.  The reviewed inventory contains 74
  barrier sites, 51 sequences, and 3,344 exact move destinations.  Every clean
  profile passes the independent oracle and covers 352/352 accesses;
  Record/Replay and Inline Shadow cover 74/74 barriers, while Sampled covers
  its 24/24 admitted barriers with typed exclusions.  The exact late-attention
  move is applied once and fails the independent oracle in every profile;
  SuperCollider, Record/Replay, and Sampled produce their precommitted
  qualified misses, while Inline produces one required diagnostic.  All
  containment health gates pass.  One-sample paired slowdowns are 1.05x,
  1.06x, 1.00x, and 1.99x, with report peaks of 4, 858,416, 902,448, and
  4,943,392 bytes respectively and no allocation, capacity, cleanup, or
  timeout failure.  Commit `8931f54bd2` also replaces the pathological compact
  first-use bitmap for ordinary gfx1250 LDS sizes with the already-qualified
  generation-tagged exact-cell protocol: full TP1 Inline execution falls from
  a 600-second timeout to 11.46 seconds without reducing its denominator.

- 2026-07-19: Promoted all four Jakub cells from yellow to green with a frozen
  bundle at `575a874c37`.  Inventory artifact
  `consan-validation-gfx1250-jakub-freeze-inventory-107`, clean artifact
  `consan-validation-gfx1250-jakub-freeze-clean-106`, contained barrier-drop
  artifact `consan-validation-gfx1250-jakub-freeze-fault-108`, and paired
  resource artifact `consan-validation-gfx1250-jakub-overhead-109` all name
  that revision and the same hook identity.  Every clean profile passes all
  three production-shaped host oracles: SuperCollider covers 62/62 accesses;
  Record/Replay and Inline Shadow cover 31/31 accesses plus 8/8 barriers; and
  Sampled covers 31/31 accesses with its typed 0/0 barrier denominator.  The
  exact reviewed signal/wait mutation is applied once and fails the
  independent oracle in every profile; SuperCollider, Record/Replay, and
  Sampled retain their precommitted qualified misses, while Inline produces
  the required diagnostic.  All containment health gates pass.  Paired
  process slowdowns are 2.77x, 1.55x, 1.51x, and 1.68x, with report peaks of
  4, 463,648, 40,608, and 12,601,920 bytes respectively and no allocation,
  capacity, cleanup, or timeout failure.

- 2026-07-19: Promoted Qwen Inline from yellow to green with a frozen bundle at
  `7ea0866fa0`.  Inventory artifact
  `consan-validation-gfx1250-qwen-inline-freeze-inventory-103`, clean artifact
  `consan-validation-gfx1250-qwen-inline-freeze-clean-102`, contained targeted
  fault artifact `consan-validation-gfx1250-qwen-inline-freeze-fault-100`, and
  paired resource artifact `consan-validation-gfx1250-qwen-inline-overhead-101`
  all name that clean revision.  The clean oracle passes in 267.36 seconds with
  1000/1000 accesses, 92/92 barriers, full static and dynamic completeness,
  and no diagnostics.  The exact barrier pair is applied once in the reviewed
  initializer slice, fails the independent oracle, produces one attributed
  write/read diagnostic, and passes both health gates.  One-sample paired
  dispatch medians are 211.17 seconds versus a 17.89-second bracketing
  baseline, or 11.80x.  Peak report memory is 6,679,616 bytes with no
  allocation, capacity, cleanup, timeout, or health failure.

- 2026-07-19: Promoted Qwen Inline from orange to mostly-working yellow.
  Commit `fdde519080` replaces the near-capacity gfx1250 compact-validity hot
  path with full-width generation-tagged local cells: no eager mirror clear,
  bitmap claim, or readiness poll is required, while each access retains the
  canonical exact exchange and complete instruction offset.  Focused layout,
  emission, atomic-token, and wide-access tests pass.  Canonical clean artifact
  `consan-validation-gfx1250-qwen-inline-freeze-clean-097` passes the Qwen
  oracle in 268.79 seconds with 1000/1000 accesses, 92/92 barriers, full static
  and dynamic completeness, zero diagnostics, and clean provenance.  Inventory
  artifact `consan-validation-gfx1250-qwen-inline-freeze-inventory-098` retains
  the reviewed barrier pair.  The first all-site fault run applied that pair
  exactly once and failed the oracle but missed its required diagnostic;
  isolating instrumentation to the mutated initializer reproduces one exact
  write/read diagnostic while preserving the end-to-end oracle failure.  The
  target fault contract now records that deterministic causal slice; its
  frozen rerun and the one-sample resource gate remain before green.

- 2026-07-19: Promoted Qwen SuperCollider and Sampled from yellow to green with
  independent frozen bundles at `0a09cd5f83`.  Shared inventory artifact
  `consan-validation-gfx1250-qwen-sc-freeze-inventory-089` retains the exact
  selected wait-side barrier mutation.  SuperCollider clean artifact
  `consan-validation-gfx1250-qwen-sc-freeze-clean-090`, contained fault
  artifact `consan-validation-gfx1250-qwen-sc-freeze-fault-091`, and resource
  artifact `consan-validation-gfx1250-qwen-sc-overhead-092` all name that
  revision.  Sampled clean artifact
  `consan-validation-gfx1250-qwen-sampled-freeze-clean-093`, contained fault
  artifact `consan-validation-gfx1250-qwen-sampled-freeze-fault-094`, and
  resource artifact `consan-validation-gfx1250-qwen-sampled-overhead-095` do
  likewise.  Both clean runs pass the Qwen oracle and cover 1000/1000
  accesses; Sampled also covers all 56 admitted barriers with typed static
  exclusions.  Each fault run applies exactly one reviewed mutation, produces
  its precommitted independent-oracle failure and qualified miss, and passes
  both health gates.  One-sample paired software-execution measurements are
  1.28x for SuperCollider and 1.09x for Sampled.  Their retained report peaks
  are 4 and 2,584,656 bytes respectively, with no allocation, capacity,
  cleanup, timeout, or health failure.

- 2026-07-19: Promoted Qwen Record/Replay from yellow to green with a frozen
  bundle at `4b11d66f1f`.  Inventory artifact
  `consan-validation-gfx1250-qwen-freeze-inventory-083`, clean artifact
  `consan-validation-gfx1250-qwen-rr-freeze-clean-081`, contained wait-drop
  artifact `consan-validation-gfx1250-qwen-rr-freeze-fault-082`, and paired
  resource artifact `consan-validation-gfx1250-qwen-rr-overhead-080` all name
  that clean revision.  Clean and overhead rows retain 1000/1000 accesses and
  92/92 barriers with complete static and dynamic coverage.  The exact
  mutation is applied once, produces the precommitted independent-oracle
  failure and qualified miss, and passes both health gates.  One-sample
  software-execution qualification measures 26.65 seconds against an
  18.07-second paired dispatch baseline, or 1.47x, with a 1,229,648-byte peak
  report allocation and complete cleanup.

- 2026-07-18: Promoted all four WMMA-attention cells directly from yellow to
  green with a frozen bundle at `0cc5c02dd8`.  Inventory artifact
  `consan-validation-gfx1250-wmma-inventory-197`, clean artifact
  `consan-validation-gfx1250-wmma-clean-198`, contained barrier-drop artifact
  `consan-validation-gfx1250-wmma-fault-195`, and paired resource artifact
  `consan-validation-gfx1250-wmma-overhead-196` all name that clean revision.
  Every clean profile passes its oracle and covers 18/18 accesses; Record/Replay
  and Inline Shadow also cover 8/8 barriers.  The exact barrier mutation fails
  the independent oracle and produces the reviewed qualified miss in all four
  profiles with healthy containment.  Paired median slowdowns are 2.27x,
  1.23x, 1.23x, and 1.25x; retained instrumentation-owned peaks are 4, 461,776,
  23,760, and 12,600,672 bytes respectively, with no resource failure.

- 2026-07-18: First contained WMMA barrier-drop execution in artifact
  `consan-validation-gfx1250-wmma-fault-194` completes exact 1/1/1 mutation,
  oracle failure, and healthy containment in all four profiles.  Unlike D128
  pressure, this mutation creates no Inline-visible conflict among WMMA's 18
  admitted shared-access sites, so all four profiles produce a qualified miss.
  The workload-specific Inline expectation is corrected to `not_detected`
  before a fresh acceptance run.

- 2026-07-18: Accepted current WMMA static inventory artifact
  `consan-validation-gfx1250-wmma-inventory-193`: eight barrier sites form four
  exact signal/wait sequences in the shared workgroup-barrier helper.  The
  reviewed first mutation precommits an independent-oracle failure and a
  qualified miss in every profile.  Contained acceptance remains before green.

- 2026-07-18: Promoted all four WMMA-attention cells from orange to yellow at
  `65a64bb1bb`.  Fresh clean artifact
  `consan-validation-gfx1250-wmma-clean-192` accepts baseline and every
  standard profile.  Current 16-bit flat access support raises each profile
  from 8/8 to 18/18 patched accesses.  SuperCollider and Inline Shadow are
  statically and dynamically complete; Inline also patches 8/8 barriers.
  Record/Replay patches 8/8 barriers and Sampled admits none, while both retain
  dynamic completeness and typed exclusions for unrelated unqualified runtime
  synchronization.  Fault, resource, and frozen-provenance gates remain before
  green.

- 2026-07-18: Promoted all four D128-pressure cells directly from yellow to green
  with a frozen bundle at `028fec503a`.  Inventory artifact
  `consan-validation-gfx1250-d128-pressure-inventory-190`, clean artifact
  `consan-validation-gfx1250-d128-pressure-clean-191`, contained barrier-drop
  artifact `consan-validation-gfx1250-d128-pressure-fault-188`, and paired
  resource artifact `consan-validation-gfx1250-d128-pressure-overhead-189` all
  name that clean revision.  All clean profiles pass their oracle and coverage
  contract.  The exact barrier mutation is accepted in all four profiles:
  Inline Shadow emits 32 race diagnostics, while the other profiles produce
  their reviewed qualified misses, and the independent workload oracle fails
  in every row.  Paired median slowdowns are 2.22x, 1.14x, 1.15x, and 1.30x;
  retained instrumentation-owned peaks are 4, 463,792, 41,904, and 13,830,816
  bytes respectively, with no allocation, capacity, cleanup, timeout, or
  health failure.

- 2026-07-18: D128-pressure contained barrier-drop execution reached all four
  profiles in artifact `consan-validation-gfx1250-d128-pressure-fault-187`.
  Exact mutation accounting is 1/1/1, both health gates pass, and the workload
  oracle fails in every profile.  SuperCollider, Record/Replay, and Sampled
  produce their expected qualified misses.  Inline Shadow emits 32 race
  diagnostics, contradicting the initial conservative `not_detected`
  expectation; its reviewed policy is corrected to require `detected` before
  the acceptance rerun.  The row remains yellow until that fresh fault run and
  the resource/frozen-revision gates pass.

- 2026-07-18: Promoted all four D128-pressure cells from orange to yellow at
  `93c00da105`.  Clean artifact
  `consan-validation-gfx1250-d128-pressure-clean-184` accepts the baseline and
  all four standard profiles from that clean revision.  SuperCollider is
  statically and dynamically complete at 40/40 accesses.  Record/Replay and
  Inline Shadow cover 32/32 accesses and 8/8 barriers; Sampled covers 32/32
  accesses.  All three are dynamically complete with their typed static
  exclusions retained.  Inline Shadow now compacts wide external publication,
  uses the target's extended descriptor-local LDS capacity, and resolves
  heuristic direct flat sites against the shared aperture at runtime.  This
  removes the prior crash and false undercoverage: the full Inline suite
  finishes in 39.28 seconds with zero dynamic-incomplete events.  Fault,
  resource, and frozen-provenance gates remain before green.

- 2026-07-18: Promoted all four D128-block cells directly from yellow to green
  with a frozen bundle at `457d512a71`.  Inventory artifact
  `consan-validation-gfx1250-d128-block-freeze-inventory-170`, clean artifact
  `consan-validation-gfx1250-d128-block-freeze-clean-171`, contained
  barrier-drop artifact
  `consan-validation-gfx1250-d128-block-freeze-fault-172`, and paired resource
  artifact `consan-validation-gfx1250-d128-block-overhead-169` all name that
  revision.  Every clean profile passes its oracle with 18/18 accesses; the
  relevant Record/Replay and Inline barrier denominators are 8/8.  Every fault
  row applies exactly one selected barrier, reaches the precommitted failing
  oracle without a false diagnostic, and passes before/after target health.
  Relative to the paired 4.891-second baseline, profile slowdowns are 1.98x,
  1.17x, 1.18x, and 1.19x.  Retained report peaks are 4, 461,776, 23,760, and
  12,600,672 bytes, with complete cleanup.

- 2026-07-18: Promoted all four D128-block cells from orange to yellow at
  `8240fd71e2`.  Gfx1250-specific 16-bit group-flat load/store instrumentation
  closes the missing short-access paths.  Clean artifact
  `consan-validation-gfx1250-d128-block-clean-165` passes every workload
  oracle: SuperCollider and Inline are statically and dynamically complete at
  18/18 accesses, with Inline also covering 8/8 barriers; Record/Replay and
  Sampled cover 18/18 accesses with dynamic completeness.  Fault and resource
  qualification is the remaining work for this row.

- 2026-07-18: Promoted all four tree atomic-OR cells from yellow to green at
  frozen revision `0669775d94`.  The same-tip bundle consists of clean artifact
  `consan-validation-gfx1250-tree-freeze-clean-156`, inventory artifact
  `consan-validation-gfx1250-tree-freeze-inventory-157`, order and scope
  artifacts `consan-validation-gfx1250-tree-freeze-order-158` and
  `consan-validation-gfx1250-tree-freeze-scope-159`, and paired resource
  artifact `consan-validation-gfx1250-tree-freeze-overhead-160`.  Baseline and
  all four clean profiles pass their independent oracle and exact dynamic
  denominators.  Both contained fault families accept all four reviewed rows
  with exact `requested=1 planned=1 applied=1` accounting and healthy target
  smokes before and after; Inline diagnoses the scope mutation and every other
  disposition is the precommitted qualified miss.  Relative to the paired
  388.5 ms baseline, SuperCollider, Record/Replay, Sampled, and Inline run at
  16.54x, 3.96x, 4.14x, and 4.12x.  Their retained report peaks are 4 bytes,
  893,936 bytes, 5,616 bytes, and 12,599,328 bytes, with zero allocation,
  capacity, or cleanup failures and zero live MOI bytes after cleanup.  Every
  source identity is clean and names the frozen revision.

- 2026-07-18: Completed tree's resource qualification while its cells remained
  yellow.  Three-sample paired bundle
  `consan-validation-gfx1250-tree-overhead-155` accepts baseline-before,
  every profile, and baseline-after with all workload oracles and coverage
  gates retained.  Relative to the paired 390/391 ms baseline medians,
  SuperCollider is 16.0x at 4 bytes of report storage, Record/Replay is 3.90x
  at 893,936 peak bytes, Sampled is 4.14x at 5,616 peak bytes, and Inline is
  4.13x at 12,599,328 peak bytes.  Every MOI buffer returns to zero live bytes
  with no allocation, capacity, or cleanup failures.  Only the complete
  clean/inventory/fault/overhead rerun at one clean committed tip remains
  before green.

- 2026-07-18: Completed tree's fault qualification while its cells remained
  yellow.  Retained order campaign
  `consan-validation-gfx1250-tree-fault-order-150` and scope campaign
  `consan-validation-gfx1250-tree-fault-scope-154` accept all eight profile
  rows against the committed policy.  Every row has exact
  `requested=1 planned=1 applied=1` accounting, the expected independent
  oracle outcome, no timeout, and healthy target-dispatch smokes before and
  after.  The order mutation is a qualified miss in all four profiles.  The
  scope mutation is a qualified miss in SuperCollider, Record/Replay, and
  Sampled, while Inline emits the required diagnostic and retains the
  numerically correct workload result.  Paired overhead, peak memory, and the
  final same-tip freeze remain open.

- 2026-07-18: Corrected the tree scope-fault oracle policy transparently after
  the first contained campaign contradicted its precommitted `fail` outcome.
  The full campaign and two additional isolated Inline repetitions all applied
  exactly one mutation, emitted the required Inline diagnostic, retained
  healthy before/after target smokes, and passed the independent numerical
  oracle.  This is semantically valid: weakening the final acquire-release
  RMW's scope creates the diagnosed cross-wave race but does not require a
  wrong numerical result, and the current deterministic schedule retains the
  producer values.  The reviewed policy now requires `detected/pass` for
  Inline.  This correction is committed before the qualifying rerun and does
  not retroactively accept the three policy-mismatching artifacts.

- 2026-07-18: Completed tree's inventory and reviewed policy before executing
  a mutation.  Fresh bounded inventory
  `consan-validation-gfx1250-tree-inventory-148` completes both admitted
  atomic fault families and records the exact agent-scope acquire-release
  helper used by the workload.  The reviewed target policy now precommits the
  order and scope expectations for every profile.  Removing only the final
  acquire-release RMW's release edge is expected to preserve the oracle and
  remain undetected because no later operation consumes that release edge;
  weakening its scope is expected to break cross-wave acquisition, with
  Inline required to diagnose it and the independent oracle expected to fail.
  No contained mutation from this policy had been run when the policy commit
  was prepared.

- 2026-07-18: Promoted all four tree atomic-OR cells from partially-working orange
  to mostly-working yellow.  Fresh retained bundle
  `consan-validation-gfx1250-tree-clean-147` accepts the baseline and every
  standard profile with the independent oracle, zero forbidden diagnostics or
  overflow, and complete required dynamic coverage.  SuperCollider patches
  4/4 accesses; Record/Replay patches 4/4 accesses, 8/8 barriers, 10/10
  atomics, and 16/16 fences; Sampled patches 4/4 accesses and 10/10 atomics;
  Inline patches 4/4 accesses, 8/8 barriers, and 10/10 atomics.  Inline's
  release-sequence chain now retains all transitive producer evidence, and a
  dedicated `+26:+27` scalar pair preserves application EXEC independently of
  the causal-token authorization masks.  The 642-test ConSan/MOI host gate
  passes.  Fault policy, contained mutations, resource gates, and a committed
  freeze remain before any tree cell can advance beyond yellow.

- 2026-07-18: Stream-K is the first completely green row.  At frozen revision
  `a8f4172f64`, artifact `consan-validation-gfx1250-streamk-frozen-clean-121`
  accepts the baseline and all four clean profiles with clean source and hook
  provenance.  Paired three-sample artifact
  `consan-validation-gfx1250-streamk-frozen-overhead-122` records a 250.5 ms
  baseline and SuperCollider, Record/Replay, Sampled, and Inline medians of
  5,559, 1,356, 1,406, and 1,420 ms: 22.19x, 5.41x, 5.61x, and 5.67x.
  Instrumentation-owned peak report storage is respectively 4, 893,936,
  5,616, and 12,599,328 bytes; all MOI reports return to zero live bytes with
  no allocation, capacity, or cleanup failures.  Frozen contained artifacts
  `consan-validation-gfx1250-streamk-frozen-order-123` and
  `consan-validation-gfx1250-streamk-frozen-scope-124` accept every profile
  with exact mutation accounting, the reviewed detector/oracle dispositions,
  30-second bounds, and successful discovery plus independent target smoke
  before and after each destructive row.

- 2026-07-18: Stream-K completes inventory and reviewed policy in all four
  yellow cells.  Fresh same-state clean artifacts accept SuperCollider at `4/4`
  accesses, Record/Replay at `4/4` accesses plus `8/8` barriers, Sampled at
  `4/4` accesses, and Inline Shadow at `4/4` accesses plus `8/8` barriers;
  every MOI profile admits and patches `10/10` atomics with no dynamic
  incompleteness.  Inline's prior clean false diagnostic came from treating a
  full-width workgroup-local shadow as though it stored the global generation
  field; both compact and full local shadows now use their persistent
  workgroup key when consulting global atomic-token tables.
- 2026-07-18: Both reviewed Stream-K fault families pass contained execution
  under all four profiles with exact `requested=1 planned=1 applied=1`
  mutation accounting and healthy discovery plus independent target smoke
  before and after every row.  Weakened order remains a precommitted qualified
  miss with a passing workload oracle.  Weakened scope is likewise a reviewed
  miss for SuperCollider, Record/Replay, and Sampled; Inline deterministically
  reports one exact conflict and the deliberately weakened cross-workgroup
  workload fails its independent oracle.  That Inline outcome reproduced in
  a second contained run before being retained in the target policy.  Paired
  overhead, resource bounds, timeout qualification, and a final clean
  committed-revision rerun remain open, so the row remains yellow rather than
  green.

- 2026-07-18: The first contained Jakub barrier mutation reached exact
  `requested=1 planned=1 applied=1`, preserved a healthy target before and
  after, and produced the precommitted SuperCollider qualified miss, but the
  mutated workload did not reach its oracle within 60 seconds.  The timeout is
  retained as a failed row, not a diagnostic.  The next precommitted campaign
  uses the directly exercised Stream-K acquire/release atomic, whose order and
  scope mutations avoid the barrier-removal execution stall.

- 2026-07-18: Every non-CLIP matrix workload now has an accepted fresh fault
  inventory.  The barrier rows retain exact target identities and logical
  pairs; both atomic workloads expose the exact agent-scope acquire/release
  helper for order and scope weakening.  The latter also proves that their
  clean 0/0 atomic denominators are not semantic absence, so clean atomic
  admission remains an implementation gap.  A reviewed Jakub policy is now
  precommitted as the first contained mutation campaign.  Inventory alone
  does not promote any cell.

- 2026-07-18: Fault inventory no longer waits for an unmodified workload after
  its required static evidence has been emitted.  The collector requires a
  family-relevant site followed by the same code-object reader's coverage
  record, terminates deliberately only after that proof, and rejects an
  ordinary timeout.  Artifact
  `consan-validation-gfx1250-jakub-fault-inventory-062` is accepted in 0.35
  seconds and retains exactly eight barrier sites plus four signal/wait
  sequences.  This advances the fault campaign but does not promote a matrix
  cell until a reviewed policy and contained exact mutation are also retained.

- 2026-07-18: Removed CLIP BF16 from the current acceptance matrix.  Short,
  uninstrumented diagnostics proved that the immediate execution problem is
  not introduced by ConSan: the default multi-executor baseline can stall
  during module loading, while a single-executor baseline promptly completes
  module loading but remains in the first inference beyond a useful short
  bound.  The existing 90/90 access and synchronization qualification evidence
  is retained below, but CLIP no longer contributes cells to the active gfx1250
  acceptance denominator.

- 2026-07-18: All four CLIP clean cells now have retained target-code
  diagnostic evidence.  Record/Replay artifact
  `consan-validation-gfx1250-clip-rr-053` patches 90/90 accesses and 48/48
  barriers.  Sampled artifact `consan-validation-gfx1250-clip-sampled-054`
  patches 90/90 accesses and all 38 qualified barriers; ten other barrier
  sites retain the typed `unqualified_sync_sequence` exclusion.  Inline
  Shadow artifact `consan-validation-gfx1250-clip-inline-055` patches 90/90
  accesses and 48/48 barriers.  No profile reports a resource or
  placement/lowering failure.  These short diagnostic runs reach 60 seconds
  before the workload oracle, so they promote the three formerly unknown
  cells only to yellow; long-bound clean acceptance remains active.
- 2026-07-18: SuperCollider's CLIP inventory exposed 16 omitted
  full-register `ds_load_u16` operations among 90 otherwise supported LDS
  accesses.  The one-dword lowering now supports that form, with a distinct
  target-generated regression from the existing partial-register `_d16`
  paths.  All 628 ConSan host tests pass.  Retained short-bound artifact
  `consan-validation-gfx1250-clip-sc-u16-052` reports 90 discovered, 90
  supported, 90 selected, and 90 patched accesses, with zero resource or
  placement/lowering failures.  It reaches the 60-second diagnostic bound
  before the independent workload oracle, so the cell remains yellow while a
  realistic long-bound run and the other three clean profiles remain open.

- 2026-07-18: Started the matrix-first CLIP campaign.  The complete doctor and
  four-profile contract audit pass with no coverage-limiting controls or
  workload tuning.  Artifact `consan-validation-gfx1250-clip-clean-050` is the
  first genuine target run: after restoring the software-device plumbing, its
  baseline compiles, loads, and executes gfx1250 code but reaches the
  600-second process bound before producing the independent oracle.  The
  SuperCollider process subsequently reached the same bound after patching
  only 74/90 accesses; the later `ds_load_u16` fix supersedes that incomplete
  static result.  This promotes only that active evidence cell to yellow;
  unexecuted profiles remain gray.
- 2026-07-18: Created the ledger.  Confirmed target-code execution through the
  workspace TheRock runtime and verified the executed binary's gfx1250 offload
  bundle.  All 44 workload/profile cells intentionally begin unknown; no
  focused or cross-architecture result was promoted into the e2e matrix.
- 2026-07-18: Linked the generated gfx1250 ISA backend into the standalone
  ConSan hook.  A fresh instrumented hip-moi dispatch now loads the hook and
  decodes both runtime and workload code objects instead of failing at dynamic
  symbol resolution.  The first SuperCollider run then fails closed at the
  expected next porting boundary: four supported accesses are selected but
  their target lowering is not yet implemented.
- 2026-07-18: Closed that first lowering boundary with target-generated
  VDS/VFLAT tests and target-specific wait, compare, readback, and report
  emission.  The instrumented ping-pong clean case now reports 4 discovered,
  4 supported, 4 selected, and 4 patched accesses, passes final validation,
  and passes its host-reference oracle.  This is retained bootstrap evidence,
  not yet a green matrix cell under the full validation contract.
- 2026-07-18: Added gfx1250 scalar indirect-call recovery, including
  wide-literal targets and shared VFLAT helper ownership, then enabled the
  Record/Replay access, prologue, and inline barrier paths.  A real clean run
  initially exposed guest-register corruption caused by using the wrong
  wave32 descriptor granularity.  Correcting field 4 to an 80-VGPR allocation
  boundary placed persistent state at v80:v81 and scratch at v82; the workload
  then passed its oracle with 4/4 accesses and 8/8 barriers patched and visible
  access records.  The broad ConSan host gate passes 610/610 tests.  The run is
  still incomplete at object scope because unsupported synchronization sites
  remain, so it is bootstrap evidence rather than a promoted workload cell.
- 2026-07-18: Enabled target-generated Sampled publication atomics and the
  carry-chain address operation used by its standard-profile multi-bank path.
  A real clean ping-pong run passed its independent oracle with 4/4 accesses
  patched and two visible dynamic records.  The same run reports all 8
  barriers and 13 atomics as unsupported, so static completeness remains false
  and no workload/profile cell is promoted.
- 2026-07-18: Enabled the gfx1250 Inline Shadow access lowering.  Its real
  standard-profile clean ping-pong run passed the independent oracle with
  analysis, static coverage, and dynamic coverage all complete: 4/4 accesses,
  8/8 barriers, and one visible evidence record.  This completes the clean
  focused probe only; fault and diagnostic acceptance remain open, so no e2e
  workload/profile cell is promoted.
- 2026-07-18: Registered the gfx1250-native hip-moi workload executables and
  filters in the validation harness.  The d128-block scoped doctor passes with
  the workspace hook, target executable, and TheRock `rocminfo`; all four
  profile commands expand through `explain`.  The full doctor intentionally
  continues to report the missing target-native Jakub workload, and no status
  cell moves until an actual retained validation run satisfies its contract.
- 2026-07-18: Ran the first retained validation row for D128 block attention.
  Baseline and all four clean profiles pass the independent workload oracle
  and the harness clean gate.  SuperCollider and Sampled patch 8/8 admitted
  accesses; Record/Replay and Inline Shadow patch 8/8 accesses plus 8/8
  barriers.  All profiles still report object-wide static incompleteness, the
  source provenance records unrelated workspace dirt, and fault/overhead
  evidence is absent.  The four row cells therefore advance to yellow,
  not accepted green.
- 2026-07-18: Completed clean D128-pressure validation for baseline and all
  four profiles.  SuperCollider and Sampled patch 8/8 admitted accesses;
  Record/Replay and Inline Shadow patch 8/8 accesses plus 8/8 barriers.  The
  Inline Shadow external-table path required target-generated lane-read and
  signed address-carry operations.  Its newly admitted B128 load then exposed
  and fixed a shared far-trampoline ordering defect in which a relocated
  consumer could overwrite the address before the deferred guest load.  The
  uncapped rerun passes all four workload variants with dynamic completeness,
  and a forced far-island gfx1250 host regression preserves the ordering.
  Static completeness, committed clean provenance, fault, containment,
  overhead, memory, timeout, and health evidence remain open, so all four
  pressure cells are yellow rather than green.
- 2026-07-18: Clean WMMA-attention validation passes baseline and all four
  profiles at `eeebbaf9fb`.  SuperCollider and Sampled patch 8/8 admitted
  accesses; Record/Replay and Inline Shadow patch 8/8 accesses plus 8/8
  barriers.  All runs pass the independent workload oracle and dynamic gate,
  but report object-wide static incompleteness.  Fault and promotion evidence
  remain absent, so the four WMMA cells advance to yellow rather than
  green.
- 2026-07-18: Clean Stream-K arrival validation passes baseline and all four
  profiles at `4eb8c6f1dd`.  All profiles patch 4/4 admitted accesses;
  Record/Replay and Inline Shadow also patch 8/8 barriers.  SuperCollider and
  Inline Shadow report static plus dynamic completeness; Record/Replay and
  Sampled are dynamically complete but object-wide statically incomplete.
  The coverage contract admits 0/0 atomics in every profile; that observation
  is not promoted to a semantic-absence claim without the reviewed fault
  inventory.  All four cells are yellow pending that review and the rest
  of the promotion contract.
- 2026-07-18: Clean tree atomic-OR validation passes baseline and all four
  profiles at `7774231dd4`.  Every profile patches 4/4 admitted accesses;
  Record/Replay and Inline Shadow also patch 8/8 barriers.  SuperCollider and
  Inline Shadow are statically and dynamically complete, while Record/Replay
  and Sampled remain object-wide statically incomplete.  As in Stream-K, the
  admitted atomic denominator is 0/0 and is not treated as proof of semantic
  absence.  All target-native hip-moi clean rows are now yellow; fault,
  atomic-inventory, provenance, and full promotion evidence remain open.
- 2026-07-18: Sharktank TP1 prefill now passes its clean baseline and all four
  profile oracles.  Record/Replay is complete at 352/352 accesses and 74/74
  barriers.  Sampled covers 352/352 accesses and all 24 supported barriers,
  with the other 48 synchronization sites retained as typed exclusions.
  Inline Shadow is statically and dynamically complete at 352/352 accesses
  and 74/74 barriers and executes 72 live-register spill patches, growing the
  high-pressure kernel's private segment through 152 bytes per lane.  A host
  backtrace proved the initial first-spill crash occurred in the software
  GPU's scratch write after it ignored descriptor-grown private memory; an
  internal model correction made the unchanged ConSan sequence pass the exact
  baseline oracle.  SuperCollider preserves the oracle but remains incomplete
  at 280/352 accesses, so all four TP1 cells are yellow rather than green.
- 2026-07-18: Retained TP1 SuperCollider validation at
  `consan-validation-gfx1250-tp1-sc-spill-006` closes that clean-coverage gap.
  The exact oracle passes, analysis is statically and dynamically complete,
  and coverage is 352/352 accesses.  All 72 formerly skipped sites execute
  seven-VGPR save/restore windows; the shared private allocation reaches 60
  bytes per lane.  A forced 256-live-VGPR host regression covers the fallback,
  and the complete host suite passes 2174/2174.  TP1 prefill is now
  clean-complete in all four profiles, but every cell remains yellow until its
  reviewed fault, containment, overhead, memory, timeout, health, and frozen
  committed-revision evidence is retained.
- 2026-07-18: TP1 decode/combined now has retained clean evidence for baseline
  and all four profiles at `consan-validation-gfx1250-tp1-decode-010`.
  SuperCollider covers 704/704 accesses; Record/Replay and Inline Shadow are
  statically and dynamically complete at 704/704 accesses and 148/148
  barriers; Sampled covers 704/704 accesses and all 48 supported barriers with
  typed exclusions for the other synchronization sites.  Every workload
  oracle passes.  Inline Shadow takes 152.8 seconds in the software GPU
  environment, so the retained run uses a 600-second bound instead of the
  generic 30-second default.  These cells remain yellow pending their complete
  fault, performance, containment, health, and frozen-provenance contract.
- 2026-07-18: TP2-family baseline and all four clean profiles pass at
  `consan-validation-gfx1250-tp2-011`.  Every profile covers 2760/2760
  accesses; Record/Replay and Inline Shadow cover 288/288 barriers, while
  Sampled covers all 48 supported barriers and retains typed exclusions for
  the rest.  SuperCollider, Record/Replay, and Inline Shadow are statically
  and dynamically complete, and every independent workload oracle passes.
  Inline Shadow takes 365.3 seconds, within the predeclared 600-second
  software-environment bound.  The four cells remain yellow until the full
  promotion contract is retained.
- 2026-07-18: The retained Qwen SuperCollider clean run at
  `consan-validation-gfx1250-qwen-sc-012` passes the exact full-logits oracle
  with static and dynamic completeness at 1000/1000 accesses.  Canonical
  Sampled artifact `consan-validation-gfx1250-qwen-sampled-015` now also passes
  the exact oracle and is accepted at 1000/1000 accesses plus all 56 supported
  barriers, with typed synchronization exclusions, complete dynamic evidence,
  no diagnostics, and no overflow in 80.5 seconds.  The Sampled fix replaces a
  fixed per-site gate reservation with a demand-derived bound and adds a
  packed 1000-site regression; the launch fix recognizes binary-compatible AMD
  extended dispatch packets and propagates their private spill requirement.
  The software GPU environment separately required its queue ABI to match the
  workspace ROCr v2 layout.  Both Qwen cells are yellow pending fault,
  containment, performance, health, and frozen committed-tip promotion
  evidence; Record/Replay and Inline Shadow remain open.
- 2026-07-18: Canonical Qwen Record/Replay artifact
  `consan-validation-gfx1250-qwen-rr-018` passes the exact full-logits oracle
  with 1000/1000 accesses, 92/92 barriers, 1994 visible access records, and
  complete static, dynamic, and host replay analysis in 84.8 seconds.  It has
  zero dropped or unsupported events, diagnostics, metadata exhaustion, and
  overflow.  The preceding run exposed that high-pressure kernels selected
  spill-private persistent epochs but Record/Replay did not attach them to its
  access records, forcing roughly 19.8 million dynamic barrier events through
  a bounded trace.  Access probes now publish the private epoch, entry probes
  initialize it, and barrier probes advance it in place; a focused regression
  proves all three share one private slot, and all 616 ConSan plus 32 hook
  tests pass.  The Qwen Record/Replay cell is yellow pending fault,
  containment, performance, health, and frozen committed-tip promotion.
- 2026-07-18: Qwen Inline Shadow remains the last open clean P0 profile.
  Empty diagnostic and undercoverage paths, local-shadow workgroup-key
  specialization, and descriptor-aware once-per-workgroup initialization now
  remove substantial cold and redundant work.  A direct filtered run of the
  151936x1024 transpose initializer passes the exact logits oracle with 3/3
  accesses, 4/4 barriers, one visible evidence event, and zero diagnostics or
  overflow; the initialization correction reduced that focused run from
  roughly 60 seconds to roughly 29 seconds.  Subsequent clean-path filtering
  retains all 626 ConSan and 32 hook tests.  Canonical artifacts through
  `consan-validation-gfx1250-qwen-inline-028` still time out at the retained
  1,200-second bound before producing an analysis verdict, so the cell is yellow
  rather than accepted and the next committed-tip canonical run is active.
- 2026-07-18: Inline aggregate-evidence publication now uses a prologue-zeroed
  persistent scalar latch when automatic allocation proves the complete
  scalar window lies above every guest-referenced SGPR.  The first executed
  access in each wave retains the original global evidence path; later sites
  skip it, while explicit and liveness-only windows keep the conservative
  per-access behavior.  All 636 ConSan-related tests and all 32 hook tests
  pass, and the filtered Qwen initializer remains exact with one visible event
  and no diagnostics or overflow.  Canonical artifact
  `consan-validation-gfx1250-qwen-inline-030` nevertheless reaches the
  1,200-second limit in the same 508th final dispatch, so the next frontier is
  its serial workgroup-local shadow initialization rather than a wider timeout.
- 2026-07-18: Commit `183904667d` replaces that serial clear with an exact
  first-x-wave initializer.  It derives its stride from the run-time active
  lane count, selects only the zero outer coordinates, and bounds-masks both
  the initial and final store batches.  The focused 151936x1024 Qwen transpose
  remains exact at 3/3 accesses and 4/4 barriers, one visible evidence event,
  and zero diagnostics or overflow; all 636 ConSan-related tests and all 32
  hook tests pass.  Its 32.75-second software GPU time is not an improvement
  over the prior roughly 29--30-second result, because the environment still
  executes the same aggregate lane/store work.  Qwen Inline remains yellow, and
  the next measured frontier is one 64-bit LDS zero store per eight-byte
  shadow slot instead of two 32-bit stores.
- 2026-07-18: Commit `036f987835` emits one 64-bit LDS store per local shadow
  slot on gfx1250 while retaining the two-store sequence required by gfx950.
  Exact target-byte and decoder tests, all 636 ConSan-related tests, and all 32
  hook tests pass; the filtered transpose remains exact in 32.20 seconds.
  Canonical artifact `consan-validation-gfx1250-qwen-inline-032` nevertheless
  reaches 1,200.51 seconds in the same final dispatch without an analysis
  verdict.  The replacement reserves a proven entry-local zero tuple above
  every guest VGPR and clears two slots per lane with a 128-bit store.  Exact
  encoder/decoder and odd-slot fallback tests pass; the full host gate passes
  2,178/2,178, the final ConSan gate passes 619/619, and the real filtered
  transpose again passes its exact oracle at 3/3 accesses and 4/4 barriers
  with one evidence event and no diagnostics or overflow.  Qwen Inline remains
  yellow until the next canonical run completes the same gates over all 508
  dispatches.  Artifact `consan-validation-gfx1250-qwen-inline-033` is not
  evidence for the 128-bit implementation: its retained provenance records a
  stale validator-default hook.  That default hook has been rebuilt from the
  current branch tip, so the replacement canonical run will have auditable
  implementation provenance.
- 2026-07-18: Replacement artifact
  `consan-validation-gfx1250-qwen-inline-034` records the rebuilt 128-bit hook
  hash, but reaches 1,200.55 seconds in dispatch 508 without a verdict.  This
  validly closes width-only clearing as an insufficient optimization.  Qwen
  Inline remains yellow while a compact exact initialization representation is
  implemented and proven.
- 2026-07-18: The target-native Jakub executable now exists at the registry's
  expected path.  Its no-pipeline, pipelined, and double-buffered production
  shapes all pass their independent host-reference oracle, and the full
  gfx1250 validation doctor succeeds.  The four profile cells remain unknown
  until retained instrumented clean runs pass their coverage and oracle gates.
- 2026-07-18: The initial Jakub profile artifact exposed 0/0 admitted accesses
  because the compiler's generic-flat shared pointers cross scratch slots,
  scalar lane reservoirs, and a 64-bit vector add address construction.  The
  tracker now preserves provenance through each gfx1250 shape, with focused
  regressions and the complete 267-test ConSan gate passing.  Retained artifact
  `consan-validation-gfx1250-jakub-clean-037` accepts Record/Replay with its
  independent oracle, 31/31 accesses, 8/8 barriers, complete dynamic evidence,
  and no coverage rejection.  The cell is yellow rather than green because
  object-wide static exclusions and the fault, containment, overhead, memory,
  health, and frozen-tip promotion gates remain open.
- 2026-07-18: Canonical committed-tip artifact
  `consan-validation-gfx1250-jakub-clean-040` accepts all four clean profiles.
  The previous SuperCollider crash at the 24th selected site was a ConSan
  instrumentation defect: a statically ambiguous generic-flat address could
  be non-group at run time, making its unconditional duplicate readback
  unsafe.  gfx1250 SuperCollider now executes the original access and gates
  only its redundant probe on the runtime group-aperture high half.  All 590
  ConSan/ConSanMoi tests pass.  SuperCollider is statically and dynamically
  complete at 62/62 accesses; Record/Replay and Inline Shadow accept 31/31
  accesses plus 8/8 barriers; Sampled accepts 31/31 accesses and its 0/0
  admitted barrier denominator.  The four cells are yellow pending reviewed
  faults, containment, overhead, peak memory, health, and one frozen full
  campaign, rather than green on clean evidence alone.
- 2026-07-18: Required-workgroup-size metadata now reaches the Inline local
  shadow layout.  A proven `64x1x1` gfx1250 kernel distributes its 128-bit
  clears across both wave32 waves, while missing or narrow metadata retains
  the conservative 32-lane path.  The focused Qwen transpose remains exact at
  3/3 accesses and 4/4 barriers with one visible event, and all 590
  ConSan/ConSanMoi tests pass.  The measured software-GPU time remains roughly
  31 seconds versus the prior 32.20 seconds, so the Qwen cell stays yellow: its
  blocker is the aggregate initialized state, not insufficient lane
  distribution.
- 2026-07-18: The near-capacity local-mirror experiment at `11d5c0d009`
  selected the pre-zeroed external exact table for gfx1250 layouts above
  48 KiB.  It retained the filtered Qwen transpose oracle, 3/3 accesses, and
  4/4 barriers while reducing that isolated run to roughly 25 seconds.
  Canonical all-dispatch artifact
  `consan-validation-gfx1250-qwen-inline-041` has exact source/hook provenance
  but still timed out at 1,200.57 seconds in the same final 2,374-workgroup
  dispatch.  The Inline cell remains yellow: neither wider clearing nor external
  exact storage meets the full-model bound.  The active implementation
  frontier is an exact local owner/epoch mirror with a small packed validity
  bitmap, so only the bitmap is cleared eagerly and metadata slots are made
  ready on first use.
- 2026-07-18: The packed-validity implementation now host-qualifies.  For the
  Qwen 21,120-byte LDS shape it retains the full 42,240-byte exact metadata
  mirror and adds a 1,328-byte bitmap, for a 64,688-byte descriptor allocation.
  Only that bitmap is cleared at entry.  Atomic initializing/ready bits order
  first-use slot zeroing before any exact metadata exchange, including
  same-cell contenders in different waves.  Focused layout, emitted-target,
  and final-ELF tests pass together with all 625 ConSan host tests.  The Inline
  cell remains yellow until focused and canonical execution prove correctness,
  completeness, and the 1,200-second bound.
- 2026-07-18: A target-native two-workgroup runtime probe with the same
  21,120-byte guest LDS footprint proves that the packed first-use protocol
  completes without a readiness deadlock: its independent oracle passes with
  1/1 admitted access, 2/2 barriers, and one visible event after the descriptor
  grows to 64,688 bytes.  The steady-state path now observes a ready cell with
  an ordinary LDS load and bypasses every atomic state transition; all 625
  ConSan host tests pass.  The canonical Qwen final kernel still exceeds a
  focused 300-second bound both before and after that fast path, without a
  verdict.  The Inline cell therefore remains yellow.  Correctness of the lazy
  state machine is established, but its eight-byte metadata exchange and
  validity traffic remain too costly at 2,374-workgroup scale; exact compact
  four-byte local state is the active implementation frontier.
- 2026-07-18: The four-byte exact local state now executes.  Its low 23 bits
  preserve the canonical kind/owner/epoch contract and its high nine bits name
  a per-kernel site; the auto-report path resolves that token to the full prior
  instruction offset before analysis and treats missing or ambiguous mappings
  as malformed.  The Qwen-shaped descriptor is 42,240 bytes rather than
  64,688, and a target-native two-workgroup probe passes its oracle with 1/1
  access, 2/2 barriers, and one visible event.  All 626 ConSan host tests pass.
  Artifact `consan-validation-gfx1250-qwen-inline-compact-042` diagnosed an
  object-wide 511-token exhaustion that left the final kernel on the lazy
  layout.  Per-direct-kernel token domains fix that without allowing ambiguous
  shared-helper provenance.  Replacement artifact
  `consan-validation-gfx1250-qwen-inline-compact-043` confirms the final
  2,374-workgroup dispatch grows from 21,120 to 42,240 bytes, but it still
  exceeds a diagnostic 300-second bound.  The Inline cell remains yellow while
  its retained 1,200-second filtered acceptance run completes.
- 2026-07-18: The retained eager-compact acceptance artifact
  `consan-validation-gfx1250-qwen-inline-compact-044` exceeded 1,200 seconds
  without a verdict.  Artifact
  `consan-validation-gfx1250-qwen-inline-compact-onesite-045` also exceeded
  300 seconds while patching only one of the final kernel's 120 sites, which
  isolates the dominant cost to eagerly clearing 21,120 bytes in every one of
  2,374 workgroups.  The replacement keeps the exact four-byte cell and adds
  a 1,328-byte packed two-bit first-use map; the descriptor is now 43,568
  bytes and only touched cells are cleared.  Compact diagnostic mappings are
  stored as bounded descriptor-qualified 16-byte records in the planned
  report allocation, with capacity and provenance failures counted as
  malformed.  All 626 ConSan host tests pass, and the target-native
  two-workgroup probe passes its oracle, 1/1 access, 2/2 barriers, one visible
  event, and process teardown with exit status zero.  The Qwen Inline cell
  remains yellow pending the isolated and canonical runs.
- 2026-07-18: Canonical compact-lazy artifacts `046` and `047` both exceeded
  300 seconds with all 840 patches because the validator deliberately strips
  diagnostic patch filters.  Direct A/B controls complete the access-free
  workload in about 19 seconds, while a genuine one-site access exceeds 300
  seconds and remains over 120 seconds with barrier and atomic tracking off.
  The first selected hot site is a two-address 64-bit LDS store.  The compact
  emitter now handles each adjacent pair with one exact 64-bit exchange,
  preserving independent readiness and diagnostic checks for both cells.
  Final-ELF validation and all 627 ConSan host tests pass.  Runtime evidence is
  intentionally pending, so this matrix cell remains yellow.
- 2026-07-20: Record/Replay no longer rejects `torch.sort` when a gfx1250
  barrier epoch body outgrows its claimed short NOP island.  The claimed
  island is retained as the indirect entry to a general appended body.
  Artifact `consan-gfx1250-pytorch-sort-rr-109` passes the exact values and
  indices oracle, patches 47,840/48,224 accesses and all 12,064 barriers,
  emits 55,256 records without loss, and is dynamically complete.  The cell
  advances from orange to yellow; its remaining blocker is static/access
  completeness.  All 661 ConSan host tests pass.  Histogram rechecks `110`
  and `111` still fail before a verdict because their selected object has no
  successful supported-site placement, proving that row has a separate
  blocker rather than the fixed oversized-body failure.
- 2026-07-20: The remaining `torch.sort` engine probes establish two more
  concrete states.  Inline Shadow artifact `113` passes the exact oracle,
  patches 48,056/48,224 accesses and all 12,064 barriers, and is dynamically
  complete, promoting that cell to yellow.  Sampled artifact `112` reaches a
  selected large object with supported sites but no successful patch and
  exits before an oracle, so that cell is orange rather than an untested
  orange.
- 2026-07-20: The P2 reduction row is now assessed in every engine.
  Record/Replay artifact `116` passes the exact 3-4-5 norm and CPU-reference
  softmax oracles, patches 4,425/4,756 accesses and 4,297/4,704 barriers, and
  is dynamically complete, promoting that cell to yellow.  Sampled `114` and
  Inline `115` both stop because their selected large reduction object has no
  globally free persistent dispatch-ID SGPR pair; those cells are orange.
  Together with the SuperCollider isolation, this identifies object-wide
  scalar-state allocation rather than literal64 decoding as the remaining
  cross-engine frontier.
- 2026-07-20: Sampled reduction artifact `118` promotes its cell from orange
  to yellow.  ConSan now reserves a legal high dispatch-ID pair and excludes
  only owner groups whose guest scalar allocation reaches it, instead of
  rejecting the complete 192-kernel object.  The exact norm and softmax
  oracle passes with dynamic completeness, 4,263/4,756 accesses, and
  4,216/4,478 barriers.  This run also exposed and fixed orphan dense-relay
  emission after resource filtering; artifact `117` records final validation
  correctly rejecting those formerly unaccounted bytes.
- 2026-07-20: Inline reduction artifact `119` confirms that the same bounded
  dispatch-ID fallback works there, then stops before patching because the
  complete object has no common automatic EXEC-save window.  The Inline cell
  remains orange with a narrower blocker; work moves to the P0 scalar-planning
  frontier rather than serializing progress on this P2 cell.
- 2026-07-20: A bounded high EXEC-save window now excludes only gfx1250 owner
  groups whose complete compiler scalar allocation reaches it.  Inline
  reduction artifact `122` therefore improves within orange: both
  exact oracles pass, two code objects emit valid replacements (including a
  6,761-patch object), and static coverage reaches 3,945/4,756 accesses plus
  4,095/4,704 barriers.  Its 1,022 dynamically incomplete sites keep the cell
  below yellow.  The matching orphan dense-host bug is fixed by retaining
  filtered reservations as inert padding; final validation artifact `121`
  demonstrates the former bytes were correctly rejected.
- 2026-07-20: The same high-window fallback removes P0 `torch.topk`'s prior
  immediate Record/Replay scalar-planning rejection.  Unrestricted artifact
  `120` spends the full 120-second bound patching the 13.3 MB, 802-kernel
  object and does not yet reach execution, so the cell remains orange rather
  than claiming a clean verdict.
- 2026-07-20: Histogram Record/Replay artifact `124` promotes its cell from
  orange to yellow with exact 64-bin counts, dynamic completeness, 112/133
  accesses, and all 168 barriers.  The dense barrier dispatcher now reserves
  fixed capacity for its one-time host-return target as well as per-barrier
  targets; artifact `123` records the prior precise capacity failure.
- 2026-07-20: Histogram Inline artifact `127` promotes the fourth clean cell
  from orange to yellow with exact counts, dynamic completeness, 112/133
  accesses, and 168/168 barriers.  Inline now allocates indirect bodies,
  direct-appended fallbacks, and dense host bodies through one placement
  cursor after the fixed relay prefix.  Diagnostic `126` localized the old
  overlap as expected byte 191,140 versus reserved-prefix end 257,316.
- 2026-07-20: Fresh Sampled sort artifact `129` advances past the former
  no-placement boundary and plans all 48,224 accesses plus 11,276 supported
  barriers.  The unrestricted 3.6 MB object then fails while growing `.text`,
  before any patch is committed or the exact oracle executes.  The cell stays
  orange, but its blocker is now object-growth capacity rather than register
  allocation or entry-island placement; work spreads to other rows.
- 2026-07-20: The scatter-reduce row did not need reshaping: fresh artifacts
  `130`--`132` confirm collision-heavy `global_atomic_add` and compare-swap
  specializations and preserve exact BF16 and FP32 sums.  All MOI engines
  patch 23/23 admitted accesses.  Sampled is dynamically complete and moves
  to yellow; Record/Replay and Inline remain orange because they emit no visible
  records.  The relaxed global atomics are inventoried but correctly remain
  unsupported as unqualified synchronization sequences.
- 2026-07-20: Unrestricted SuperCollider top-k artifact `133` promotes the P0
  cell from orange to yellow.  Both the FP64 spill case and BF16 coverage case
  pass exact sorted-value/index oracles, execution is dynamically complete,
  and 2,991/135,384 accesses are patched.  Static access completeness remains,
  but the prior scalar-planning diagnosis is no longer current.
- 2026-07-20: Reduced SGEMM Inline artifact `134` cross-checks artifact `085`
  with an independent software backend.  It accepts the same legal 92 KiB
  workgroup allocation and remains compute-active through 120 seconds instead
  of crashing immediately.  The orange cell is therefore a backend-dependent
  execution blocker, not justification for lowering gfx1250's architectural
  LDS limit or changing ConSan's exact shadow representation.
- 2026-07-20: Bounded SuperCollider SPMM-F8 artifact `135` completes the first
  contraction orientation with exact numeric results, dynamic completeness,
  and 298/4316 static accesses, then continues into the second orientation at
  the 120-second bound.  This replaces the vague pending state with measured
  partial evidence while keeping the cell orange until the complete runner
  reaches a verdict.
- 2026-07-20: Record/Replay SPMM-F8 artifact `136` advances the corresponding
  cell from pending orange to a concrete orange correctness blocker.  Six
  candidates pass, but five later candidates return incorrect tensors before
  the 120-second bound.  Baseline artifact `137` passes every matching
  orientation and problem size it reaches, including the complete first three
  orientations, so the failures are instrumentation-specific rather than an
  emulator-duration artifact.  The next step isolates the first failing
  candidate instead of repeating the 186 MB full-object trace.
- 2026-07-20: SPMM-F8 Record/Replay advances from orange to yellow after
  isolating the full-object failure.  Access-only artifact `162` completes
  with 16 passes and no failures; standalone barrier-record artifact `163`
  completes with 18 passes and no failures; and relay-only diagnostics
  exonerate both routing and spill preservation.  Direct full-object solution
  selection then proves that advancing private epoch state reproduces the two
  wrong-result cases.  Full-VGPR gfx1250 Record/Replay now keeps owner/epoch in
  fresh scalar registers and records that scalar epoch at accesses.  Targeted
  artifact `174` passes the formerly failing solution 3/3, while unrestricted
  artifact `175` reaches 13 passes, zero failures, and complete static
  analysis before the 180-second bound.  The cell remains yellow, not green,
  because the unrestricted workload has not yet completed.
- 2026-07-20: `torch.sort` Record/Replay advances from yellow to green.  The
  remaining 384 access misses were all `ds_store_2addr_b64` sites: resource
  planning correctly found scratch registers between two disjoint data
  tuples, but the later overlap guard had forgotten the second tuple and
  modeled both as one contiguous four-VGPR range.  Preserving the second data
  operand removes the false overlap.  Clean artifact `204` passes the exact
  values and indices oracle with 48,224/48,224 accesses, 6,032/6,032 barriers,
  and complete static and dynamic analysis.  Paired artifact `205` accepts
  both baselines and Record/Replay, measuring about 53.3 seconds versus a
  131 ms paired baseline in software execution.  Inventory `206` and reviewed
  barrier-drop artifact `208` complete the bundle with exactly one mutation,
  the precommitted no-diagnosis/pass-oracle result, and healthy before/after
  gates.  The first reviewed site in artifact `207` is retained as rejected
  evidence because its oracle passed despite a precommitted failure; no
  outcome was relabeled after observation.
- 2026-07-20: The same disjoint-tuple fix promotes `torch.sort` Inline Shadow
  from yellow to green without another source change.  Clean artifact `209`
  passes the exact values and indices oracle with 48,224/48,224 accesses,
  6,032/6,032 barriers, and complete static/dynamic analysis.  Paired artifact
  `210` accepts both baselines and Inline Shadow, measuring about 52.1 seconds
  versus a 134 ms paired baseline in software execution.  Reviewed
  barrier-drop artifact `211` completes the bundle with exactly one mutation,
  the precommitted no-diagnosis/pass-oracle result, and healthy before/after
  gates.
- 2026-07-20: Sampled sort no longer fails the executable-growth transaction.
  Current-tip artifact `212` reproduced the old refusal and its subsequent
  required-patch process failure in about 38 seconds.  Debugger measurement
  `216` found an exact 75,747,740-byte growth request, only 8.6 MB beyond the
  generic 64 MiB safety cap.  The cap is now a still-bounded 96 MiB.  The
  Sampled regression slice passes, and unrestricted artifact `217` advances
  through patch construction and remains compute-active until the canonical
  300-second bound.  The cell stays orange because it has no oracle or
  coverage verdict yet, but its blocker is now software-runtime duration
  rather than code-object growth.
- 2026-07-20: All four scatter-reduce cells advance from yellow to green.
  Reviewed artifacts `329` and `330` accept typed N/A dispositions for
  atomic-order and atomic-scope weakening in every profile.  The inventory
  proves that the workload's global reduction atomics are relaxed singleton
  sequences, so there is no synchronization edge to weaken.  This completes
  the fault gate without inventing a mutation; clean, paired, exact-oracle,
  and complete 23/23-access evidence remains in artifacts `305`, `306`,
  `322`, `323`, and `324`.
- 2026-07-20: F8 quick-GEMM Record/Replay advances from yellow to green.
  Paired artifact `332` accepts the exact oracle and complete 1772/1772
  accesses, 44/44 barriers, and 16/16 fences in 222.7 seconds between 46.4-
  and 45.2-second baselines.  Reviewed artifact `331` applies exactly one
  barrier drop, preserves the oracle with the precommitted no-diagnosis
  outcome, and passes containment health.  The former aggregate quick-GEMM
  row is split by configuration so this per-workload result is visible.
- 2026-07-20: F8 quick-GEMM Sampled advances from yellow to green.  Paired
  artifact `335` accepts the exact oracle and complete 1772/1772-access plus
  80/80-barrier coverage in 213.7 seconds between 45.2- and 45.6-second
  baselines.  Reviewed artifact `334` applies exactly one barrier drop,
  preserves the oracle with the precommitted no-diagnosis outcome, and passes
  containment health.  Rejected artifact `333` records a transient unhealthy
  preflight with zero mutations; a direct target smoke passed before the one
  successful retry.
- 2026-07-20: HGEMM quick Record/Replay advances from yellow to green.  Fresh
  inventory `336` supplies a current-tip selector.  Dropping its first barrier
  pair preserves the exact oracle but loses postflight device health in
  artifact `338`; the materially different late-pair artifact `339` applies
  exactly one mutation, preserves the precommitted pass-oracle/no-diagnosis
  outcome, and passes containment.  Paired artifact `340` accepts complete
  8162/8162 accesses, 292/292 barriers, and 80/80 fences in 199.9 seconds
  between 113.2- and 125.3-second baselines.
- 2026-07-20: HGEMM quick Sampled advances from yellow to green.  Reviewed
  late-pair artifact `341` applies exactly one barrier drop, preserves the
  pass oracle with the precommitted no-diagnosis outcome, and passes
  containment health.  Paired artifact `342`, using one repetition throughout,
  accepts complete 8162/8162 accesses and 544/544 barriers in 207.6 seconds
  between 128.7- and 123.3-second baselines.
- 2026-07-20: A current-tip `torch.mode` Record/Replay paired attempt `327`
  preserves the exact oracle, 28,939/28,939 supported accesses, and
  4,446/4,446 barriers, but correctly rejects static completeness.  The two
  remaining executed sites are `ds_add_u32` LDS atomics rejected as
  `non_flat_atomic_address`.  Promoting this cell therefore requires a typed
  LDS communication token in Record/Replay, not relabeling the sites or
  reusing the flat/global effective-address representation.
- 2026-07-21: SPMM-all Inline Shadow is reclassified from an obsolete LDS-
  capacity failure to a measured software-execution limit.  The corrected
  gfx1250 topology admits the architectural 160 KiB CU LDS capacity.  Commit
  `00be0d6000` fixes spill-backed dense routing by keeping its pre-save return
  pair and dispatch key in structurally fresh SGPRs, and the formerly corrupt
  16-workgroup MT32 case now passes with zero numeric failures.  The next MT64
  case remains impractical for complete all-site validation: focused
  one-repetition probes pass numerically through 44 selected accesses, but
  identical solo runs vary from about three seconds to beyond 60 seconds.
  The 300- and 900-second full runs stop at the same boundary without a failure
  or final verdict.  The cell remains red with no overhead rather than
  treating containment expiry as detection or changing ConSan semantics to
  accommodate unstable software-runtime cost.
- 2026-07-21: Norm/softmax Inline Shadow improves without changing its orange
  status.  Commit `e2c1e026bc` replaces the persistent dispatch-ID SGPR pair
  with the report's stable literal identity for non-atomic gfx1250 Inline
  probes.  Clean-tree one-repetition artifact
  `consan-green-expansion-20260721-norm-softmax-inline-literal-dispatch-independent-052`
  passes the exact oracle in 35.17 seconds and improves coverage from
  4,254/4,756 accesses plus 2,145/2,352 barriers to 4,282/4,756 plus
  2,163/2,352.  The remaining 663 static plans need 16--19 temporary VGPRs
  without enough site-local vector space, while the separate 1,022 bounded
  publication-contention failures remain unchanged.  The cell rotates rather
  than opening both a VGPR-spill mechanism and a publication redesign at once.
- 2026-07-21: HGEMM SuperCollider remains orange after a clean-tree,
  one-repetition reassessment.  Artifact
  `consan-green-expansion-20260721-tensile-hgemm-sc-clean-independent-053`
  records 136 exact numeric passes and zero failures through a fixed
  300-second bound, but the first 143-solution problem is still active and the
  aggregate lacks its second applicable-object record.  This supersedes the
  prior 150-second duration evidence without motivating another timeout
  increase or a gfx1250-specific semantic change.
- 2026-07-21: Torch.mode Inline Shadow advances from orange to
  assessed/partial orange.  Artifact `054` exposes 744 fail-closed final-
  validation errors: workgroup-local exact shadows emit the expected exchange
  and claim for atomic access candidates, but the validator searches only read
  and write candidates.  Commit `a6721f8e76` admits the already-supported
  Atomic kind and adds focused coverage.  Clean-tree one-repetition artifact
  `consan-green-expansion-20260721-pytorch-mode-inline-atomic-validation-independent-055`
  then passes the exact values/indices oracle in 32.26 seconds with complete
  28,939/28,939 access and 4,446/4,446 barrier lowering.  A subsequent
  controlled 32-to-128 exact-bank experiment leaves all 13,342 undercoverage
  events unchanged, superseding the initial publication-contention diagnosis.
  Debug classification and captured object metadata instead locate them in
  local-shadow bounds handling for a kernel whose 2-byte fixed LDS is extended
  to 1,540 bytes dynamically at dispatch.  Dynamic-LDS-aware shadow selection
  is active; no yellow/mostly-working claim is made.
