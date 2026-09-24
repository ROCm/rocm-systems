# RDNA4 SuperCollider investigation — September 24, 2026

A later [sensitivity experiment](SC_SENSITIVITY_RDNA4_20260924.md) implements
opt-in wave-dependent replay delay and records its qualification results. The
historical audit below precedes that change; [STATUS_RDNA4.md](STATUS_RDNA4.md)
contains the current per-workload results.

The September 23 revalidation has 3 green SuperCollider cells, compared with
16 in the historical ledger (commit `41c6a6e5e46`). These colors do not establish
an across-the-board detector regression: the fault contracts and some workloads
changed. This investigation separates those changes from detection behavior.

## Historical greens included expected misses

Retained July 16 artifacts under
`/home/benoit/workspace/consan-validation/final-tip-640e575-faults*`
contain accepted SuperCollider profiles with `detector_policy: not_detected`,
`detections: 0`, and `trials: 1` for D128 block, D128 pressure, WMMA attention,
Jakub, Stream-K arrival, and tree atomic-OR. Stream-K and tree include both
atomic-order and atomic-scope weakening. The current contract requires at least
6 detections in 8 admitted/reached trials. These six historical greens therefore
cannot support a claim that these workloads previously detected faults reliably.

Other retained bundles contain genuine SuperCollider mismatch reports, including
Qwen, TP1 prefill/decode, TP2, and CLIP. Those accepted profiles each had one
trial. Some earlier attempts expected a miss and were rejected *because a fault
was detected*; later specifications changed that expectation. This is another
reason to compare raw evidence rather than the old `accepted` flag.

## Faults changed, even where the code object did not

| Workload | Historical positive fault | September 23 fault |
| --- | --- | --- |
| Qwen | `_initializer_11_dispatch_0_transpose_1024x3072_f32`, ELF `de4a0fc370f12a36` | `main$async_dispatch_0_reduction_5x1024_f32`, ELF `5d29deb0b917dbeb` |
| TP1 prefill | matmul dispatch 2, `.text+0xaa4` | matmul dispatch 13, `.text+0x9f24` |
| TP1 decode | matmul dispatch 30, `.text+0x14c30` | attention dispatch 24, `.text+0x133a0` |
| TP2 | matmul dispatch 127, `.text+0x10c64` | matmul dispatch 27, `.text+0x9b70` |
| CLIP | batch matmul dispatch 11, `.text+0x562c` | matmul dispatch 3, `.text+0x1b98` |

TP1, TP2, and CLIP retain the respective code-object identities across these
comparisons. Qwen's old initializer is absent from the current native allowlist.
A regression claim needs the same executed binary, fault, controls, and comparable
trials. The existing comparison changes several of these simultaneously.

## Historical clean runs also reported mismatches

The retained `final-tip-640e575-clean-all-20260716` clean logs reveal a more
substantial qualification problem:

| Workload | Clean mismatch buffers | Historical fault mismatch buffers |
| --- | ---: | ---: |
| TP1 prefill | 1/1 | 1/1 |
| TP1 decode/combined | 2/2 | 2/2 |
| TP2 | 6/6 | 6/6 |
| CLIP | 1/1 | 1/1 |

These clean runs were marked `accepted: true`. At `640e575`, `_run_row` accepted
zero exit status plus `_coverage_summary`; neither rejected SuperCollider report
markers. Automatic reporting need not cause a nonzero process exit. Thus these
four historical greens did not demonstrate detection specific to an injected
fault. This does **not** establish whether the clean alarms were detector false
positives, real races in the clean workload, or a reporting problem. It establishes
that the old qualification failed to distinguish them.

Qwen's retained clean report has zero mismatches, so this explanation does not
apply to its historical positive. Its executable and fault site changed, and its
old positive was only one trial.

The same omission persisted in the current validation runner. This investigation
adds rejection of mismatch reports, incomplete summaries, and report allocation,
read, or cleanup failures, including the historical `ConSan SC` spelling. The
three current SuperCollider greens (split softmax, matvec, CLIP) were re-audited
against this rule: all six matching clean comparators pass. No current green was
lost. Historical clean-summary records are retained in
`historical-clean-sc-reports.json` in the artifact directory below.

## Matched historical TP2 fault experiment

The experiment was declared before running: use the historical dispatch-127
barrier removal with the current hook and workload, delay=0, eight fault trials,
a 6/8 threshold, and a matching clean run. The pristine ISA confirms the exact
signal/wait pair between cooperative tile stores and consuming LDS loads. The
current code object retains the historical identity `2c96234e60a2a4dd`.

| Current-hook configuration | Clean | Admitted/reached | Detections | Numerical fault manifestations |
| --- | --- | ---: | ---: | ---: |
| Native rocprofv3 allowlist | Pass; zero SC mismatches | 8/8 | 0/8 | 0/8 |
| No kernel allowlist | Pass; zero SC mismatches | 8/8 | 0/8 | 0/8 |

Both arms pass matching provenance, complete coverage, and before/after GPU
health checks. Reach uses the reviewed unconditional final ISA witness, rather
than a runtime per-site counter. Both instrument 2,976/2,976 supported accesses.
Removing the allowlist did not restore the old positive. Because the old clean
run already alarmed, this comparison cannot establish a loss of valid detection.
Nor does reaching a removed barrier guarantee a race manifests on every run.
The current dispatch-27 fault remains a separate miss.

The historical hook SHA-256 was
`c45aa0fece5a9aa7ef8b3ad24bcbb2077e477586df6b4eecf12990f7fafa693d`.
The binary now at its recorded path has a different hash, so it was not used as
an old-version comparator. The current hook is
`69566b53a48b3d65aa1835b2ae206c4f3cc5799952dcdb34560dfc5fe328e803`.
This investigation has not isolated a particular detector or runtime change
responsible for the disappearance of the historical clean alarms.

## Current misses and useful improvements

SuperCollider compares an LDS access with a duplicate load/readback, separated
by an optional delay. It detects a value changing between those observations.
If both observations see the same stale or otherwise wrong value, it can miss
the race. Full static instrumentation coverage does not eliminate this limit.
The relevant implementation is
`lib/rocjitsu/src/rocjitsu/code/patch/consan/supercollider/consan_supercollider_common.inc`.

Top-k is a particularly useful sensitivity test: each of the existing sleep=1,
4, and 15 batches has 8/8 admitted/reached trials and 8/8 numerical fault
manifestations, but 0/8 SuperCollider detections. This is stronger evidence of a
sensitivity gap than a fault that neither changes output nor triggers a report.
Default's preset ladder does not tune SuperCollider; increasing those presets
cannot solve this gap. Lowering the 6/8 bar also cannot rescue a 0/8 result.

Recommended next detector experiment:

1. Start with top-k's unchanged, output-corrupting fault and matched clean
   controls. Keep the current fault and qualification bar fixed.
2. Try timing perturbations at synchronization edges. The existing
   `RJ_CONSAN_SC_PERTURB_KIND`, `EDGE`, `INDEX`, `MAX`, and `SLEEP` controls provide
   a narrower place to perturb than delaying every replayed access. Their
   selector admission and final patch placement must be checked; the validation
   runner would need explicit support for matched clean/fault settings.
3. If uniform edge delays still fail, explore wave-dependent or site-dependent
   timing perturbations. Current replay delays and existing edge sleeps are
   unconditional. Adding the same delay across cooperating waves may preserve
   their relative timing; asymmetric perturbations could expose transitions
   between the two observations. This is a hypothesis, not a measured gain.
4. Require improved detection across repeated trials with no new clean alarms.
   For misses inherent to value comparison, Default's conflict tracking remains
   the already measured alternative; a larger SC delay alone is not a general
   solution.

No new delay scheme has been implemented or claimed qualified by this deep dive.
The immediate implemented improvement is the clean-report validation fix.

## Status presentation correction

Previous updates overwrote yellow cells with the latest delay trial even when
it detected fewer faults. The table now retains the strongest measured matched
results: FP16 sleep=1, 2/8; FP8 sleep=4, 4/8; TP1 decode sleep=1, 2/8; compiled
softmax sleep=1, 4/8 with the stated same-value policy. These are exploratory best
observations, not newly qualified greens. They do not change the green count.

## Reproduction and artifacts

Artifacts, frozen specifications, raw logs, historical extracts, and the audit:
`/home/benoit/workspace/consan-validation/sc-deep-dive-20260924/`.

- `historical-final-tip-faults.json`: historical fault policies and outcomes.
- `historical-clean-sc-reports.json`: historical clean summaries and hook hashes.
- `historical-tp2-spec.json`: exact historical fault selection and trial contract.
- `tp2-historical-{clean,fault}/`: current allowlisted comparison.
- `tp2-no-allowlist-plan.json` and `tp2-no-allowlist-{clean,fault}/`: comparison
  without the allowlist, declared before execution.
- `audit.py` and `current-clean-and-tp2-audit.json`: re-audit of current SC green
  clean controls and the two TP2 arms.
- `best-yellow-display-correction.json`: provenance of corrected yellow cells.
- `validator-tests.log`: validation runner regression tests.
