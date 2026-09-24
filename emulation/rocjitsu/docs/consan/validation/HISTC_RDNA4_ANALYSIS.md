# RDNA4 histogram Default-engine investigation

September 24, 2026, gfx1201. This investigation keeps the original
`pytorch-torch-histc` inputs, native kernel allowlist, barrier-removal fault, and
qualification threshold of 6/8 admitted, reached, healthy trials. A passing clean
run with matching controls and complete coverage is also required. Results here
are end-to-end validation evidence; unit tests alone do not qualify a table cell.

## Two detector issues

1. **Missing LDS atomic access coverage.** The gfx1201 capability profile omitted
   `RelaxedLdsAtomicAccess`, so the histogram's `ds_add_f32` operations were not
   recorded. The initialization stores were recorded, but their conflicting
   atomic updates were absent. Existing exact relaxed LDS RMW lowering is now
   enabled on RDNA4. These are access observations, not synchronization edges;
   atomic/atomic pairs remain nonconflicting. Clean histogram access coverage
   increases from 5/5 to 6/6. The old denominator described the narrower admitted
   capability, so it did not expose this missing semantic form.
2. **Systematic evidence-bank collisions.** The retention hash combined the
   workgroup coordinates and wave owner with plain XOR before its final mix.
   Workgroup 0/wave 1 and workgroup 1/wave 0 therefore had identical hashes at
   every bank count. For this two-workgroup, 16-waves-per-workgroup kernel, 32
   identities collapsed to 16 before any capacity truncation. The hash now
   multiplies the workgroup/dispatch component before combining the owner.
   Workgroup sampling itself is unchanged. Banks remain bounded retention
   buckets, not identities: full dispatch/workgroup/owner checks still decide
   whether two retained observations conflict.

The second issue was established from GPU access records: after enabling atomic
coverage, `higher` with 256 banks detected 4/8, and `max` with 256 banks detected
2/8. Detection exactly followed retention of a same-workgroup, overlapping,
unequal-owner store/atomic pair. Missed trials had no such retained pair. This
was loss of evidence, not an acquire/release or barrier rule suppressing a
recorded race. Increasing the preset alone could not repair either issue.

The fault removes the initial histogram publication signal/wait at
`.text+0x3d8c8` and `.text+0x3d8e4` in native object
`fnv1a64:6d2e968a699a17e3`, retaining the final reduction barrier. Relevant accesses
are initialization `ds_store_b32` at `.text+0x3d890` and accumulation `ds_add_f32`
at `.text+0x3df90`. The numerical result can still be correct despite the missing
barrier; race detection, rather than output corruption, is the qualification
oracle.

## Qualification after both fixes

| Preset | Banks | Strict clean | Fault detections | Result |
| --- | ---: | --- | --- | --- |
| default | 8 | Pass | 0/8 | Below bar; no sampled access records |
| high | 8 | Pass | 0/8 | Below bar; no sampled access records |
| higher | 8 | Pass | 0/8 | Insufficient retained evidence |
| higher | 256 | Pass | **6/8** | **Green** at the existing 6/8 bar |

All fault trials in this table were admitted and reached, with complete evidence
and healthy GPU checks. `higher` is the lowest preset reaching green in this
search; its bank override is essential and is shown in the status cell. The
smaller presets do not produce access observations for this workload, which
increasing retention capacity cannot repair. This does not calibrate the minimum
bank count: intermediate bank counts were not searched.

In the successful eight-trial batch, each detection retained all four sampled
initialization stores across the two workgroups. Both missed trials retained
only the two stores from workgroup 0. Finite-bank collisions still occur; the
change removes the unconditional XOR alias rather than promising collision-free
retention for arbitrary identities. The numerical oracle passed in every trial.

Reproduce the green configuration with the existing environment and allowlist:

```sh
export CONSAN_VALIDATION_DEFAULT_PRESET=higher
export CONSAN_VALIDATION_WATCHPOINT_BANKS=256
python emulation/rocjitsu/tests/dbi/consan/consan_validation.py --target gfx1201 run \
  --workload pytorch-torch-histc --profile default --phase clean \
  --artifact-root /path/to/fresh-artifacts
python emulation/rocjitsu/tests/dbi/consan/consan_validation.py --target gfx1201 fault \
  --workload pytorch-torch-histc --profile default \
  --spec emulation/rocjitsu/tests/dbi/consan/consan_validation_faults_gfx1201.json \
  --fault barrier-drop-histogram-initialization-preset-higher-banks-256 \
  --allow-destructive --artifact-root /path/to/fresh-artifacts
```

## Checks for regressions in existing green rows

Both fixes affect shared access/retention paths, so two other workloads were
rerun with their established presets and ordinary eight-bank budget:

| Workload | Preset | Strict clean | Fault detections | Evidence / health |
| --- | --- | --- | --- | --- |
| Stream-K arrival | higher | Pass | 8/8 | Complete; healthy |
| Production FP16 matmul | high | Pass | 8/8 | Complete; healthy |

Stream-K exercises atomic publication ordering and the corresponding banked
synchronization metadata. FP16 matmul exercises ordinary LDS barrier ordering
and the production-size numerical oracle. Their existing green qualifications
survive the histogram fixes. Artifacts are in `regression/`, with an aggregate
`regression-audit.json` checking all 16 trials and retaining hook hashes.

## Verification and artifacts

- Capability repair: `4e3ce0b6e0e`.
- Retention repair: `1531e7a780f`.
- Regression suite: 977 passed, two opt-in benchmark tests skipped, using
  `--gtest_filter='ConSan*.*:ObservationPlan.*'`.
- The new bank regression executes emitted instructions on every supported
  target, checks separate retention of the 32 histogram identities for four
  dispatch values, and checks that owner and workgroup inputs are preserved.
- Artifact root on this host:
  `/home/benoit/workspace/consan-validation/histc-default-20260924/`.
  `preset-*` directories contain the atomic-coverage-only experiments;
  `hash-fixed-*` contain both fixes. Each holds clean results, eight individual
  fault results, aggregate acceptance, coverage, health, and provenance.
- Hook SHA-256 after both fixes:
  `cf1c6916f8ec77ae42f334dbb4d8d20dd46b7259ca6e0be1fdb2394f6c210319`.

The machine-readable `qualification-audit.json` checks clean acceptance, all
32 fault trials’ completeness and GPU health, detection counts, and hook hashes.

See [STATUS_RDNA4.md](STATUS_RDNA4.md) for the latest qualification state.
