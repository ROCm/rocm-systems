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

See [STATUS_RDNA4.md](STATUS_RDNA4.md) for the latest qualification state.
