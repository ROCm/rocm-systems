# Publication test pass — September 24, 2026

This campaign rebuilds and runs the full RocJitsu CTest suite with GCC,
Clang + UBSan, and Clang + ASan, including ConSan device tests on the emulator
and the physical gfx1201. It is a regression gate, not a replacement for the
external-workload qualification tables.

**In progress:** the full rerun is active after successful builds of commit
`5d45a79720b`. Known test failures remain; this is not a passing publication gate.
The interrupted first attempt is retained in `attempt1/`.

| Configuration | Compiler | Build | Full suite | ConSan emulator | ConSan physical gfx1201 |
| --- | --- | --- | --- | --- | --- |
| GCC | GCC 15.2.0 | Passed | Running | Running; failures found | Running; failures found |
| Clang + UBSan | Clang 21.1.8 | Passed | Running | Running; failures found | Running; failures found |
| Clang + ASan | Clang 21.1.8 | Passed | Running | Running; failures found | Running; failures found |

## Configuration and evidence

All builds use `RelWithDebInfo`, assertions, expensive checks, and
`BUILD_TESTING=ON`. Clang sanitizer runtimes are shared; undefined-behavior
checks use the project's `-fno-sanitize-recover=all`. Test-specific runtime
settings, including existing HIP/ASan leak-check exclusions, are retained.
The ROCm SDK is from `/home/benoit/venv`.

Artifacts are under
`/home/benoit/workspace/consan-validation/publish-tests-20260924/`:
compiler configuration commands, build logs, per-build CTest inventories,
the aggregate test manifest and generator, and test results. Build directories
are `/home/benoit/workspace/rocjitsu-{gcc,ubsan,asan}-build`.

The combined CTest scheduling domain preserves test commands and test oracles.
CPU and emulator tests may run concurrently. Physical tests share a CTest
resource lock and `/tmp/rocjitsu-consan-destructive-gpu.lock`, including native
DBI and hardware-backed translation tests outside ConSan. Existing simulator
`RUN_SERIAL` properties are removed for this campaign. Disabled tests and
runtime skips will be reported separately from passes.

The final inventory contains 11,444 tests each for GCC and UBSan and 11,458
for ASan (34,346 total), including 2,010 ConSan emulator tests and 447 ConSan
physical tests per build. All 469 physical tests per build share the GPU lock.

## Issues found

- GCC 15 rejected reachable-CFG range initialization under
  `-Werror=free-nonheap-object`. Commit `673681e22eb` constructs the fallback
  vector directly instead of appending to an empty range-constructed vector,
  retaining input validation and behavior. The affected translation unit now
  compiles with GCC; full regression results are pending.

- The venv SDK contains an older `librocjitsu.so`, which shadowed the build's
  library through `LD_LIBRARY_PATH` (for example, `RjWaitcheck.CHeader` failed
  with a missing `rj_waitcheck_options_init` symbol). The campaign now prepends
  the matching build directory for every test. The interrupted attempt is not
  accepted as a complete regression result.
- Physical and emulated gfx1201 atomic-publication cases found valid GFX12 FLAT
  stores with nonzero immediate displacements. Publication capture rejected
  those addresses, correctly making the trace incomplete. Commit `5d45a79720b`
  materializes the signed effective address and retains the older-encoding
  restriction. An isolated atomic-arrival case reproduced on all three builds.

## Remaining failures under investigation

- Atomic-arrival's clean case now has a complete trace but reports a conflict.
  The observed release RMW has `release=false`; the generated code has a global
  store wait but no adjacent LDS-completion wait. Its fixture writes LDS through
  inline assembly. The captured acquire RMW is correctly marked as an acquire.
  Other clean atomic-publication fixtures also fail; they need individual
  memory-model/fixture review before being classified as detector false positives.
- Histogram's missing publication modification is
  `global_atomic_pk_add_bf16`, leaving the trace incomplete despite successful
  capture of the other seven modifications. Correct and incorrect cases fail.
- Top-k's trace exceeds its allocation: an isolated clean run produced 4,671
  publication events against capacity 4,320, with the dropped flag set. Correct
  and incorrect cases fail the completeness gate.
- RCCL daemon tests emulate gfx950, while the selected SDK only includes
  `.kpack/rccl_lib_gfx1201.kpack`. The first attempt's five collective tests
  failed with `invalid device function`; a matching gfx950 RCCL archive is a
  prerequisite for qualifying those cases.

The focused rerun is recorded in `focused.log` and `focused.xml`; additional
atomic and coverage diagnostics are in `atomic-full-events.log` and
`coverage-debug.log`. Temporary expanded event logging was reverted before the
full rerun. No test oracle has been relaxed.

### Progress checkpoint

At the user's commit checkpoint, 28,224 of 34,346 cases had completed (82%).
Four additional ASan-only assertion failures reproduced in a serial, isolated
rerun; the same cases passed with GCC and UBSan:

- `L2CacheTest.LinkedHierarchyCachesOnlyAccessibleBytesOfIncompleteLines`
- `LegacySubPageCacheTest.VectorAccessCachesOnlyAccessibleBytesOfIncompleteLine`
- `LegacySubPageCacheTest.ScalarAccessCachesOnlyAccessibleBytesOfIncompleteLine`
- `KfdIoctlTest.DbgTrapUnpublishableStopRollsBackInsteadOfStrandingTheWave`

These are assertion failures, not ASan memory-error reports. Evidence is in
`asan-assertion-rerun.log` and `asan-assertion-rerun.xml`. The normally disabled
`SimdCoverage.InstructionTimings` test was explicitly run separately and passed
in all three builds (`timing-opt-in.xml`); its timings are not performance data.
Eight further gfx1201 emulator preset/selection cases also overflow publication
buffers, in addition to the top-k cases described above.
