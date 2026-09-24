# Publication test pass — September 24, 2026

This campaign rebuilds and runs the full RocJitsu CTest suite with GCC,
Clang + UBSan, and Clang + ASan, including ConSan device tests on the emulator
and the physical gfx1201. It is a regression gate, not a replacement for the
external-workload qualification tables.

**Initial baseline, before repairs.** All builds passed and the complete registered matrix
ran against the production sources at `5d45a79720b`. The full test run took
818 seconds, excluding builds and focused diagnostics. The interrupted first
attempt is retained in `attempt1/` and is not included in these results.

Counts below are **passed / failed**. Failures include timeouts and aborts.

| Configuration | Compiler | Build | Full suite | ConSan emulator | ConSan physical gfx1201 |
| --- | --- | --- | --- | --- | --- |
| GCC | GCC 15.2.0 | Passed | 11,359 / 54 | 1,989 / 21 | 419 / 28 |
| Clang + UBSan | Clang 21.1.8 | Passed | 11,359 / 54 | 1,989 / 21 | 419 / 28 |
| Clang + ASan | Clang 21.1.8 | Passed | 10,575 / 849 | 1,333 / 677 | 287 / 160 |

The full suite additionally reported 30 / 30 / 33 runtime skips and one disabled
instruction-timing test per configuration. That disabled test was explicitly
run separately and passed in all three builds. No ConSan device case was skipped
or disabled. Runtime skips are in architecture-specific arithmetic and
launcher/interposer tests; their exact reasons are retained in `skips.json`.

GCC and UBSan failed the same 54 tests: 49 ConSan cases and five RCCL cases.
No UBSan runtime violation was reported. ASan had 779 failing cases containing
LeakSanitizer diagnostics, plus assertion, diagnostic-oracle, and timeout
failures. A LeakSanitizer-containing failure can also have functional errors;
these counts do not classify every such case as leak-only.

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
runtime skips are reported separately from passes.

The final inventory contains 11,444 tests each for GCC and UBSan and 11,458
for ASan (34,346 total), including 2,010 ConSan emulator tests and 447 ConSan
physical tests per build. All 469 physical tests per build share the GPU lock.

## Issues found

- GCC 15 rejected reachable-CFG range initialization under
  `-Werror=free-nonheap-object`. Commit `673681e22eb` constructs the fallback
  vector directly instead of appending to an empty range-constructed vector,
  retaining input validation and behavior. The affected translation unit now
  compiles with GCC, and the CFG regression tests pass.

- The venv SDK contains an older `librocjitsu.so`, which shadowed the build's
  library through `LD_LIBRARY_PATH` (for example, `RjWaitcheck.CHeader` failed
  with a missing `rj_waitcheck_options_init` symbol). The campaign now prepends
  the matching build directory for every test. The interrupted attempt is not
  accepted as a complete regression result.
- Physical and emulated gfx1201 atomic-publication cases found valid GFX12 FLAT
  stores with nonzero immediate displacements. Publication capture rejected
  those addresses, correctly making the trace incomplete. Commit `5d45a79720b`
  materializes the signed effective address and retains the older-encoding
  restriction. The classifier regressions pass in all three builds; atomic-arrival
  now has a complete trace but retains the ordering failure described below.

## Initial failures (before the repair passes below)

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
  `.kpack/rccl_lib_gfx1201.kpack`. All five collective tests in each build
  failed with `invalid device function`; a matching gfx950 RCCL archive is a
  prerequisite for qualifying those cases.

The focused rerun is recorded in `focused.log` and `focused.xml`; additional
atomic and coverage diagnostics are in `atomic-full-events.log` and
`coverage-debug.log`. Temporary expanded event logging was reverted before the
full rerun. No test oracle has been relaxed.

### ASan-specific findings

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
Eight gfx1201 preset/selection cases fail on both emulator and physical GPU,
with publication-buffer overflow observed in the selected-address case. Seven
older physical atomic/fence/helper cases also fail in GCC and UBSan. These are
included in the ConSan totals above.

ASan also aborts in `WaveDebugTest.DeclinedMemoryViolationStillIssuesTheAccess`
after an unexpected PC and an access to a freed register block. Clean graph
replay on gfx942 and gfx950 reports ConSan conflicts, and
`ConSanLdsTest.DbiPaddedRacyStoreTripsTrap` fails its trap-output contract.
Thirteen ASan emulator stress cases timed out at their configured limits under
24-way test concurrency. A separate four-worker rerun with unchanged deadlines
passed five cases (four dense-branch cases and gfx1100 many-owner Correct) and
still timed out in eight cases: gfx942/gfx950/gfx1250 many-owner Correct/Incorrect
and gfx950 large-LDS-pipeline Correct/Incorrect (`timeout-rerun.xml`). The full
matrix counts above retain the original 13 timeouts.

LeakSanitizer reports include ROCr signal/event allocations and allocations in
the system `/usr/lib/cargo/bin/coreutils/echo`. An isolated
`ConSanGfx942Sim.BaselineScalarSpillExec.Correct` fails with leak checking and
passes with `detect_leaks=0` (`lsan-diagnostic.xml`). That diagnostic control
has **not** replaced its failing result or disabled leak checking in the full
campaign.

Machine-readable results are `results.xml`, `summary.json`, `failures.json`,
and `skips.json`. `failures.json` retains every failing test's output. All 469
physical GPU tests per build completed under the shared lock; the 22 outside
ConSan passed in every build.

## GCC repair pass — resolved

The [other host's report](https://gist.github.com/bjacob/e96ca161cf0623a2824a96a140b18095)
reproduces the same 21 gfx1201 emulator failures. GCC repairs were qualified
before proceeding to UBSan and ASan. Evidence is retained locally in
`/home/benoit/workspace/consan-validation/publish-fixes-20260924/`.

- Python 3.10 Tensile launches no longer require `-P`; 37 Tensile tests pass,
  including import-path isolation. The three torch-dependent Aorta oracle
  tests also pass in the PyTorch venv.
- Five legacy gfx950 physical tests now have physical labels and resource locks.
- Publication logging is limited to kernels with publication observers. All
  modifications in participating kernels remain subject to completeness checks.
  This clears 24 original GCC failures across emulator and physical gfx1201.
- HIP custom commands now track included headers through compiler depfiles.
- Identity RMWs can acquire an existing proven release sequence; their own
  ambiguous release is not used to establish outgoing ordering.
- Several fixture LDS writes were invisible to compiler wait-counter insertion.
  Compiler-visible stores and explicit LDS waits in assembly publication helpers
  fix their missing LDS completion before release.
- Aliased atomic input/output registers survive publication capture and spilling.
  Mutation validation accepts the precise composed observation rewrite.
- Qualified release stores and CAS outcomes are captured. Stores break preceding
  release sequences; failed CAS cannot publish. Hardware owner IDs are opaque
  equality keys, not bounded wave ordinals.
- The many-owner fixture retains eight kernel owners and two conflicting waves,
  with one workgroup per kernel to avoid incidental sampling competition.
  The physical incorrect case passed 20 consecutive repetitions.
- A matching `rocm-sdk-device-gfx950==10.2.0a20260915` wheel was obtained from
  AMD's nightly index; only its RCCL code pack was added to this host's SDK.
  All five RCCL tests now pass.

The first complete GCC rerun (`gcc-full1.xml`) reduced failures to five.
All five passed in the focused 97-test run (`gcc-focused12.xml`). The second
complete run (`gcc-full2.xml`) passed 11,420 tests, with one failure, 30 runtime
skips, and one disabled test. All 2,010 ConSan emulator and 447 physical gfx1201
cases passed. The only failure was a Python capability-probe timeout fixture
that required child startup within 50 ms. It now checks the parent-observed
PID instead; all 33 VFIO launcher unit tests, 24 concurrent timeout repetitions,
and its CTest rerun (`gcc-vfio-fixed.xml`) pass. No GCC failure remains.

Sanitizer rebuilds and qualification are now underway, using the GCC repairs.
The original matrix above remains the pre-repair baseline.

## Sanitizer repair pass — in progress

- ASan cache validation must preserve demand-access semantics: missing backing
  remains retryable, and a refused read leaves its destination untouched.
  `c2c35c72a66` separates this from debugger probes. The focused run passes all
  81 cache, VM, debugger, and scratch cases (`asan-cache-focused3.log`).
- Concurrent HIP graph queues previously shared the process-wide scratch
  reservation when instrumentation introduced private memory. This corrupted
  ConSan epochs and produced false conflicts. `21c709eec97` honors ROCr's existing
  per-queue allocation, resize, and reclaim protocol, even with local KFD backing
  helpers installed. Both failing ASan graph replay cases pass; a deterministic
  allocation-protocol regression checks both helper configurations.
- The physical deliberate-trap test now distinguishes its expected GPU abort
  from host sanitizer failures. Its checked runner requires trap evidence and
  successful instrumentation, and rejects sanitizer diagnostics. Five Python
  regressions cover the runner's acceptance and rejection paths.
- The installed ROCr predates this branch's existing `069613ef876` signal-pool
  LeakSanitizer root registration. A fresh branch ROCr build fixes the isolated
  live-signal and gfx942 HIP leak cases with leak checking enabled. ASan tests
  use an isolated runtime overlay in the repair artifacts' `runtime/` directory:
  a copy of SDK `libamdhip64.so.7` with RUNPATH instead of its original RPATH,
  and symlinks to the freshly built `libhsa-runtime64.so.1`. This prevents HIP's
  `$ORIGIN` RPATH from silently selecting the old SDK ROCr. The SDK libraries
  themselves are unchanged; no new leak suppressions were added.

The first sanitizer rerun (`sanitizers-full1.log`) was interrupted after exposing
RDNA scratch resize loops: the emulator demanded 1 KiB alignment while ROCr
provisions 256-byte units on RDNA. `b4517a899ef` uses the ISA's scratch granule
consistently, tests small allocations, and fixes the checked runner's Python
registration. All 152 focused scratch tests pass in UBSan and GCC. The shell
spawn fixture now uses builtin `printf`, preserving its process/pipe assertions
without testing Rust coreutils' allocator lifetime.

The uninterrupted rerun is `sanitizers-full2.log`, with 24 CPU workers and all
physical GPU tests serialized by the shared lock. UBSan has completed with
11,423 passes, zero failures, 30 runtime skips, and one disabled test. All 2,010
ConSan emulator and 447 physical gfx1201 cases passed. The disabled instruction
timing test passed separately in both sanitizer builds. ASan is still running.
The original baseline above is retained separately from these repair results.

### Follow-up ASan findings

The first uninterrupted repaired run completed with 11,416 ASan passes, 18
failures, 33 skips, and one disabled test. All 447 physical ConSan cases passed.
The failures were 13 emulator stress timeouts and five RCCL rank timeouts.
All 13 stress cases passed unchanged correctness checks in a four-worker
600-second diagnostic run, taking 37–107 seconds. The three affected workload
families now receive 300 seconds in ASan emulator configurations; physical and
other compiler limits are unchanged.

A new VM regression also found that cache validation refused valid client-owned
memory without local mappings. `b23e53c1289` validates through the existing client
read conduit; inaccessible client memory reports a fault instead of retrying
forever. All 77 focused VM/cache cases pass after this fix.

RCCL's longer diagnostic completed its numerical checks but exposed the main
ASan slowdown: the daemon repeatedly parsed the entire textual VMA table to
check write permissions. A direct, uncached
[PROCMAP_QUERY](https://docs.kernel.org/filesystems/proc.html) query avoids that
scan on Linux 6.11+; the existing parser remains the fallback for older headers,
kernels, or policies that reject the ioctl. Mapping protection is still checked
on every access.

RCCL also has its own RPATH selecting the SDK's old ROCr. The isolated ASan
overlay therefore includes a copy of `librccl.so.1` with RUNPATH, as well as HIP.
`ldd` confirms that the RCCL client resolves HIP and ROCr through the overlay.
The final rerun is pending these repairs; the 18 failures are not counted as
resolved by diagnostic timeout overrides alone.
