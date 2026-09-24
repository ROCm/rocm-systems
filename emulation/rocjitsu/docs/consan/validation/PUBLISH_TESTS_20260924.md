# Publication test pass — September 24, 2026

This campaign rebuilds and runs the full RocJitsu CTest suite with GCC,
Clang + UBSan, and Clang + ASan, including ConSan device tests on the emulator
and the physical gfx1201. It is a regression gate, not a replacement for the
external-workload qualification tables.

**In progress:** all three builds are running; no full-suite result yet.

| Configuration | Compiler | Build | Full suite | ConSan emulator | ConSan physical gfx1201 |
| --- | --- | --- | --- | --- | --- |
| GCC | GCC 15.2.0 | Running after CFG initialization fix | Pending | Pending | Pending |
| Clang + UBSan | Clang 21.1.8 | Running | Pending | Pending | Pending |
| Clang + ASan | Clang 21.1.8 | Running | Pending | Pending | Pending |

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

## Issues found

- GCC 15 rejected reachable-CFG range initialization under
  `-Werror=free-nonheap-object`. Commit `673681e22eb` constructs the fallback
  vector directly instead of appending to an empty range-constructed vector,
  retaining input validation and behavior. The affected translation unit now
  compiles with GCC; full regression results are pending.
