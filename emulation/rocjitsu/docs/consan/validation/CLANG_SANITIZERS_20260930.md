# Clang sanitizer validation — September 30, 2026

Source: initial `ca904b02dbe`, rebased to `72a843d3b61` on `shared/rocjitsu/sanitizers`. The only tree difference is validation documentation; executable sources are identical (see `source-history.json`).

| Build | Passed | Runtime skips | Disabled | Failed |
| --- | ---: | ---: | ---: | ---: |
| Clang 21.1.8 + UBSan | 12,113 | 90 | 1 | 0 |
| Clang 21.1.8 + ASan | 12,123 | 94 | 1 | 0 |

Both full rocjitsu suites include 478 physical gfx1201 tests, all passing.
CPU/emulator tests use 16 workers; physical tests run serially under the GPU
lock. Builds use 32 jobs inside the shared 40 GiB memory cap. RelWithDebInfo,
assertions, expensive checks, and x86-64-v3 are enabled. UBSan is nonrecovering.
Test-specific sanitizer options, skips, and disabled tests are preserved.

The first ASan run accidentally used the SDK's older ROCr, producing 647
CPU/emulator and 131 physical failures containing LeakSanitizer reports.
The corrected full rerun uses a fresh build of this branch's ROCr, including
its existing signal-pool root registration (`069613ef876`). The previously
established runtime overlay gives HIP a RUNPATH and prioritizes this ROCr;
per-test environment modifications preserve that priority. No new leak
suppressions or source changes were needed. Initial results remain under
`asan-sdk-runtime/`; they are not the accepted validation result.

Artifacts: `/home/benoit/workspace/consan-validation/rdna4-reevaluation-20260930/`.
See `sanitizer-configurations.json`, `asan-runtime.json`, the per-build
inventories, CPU/GPU JUnit XML and logs, summaries, and build logs. The ASan
CTest wrapper preserves the original commands and all test properties,
appending only the required runtime and build-library search paths.
