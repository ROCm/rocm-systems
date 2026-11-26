# ROCr tests against rocddi/libhsa on gfx1201

Test date: November 19, 2026. Branch: `users/agutierr/rocr-test-fixes`.
Machine: Radeon RX 9070 XT (gfx1201), 16 GiB VRAM, Linux 6.17.0-23-generic,
amdgpu 6.19.14.31400000, KFD ABI 1.23. ROCr test invocations were launched
serially.

The consolidated `users/agutierr/rocddi-rocr-test-fixes` branch includes only
changes under `runtimes/`. The ROCr lifecycle, core-dump, and SVM harness
corrections described here remain on `users/agutierr/rocr-test-fixes`; they are
not included in the consolidated branch. The results below were obtained
with that corrected harness.

PR 16293 review subsequently added source-ELF and retained-trampoline core
regions, omitted reads of managed SVM pages, and batched checked VRAM zero
fills. The core changes have fresh focused validation described in the
[core report](rocr-gfx1201-core-dump.md). The revised kernel patch has build
and host-test coverage, but has not been loaded; the 133-case sweep below
qualifies the earlier `DE114CD2BB59945654162BC` module. See the
[batching qualification limits](rocr-gfx1201-svm-prefetch.md#pr-review-batch-checked-zero-fills).

The remaining SVM failure is fixed: the driver now provides checked migration,
and rocddi negotiates that exact operation before libhsa accepts prefetch work.
The GPU core-dump and earlier runtime corrections remain qualified. Three
separate test-harness fixes make lifecycle, core-dump, and SVM failures visible.

## Results and coverage

A fresh serial sweep executed all 133 selected gfx1201 cases against one
immutable runtime snapshot and the combined driver correction:

| Result | Cases |
|---|---:|
| Pass | 118 |
| Skip or partial capability coverage | 15 |
| Fail, crash, or watchdog timeout | 0 |

All cases exited zero with no gtest failure assertions. The SVM prefetch
subtests now pass; its XNACK-dependent batch-discard skip accounts for the
change from 14 to 15 partial outcomes. This table represents one fresh run,
not a combination of older targeted results.

The sweep used runtime revision `b8ac2193d6e`, module srcversion
`DE114CD2BB59945654162BC`, the corrected test binary, and no preload. Runtime
SHA256: `300aac99a8796aff7faf3690001b2846918d6e7d7239b5ed37cda603d3556615`.
Summed case execution time was 391.896 seconds. All 133 post-case GPU-handle
audits were empty. Kernel messages were limited to 256 expected illegal-opcode
reports from malformed-queue testing, 16 workqueue-latency notices on those
paths, and the expected address-zero faults in the seven core-dump cases.
There were no new resets, MES timeouts, or unexpected faults.

The manifest, per-case JSON/XML/logs, and health audit are in
`target/rocr-test-fixes/runs/checked-final-133/`. The per-case CSV and summary
are `target/rocr-test-fixes/results.csv` and `summary.json`.

The prior broad nonfaulting run had 105 passes, 14 skips/partial outcomes,
and the SVM failure. Later targeted runs closed the trap and core-dump gaps.
With checked prefetch, `SvmMemory_Basic_Test` now exits zero, including both
prefetch subtests; its separate XNACK-dependent batch-discard section remains
skipped. The seven core-dump cases execute actual intentional GPU faults and
validate real captures, rather than counting child setup errors as passes.

## Fixes

| Commit | Root cause and correction | Relevant verification |
|---|---|---|
| `dc5351417c8` | No native snapshot/ELF writer or pattern handling existed. Retain owners, stop queues, refresh CWSR and stream actual CPU aliases at their GPU addresses. | All seven corrected gfx1201 dump cases pass; native/ELF/pipe tests and independent saved-wave inspection pass. |
| `02bda161e93` | Code-cache publication's PM4 IB lacked GPU execute permission, causing initialization timeout/status 46. Grant executable backing. | New native regression fails before and passes after; rocminfo and kernel dispatch work. |
| `b71be882880` | Modern code objects were assigned a fabricated BASE profile, rejecting FULL executables with 4109. Preserve absent metadata; reject the deprecated profile query without changing its output. | Loader tests, unchanged ROCr kernel/memory cases, and C API comparison with reference ROCr. |
| `aa67dc5e6d4` | `HSA_ENABLE_YIELDING_POLL` was ignored. Add opt-in scalar polling backoff with hint-dependent hot windows and timeout clamping. | All five ROCr polling cases pass, including CPU-use and default-policy checks; host cancellation/backoff tests pass. |
| `723fd3ef841` | VM retain looked up only mapping bases. Accept interior addresses using predecessor lookup and half-open bounds. | Unchanged VM Basic test covers base/interior/last/end and balanced releases. |
| `95bf8f0cfc2` | VM map rejected unused nonzero flags accepted by reference ROCr. Ignore the unsupported argument. | Unchanged NonContiguousChunks test passes. |
| `45fd3b59b45` | `GPU_MAX_HW_QUEUES` was ignored, preventing required sharing. Sample the per-agent/per-priority limit at initialization. | All nine counted-queue tests pass, including concurrent dispatch and wraparound; parser boundaries pass. |
| `7ad0b0d4857` | The KFD exception signal was decoded using legacy CP bits. Separate protocols and preserve public status translations. | All four negative queue cases pass; native decoder and frontend status tests pass. |
| `a62bbfee15a` | Imported VM placement was exposed as known allocation ownership. Clear unknown public owner/flags while preserving internal copy placement. | Host/device DMA-BUF pointer tests pass; native zero-agent copy verifies data and direct routing. Fabric is unavailable on this GPU. |
| `2941a70fa28` | AQL events with no application callback were discarded, leaving a faulting child blocked. Add ROCr default diagnostic/abort policy with its exact-zero opt-out. | Host callback/subprocess tests, CTest, and seven explicit-callback GPU regressions pass. Corrected DisableFlag changes from watchdog failure to pass in 1.618 s. |
| `a2431eb13c0` | Core-dump setup exits, watchdog kills, and normal completion could pass. Require pre-publication readiness plus SIGABRT and retry spurious signal waits. | 27 parent/caller and four actual-child host injection checks pass; five old false-pass controls establish the regression. |
| `fc66091f907` | Legacy KFD prefetch did not prove complete migration. Add an opt-in checked operation and native qualification probe. | Exact-source build/ABI checks, reviews, and all eight native/public-HSA probes pass with clean GPU health. |
| `b8ac2193d6e` | Gate Migration on the exact native extension and preserve strict errors without fallback. Keep host-only backing independent of activation. | 660 host tests pass with seven existing ignored; CTest, warnings-denied Clippy, and formatting pass. |
| `c1f7626a455` | SVM prefetch spun on negative completion and ignored an invalid query. Bound polling and check the last-prefetch query. | Negative, stuck, and query-error controls produce bounded failures with clean GPU health. |
| `424461362ca` | Concurrent_Shutdown and Reference_Count printed HSA errors but passed. Record nonfatal gtest failures. | Four host-injected error cases pass before and fail after; both normal host-only controls pass. |

The runtime fixes were independently reviewed with owner and curator passes.
The imported-pointer review caught and resolved an initial copy-routing
regression before commit. The lifecycle harness was compiled with the existing
GNU C++17 settings, preserving its original line endings and cleanup flow.

The separately reviewed kernel fix in `e8306c3079c` waits for the initial
BOOKKEEP fence before switching a compute VM from SDMA to CPU page-table
updates. Hardware qualification is committed in `5e2f1333b30`. The temporary
reload and hardware qualification are recorded in the
[patch validation](patches/README.md) and [trap finding](rocr-gfx1201-trap-recovery.md).

## Failure resolution and coverage limits

| Case | Established finding | What remains |
|---|---|---|
| `SvmMemory_Basic_Test` | The original driver cannot prove migration completion, so libhsa rejected admission with status 47. | Resolved by `fc66091f907` and `b8ac2193d6e`; native/public-HSA probes and the corrected ROCr prefetch subtests pass. XNACK-dependent batch discard remains skipped. See [SVM qualification](rocr-gfx1201-svm-prefetch.md). |

The [core-dump implementation](rocr-gfx1201-core-dump.md) now passes file,
substitution, content, disable, invalid-path and pipe cases. Independent ELF
inspection confirms real queue indices, dispatch bytes, refreshed CWSR and
native runtime/device records. An additional pipe control with an
HSA-querying `pthread_atfork` prepare handler validates spawn outside the
runtime registry lock. Reference ROCr's earlier DefaultPattern control also
passed on this patched driver (3.171 s, 161,559,936-byte core).

A default process-wide observer without application registration remains a
separate coverage gap; these dump tests use faults reported through AQL
queue events. Unavailable private VRAM is represented by an ELF memory extent
with no file bytes. It is not replaced with zero-filled or staged data.

Skip/partial outcomes are retained, rather than counted as full passes.
They include unavailable fabric/multi-GPU/signal-hint/other optional coverage. The
per-case logs contain the exact reason; gfx1201 remains applicable to the
default core-dump tests as well as the now-passing shader-trap tests.

The report audit corrected four false skip classifications caused by "skipped"
appearing only in test descriptions. The three multi-signal wait cases and
deprecated-doorbell discovery case executed successfully. The raw run records
remain unchanged; `summary.json` retains each `recorded_status` alongside its
corrected result, and the runner now excludes descriptions from skip detection.

## Verification and artifacts

- Checked SVM qualification covers zero/data round trips, exact subranges,
  pinned-page partial migration (`EAGAIN`), failed split preparation (`EFAULT`),
  recovery/retry, and real negative HSA completion. All eight bounded probes
  pass with empty kernel deltas and GPU-handle audits. Native/loader/libhsa
  host tests report 660 passes and seven existing ignored tests.
- The running module is `DE114CD2BB59945654162BC`, containing both VM-clear and
  checked-SVM corrections. It is loaded temporarily; the installed boot module
  remains original, and reboot restores that original driver. The patches and
  exact qualification limits are in the [driver record](patches/README.md).

- Runtime CTest: 10 passes, 2 expected skips, no failures. This includes the
  Rust workspace and ABI checks; the skipped tests require hardware/runtime
  facilities absent from the sandbox.
- Workspace Clippy with warnings denied and rustfmt checks pass. See the
  [verification correction](rocr-gfx1201-clippy.md) (`6d0bddf9e1c`). The final
  Clippy invocation removes the build-only `--print=native-static-libs` flag;
  that flag makes clippy-driver delegate directly to rustc, so the earlier
  invocations did not establish lint coverage. The corrected check also
  identified minor documentation/test-lint cleanup from earlier fixes.
- Targeted native regressions cover executable code-cache publication and
  imported VM copy routing/data integrity.
- The rebuilt lifecycle harness rejects all four injected init/shutdown
  failure scenarios with gtest assertions and exit status 1, without GPU work.
- The core-dump harness passes 27 parent/caller and four host-backed child
  injection checks. These validate failure reporting and ordering without
  executing GPU work or supplying fake dump output. Its corrected DisableFlag
  comparison initially failed for libhsa (10.379 s) while reference passed
  (1.618 s). With the default handler, libhsa also passes in 1.618 s.
- Seven authorized device-specific MODE1 recoveries completed successfully.
  Independently recovered reference Abort controls pass with debugging off
  and on; Rust failed on the original driver. A read-only live page-table
  comparison identifies an invalid Rust root entry for the CWSR trap address. Delaying ACQUIRE_VM
  restores that mapping. The delayed Abort and Generic controls both pass
  with no new kernel messages. This timing diagnostic is not a committed
  workaround and does not qualify the unmodified driver as fixed.
- With the corrected module loaded, NoTrap and Abort pass in 0.415 s each;
  Generic passes in 0.465 s. These use the unchanged runtime binary, no preload,
  and `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1`. A separate read-only PTE diagnostic
  finds a valid readable/executable trap leaf without a pre-acquire delay.
- The 41-case patched-driver regression completes with 39 passes and two
  expected fabric skips. The kernel delta contains 256 illegal-opcode messages
  from negative queue testing and 16 workqueue latency warnings, with no new
  VM fault, MES timeout, or reset. The final GPU handle audit is empty.
- The default-handler runtime passes 35 queue unit tests and the full host
  CTest/Clippy/format checks. All seven explicit-callback GPU regressions pass.
  The seven core-dump runs finish in 1.6–2.2 s, with no watchdog termination.
  Per-case health checks show only intended address-zero faults and empty GPU
  handle audits; there are no new MES timeouts or resets.
- The earlier process audit found four orphaned `MemoryTest1` children from the
  earlier canceled HIP recovery batch still holding GPU handles. Their exact
  executable, case-list path, and output logs established ownership. All four
  were terminated through checked pidfds; `fuser` then reported no users of
  `/dev/kfd` or `renderD128`. This cleanup did not rerun HIP or reset the GPU.
  Their earlier presence is another limitation on clean-state comparisons.

Local artifacts under `target/rocr-test-fixes/`:

- `runs/core-writer-callback-regressions/`: all three trap cases and all four
  negative queue cases pass after capture integration; health audit is empty.
- `runs/core-writer-final-*` and `kernel/core-writer-final-health.json`: all
  seven final dump cases, with per-case health/handle audits.
- `core-writer-final-integrity.gpu`: retained real GPU core for inspection;
  `core-all-host-tests.log`, `core-final-clippy.log`, and
  `core-final-ctest.log`: host/ABI/formatter qualification. The native,
  libhsa and loader host tests pass 658 cases with seven existing ignores.
- `runs/core-writer-atfork-query/`, `core-atfork-query.c`, and
  `kernel/core-writer-atfork-health.json`: the additional fork-lock control;
  this preload is not required for the seven plain passing cases.

- `runs/default-queue-*`, `default-queue-{host,ctest,clippy,build}.log`, and
  `kernel/default-queue-coredump-health.json`: earlier callback and dump results,
  host checks, and serial per-case health audits.
- `summary.json` and `results.csv`: all 133 results from the fresh sweep.
- `runs/checked-final-133/`: current manifest, immutable library, per-case
  logs/XML, and all 133 health audits.
- `kernel/svm-{manifest,reload-result,native-health,frontend-health}.json`:
  combined module/source identity, loading, and native/public-HSA qualification.
- `svm-{host-tests-final,ctest-final,clippy-final-2}.log`: host qualification.
- `runs/final-nonfaulting/`: complete 120-case rerun, logs/XML, selection,
  revision/dirty state, environment, and immutable library hash.
- `runs/queue-decoder/`: four corrected negative queue cases.
- `runs/reference-traps/` and `reference-kernel.log`: reference control and
  kernel evidence, with the clean-state limitations described above.
- `final-kernel.log` and `orphan-hip-cleanup.json`: final diagnostics and the
  verified cleanup of the earlier test children.
- `vmem-import-copy-final.log`, `runs/vmem-public-metadata/`,
  `final-host-ctest.log`, and `final-clippy.log`: final targeted validation.
- `harness-before.log`, `harness-after.log`, `harness-normal.log`,
  `harness-build-commands.json`, and `harness-hashes.json`: false-pass proof,
  corrected outcomes, exact compile/link commands, and binary identities.

The original broader CI baseline remains under `target/ci-gfx1201-20261119/`.
HIP was not resumed; this work follows the user's ROCr-first direction.

The [kernel correction](patches/README.md) has passed an isolated full driver
build, independent review, and temporary-load hardware qualification. The
currently loaded combined module has srcversion `DE114CD2BB59945654162BC`. The installed
boot module remains original (`88983A47E73AF492726AD7C`); a reboot loses both driver corrections.
No persistent driver installation was performed. Runs and kernel snapshots
are under `runs/checked-final-133/`, earlier `runs/patched-kmd-*`, and `kernel/`;
each run records the loaded
srcversion and immutable runtime hash. The PTE diagnostic is recorded
separately and is excluded from the plain test totals.
