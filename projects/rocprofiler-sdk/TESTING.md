# Testing Strategy — rocprofiler-sdk

- **Status:** Draft
- **Owner:** @itrowbri
- **Technical Lead:** TBD
- **Last Updated:** 2026-08-06

This document is the reference for how **rocprofiler-sdk** is validated and how engineers should write
tests for it. It describes what exists today, the tooling behind it, when tests run, and the
conventions to follow.

> **Scope.** rocprofiler-sdk is a GPU profiling/tracing library plus the `rocprofv3` CLI and `rocpd`
> tooling. It is an in-process interposer sitting directly on HIP, the HSA/ROCr runtime, and
> `aqlprofile`, and most of its behavior is only observable on real AMD GPU hardware. That shapes
> everything below: unit tests cover the CPU-side logic that can be isolated, and the bulk of
> confidence comes from integration tests that run instrumented applications on-device and validate
> the emitted output.

For the day-to-day mechanics (the CMake recipe, reconstructing a single test's command under `gdb`),
the authoritative source is [CONTRIBUTING.md → "Testing Guidelines"](CONTRIBUTING.md#testing-guidelines);
this document is the strategy layer and does not duplicate it.

---

## 1. Unit Testing

**What is tested.** CPU-side library logic exercisable without a full profiling session:

- Counter/metric machinery — AST evaluation (`evaluate_ast_test.cpp`), metric and dimension
  definitions (`metrics_test.cpp`, `dimension.cpp`), the counter-expression parser.
- PC-sampling parser (`source/lib/rocprofiler-sdk/pc_sampling/parser/tests`), incl. per-arch decode.
- Buffer management — serial/parallel producer-consumer paths and save/load round-trips
  (`source/lib/tests/buffering/`).
- Code-object loading (`source/lib/tests/codeobj`), AQL packet construction (`.../aql/tests`),
  KFD/platform discovery, and shared utilities (`source/lib/tests/common`).

**Tooling.** GoogleTest. Registered through the `rocprofiler_add_unit_test` helper in
`cmake/rocprofiler_utilities.cmake`, which `include(GoogleTest)`, registers cases with
`gtest_add_tests`, applies the `unittests` label, and prefixes the CTest name with `unit.`; each
test's own `CMakeLists.txt` links `GTest::gtest`/`GTest::gtest_main` itself. All discoverable/runnable
through CTest (`ctest -L unittests`).

**Location & convention.** Tests live *next to the code they test*, in the module's `tests/`
subdirectory (15 directories under `source/lib/`). The test file is named after the source file it
covers, the first arg to `TEST(<group>, <name>)` is that file or the executable name, and one
executable is built per test folder. `add_subdirectory(tests)` is guarded with
`if(ROCPROFILER_BUILD_TESTS)`.

**Caveat — "unit" is not always GPU-free here.** Some unit tests compile real device kernels (the
counters tests build `.hsaco` objects per `GPU_TARGETS` with `amdclang++`), initialize a driver, or
take a `RESOURCE_LOCK`, and will `SKIP` when no GPU/permissions are present. For an interposer, the
"pure CPU" vs "pure device" split is not clean — part of the unit layer needs the ROCm toolchain to
build and a GPU to run fully.

**Coverage expectations.** Every non-trivial hardware-independent path (parsers, evaluators,
serialization, data-structure logic) should have a unit test, and new off-GPU-testable logic should
not lean solely on integration tests to catch regressions. The long-term ROCm-wide goal is >95% line
coverage of hardware-independent paths — not mandated initially, approached in phases from measured
baselines. gcovr sees host code only, so device paths (validated by integration tests) are invisible
to the number; it understates the true tested surface. See §6.

---

## 2. Integration Testing

The primary confidence mechanism, because profiling behavior is only fully observable end-to-end. The
point is not "the profiler ran" but "the profiler collected the *right data*."

**How it works — execute/validate.** Each test is a pair of CTest entries chained by fixtures:

1. **Execute** — a workload is run under the profiler. There are two harness styles (see below): the
   library-level tests preload a tool library directly, while the `rocprofv3` tests drive the shipped
   CLI. Either way the run emits trace/counter output as JSON, Perfetto, OTF2, or `rocpd` (SQLite).
2. **Validate** — a pytest `validate.py` parses the emitted file and asserts on its contents (agents
   present, API traces recorded, kernel dispatches captured, counter values sane).

**Validation philosophy — structural, not golden.** Validators assert *invariants and bounds*, not
byte-for-byte matches against stored golden files (there are none). Typical assertions: row/record
counts and non-emptiness, correlation-id ordering and contiguity, timestamp monotonicity
(`end >= start`), membership in a known set (kernel names, API domains), fixed launch dimensions
implied by the workload (e.g. grid/workgroup sizes), and counter values checked against an
expectation — sometimes an exact workload-derived value (`tests/counter-collection/validate.py`
asserts each non-zero counter equals `1 * scaling_factor`, a wavefront-size constant), more often just
a `> 0` / uniform-across-identical-dispatches bound. There are no stored golden files to diff against.
This is a deliberate strategy and it bounds what these tests catch: they will flag a
missing/mis-ordered/absent record or an out-of-expectation value, but a *plausible-but-wrong* value on
a path that only checks `> 0` can pass. It also keeps the suite robust across architectures, where
exact values legitimately differ.

**Two execute harnesses.** The distinction matters when you add a test:

- **Library-level (tool-preload).** An app from `tests/bin/` (e.g. `hip-graph`,
  `reproducible-runtime`) is launched with a tool library preloaded
  (`LD_PRELOAD=librocprofiler-sdk-json-tool.so`), driven directly by env vars such as
  `ROCPROFILER_TOOL_OUTPUT_FILE` and `ROCPROFILER_TOOL_CONTEXTS`. This exercises the SDK's
  interception layer straight, with no CLI in the path. Most directories under `tests/` use this style.
  The preloaded tool libraries are themselves built from `tests/tools/` (`json-tool.cpp`,
  `late-start-tool.cpp`, `thread-trace-*-tool.cpp`, `c-tool.c`) — that is where a new library-level
  harness's `.so` is authored.
- **CLI-level (`rocprofv3`).** The `tests/rocprofv3/` tree instead invokes the shipped `rocprofv3`
  tool — it does **not** preload the json-tool. See [§2a](#2a-rocprofv3-cli-tests) below.

**Registration.** Two wrappers in `tests/common/CMakeLists.txt` —
`rocprofiler_add_integration_execute_test` and `rocprofiler_add_integration_validate_test` — chain via
`FIXTURES_SETUP`/`FIXTURES_REQUIRED`, tag with the `integration-tests` label (plus one of many
sub-labels — `rocpd`, `pc-sampling`, `stochastic`, `thread-trace`, `multi-gpu`, `application-replay`,
`counter-collection`, `late-start`, `openmp-host`/`openmp-target`, `pause-resume`, `selected-regions`,
`attachment`, …; run `ctest --print-labels` for the authoritative list, since these labels are also
the mechanism CI uses for per-arch `-LE` exclusion), set a `TIMEOUT` (default 60s; the unit-test
default is 45s), and match against `ROCPROFILER_DEFAULT_FAIL_REGEX` (defined in
`cmake/rocprofiler_options.cmake` and overridden with a slightly broader pattern in
`tests/common/CMakeLists.txt`, which the integration wrappers use). `LD_PRELOAD` must be passed via
the wrappers' `PRELOAD` argument (it is rejected in `ENVIRONMENT`), which also lets the sanitizer
runtime be prepended automatically. Parse output with the readers in
`tests/pytest-packages/pytest_utils/` (`perfetto_reader.py`, `rocpd_reader.py`, `otf2_reader.py`,
`dotdict.py`) rather than by hand. `tests/pytest-packages/` is itself an installed pytest package
(consumed via `find_package(rocprofiler_sdk_pytest REQUIRED)`) that also ships shared test helpers and
PC-sampling fixture assets, so validators can `import` it rather than duplicating logic.

**What is covered** (non-exhaustive — see the full `tests/` tree). Each dependent system/feature has a
dedicated directory: tracing (`kernel-tracing`, `async-copy-tracing`, `hip-host-tracing`,
`hip-graph-tracing`, `scratch-memory-tracing`, `hsa-memory-allocation`, `code-object-multi-threaded`,
`late-start-tracing`, `hipfile`), counters (`counter-collection`), PC sampling (`pc_sampling`), thread
trace/SPM (`thread-trace`, `spm`), parallelism (`openmp-tools`), downstream libs (`rocdecode`,
`rocjpeg`, `rocshmem-trace`), and the DB/attach tooling (`rocpd`, `rocpd-api`, `rocattach`). Two
directories carry a distinct strategic role worth calling out:

- **`tests/c-tool/`** — a **C** tool (`tests/tools/c-tool.c`) preloaded to assert the SDK's
  C-linkage/priority/version contract (it checks the reported `rocprofiler-sdk vX.Y.Z`). This is the
  dedicated guard for the C-ABI surface that CONTRIBUTING calls out (public API structs lead with a
  `uint64_t size` for runtime ABI checks), so a "new public API" change should keep it green.
- **`tests/environment/`** — regression coverage for a bash-specific `getenv()` bug; it runs
  `rocprofv3` with `bash` as the target and deliberately **does not** guard on `find_program(bash)`,
  so the bug it exists to catch can't be silently skipped. A concrete example of the "regression test
  that fails without the fix" policy, and an intentional exception to the usual skip-when-absent rule.

Shared test workloads live in `tests/lib/` (`transpose`, `vector-operations`); the tool `.so`s are
built from `tests/tools/`. The end-to-end CLI coverage lives in `tests/rocprofv3/` and
`tests/rocprofv3-avail/` — covered in [§2a](#2a-rocprofv3-cli-tests).

**GPU requirement.** Requires real AMD GPU hardware; does not run CPU-only. There is no `REQUIRES gpu`
flag — CI excludes cases per-GPU via ctest `-LE`/`-E` regex env vars (`navi3_`, `mi200_`, `mi300_`,
`mi300a_`, `mi325_`, …); unsupported cases are disabled in CMake, not in the CI invocation. Some
features are hardware-gated (e.g. PC sampling on mi200/mi300a only). Multi-GPU cases are label-tagged
(`multi-gpu`) and hard-disabled or arch-gated in CMake, then enabled per-runner — the harness does not
auto-detect a second GPU.

**Optional-dependency gating (a green run may have skipped whole areas).** Several suites `find_package`
an optional dependency and **silently disable themselves** when it is absent from the build
environment — MPI (`mpi-ranks`, `rocshmem-trace`), rocDecode/rocJPEG (`rocdecode`, `rocjpeg`), the ATT
decoder (`advanced-thread-trace`, gated on `attdecoder_FOUND`), and OpenMP target (gated on a gfx-arch
regex). A passing `ctest` therefore does **not** imply these ran; check the configure log for what was
found. `tests/environment/` is the deliberate exception — it intentionally omits the guard so its
regression coverage can't be silently skipped.

**Size/duration.** Keep within the 60s default `TIMEOUT` where possible; prefer a few representative
workloads (vary threads/streams/problem size via the app's own arguments) over many near-duplicates —
these tests are expensive and hold a GPU.

**Samples as a third category.** The `samples/` programs (`api_buffered_tracing`, `counter_collection`,
`pc_sampling`, `thread_trace`, …) double as end-to-end tests: each is registered with plain `add_test`
under the `samples` label and demonstrates the public API the way an external tool would consume it.
Unlike the integration tests, they have **no external `validate.py`** — any correctness checking is
self-contained in the sample's own `client.cpp`/`main.cpp`, and the ctest is gated only on clean exit
plus `FAIL_REGULAR_EXPRESSION`. So a sample is a strong smoke test of the public API surface but a
weaker output check than the execute/validate pairs; pair a public-API change with an integration
validator when output correctness (not just "runs clean") matters.

---

## 2a. rocprofv3 CLI Tests

`rocprofv3` is the command-line tool shipped with the SDK — a `python3` front-end
(`source/bin/rocprofv3.py`, installed as `rocprofv3`) that configures collection, launches (or
attaches to) the target application with the SDK tool library loaded, and post-processes the results
into the requested output formats. The `tests/rocprofv3/` tree (40+ subdirectories) is the end-to-end
coverage for that tool, and it is the largest single test area in the repo.

**How these differ from library-level integration tests.** The library-level tests preload a tool
library (`LD_PRELOAD=librocprofiler-sdk-json-tool.so`) directly onto a `tests/bin/` app. The rocprofv3
tests do **not** — the execute step invokes the `rocprofv3` CLI, which sets up the environment and
loads `librocprofiler-sdk-tool.so` itself. So these tests validate the *tool's* behavior (its
arguments, its `-i <config.json>` input mode, its output writers, its attach flow) on top of the SDK,
not just the SDK's interception layer. Because the CLI owns `LD_PRELOAD`, the sanitizer runtime is
injected through a different channel: the harness maps `LD_PRELOAD=` to `ROCPROF_PRELOAD=` (which
rocprofv3 reads via its `--preload` option and prepends), so sanitizer builds still work end-to-end.

**Registration.** The same wrappers are used (`rocprofiler_add_integration_execute_test` /
`_validate_test`), but the execute `COMMAND` is
`$<TARGET_FILE:rocprofiler-sdk::rocprofv3> <flags> -- $<TARGET_FILE:<workload>>` rather than a bare app
with a preload. The canonical example is `tests/rocprofv3/tracing/`, which runs `simple-transpose`
under several flag combinations — full trace (`-M --hsa-trace --kernel-trace ...`), `--sys-trace`, and
a JSON-config input (`-i input_trace.json`) — each emitting `pftrace csv json rocpd` and validated by
one shared `validate.py`. Validators receive the produced files by path via CTest `ARGS`
(`--hsa-input`, `--kernel-input`, `--json-input`, `--pftrace-input`, `--rocpd-input`, …) and can skip
individual checks with `DISCOVERY_ARGS -k "not <test>"`.

**What the tree covers.** Output formats and the CSV/JSON/Perfetto/OTF2/rocpd writers; the input-file
config mode; trace domains and combinations (`tracing`, `tracing-plus-counter-collection`,
`counter-collection`, `pc-sampling`, `spm`, `advanced-thread-trace`); CLI features
(`kernel-rename`, `summary`, `agent-index`, `collection-period`, `minimum-bytes`,
`negate-aggregate-tracing-options`, `kernel-trace-duration`); parallelism (`mpi-ranks`, `ompt`);
downstream libraries (`rocdecode-trace`, `rocjpeg-trace`, `rocshmem-trace`, `hipfile-trace`); the
`rocpd*` database outputs (`rocpd`, `rocpd-kernel-rename`, `rocpd-scratch`, `rocpd-virtual`) and the
`conversion-script`; the legacy-compat `roctracer-roctx` / `roctx-pause-resume` paths; and
`python-bindings`.

**Dynamic attach.** A distinct rocprofv3 capability — attaching to an already-running process rather
than launching it (`--attach`/`--pid`/`-p`) — is exercised under `tests/rocprofv3/attachment/`
(`attach-once`, `attach-twice`, `attach-tree`, `attach-once-att`) via helper scripts, and tagged with
the `attachment` sub-label. `tests/rocattach/` covers the lower-level attach entry point.

**Application replay / determinism.** Collecting more counters than fit in one pass makes rocprofv3
re-run the application (multi-pass "application replay"), so determinism across runs is a real
correctness dimension. It is exercised under `tests/rocprofv3/tracing-plus-counter-collection/`
(tagged `application-replay`), backed by the `reproducible-runtime` / `reproducible-dispatch-count`
workloads in `tests/bin/`, whose validators assert the same dispatches/counts appear on each pass.

**Availability tool.** `tests/rocprofv3-avail/` covers `rocprofv3-avail`, the companion CLI that lists
available counters/agents. These query the device directly (`info`, `-d 0`) rather than running a
profiled workload, but are still labeled `integration-tests` since they need real hardware present.

---

## 3. Performance Testing

A standalone benchmark project under `benchmark/` (its own CMake project, built only when
`ROCPROFILER_BUILD_BENCHMARK=ON`, default OFF; **not** Google Benchmark). It measures; nothing gates
on the result.

- **Driver:** `benchmark/source/bin/rocprofv3-benchmark.py`, run as
  `rocprofv3-benchmark -i <config>.yml -n <iterations>`.
- **Workloads:** a `mandelbrot` HIP app and a vLLM inference config (`vllm.yaml`).
- **Measured:** *profiler overhead* — the cost the tool adds to an application, captured with the
  bundled `timem` (`ROCPROFILER_BENCHMARK_INSTALL_TIMEM`).
- **Results:** written to SQLite (`benchmark_metrics`, `benchmark_statistics`; schema in
  `benchmark_tables.sql` / `benchmark_views.sql`) for run-over-run comparison.

**Baselines & gating.** Metrics land in SQLite so a baseline can be set and later runs compared, but
comparison is manual — run the relevant workload before/after a perf-sensitive change and confirm
overhead has not regressed. Absolute numbers are not comparable across GFX. There is no automated
threshold, no shared per-architecture baseline store, and no PR or nightly gate — a known gap (§8).

---

## 4. When Tests Run

CI workflows live at the monorepo root in `.github/workflows/`, prefixed `rocprofiler-sdk-*`.
Build+test is driven by `source/scripts/run-ci.py`, which submits to CDash; installed packages are
re-validated via `ctest --test-dir`.

| Workflow | Trigger | Purpose | Blocks merge? |
|---|---|---|---|
| `...-continuous_integration.yml` | PR, push `develop`, nightly `cron '0 7 * * *'`, dispatch | Build + unit + integration across arch/OS matrix; `sanitizers` job nightly/dispatch only | **Yes** (build+tests) |
| `...-code_coverage.yml` | PR, push `develop`, dispatch | gcovr coverage on mi325/ubuntu-22.04, diffed vs base SHA, posted as PR comment; asserts every test is labeled and `samples`/`tests` don't overlap | Label check yes; coverage no |
| `...-formatting.yml` | PR | clang-format-11, cmake-format, black, trailing-newline | **Yes** |
| `...-python.yml` | push / PR | flake8 across Python 3.8 / 3.10 / 3.12 | **Yes** |
| `...-restrictions.yml` | PR | Code-restrictions check | **Yes** |
| `...-codeql.yml` | PR, push `develop`, dispatch | CodeQL — **python + actions only, no C++** | Informational |
| `...-docs.yml` / `...-rocm_release_compatibility.yml` | PR, push `develop`, dispatch | Docs build / release compat | Informational |

- **Per-PR:** build, unit, and integration run on the changed branch, plus formatting, flake8,
  restrictions, coverage (comment only), CodeQL.
- **Nightly (`CI_MODE=Nightly`, 07:00 UTC):** fuller RPM distro matrix, the sanitizer lanes, coverage,
  and install-then-retest of packaged artifacts. Continuous mode is a lighter subset.
- **The fork caveat.** The build/unit/integration jobs are guarded by
  `!github.event.pull_request.head.repo.fork` — they need internal GPU runners and secrets, so **a PR
  from a fork skips them entirely** and can show all-green while nothing was built or tested. Review
  fork contributions by running the suite from a main-repo branch.
- **GPU archs (`GPU_TARGETS`):** `gfx906 gfx908 gfx90a gfx942 gfx950 gfx1030 gfx1100 gfx1101 gfx1102
  gfx1201`, on self-hosted per-arch runners in `rocm/rocprofiler-private` containers. Primary lanes
  run on mi325 (gfx94X); navi3/navi4 rows are temporarily disabled. Those containers are built and
  refreshed by `...-build-ci-docker-images.yml` (nightly + push/PR/dispatch) — infrastructure rather
  than a test lane.
- **OS matrix:** ubuntu-22.04 (DEB), rhel-8.8/9.5, sles-15.6 (RPM).

**Supported configurations, explicitly not tested:** Windows (unsupported for this component);
CPU-only machines (unit tests only); individual XCC mode; fork-PR validation; any `GPU_TARGETS` arch
without a runner (nightly at best).

---

## 5. How to Write Tests

**Decide the layer first.**

- **Hardware-independent** (parser, evaluator, data structure, serialization) → a **GoogleTest unit
  test** next to the code, in the module's `tests/` directory, registered with
  `rocprofiler_add_unit_test`.
- **Only observable on-device** (a trace domain, counter, CLI feature, output format) → an
  **integration test** under `tests/`:
  1. Add or reuse a workload in `tests/bin/`.
  2. Register an execute+validate pair (`rocprofiler_add_integration_execute_test` +
     `rocprofiler_add_integration_validate_test`).
  3. Write `validate.py` (`test_*` functions) + `conftest.py` + `pytest.ini`, parsing output with the
     readers in `tests/pytest-packages/pytest_utils/`.

**Sufficient coverage by change type:**

| Change | Expected validation |
|---|---|
| New counter/metric | Unit test for its expression/AST **and** an integration test proving a plausible on-device value |
| New trace domain / API interception | Integration test asserting the expected records appear |
| New `rocprofv3` / `rocpd` feature | End-to-end test through the CLI under `tests/rocprofv3/` (§2a), not a tool-preload test |
| New public API | Integration validator for output correctness, plus a sample that consumes it (smoke-level, self-checking); keep `tests/c-tool/` green for the C-ABI/version contract |
| Bug fix | Regression test that fails without the fix |
| Performance-sensitive path | Run the relevant `benchmark/` workload before/after |
| New GPU/OS support | Validate on that config; update §4's config list |

Give every integration test a `LABEL` and a realistic `TIMEOUT`; tag known-flaky cases `UNSTABLE`
rather than letting them fail intermittently.

**Local workflow:**

```bash
cmake -B build -DROCPROFILER_BUILD_CI=ON -DROCPROFILER_ENABLE_CLANG_TIDY=ON .
cmake --build build --target all -j
ctest --test-dir build --output-on-failure        # -L unittests / -L integration-tests to filter
```

`ROCPROFILER_BUILD_CI=ON` forces tests + samples + `-Werror` on. (Note: if you first configured with
it OFF, turning it ON later does not retroactively enable tests/samples — set them explicitly or start
fresh.) Other knobs in `cmake/rocprofiler_options.cmake`:
`ROCPROFILER_MEMCHECK=<Address|Thread|Undefined|Leak>` (sanitizer build; opt out per test with
`DISABLED_MEMCHECKS`), `ROCPROFILER_BUILD_CODECOV=ON`, `ROCPROFILER_DISABLE_UNSTABLE_CTESTS=ON`
(default, skips `UNSTABLE` tests). Separately, `ROCPROFILER_BUILD_INTEGRATION_TESTS=OFF` (a
non-standard cache variable set in the top-level `CMakeLists.txt`, defaulting to
`ROCPROFILER_BUILD_TESTS`) builds/installs only the unit tests — the closest thing to a GPU-free
build.

See **CONTRIBUTING.md → "Testing Guidelines"** for the canonical instructions.

---

## 6. Coverage

Measured with **gcovr** (not lcov, not a hosted service). Enable with `ROCPROFILER_BUILD_CODECOV=ON`;
`source/scripts/run-ci.py --coverage {all,tests,samples}` generates the reports (needs `gcovr` and a
matching `gcov-<ver>`), honoring `codecov_exclude` lists. The `...-code_coverage.yml` workflow builds
on mi325/ubuntu-22.04, caches coverage keyed on the base SHA so a PR is diffed against its base, and
posts a PR comment. Its one gating step ("Verify Test Labels") asserts that every test is classified:
none is un-labeled (each carries `samples`, `unittests`, or `integration-tests`), and there is no
overlap between the `samples` and `tests` groups. Tests opt out with `DISABLED_CODECOV`.

**Code coverage ≠ test coverage.** Code coverage is lines executed (700/1000 → 70%); test coverage is
scenarios exercised (arch families, trace domains, output formats, multi-GPU, attach-vs-launch).
Because device paths are invisible to host-side gcovr, this component can show high code coverage while
on-device test coverage is incomplete. Measured on Linux only (Windows unsupported).

---

## 7. Sanitizers

Selected with `-DROCPROFILER_MEMCHECK=<type>` (`cmake/rocprofiler_memcheck.cmake`):
**AddressSanitizer** (host memory errors), **ThreadSanitizer** (races — high value for an in-process
interposer in multithreaded apps), **LeakSanitizer**, **UndefinedBehaviorSanitizer**. The runtime is
`LD_PRELOAD`ed into tests (translated to `ROCPROF_PRELOAD` for rocprofv3 tests); suppression files and
`setup-sanitizer-env.sh` live in `source/scripts/`.

```bash
cmake -B build-tsan -DROCPROFILER_BUILD_CI=ON -DROCPROFILER_MEMCHECK=ThreadSanitizer .
source source/scripts/setup-sanitizer-env.sh
cmake --build build-tsan --target all -j && ctest --test-dir build-tsan --output-on-failure
```

**In CI:** nightly / manual dispatch only — **never on a PR**. Not run by any lane: MSan (it is a
selectable `--memcheck` choice in `run-ci.py`, but no workflow uses it); device-side memory errors
beyond what a host sanitizer surfaces. Tests opt out with `DISABLED_MEMCHECKS`.

---

## 8. Known Gaps / Maturity

| Gap | Impact | Mitigation today |
|---|---|---|
| No automated performance-regression gate at any cadence, and no shared per-arch baseline | Medium | Manual before/after comparison via `benchmark/` |
| Integration coverage bounded by per-arch runner availability; several `GPU_TARGETS` archs only nightly, navi3/navi4 lanes disabled | High | Nightly coverage where runners exist; documented in §4 |
| Fork PRs skip build/unit/integration entirely | High if unnoticed | Reviewer runs the suite from a main-repo branch |
| Sanitizers nightly only, not per-PR | Medium | Nightly ASan/TSan/LSan/UBSan lanes |
| No C++ static analysis on PRs (CodeQL is python+actions only) | Medium | clang-tidy runs in the build lane; code review |
| `UNSTABLE` tests lack a per-case owner+ticket+expiry; nothing reports what is quarantined | Medium | `ROCPROFILER_DISABLE_UNSTABLE_CTESTS=ON` keeps them out of gating |
| Flaky *detection* is manual/local, not an automated gate | Medium | `run-ci.py` supports `--repeat-until-{pass,fail}` / `--repeat-after-timeout` (→ `ctest --repeat`), but no CI lane invokes them |
| Device paths invisible to host-side gcovr | Medium | Integration tests validate device paths |

A flaky test tagged `UNSTABLE` is a temporary quarantine, not a final state — it should have an owner
and a tracking bug and be re-enabled once fixed. Flaky shake-out today is a local step: `run-ci.py`
exposes `ctest --repeat` (`--repeat-until-pass`, `--repeat-until-fail`, `--repeat-after-timeout`), and
CONTRIBUTING documents `ctest --repeat until-fail:<N>`, but no workflow passes these — so nothing
detects a newly flaky test automatically before it lands.

---

## Review Cadence

Review this document when: a major architectural change lands (especially anything changing what is
testable without a GPU); a new test pattern or CI lane is introduced, or a lane changes what it gates;
a regression escapes to a downstream consumer or to release; before a major release, alongside the
known-gap review. At minimum, revisit §8 quarterly — the strategy is working if that table shrinks.
