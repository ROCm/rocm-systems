# Testing Strategy - rocprofiler-sdk

This document is the authoritative reference for how `rocprofiler-sdk` is
validated and how engineers should write tests for it. It describes what exists
today in the repository, the tooling behind it, when tests run, and the
conventions to follow when adding coverage.

> **Scope.** `rocprofiler-sdk` is a GPU profiling/tracing SDK plus the
> `rocprofv3` CLI, `rocattach`, `rocpd` tooling, Python bindings, installed-test
> packaging, samples, and benchmark infrastructure. It sits directly on top of
> HIP, the HSA/ROCr runtime, and `AQLProfile`, and most of its behavior is only
> observable on real AMD GPU hardware. That shapes everything below: unit tests
> cover CPU-side logic that can be isolated, and the bulk of confidence comes
> from integration tests that run instrumented applications on-device and
> validate emitted output.

The exact CI implementation lives in repository workflows, CMake files, and
`source/scripts/run-ci.py`. This document defines the intended strategy. If
automation and this document disagree, update one of them rather than treating
the mismatch as an undocumented exception.

## 1. Unit Testing

**What is tested.** CPU-side library logic that can be exercised without a full
profiling session:

- Counter/metric machinery: AST evaluation (`evaluate_ast_test.cpp`), metric and
  dimension definitions (`metrics_test.cpp`, `dimension.cpp`), and the counter
  expression parser.
- PC-sampling parser behavior
  (`source/lib/rocprofiler-sdk/pc_sampling/parser/tests`).
- Buffer management: producer-consumer paths, save/load paths, and record
  bookkeeping (`source/lib/tests/buffering`).
- Code-object loading (`source/lib/tests/codeobj`), AQL packet construction
  (`source/lib/rocprofiler-sdk/aql/tests`), KFD/platform discovery, shared
  utilities (`source/lib/tests/common`), configuration and environment parsing,
  context state machines, timestamp/correlation transforms, and isolated bug
  regressions.

**Tooling.** GoogleTest and CTest. Prefer `rocprofiler_add_unit_test` from
`cmake/rocprofiler_utilities.cmake` so labels, timeouts, installation, data
files, and default failure matching stay consistent. The helper uses
`gtest_add_tests` and adds the `unittests` label by default.

**Location & convention.** Unit tests live next to the code they test, in a
`tests` subdirectory of the owning library module. Examples include
`source/lib/rocprofiler-sdk/tests`, `source/lib/rocprofiler-sdk/counters/tests`,
`source/lib/rocprofiler-sdk/pc_sampling/tests`,
`source/lib/rocprofiler-sdk/thread_trace/tests`, `source/lib/tests/common`,
`source/lib/tests/buffering`, `source/lib/tests/codeobj`, and
`source/lib/aqlprofile/*/tests`. If a test targets one source file, use the same
base filename when practical. Existing test files use several conventions,
including `_test.cpp`, `_tests.cpp`, and implementation-style names such as
`buffer.cpp`, `dimension.cpp`, and `queue_interposition.cpp`; follow the local
module convention. Compile sources in a unit-test directory into one executable
unless there is a clear dependency, isolation, or runtime reason to split them.
Guard parent `add_subdirectory(tests)` calls with `if(ROCPROFILER_BUILD_TESTS)`.

**Note on GPU coupling.** Some "unit" tests still compile real device kernels.
For example, counter tests build `.hsaco` objects per `GPU_TARGETS` with
`amdclang++`. These validate library logic but require the ROCm toolchain to
build.

**Coverage expectations.** Every non-trivial, hardware-independent code path
should have focused unit coverage when practical. Changed decision branches with
distinct observable behavior should have tests; new public APIs and options
should have positive and negative coverage; bug fixes should include a
regression test when practical. Coverage is measured with `gcovr` (see
`Coverage Tooling`). If a path cannot be tested at this layer, state the gap and
compensating validation in the PR.

## 2. Integration Testing

Integration testing is the primary confidence mechanism for `rocprofiler-sdk`
because profiling behavior is only fully observable end-to-end.

**How it works.** The standard pattern is an execute/validate pair wired
through CTest fixtures:

1. Build or reuse a workload, usually from `tests/bin`.
2. Run the profiler, runtime scenario, or test application.
3. Write an output artifact.
4. Validate the artifact with pytest.
5. Connect setup and validation with CTest fixtures.

**Dependent systems exercised.** Integration tests exercise ROCr/HSA, HIP,
`AQLProfile`, `rocprofiler-register`, tool registration, callbacks, buffering,
`rocprofv3`, `rocattach`, Python bindings, SQLite/`rocpd`, OpenMP tools,
`rocDecode`, `rocJPEG`, Linux process and `ptrace` behavior, installed CMake
packages, supported GPUs, firmware, and ROCm releases. Directories under
`tests` cover kernel tracing, async copy, counter collection, thread trace, PC
sampling, hip graph tracing, scratch memory, `rocdecode`, `rocjpeg`,
`openmp-tools`, `rocpd`, `rocpd-api`, `rocprofv3`, `rocprofv3-avail`,
`rocattach`, and Python bindings.

**Tooling.** CTest orchestrates execution and pytest validates output. Use
`rocprofiler_add_integration_execute_test` and
`rocprofiler_add_integration_validate_test` from `tests/common/CMakeLists.txt`.
These helpers support labels, timeouts, fixtures, environment variables,
preloads, memcheck exclusions, coverage exclusions, and unstable-test handling.
pytest validators generally use `validate.py`, with `conftest.py` and
`pytest.ini` when needed. Prefer helpers from `tests/pytest-packages` instead
of reparsing complex output by hand.

**Location & convention.** Integration tests live under the top-level `tests`
directory. Workload applications live under `tests/bin`; shared integration
helpers and tool libraries live under `tests/common`, `tests/lib`, and
`tests/tools`; pytest helper packages live under `tests/pytest-packages`.
Samples live under `samples` and should remain simple unless a sample is
intentionally used as validation for public behavior.

`validate.py` files should follow the existing fail-fast entry point:

```python
if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
```

**Validation expectations.** Validators should check semantic correctness, not
just file existence. Depending on the output, assert required records and
domains, schema and field types, identifier and correlation relationships,
timestamp ranges, expected dispatches/copies/API calls/counters, per-agent or
per-queue attribution, absence of duplicate or forbidden records, graceful
handling of empty or partial data, and consistency across CSV, JSON, pftrace,
OTF2, or database outputs where supported. Avoid exact full-file golden
comparisons for nondeterministic fields such as process IDs, addresses,
timestamps, and runtime-generated identifiers.

**Hardware requirements.** Many integration tests require real AMD GPU
hardware. Use `GPU_TARGETS` to control compile targets; otherwise the build uses
the default target list from `tests/cmake/rocprofiler_test_gpu_targets.cmake`.
Hardware-specific tests should detect unsupported hardware explicitly, skip only
when the required capability is genuinely unavailable, provide a useful skip
reason, identify the minimum architecture or service required, use counters
valid for the architecture, and preserve logs and generated artifacts on
failure.

**Lifecycle and packaging.** Coverage should include relevant lifecycle and
concurrency cases: init/finalize, start/stop/flush/pause/resume/restart,
callback registration/removal, concurrent queues/streams/threads/processes,
process exit and signals, attach/detach during active work, partial output after
failure, repeated creation/destruction, multiple contexts or tools, and delayed
asynchronous callbacks. For public or packaged features, add installed-package
evidence when a change can affect installed headers, exported CMake targets,
`RPATH`, runtime discovery, package contents, or consumer behavior.

## 3. Performance Testing

**What exists.** Performance testing uses the benchmark suite under
`benchmark`. It is its own CMake project, orchestrated through CTest and
`rocprofv3-benchmark`; it is not based on Google Benchmark.

- Driver: `benchmark/source/bin/rocprofv3-benchmark.py`, invoked as
  `rocprofv3-benchmark -i <config>.yml -n <iterations>`.
- Workloads/configs: `benchmark/example.yml`, `benchmark/minimal.yaml`,
  `benchmark/vllm.yaml`, and benchmark workloads such as `mandelbrot`.
- Results storage: SQLite tables and views such as `benchmark_metrics` and
  `benchmark_statistics`.

**What is measured.** Run performance tests for changes that can affect
application wall-clock overhead, profiler startup/shutdown/attach/detach
latency, callback or buffered tracing overhead, counter collection, PC sampling,
thread trace, SPM, `rocpd`, buffer flush throughput, dropped records, output
serialization, database inserts, output size, CPU utilization, synchronization
contention, memory growth, or scaling across threads, queues, streams, agents,
processes, and duration.

**Baselines & regression thresholds.** Baselines must use comparable hardware,
ROCm versions, GPU visibility settings, workload inputs, build type, and
profiler options. For noisy workloads, increase the repeat count and compare
medians or aggregate statistics rather than a single run. The benchmark harness
records data into SQLite so baselines can be established and compared.

Formal automated performance gating is a known maturity gap. Until
component-specific gates exist, performance-sensitive PRs should include
before-and-after benchmark evidence or explain why the change cannot affect
runtime or profiler overhead. As review guidance, a stable runtime or
profiler-overhead regression greater than 5% should be explained, mitigated, or
accepted by the component owner; greater than 10% should be treated as blocking
unless it is an intentional tradeoff.

**How to run.**

```bash
cmake -B build-rocprofiler-sdk-benchmark \
  -DROCPROFILER_BUILD_BENCHMARK=ON \
  -DROCPROFILER_BUILD_SAMPLES=ON \
  .
cmake --build build-rocprofiler-sdk-benchmark --target all --parallel "$(nproc)"

cd build-rocprofiler-sdk-benchmark/benchmark
export PATH=${PWD}/bin:${PATH}
rocprofv3-benchmark -i ./example.yml -n 2
```

## 4. When Tests Run

**Cadence.** Test cadence depends on risk, runtime, and hardware requirements:

- Before PR: contributors should run relevant unit tests, targeted integration
  tests, formatting checks, and repeated runs for concurrency or flake-sensitive
  changes.
- Per PR: build the component and run unit and integration tests relevant to the
  changed behavior.
- Per PR when relevant: changes to `rocprofv3`, public SDK behavior, output
  formats, `rocpd`, profiler runtime services, packaging, or installed tests
  should include integration or package-test evidence.
- Per PR when relevant: deterministic library changes should include unit-test
  evidence; if not isolatable, explain why integration coverage is the right
  level.
- Scheduled, nightly, release, or explicit CI: sanitizer, coverage, multi-GPU,
  stress, unstable-test signal, compatibility, hardware-specific jobs, and
  expensive benchmarks.
- Release validation: full configured CTest suite, installed tests when
  packaging changes are present, known-gap review, and benchmark workloads
  relevant to release risk.

**PR evidence.** Static analysis, formatting, documentation, and Python checks
run according to repository workflows and are part of the quality signal. PR
descriptions for behavior changes should state the relevant unit, integration,
performance, and known-gap evidence. When a required cadence is not available,
document the gap and identify the follow-up CI, nightly, manual, or release
validation that will cover it.

## 5. Running Tests Locally

```bash
cmake -B build-rocprofiler-sdk \
  -DROCPROFILER_BUILD_TESTS=ON \
  -DROCPROFILER_BUILD_SAMPLES=ON \
  .
cmake --build build-rocprofiler-sdk --target all --parallel "$(nproc)"
ctest --test-dir build-rocprofiler-sdk --output-on-failure -O ctest.all.log
```

Useful CTest commands:

```bash
# List tests.
ctest --test-dir build-rocprofiler-sdk -N

# Show commands, working directories, labels, and environments.
ctest --test-dir build-rocprofiler-sdk -N -V

# Run unit or integration tests.
ctest --test-dir build-rocprofiler-sdk -L unittests --output-on-failure
ctest --test-dir build-rocprofiler-sdk -L integration-tests --output-on-failure

# Run a specific scenario or repeat a suspected flaky failure.
ctest --test-dir build-rocprofiler-sdk -R rocprofv3-test-trace --output-on-failure
ctest --test-dir build-rocprofiler-sdk -R '<test-regex>' \
  --repeat until-fail:20 --output-on-failure
```

`ROCPROFILER_BUILD_TESTS=ON` enables source-level unit tests. Top-level
integration tests are controlled by `ROCPROFILER_BUILD_INTEGRATION_TESTS`, which
defaults to `ROCPROFILER_BUILD_TESTS`. Use
`-DROCPROFILER_BUILD_INTEGRATION_TESTS=OFF` when a change only needs unit-test
build coverage.

Useful options:

- `ROCPROFILER_BUILD_CI=ON` enables CI defaults and stricter build options.
- `ROCPROFILER_MEMCHECK=<AddressSanitizer|ThreadSanitizer|LeakSanitizer|MemorySanitizer|UndefinedBehaviorSanitizer>`
  selects sanitizer-style builds where supported.
- `ROCPROFILER_BUILD_CODECOV=ON` enables coverage-oriented builds.
- `ROCPROFILER_DISABLE_UNSTABLE_CTESTS=ON` disables unstable tests. This is
  currently the default in the integration and sample test trees.
- Use CTest `-L`, `-LE`, `-R`, and `-E` to select tests by label or name.
- Other useful CTest options include `--rerun-failed`, `--stop-on-failure`,
  `--show-only` (`-N`), `--verbose` (`-V`), `--extra-verbose` (`-VV`), and
  `--print-labels`.

Use `source/scripts/run-ci.py --disable-cdash` to reproduce CI-style jobs
locally without submitting to CDash. For sanitizer jobs, prefer
`source/scripts/run-ci.py`; if configuring manually, use
`source/scripts/setup-sanitizer-env.sh` to match the sanitizer runtime
environment.

For single-test debugging, start with `ctest --test-dir build-rocprofiler-sdk -N
-V -R <test-name>` and use the printed command, working directory, labels, and
environment to reproduce the test in a shell or debugger. If the test command
uses `rocprofv3`, debugging may require `gdb --args python3 /path/to/rocprofv3
...`; if `rocprofv3` replays the application, use `set follow-fork-mode child`
inside `gdb`.

## 6. How to Write Tests

**Choose the tier first.** Unit test isolated deterministic logic, add component
integration tests for runtime interaction, add end-to-end tests for user-visible
workflows, and benchmark changes that can affect cost or scale. Do not replace
a missing unit test with a slow end-to-end test when the logic can be isolated,
and do not replace required end-to-end validation with mocks when the defect can
occur only through real runtime interaction.

**Structure.** Tests should follow Arrange, Act, Assert. Shared fixtures should
contain setup mechanics, not hidden behavioral decisions.

**CTest registration.** Every registered test should define the target or
command, category and component labels, timeout, environment, fixtures, hardware
or dependency requirements, cleanup behavior, and any intentional unstable,
sanitizer, or coverage exclusions. Prefer existing `rocprofiler-sdk` helper
functions. Do not use a source-tree path as an accidental runtime dependency of
an installed test; install test assets needed after package installation.

**Assertions.** Verify the strongest stable contract available. Prefer exact
values for deterministic results, ranges for variable-duration events, set or
relationship checks for unordered records, explicit error contracts,
workload-tied record counts, cross-field invariants, and absence checks for
forbidden behavior. Avoid "file is non-empty" or process exit status as the only
validation, broad substring matches, implementation-private ordering unless it
is the contract, or tolerances so wide that defects pass.

**Negative and regression tests.** For new inputs, options, APIs, or file
formats, consider malformed input, missing input, unsupported values, duplicate
or conflicting configuration, permission failure, dependency unavailability,
partial initialization, shutdown during operation, output path failure, and
resource exhaustion when realistic. Every defect fix should add a regression
test at the layer closest to the root cause when practical.

**Test data and cleanup.** Workloads should be minimal but representative,
deterministic where possible, fast enough for their intended cadence, explicit
about kernel or stream/queue counts, and reusable only when sharing does not
obscure the scenario. Generated test data is preferred over large checked-in
binaries; checked-in golden data should have a documented generator and update
procedure. Tests that create processes, threads, files, shared libraries, GPU
resources, signals, or temporary directories should guarantee cleanup on success
and failure.

## 7. Coverage Tooling

**Tooling.** Coverage builds use the `Coverage` build type and
`ROCPROFILER_BUILD_CODECOV=ON`. `source/scripts/run-ci.py` supports
`--coverage` modes and uses `gcovr` when available, with `codecov_exclude` lists
and XML/HTML report generation. Integration tests can opt out of coverage runs
with `DISABLED_CODECOV` when instrumentation changes behavior or makes the test
unsuitable.

**How to run.**

```bash
cmake -B build-rocprofiler-sdk-coverage \
  -DCMAKE_BUILD_TYPE=Coverage \
  -DROCPROFILER_BUILD_TESTS=ON \
  -DROCPROFILER_BUILD_CODECOV=ON \
  .
```

**Expectations.** For coverage-sensitive changes, add focused unit coverage for
deterministic logic when feasible, add integration coverage when behavior only
exists through profiler execution or generated output, keep output validation
semantic, and explain intentionally uncovered paths when they require
unavailable hardware, privileged system state, or external runtime behavior.

## 8. Component Validation Matrix

**Minimum expectations.** Component owners should keep this matrix accurate. New
capabilities should update it before release.

| Component / capability | Minimum unit coverage | Required integration coverage | Performance / reliability coverage |
|---|---|---|---|
| Core SDK API and lifecycle | Argument validation, state transitions, bookkeeping, error propagation | Tool registration, contexts, buffers, callbacks, start/stop/flush/finalize | Callback and record overhead, contention, repeated lifecycle |
| Kernel and runtime tracing | Filtering, correlation, record conversion | HIP/HSA workloads, multi-stream/queue, output validators | Event rate, trace overhead, dropped records, output volume |
| Counter collection | Counter parsing, configuration, state management | Real `AQLProfile`/counter collection on supported GPUs; output-to-dispatch validation | Per-dispatch overhead, throughput, multi-queue stress, detach/stop reliability |
| `rocprofv3` | Option/config parsing and normalization | CLI launch, supported output formats, installed executable, abnormal application exit | Startup/finalization, serialization throughput, output size |
| `rocattach` | Path, ELF/symbol, maps, and input validation | Attach/detach, signals, process tree, active work, installed tool path | Attach latency, repeated attach/detach, active-work stress |
| Python bindings | Argument conversion, exceptions, object lifecycle | Import from installed package and representative API workflows | Import/startup and high-volume callback overhead when relevant |
| `rocpd` / database output | Schema helpers, conversion, error paths | SQLite creation, relational integrity, queries, incomplete/failure handling | Insert/flush throughput, database size, long-run integrity |
| PC sampling, thread trace, SPM | Capability and configuration logic | Supported hardware, explicit unsupported-hardware behavior, semantic output validation | Sampling/trace overhead, buffer pressure, long-duration stability |
| OpenMP, `rocDecode`, `rocJPEG`, optional integrations | Configuration and adapter logic | Dependency enabled/disabled and representative real workload | Added overhead and sustained event collection |
| Packaging and CMake exports | Package/version/config helper logic where practical | Build-tree and installed consumer builds, runtime discovery, test-asset installation | Configure/build time only when materially affected |

## 9. Review Checklist

**Before approval.** Reviewers should check that tests are at the lowest
effective layer; user-visible behavior has integration coverage;
installed-package behavior is tested when relevant; the test would fail without
the change; assertions validate semantics; negative, lifecycle, and concurrency
paths are represented where applicable; cleanup is guaranteed; labels, timeout,
dependencies, fixtures, hardware requirements, skips, and test assets are
correct; coverage and performance evidence are provided when required; bug fixes
include regression tests; known gaps have owners or tracking issues; and the
validation matrix remains accurate.

## 10. Unstable, Disabled, and Known-Gap Tests

**Policy.** `ROCPROFILER_DISABLE_UNSTABLE_CTESTS` defaults to `ON` in the
integration and sample test trees. Use unstable-test disabling sparingly and
only for tests with a known environmental or infrastructure issue.

**Tracking.** An unstable, excluded, or disabled test should have an owner,
failure signature, failing environments, stabilization or replacement plan, and
tracking issue when the gap is not short-lived. New functionality should not
depend solely on a disabled or unstable test.

**Flake handling.** A flaky test should not be "fixed" by broad retries,
arbitrary sleeps,
unjustified timeout increases, weakened assertions, or unnecessarily broad
architecture exclusions. Retries may be used temporarily to collect diagnostics,
but the first failing attempt should remain visible.

## 11. Maintaining This Document

**When to update.** Review this document when a PR changes validation
expectations. Update it when a component or public capability is added;
supported GPU, OS, ROCm, or dependency expectations change; test frameworks, CI
cadence, coverage, or benchmark strategy changes; a test category moves between
PR, nightly, manual, or release qualification; a recurring test gap is
discovered; or an exception becomes permanent policy.

**Related docs.** Component-specific READMEs may provide detailed commands and
fixtures, but they should link back to this strategy and should not contradict
it.
