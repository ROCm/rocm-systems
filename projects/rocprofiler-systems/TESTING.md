# Testing Strategy - rocprofiler-systems

- **Status:** Draft
- **Owner:** TBD
- **Technical Lead:** David Galiffi
- **Last Updated:** 2026-11-16

## Component overview

ROCm Systems Profiler collects system-level profiling information - CPU and
GPU activity, hardware counters, and system, process and thread metrics - from
an unmodified application, as a trace or a summary profile. Linux only.

It sits at the top of the ROCm stack, as a consumer rather than a provider: it
depends on rocprofiler-sdk, amd-smi, HIP/ROCr, hipFile and rocDecode/rocJPEG
from ROCm, on Dyninst for instrumentation, on PAPI for CPU and NIC counters,
and on Perfetto and rocpd for output. Nothing in ROCm depends on it, so a
break here reaches users directly rather than through another component.

The testing strategy follows from two facts:

- Most of what the tool does can only be observed by profiling a real process.
  Much of it also needs a GPU.
- The hardware-independent logic sits behind backend wrappers
  (`source/lib/backends/<library>/`) and policy concepts. Unit tests swap
  those for mocks.

So unit tests cover host-side logic for every feature, GPU domains included -
about 636 of the 2,075 cases drive the rocprofiler-sdk, amd-smi, hipFile and
GPU counter paths against mocks. What they cannot reach is real library,
driver and device behavior, which is what the pytest-driven end-to-end tests
on real processes and real GPUs are for.

## Development workflow

Run these steps from `projects/rocprofiler-systems/`:

```bash
cmake --preset debug                                    # or debug-optimized, coverage, ...
cmake --build build/debug --parallel $(nproc)           # whole project; never --target
./build/debug/bin/rocprof-sys-unit-tests --gtest_filter='<pattern>'
ctest --test-dir build/debug --output-on-failure -R '<name>|<name>'   # affected E2E tests
pre-commit run --files <changed files>
./scripts/clang-tidy-check.py --build-path build/debug --base develop -j $(nproc)
```

`.claude/commands/` provides the same steps as `/build`, `/test`,
`/coverage` and `/clang-tidy`.

## Unit tests

| Item | Detail |
| --- | --- |
| Framework | GoogleTest + GMock. `ROCPROFSYS_BUILD_GTEST` (ON by default) builds v1.17.0 from the `external/googletest` submodule; OFF falls back to the system GTest and links GMock only if `GTest::gmock` exists. |
| Location | `tests/` next to the module (`source/lib/**/tests`, `source/bin/**/tests`); 140 files and about 2,075 cases. The compiled set depends on the configuration: several suites are guarded by `TARGET_EXISTS`, and the SDK PMC collector suite is only built for ROCm ≥ 6.4. |
| Build | Each `tests/` dir builds an `OBJECT` library that has to be listed by name in `UNIT_TEST_OBJECTS` in `source/tests/CMakeLists.txt`; that list is hand-maintained and the target names do not track the paths. Everything links into one binary, `rocprof-sys-unit-tests`. Do not create a standalone test executable, and do not link `rocprofiler-systems-object-library` — `library.cpp`'s static initializers deadlock TSan at exit, so the configure step fails if it reaches the link line. |
| Isolation | External libraries (rocprofiler-sdk, amd-smi, hipFile) are reached only through `backend<Wrapper>`. Tests pass a mock wrapper that does not include the library header: `mock_wrapper.hpp` for amd-smi, hipfile and core, `mock_sdk.hpp` for rocprofiler-sdk. The suite must compile on a machine where those libraries are not installed. |
| Run | `./build/<preset>/bin/rocprof-sys-unit-tests` directly. In CTest it runs as one test named `unit-tests` and labeled `unit_tests`. The CTest name is not the binary name; `-R` filters must use `unit-tests`. |
| GPU needed | No. Tests that depend on the environment use `GTEST_SKIP` (root, PAPI net events, SDK version). |
| Not covered | Real library/driver behavior, Dyninst instrumentation, `LD_PRELOAD` interception, and anything that runs on the device |
| Python | Separate from the GTest binary: `source/python/rocprofsys/tests/test_main.py` has 24 cases for `__main__.py` argument parsing and environment handling, needing neither the native bindings nor a GPU. CTest registers them as `rocprofsys-python-unit-tests`, and omits them without failing if Python3 or pytest is missing at configure time. |
| Coverage | 27.7% of lines repo-wide, measured per PR; see [Code coverage by tool](#code-coverage-by-tool) for the per-area split |

**Unit-testability:** design new code so it can be unit-tested. Put each
external call behind a backend wrapper or a policy concept, and keep the
logic in code that has no GPU or library dependency. Suites and tests are
named `lower_case`, as `<unit>_<scenario>_<exact_result>`. The mock and
expectation rules for new tests are in
`.claude/rules/testing-conventions.md` and
`.claude/skills/testing-gtest-gmock/`. Coverage expectations are in
[Coverage](#coverage).

## Integration / end-to-end tests

The tests are written in pytest and run by CTest. At build time,
`tests/pytest/conftest.py` (the `generate-pytest-ctests` target) generates
`build/<preset>/tests/CTestTestfile.cmake`. Most pytest markers become CTest
labels; `conftest.py` keeps an exclusion set (`timeout`, `class_name`,
`depends_on`, the `*_min_version` gates and others) that is used for CTest
properties instead, and it adds the instrumentation mode as a label, so
`ctest -L sys_run` works. Run the suite with `ctest`, not pytest. Details are
in `tests/README.md` and `tests/pytest/DEVELOPMENT.md`.

- **Flow:** build an app from `examples/`, run it under a runner mode
  (`baseline`, `sampling`, `binary_rewrite`, `runtime_instrument`, `sys_run`,
  `causal`, `python`), then validate the output.
- **Validation:** helper scripts check the output's structure and content:
  - `validate-perfetto-proto.py` (uses `trace_processor_shell` v47.0)
  - `validate-rocpd.py` (JSON rules in `tests/rocpd-validation-rules/`)
  - `validate-timemory-json.py`
  - `validate-causal-json.py`
  - `validate-unified-memory.py`
  - regex checks on stdout and logs

  No golden files are compared.
- **Scope:** `tests/pytest/test_*.py` has 44 files. Three of them
  test the harness and the validators rather than the tool
  (`test_pytest_impl.py`, `test_units.py`, `test_validate_unified_memory.py`);
  they are registered as CTest tests like any other. The
  [Test matrix](#test-matrix) maps each one to the environments where it runs.
- **GPU gating:**
  - `@pytest.mark.gpu` skips when `rocminfo` finds no GPU.
  - `run_if_gpu_category`, `disable_archs` and `multi_gpu(N)` gate tests by GPU type, architecture and count.
  - `check_target_arch` skips tests whose example binary was not built for the current GPU.
- **Optional dependencies:** a test is skipped and returns code 77 (a CTest
  "skip") when its dependency or system capability is missing:
  - MPI, PAPI, UCX, OpenSHMEM or Julia is not installed
  - `perf_event_paranoid` or `ptrace_scope` is too restrictive
  - a version gate fails (`*_min_version`)

  **A green `ctest` does not mean every area ran.** Check the skipped tests.
- **Timeouts:** 300 s per test by default (+30 s buffer). Override per test
  with `@pytest.mark.timeout(N)` or for the whole run with
  `ROCPROFSYS_CI_TIMEOUT`.
- **Filtering:** at run time, use `ctest -L/-LE/-R/-E` with labels such as
  `gpu`, `mpi`, `slow`, `network` or `thread_limit`. At configure time, use
  `ROCPROFSYS_TEST_LABELS_{INCLUDE,EXCLUDE}`,
  `ROCPROFSYS_TEST_KEYWORDS_{INCLUDE,EXCLUDE}` or
  `ROCPROFSYS_DISABLE_EXAMPLES`. Tier labels (`quick`, `standard`,
  `comprehensive`, `full`) come from `tests/test_categories.yaml`. TheRock uses
  them, and that file must stay in sync with TheRock's runner.
- **Installed-package tests:** `scripts/test-install.sh` and
  `scripts/test-find-package.sh`. With `ROCPROFSYS_INSTALL_TESTING=ON`, the
  suite installs to `share/rocprofiler-systems/tests`.

## What a user can rely on

The [Test matrix](#test-matrix) below is organized by component. This is the
same information from the user's side - which workflows are validated before
a change ships.

Validated on every PR:

- Trace or sample an unmodified CPU application through sampling, `sys_run`,
  binary rewrite or runtime instrumentation, and read the Perfetto, rocpd or
  timemory output.
- Trace a HIP application on gfx950: HIP and HSA APIs, kernel dispatches,
  memory copies, ROCTx regions, GPU counters, RCCL, rocJPEG, hipFile,
  OpenMP target offload and SDMA.
- Attach to an already-running process. The `rocattach-attach-loop-1/2/3` and
  `rocattachreattachformat-*` tests run and pass on the gfx950 lane, whose
  container runs `--privileged`, so ptrace attach is permitted there.
- Profile a Python application, and run causal profiling - CPU only.
- Install the package and consume it through `find_package` and the
  modulefile.

Validated nightly only:

- The same GPU workflows on gfx94X, gfx103X, gfx110X and gfx120X, plus the
  cases labeled `slow`.

Not validated anywhere today:

- Multi-GPU and multi-node runs: rocSHMEM, TransferBench and the two-GPU
  Jacobi cases skip for want of a second GPU - the PR runner is
  `linux-gfx950-1gpu-ccs-ossci-rocm` - and the RCCL `alltoall` `sys_run` case
  is excluded by name. The other RCCL collectives do run on one GPU.
- KFD events, unified-memory output, scratch memory, and rocDecode video
  decode on a GPU runner. `scratch-memory-*` and `video-decode-*` skip with
  exit 77 on the gfx950 lane, and `kfd-.*` is in `EXCLUDED_TESTS`.
- NIC and AI NIC counters.
- MPI, PAPI, UCX, OpenSHMEM and Julia workflows on any image that lacks the
  dependency - they skip rather than fail (exit 77).
- Profiling overhead: no benchmark or regression gate at any cadence.

## Key quality concerns

The five things that must not break, and where the confidence comes from.

| Concern | Why it matters | How it is validated |
| --- | --- | --- |
| The tool must not break the application it profiles | A profiler that crashes, hangs or deadlocks the target is worse than no profiler. Instrumentation, `LD_PRELOAD` interception and signal-based sampling all run inside an application process. | E2E tests run every runner mode against real apps and fail on a non-zero exit or an error pattern in the log; ASan and UBSan cover 22 of those tests; `test_fork`, `test_pthread` and `test_thread_limit` cover process and thread edge cases |
| Collected data must be correct, not merely present | A trace that silently drops dispatches or misattributes time sends users chasing the wrong bottleneck. | Validators assert record presence, counts, ordering and ranges, with 179 exact `equals` assertions; unit tests cover the host-side processors against mocks |
| Output must stay readable by its consumers | Perfetto UI, rocpd tooling and timemory consumers all parse our output; a schema change breaks them silently. | `validate-perfetto-proto.py` parses with a pinned `trace_processor_shell` v47.0; `validate-rocpd.py` checks required tables and columns |
| GPU coverage must hold across architectures | A change that works on gfx950 can fail on Navi or on an Instinct part with different queue or counter behavior. | Nightly dispatches five architecture families but is currently failing; PRs run gfx950 only, which is a known gap |
| The shipped package must be usable | Users consume an installed package, not a build tree. | `scripts/test-install.sh` and `scripts/test-find-package.sh` on CPU PR lanes, plus the modulefile test. No API or ABI compatibility check exists - also a known gap |

## Test matrix

This table shows which functionality is tested, and where. It was built from
the pytest markers and the CI filters:

- CPU jobs run `-LE "network|gpu"`.
- GPU jobs run `-L gpu`, plus the `-E` exclude list.
- Sanitizer jobs run the `-R` lists.

When a CI filter or a test marker changes, update this table.

Legend:

- **✓** runs.
- **~** partial: only some cases run.
- **skip** is selected, but skips at runtime in CI.
- **excl** is removed by a CI exclude.
- **—** not tested there.
- **†** needs a dependency the CI image may not have (MPI, PAPI, perf events,
  OpenSHMEM, UCX, Julia). If it is missing, the test skips (exit 77).
  Check the skipped count in the JUnit summary.

| Functionality (tool) | pytest file(s) | Unit | CPU (PR, 6 distros) | GPU gfx950 (PR, 24.04) | GPU nightly (5 archs) | Sanitizers |
| --- | --- | --- | --- | --- | --- | --- |
| **Tools and CLI** | | | | | | |
| `rocprof-sys-avail` | `test_binaries` (avail) | — | ✓ | ~ (`settings-rocm-available`) | ~ | ASan, UBSan, TSan (`avail-all*`) |
| `rocprof-sys-instrument`: binary rewrite | `test_binaries`, `test_rewrite_caller`, `test_code_coverage`, `test_minimal`, `test_time_window`, `test_transpose` | ~ (dependency listing only) | ✓ | ✓ (`test_transpose`) | ✓ | ASan, UBSan |
| `rocprof-sys-instrument`: runtime instrumentation | `test_minimal`, `test_code_coverage`, `test_lulesh`, `test_thread_limit`, `test_transpose` | — | ✓ | ✓ (`test_transpose`) | ✓ | ASan, UBSan (`code-coverage`) |
| `rocprof-sys-sample` | `test_pthread`, `test_openmp`, `test_lulesh`, `test_mpi`, `test_transpose` | — | ✓ | ✓ (`test_transpose`) | ✓ | — |
| `rocprof-sys-run` (`LD_PRELOAD`) | most files (`sys_run` mode) | — | ✓ | ✓ | ✓ | ASan, UBSan (`rewrite-caller`) |
| `rocprof-sys-causal` | `test_causal` (incl. `causal-e2e`, `slow`) | — | ✓ | — | — | — |
| `rocprof-sys-python` | `test_python` + `rocprofsys-python-unit-tests` | pytest only | ✓ | — | — | — |
| `rocprof-sys-attach` / `rocattach` | `test_rocattach` | — | — | ✓ (needs ptrace) | ✓ | — |
| **CPU runtime features** | | | | | | |
| Fork, pthreads, thread limit | `test_fork`, `test_pthread`, `test_thread_limit` | ~ (gotchas) | ✓ | **excl** (`fork.*` removes the `gpu` fork cases; `thread_limit` label excluded) / — (pthreads not `gpu`) | **excl** / — | — |
| PAPI counters, overflow, annotate | `test_annotate`, `test_overflow` | — | ✓† | — | — | — |
| MPI, MPIP, rank filter | `test_mpi`, `test_rank_filter` | — | ✓† | — | — | — |
| UCX, OpenSHMEM | `test_ucx`, `test_shmem` | ~ (gotchas) | ✓† | — | — | — |
| OpenMP (OMPT) host | `test_openmp` (CG, LU, Fortran) | ~ (OMPT decoder) | ✓ (ROCm ≥ 6.4) | — | — | — |
| Julia, LULESH | `test_julia`, `test_lulesh` | — | ✓† | skip† | skip† | — |
| **GPU tracing and counters** | | | | | | |
| HIP/HSA API, kernel dispatch, memory copy | `test_transpose` | ✓ (callback, buffered) | — | ✓ | ✓ | — |
| GPU perf counters | `test_transpose` (`gpu-perf-counters`) | ✓ | — | ✓ (needs `CAP_PERFMON`) | ✓ (not gfx1101/1102/115x) | — |
| ROCTx, selective regions, pause/resume | `test_roctx`, `test_selective_regions` | ✓ | — | ✓ | ✓ | — |
| RCCL | `test_rccl` | ✓ | — | ✓ (`alltoall-perf` excl) | ✓ (`alltoall-perf` excl) | — |
| rocDecode, rocJPEG | `test_videodecode`, `test_jpegdecode` | ✓ | — | ~ (rocJPEG only; `video-decode-*` skips) | ~ | — |
| hipFile | `test_hipfile` | ✓ (mock backend) | — | ✓ | ✓ | — |
| OpenMP target offload | `test_openmp` (`openmp-target`) | — | — | ✓ | ✓ | — |
| HPC apps (matrix-exponential, split-copy) | `test_hpc` | — | — | ✓ | ✓ | — |
| SDMA | `test_sdma` | — | — | ✓ (Instinct) | ~ (Instinct only) | — |
| KFD events | `test_kfd` | ✓ | — | **excl** (`kfd-.*`) | **excl** | — |
| Unified memory | `test_unified_memory` | ✓ | validator self-test only | **excl** | **excl** | — |
| Jacobi (USM, ROCTx, HIP+MPI) | `test_hpc` | — | — | **excl** / skip (needs 2 GPUs) | **excl** / skip | — |
| Scratch memory | `test_scratch_memory` | ✓ | — | skip (`no_docker`; CI runs in containers) | skip | — |
| **Multi-GPU and system** | | | | | | |
| rocSHMEM, TransferBench (XGMI) | `test_rocshmem_tracing`, `test_transferbench` | — | — | skip (needs ≥ 2 GPUs) | skip | — |
| AMD-SMI GPU metrics | no dedicated test file | ✓ (mock backend + provider) | — | indirect | indirect | — |
| NIC, AI NIC | `test_nic_perf`, `test_ainic` | ✓ (NIC collector) | ~ (AI NIC settings check selected, subject to AMD-SMI ≥ 26.3; counter tests excluded by `network`) | — (not `gpu`) | — | — |
| **Output and packaging** | | | | | | |
| Perfetto, rocpd, timemory output | validators in most E2E tests | ✓ (perfetto, trace_cache) | ✓ | ✓ | ✓ | — |
| Install, modulefile, `find_package` | `scripts/test-install.sh`, `test-find-package.sh` | — | ✓ | — | — | — |

The GTest binary does not appear in the Sanitizers column, because it never
runs there. The sanitizer jobs select
`^(rewrite-caller|code-coverage|trace-time-window|rocprofiler-systems-avail-all|rocprof-sys-unit-tests)`,
but the binary is registered in CTest as `unit-tests`
(`tests/pytest/test_units.py`), so the anchored alternative never matches.


## CI gates

The workflows are in the monorepo `.github/workflows/rocprofiler-systems-*.yml`.
Each run reports to CDash through `scripts/run-ci.py`, and the results are
posted in a sticky PR comment.

Three more workflows exist but are infrastructure rather than test lanes:
`containers.yml` and `ghcr.yml` build and publish the CI images (the only
workflows that use repository secrets), and `skill-check.yml` runs skill
evaluations on `skills/**`.

| Workflow | When | What | Blocks merge? |
| --- | --- | --- | --- |
| `build.yml` → `build-group.yml` | PR, push to develop | Build and run `ctest -LE "network\|gpu"` on GitHub-hosted CPU runners across Ubuntu 22.04 and 24.04, Debian 12, and RHEL 8.10, 9 and 10 with ROCm 6.3–7.2. `.github/ci-build-matrix.json` holds the exact pairs: 26 primary configurations (not a full cross product — Debian 12 starts at ROCm 6.4, RHEL 10 at 7.1) plus 10 system-deps configurations (g++ and amdclang++ on ROCm 7.2). Also builds RHEL with gcc-toolset 13, 14 and 15 (build only), then runs install and `find_package` tests. | Not recorded* |
| `continuous-integration.yml` | PR, push, nightly 06:00 UTC | `ctest -L gpu` on GPU runners. On PRs and pushes: MI355 (gfx950) only, with the `slow` label excluded. Nightly adds MI325, Navi2x, Navi3x and Navi4x. | Not recorded* |
| `ubuntu-noble-sanitizers.yml` | PR, push | ASan, UBSan and TSan on a subset of tests (see below) | Not recorded* |
| `code-coverage.yml` | PR, push | gcovr: compares unit-test coverage against the base branch, plus diff-only and E2E reports, posted as a PR comment | No (informational) |
| `clang-tidy.yml` | PR | `clang-tidy-check.py` on changed lines (clang-tidy 22.1.8) | Not recorded* |
| `formatting.yml` | PR | clang-format-18, gersemi, black, markdownlint, spellcheck, and include and fixed-width-type checks | Not recorded* |
| `python.yml` | PR, push | `ruff check` | Not recorded* |


**Retries:** `run-ci.py` reruns a failed test up to 3 times
(`--repeat until-pass:3 after-timeout:2`). Sanitizer jobs pass
`--repeat-until-pass 1 --repeat-after-timeout 1`, so each test gets a single
attempt there.

**Flaky tests:** there is no `UNSTABLE` label. Known-bad tests are excluded
with `EXCLUDED_TESTS` / `EXCLUDED_LABELS` in the GPU workflow and with
excludes in `test_categories.yaml`. None of these exclusions has an owner or
a ticket today.

Policy:

- Tag every flaky or excluded test.
- Give it an owner and a tracking ticket.
- Keep it out of blocking runs until it is fixed.

Excluding a test is temporary, not a final state.

## Nightly validation

The nightly GPU run differs PR runs in exactly three ways:

- **More hardware:** gfx950 on Ubuntu 22.04 and 24.04, gfx94X on both, and
  gfx103X, gfx110X and gfx120X on 24.04 - seven configurations across five
  architecture families, against gfx950 on 24.04 for a PR.
- **More tests:** the `slow` label is no longer excluded, which brings in the
  long `causal-e2e` cases and other extended runs.
- **Same exclusions:** `EXCLUDED_TESTS` and the `thread_limit` label are
  excluded nightly too, so the KFD, unified-memory, GPU fork and Jacobi cases
  do not run at night either.

**Current state: the lane is not green.** The ten most recent scheduled runs
(8-17 November 2026) all concluded `failure`; in the 17 November run the
`Test` step failed in six of the seven jobs, and no nightly build was
reported to CDash on 16 or 17 November. Until that is fixed, nightly
dispatches the extra hardware without producing evidence for it.

Nothing else is nightly-only: there is no larger-workload lane, no
multi-GPU or multi-node lane, no soak or stress run, and no downstream or
cross-component validation of consumers.

**Shared infrastructure:** TheRock consumes `tests/test_categories.yaml` for
the `quick`, `standard`, `comprehensive` and `full` tiers, and runner labels
resolve from `ROCm/therock-ci-config` at workflow runtime. Changing either
file without TheRock's runner in mind breaks their lane, not ours, and
nothing in CI checks that the two stay in sync.

## Release qualification

Not defined. There is no release branch validation procedure, no QA handoff or
sign-off expectation, no documented known-gap review before a release, and no
performance regression review, recorded for this component.

What exists today and would form the basis of one: the nightly GPU matrix, the
CPU build-and-test matrix across 26 configurations, and the install and
`find_package` tests. Defining the sign-off criteria - who approves, against
which run, with which gaps accepted - is an open item owned by the technical
lead.

## Sanitizers

Enable with `-DROCPROFSYS_USE_SANITIZER=ON -DROCPROFSYS_SANITIZER_TYPE=<address|thread|undefined|leak|...>`.
Suppression files are
`scripts/rocprof-sys-{asan,leak,lsan,thread,ubsan}.supp`, plus the
`scripts/rocprof-sys-ubsan-ignorelist.txt` ignorelist. Most of the
suppressions are for Dyninst. `ASAN_LIBRARY` is prepended to `LD_PRELOAD` for
`sys_run` tests.

| Sanitizer | CI scope (PR, CPU only) | Notes |
| --- | --- | --- |
| ASan (+LSan) | `rewrite-caller`, `code-coverage`, `trace-time-window`, `rocprofiler-systems-avail-all*` - 22 tests | Fails on any `ERROR`/`SUMMARY` line in the log |
| UBSan | Same as ASan | `-fno-sanitize=vptr` (Dyninst RTTI) |
| TSan | `rocprofiler-systems-avail-all*` only | Kept small because of the 5–15x slowdown |

The workflow regex intends to include the GTest binary but does not select it
(see the [Test matrix](#test-matrix) note above), so no unit test runs under
any sanitizer today.

Not covered: sanitizers on GPU runners, MSan, and most E2E instrumentation
modes. There is no sanitizer CMake preset.

## Coverage

Coverage is measured with gcovr ≥ 8.4 on Linux host code only, using
`ROCPROFSYS_BUILD_CODECOV=ON`, which the `coverage` preset sets.
`scripts/generate-coverage.py` writes JSON, Cobertura, HTML and Markdown
reports to `.codecov/` and excludes `external/`, `examples/`, `tests/`,
`/usr` and `/opt`. To run it locally, use `/coverage` or follow
`.claude/commands/coverage.md`.

- **Current baseline:** 27.7% of lines from unit tests and 73.7% with E2E, on
  33,238 lines across 332 files. See
  [Code coverage by tool](#code-coverage-by-tool) for the per-area split and
  provenance.
- **Target:** the ROCm-wide goal is >95% of hardware-independent code. There
  is no enforced threshold yet.
- **How the report is built:** `code-coverage.yml` builds the base SHA first
  and runs the unit tests to produce `unit-tests-baseline.json`, then rebuilds
  the PR head, deletes the baseline `.gcda`, and reruns. It publishes three
  reports: unit tests with a delta against that baseline, unit tests
  restricted to the files the PR touched (`--diff-only`), and a CPU `ctest`
  run with no baseline, so the E2E number is the only one without a delta.
  The diff-only delta is not meaningful: files new in the PR read as +94% in
  the example above only because the base had no coverage for them.
- **What the E2E number excludes:** the coverage build configures with
  `ROCPROFSYS_DISABLE_EXAMPLES` set to twelve example groups - `transpose`,
  `rccl`, `videodecode`, `jpegdecode`, `openmp`, `roctx`, `scratch-memory`,
  `unified-memory`, `transferBench`, `hpc`, `sdma_test` and `annotate` - so
  it is narrower than the `ctest` run on a CPU build lane, and narrower than
  "all CPU tests" suggests.
- **Code coverage vs. test coverage:** these differ.
  - Code coverage counts the host lines executed. It cannot see device-side
    or interception paths that run inside an instrumented process.
  - Test coverage counts the scenarios exercised: architectures, runner
    modes, MPI, and output formats. On PRs, GPU runs use only gfx950.

**Gating status:**

| Gating level | Status | Notes |
| --- | --- | --- |
| PR-level automated gate | No | Known gap |
| Nightly automated comparison | No | No baseline to compare against |
| Manual nightly review | No | Nobody reviews overhead on a schedule |
| Release qualification | No | See [Release qualification](#release-qualification) |
| Per-architecture baselines | None stored | Absolute numbers are not comparable across GFX, so a baseline per architecture is a prerequisite |

## Supported configurations

| Configuration | Validation level | Frequency | Notes |
| --- | --- | --- | --- |
| Ubuntu 22.04 / 24.04, Debian 12, RHEL 8.10 / 9 / 10 × ROCm 6.3–7.2 | build + CPU tests, no GPU | PR | 26 configurations; `network`-labeled tests excluded |
| gfx950 (MI355), Ubuntu 24.04 | GPU tests minus `slow` and the exclude list | PR | The only GPU architecture validated per change |
| gfx950 + gfx94X (22.04 / 24.04), gfx103X, gfx110X, gfx120X (24.04) | GPU tests including `slow` | Nightly | Seven configurations, five architecture families |
| g++ and amdclang++ "system deps" builds (ROCm 7.2) |  build + CPU tests | PR | 10 configurations on Ubuntu 22.04/24.04, Debian 12, RHEL 9/10 |
| Any configuration | Full | Never | No lane runs the whole suite with every optional dependency present |

The following configurations are **not tested**:

- Windows and macOS (not supported)
- openSUSE (release images only)
- multi-GPU and multi-node runs (tests marked `multi_gpu` skip on single-GPU runners)
- tests excluded by `thread_limit` and by `EXCLUDED_TESTS`
- fork PRs on GPU runners (behavior not verified)

## Known gaps

| Gap | Risk | Mitigation today | Owner | Target |
| --- | --- | --- | --- | --- |
| No performance benchmark or regression gate | Medium | Manual before/after comparison | TBD | TBD |
| PR GPU coverage is gfx950 only; other architectures run nightly | High | Nightly matrix, which has concluded `failure` on every scheduled run from 8-17 November 2026 and reports nothing to CDash | TBD | TBD |
| Multi-GPU and multi-node paths are not exercised in CI | High | Manual validation | TBD | TBD |
| Unit coverage is 27.7% of lines against a >95% ROCm goal; CLI tools are at 0-1.8% and `rocprof-sys-rt` at 18.5% even with E2E | High | E2E raises the total to 73.7%; new code lands near 94% diff coverage | TBD | TBD |
| No coverage threshold is enforced, and `therock-pr-bot` lists Code Coverage as "To Be Enabled" | Medium | Diff-coverage PR comment, reviewed by hand | TBD | TBD |
| Python bindings (`source/python/libpyrocprofsys.cpp`, 453 lines) have 0% coverage in both columns | Medium | `test_python` covers the front end; `rocprofsys-python-unit-tests` covers `__main__.py` | TBD | TBD |
| Release qualification is undefined: no sign-off criteria, QA handoff or pre-release gap review | High | Nightly GPU matrix and the CPU build matrix are the de facto evidence | TBD | TBD |
| KFD, unified-memory and GPU fork E2E tests are excluded from GPU CI; scratch-memory skips in containers | High | Unit tests for the processors | TBD | TBD |
| NIC and AI NIC E2E tests never run in CI (`network` label is excluded from CPU jobs; tests are not `gpu`-labeled for GPU jobs) | Medium | NIC collector unit tests | TBD | TBD |

The pilot asks for an owner and a planned resolution date for every gap, with
a tracking issue behind each one. A gap with no owner is neither closed nor
accepted, so the table above is not yet usable for tracking. The table should
get shorter over time.

## Review cadence

Update this file when any of these change: a CI workflow or matrix, the test
framework or conventions, the supported OS, ROCm or GPU list, or a
known gap. Review it again after any regression that escaped CI, and
before each ROCm release.

Independently of those triggers, the technical lead reviews
[Known gaps](#known-gaps) quarterly. The strategy is working if that table
shrinks.
