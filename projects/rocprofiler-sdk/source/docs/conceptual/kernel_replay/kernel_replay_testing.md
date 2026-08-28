# Kernel replay test coverage

What kernel replay is tested for today, what it is not, and where a new test belongs. This is a
companion to [the performance assessment](kernel_replay_performance.md), which covers why the
performance tests are shaped the way they are.

## Where the tests live

Kernel replay is tested at four levels, and they differ mainly in what they need to run.

| Level | Location | Needs |
| --- | --- | --- |
| Public ABI and header contract | `source/lib/rocprofiler-sdk/kernel_replay/tests/replay_abi.cpp`, `replay_abi_c.c` | nothing |
| Pure logic (context overrides, configuration) | `.../tests/local_context.cpp`, `replay_configure.cpp` | nothing |
| Snapshot and tracker behaviour | `.../tests/snap_restore.cpp`, `snap_bandwidth.cpp` | a GPU |
| End-to-end through `rocprofv3` | `tests/rocprofv3/kernel-replay/` | a GPU, except the CLI tests |
| Performance regressions | `tests/kernel-replay-perf/`, `tests/queue-hooks-perf/` | a GPU |
| Performance harness helpers | `tests/perf-common/test_perf_stats.py` | nothing |

The distinction matters more than it looks. Most contributors and most pre-merge checks do not have
a GPU attached, so a bug that is only reachable by a GPU-gated test is a bug that will be found late.
Several of the tests listed above as needing nothing were written specifically to pull checks out of
that category.

## What the GPU-free tests cover

### The public record's layout

`rocprofiler_callback_tracing_kernel_replay_data_t` crosses the library boundary: the SDK fills it
in and a separately compiled tool reads it. Moving or resizing a field does not break any build —
it makes a tool compiled against the older header misread every field after the change. The `size`
member exists so a tool can notice, but nothing enforced that the rest of the layout stayed put.

`replay_abi.cpp` pins the field order, the offsets, the size and alignment, the callback signatures
and the operation enum values, plus the documented semantics that do not need a device: that a
zero-initialized record means "do not replay this dispatch", that `total_passes == 0` means an
indefinite loop and is distinguishable from a single pass, and that `current_pass` is 0-indexed
against `total_passes`.

### The header as C

The public headers are meant to be consumed by C tools. Nothing else in the tree compiles them as C
— the directory named `c-tool` is a `LANGUAGES CXX` project — so a C++-only construct reaching a
public header would not be noticed until a downstream tool failed to build.

`replay_abi_c.c` is compiled as C for exactly that reason, and its existence as a build input is
half of what it is for. The other half is that it reports what C computes for the record's layout,
which the C++ side compares field by field. That catches the case a compile check alone cannot: a
header change both languages accept but interpret differently, where a C tool reads the wrong bytes
with nothing failing to build.

The pattern is not new here: `source/lib/aqlprofile/core/tests/aql_profile_v2_c_test.c` mixes a C
translation unit into a gtest executable the same way, with `C_STANDARD` pinned on the target so
the check means the same thing regardless of the toolchain's default dialect. That is the only
other place in the tree where a header is compiled as C, and it covers an internal aqlprofile
header rather than a public SDK one.

### The performance harness itself

`perf_stats.py` and the two cost models decide whether a performance test passes. A bug there either
hides a regression or fails a healthy build, and because the performance tests need a GPU, nothing
would catch such a bug on a machine without one. `tests/perf-common/test_perf_stats.py` covers the
sampling, the median, the ceiling modes and the results writing directly.

Writing those tests found four bugs in the helpers, which is the argument for having them.

### The `rocprofv3` command line

`tests/rocprofv3/kernel-replay/test_kernel_replay_cli.py` imports `rocprofv3.py` as a module and
exercises its argument handling without a GPU or a built SDK. It covers the two decisions unique to
replay: which services cannot be collected in the same run, and how an input file's counter groups
collapse into the single application run that replay produces.

## What is not covered

These are known gaps, listed so they are not rediscovered:

- **No GPU-free exercise of the replay window.** The drain, snapshot, pass loop and restore sequence
  in `hsa/queue.cpp` is only reachable with a device, and the loop's control flow — decline paths,
  early exit, the restore-failure abort — is untested except on hardware.

  There is no mock HSA stack anywhere in the SDK to build such a test on. What exists is a set of
  partial fakes, and none of them reach far enough: the `FakeQueue` used by the counters, SPM and
  thread-trace local-context tests supplies only an agent and a queue id and still calls
  `hsa_init()`, so it needs ROCr even though it never dispatches; `counters/tests/hsa_tables.cpp`
  builds an API table out of the *real* `hsa_*` function pointers and deliberately leaves the
  intercept-registration entries unwired. The closest thing to a synthetic packet flow is
  `source/lib/rocprofiler-sdk/tests/queue_interposition.cpp`, which drives ring buffers and
  doorbells with no device — but it exercises `process_doorbell_impl`, not `Queue::WriteInterceptor`,
  which is where replay lives.

  A harness for the replay window would need a `Queue` test double that can be handed synthetic
  packet batches, host-backed stand-ins for the memory tracker and snapshot, and an HSA table stub
  whose intercept registration actually works. That is new infrastructure rather than an extension
  of what is there.
- **No build test for the public headers as a whole.** CI's "Test Install Build" and "Test Installed
  Packages" steps rebuild `samples/` and `tests/` against the install tree, which does compile the
  public headers, but only as far as those trees happen to include them. Nothing asserts that every
  public header is self-contained, and outside the kernel replay header added here, none of them are
  compiled as C. There is no `try_compile` or `check_cxx_source_compiles` anywhere in the SDK's CMake
  — the only compile probes are `check_cxx_compiler_flag` calls for warning flags.
- **No compile-time budget.** Build time is not measured, so a template or header change that makes
  the SDK slower to build is invisible. There is no `CMAKE_RULE_LAUNCH_COMPILE` wrapper or
  build-duration reporting in CI.
- **No ABI checker.** Layout is asserted by hand where someone thought to do it — `offsetof` tests
  for `rocprofiler_agent_t`, `static_assert`s on the HSA packet types, the `ROCP_SDK_ENFORCE_ABI`
  macros for dispatch tables, and now the replay record — but nothing compares built artifacts
  across versions the way `abidiff` would.
- **No multi-GPU coverage at any level.**
- **No test that a HIP graph launch declines replay visibly.** The behaviour is documented and
  warns once, but nothing asserts it, so a workload that captures graphs — which is the default for
  much of PyTorch and vLLM — would silently get no replay.

Static analysis is in better shape than the build checks: clang-tidy runs on a GPU CI matrix
entry (`--linter clang-tidy`, opt-in locally via `ROCPROFILER_ENABLE_CLANG_TIDY`), and CodeQL has its
own workflow. The `kernel_replay/tests` directory calls `rocprofiler_deactivate_clang_tidy()`, as
every test directory does, so the tests themselves are not linted — only the code they exercise.

## Choosing a benchmark framework

rocprofiler-sdk does not use Google Benchmark anywhere. The `benchmark/` directory is a separate
thing entirely: a YAML-driven suite that runs whole applications under `timem` and records wall time
and peak RSS into SQLite. It measures end-to-end profiling overhead, not function-level cost.

Google Benchmark is used elsewhere in the monorepo, by `profilers/profiler-hub`, which is the
template to follow if it is ever wanted here. That project pins version 1.8.3 in
`cmake/benchmark.cmake`, prefers a system package via `find_package(benchmark QUIET)` and falls back
to `FetchContent` from upstream, and gates the whole thing behind `PROFILER_HUB_BUILD_BENCHMARKS`.

For kernel replay specifically, Google Benchmark would suit the parts whose cost is a function call
rather than a run: snapshot inventory construction, the tracker's allocate and free hooks, the
context override map. It would not suit the thing that actually dominates replay cost, which is
bytes moved per dispatch across the host link — that needs a real workload and a real device, which
is what `tests/kernel-replay-perf/` and the `benchmark/` suite already do.

## Adding a test

The question worth asking first is whether the test needs a GPU. If the property being checked is
about layout, argument handling, or a decision the SDK makes before it touches a device, it almost
certainly does not, and putting it in a GPU-gated file is the difference between a check that runs
on every machine and one that runs on a few.

Long-running performance sweeps are registered only when `ROCPROFILER_BUILD_NIGHTLY_PERF_CTESTS` is
on, and carry the `perf-nightly` label. They take a large share of the CI test budget and what they
produce is a trend across runs rather than a per-commit signal, so a single noisy run failing a pull
request costs more than it catches.
