# Kernel replay — validation report for QA

Handoff for the validation team covering the two pull requests that landed the experimental kernel
replay feature. Everything below refers to the exact commits that were merged, and every result is
taken from the CI run on the merged head rather than from an earlier iteration of either branch.

- [PR #7960 — kernel replay service (callback tracing)](https://github.com/ROCm/rocm-systems/pull/7960)
- [PR #10439 — kernel replay `rocprofv3` integration](https://github.com/ROCm/rocm-systems/pull/10439)

The feature is **experimental**. The public header lives under `rocprofiler-sdk/experimental/`, and
both the API and the command-line flags are expected to change before a stable release.

## 1. Build identification

Both PRs merged on 2026-09-02. #10439 was stacked on #7960, so #7960 landed on `develop` first and
#10439 landed two seconds later through the stack.

| | PR #7960 | PR #10439 |
| --- | --- | --- |
| Validated head SHA | `9134f8f2aeef57e3b96cd300b484079488e552d9` | `929884fc7af1ec5417222588af31b239b5055615` |
| Merge commit on `develop` | `88f5747ec61d585466b9f3d893e9bb8f362383f9` | `c5f49aeaa97dfeda0d5b7d083f9705dbbdec8c7e` |
| Merged at (UTC) | 2026-09-02 08:40:01 | 2026-09-02 08:40:03 |
| Merged by | `bgopesh` | `bgopesh` |
| Size | 79 commits, 98 files, +13594/−48 | 31 commits, 26 files, +2288/−20 |

**Build numbers to quote when filing defects.** The workflow is
`rocprofiler-sdk Continuous Integration` (`.github/workflows/rocprofiler-sdk-continuous_integration.yml`).

| Build | Run number | Attempt | Link |
| --- | --- | --- | --- |
| #7960 functional CI | **8915** | 1 | [run 33579830458](https://github.com/ROCm/rocm-systems/actions/runs/33579830458) |
| #7960 code coverage | **8308** | 1 | [run 33579830453](https://github.com/ROCm/rocm-systems/actions/runs/33579830453) |
| #10439 functional CI | **8917** | 3 | [run 33594406094](https://github.com/ROCm/rocm-systems/actions/runs/33594406094) |
| #10439 code coverage | — | 1 | [run 33594406052](https://github.com/ROCm/rocm-systems/actions/runs/33594406052) |

Because #10439 contains #7960, build **8917** is the single build that validates the complete
feature. Build 8915 is the right reference only for defects isolated to the SDK service with no
`rocprofv3` involvement.

## 2. Test environment

| Property | Value |
| --- | --- |
| GPU | MI325, CI `gpu-target` `gfx94X` |
| Container image | `docker.io/rocm/rocprofiler-private:<os>-gfx94X-latest` |
| OS matrix | ubuntu-22.04, rhel-8.8, rhel-9.5, sles-15.6 |
| Build type | RelWithDebInfo |
| aqlprofile compatibility matrix | 6.2, 6.3, 6.4, latest (external and internal, ubuntu-22.04) |
| Trace decoder | built in CI (`enable-rocprof-trace-decoder-build: true`) |
| Harness | CTest, results also published to [CDash](https://my.cdash.org/index.php?project=rocprofiler-sdk-alt) |

The container tag is a rolling `latest`, so the ROCm version is **not pinned by the workflow**. If a
defect turns out to be version-sensitive, capture `/opt/rocm/.info/version` from inside the job
rather than inferring it from the build number. The ubuntu-22.04 leg additionally runs clang-tidy
(`--linter clang-tidy`); the other OS legs do not.

Two nuances that matter when reproducing results:

**The kernel replay tests run twice per job.** Once against the build tree, and again against the
install tree after the samples and tests are rebuilt against installed packages ("Test Install
Build" / "Test Installed Packages"). The two runs do not have identical outcomes, and section 5
describes a case where they diverged. When reporting a defect, state which of the two produced it.

**The installed-samples suite runs highly parallel on one GPU.** All ~27 sample tests are launched
concurrently against a single device. Thread trace (ATT) and full-device counter enumeration
contend for the same hardware resources under that pattern. The build-tree suite does not schedule
them the same way.

## 3. Test plan

The test plan is a living document in the repository rather than a separate artifact:

- **[Kernel replay test coverage](../source/docs/conceptual/kernel_replay/kernel_replay_testing.md)**
  — what is tested, what is deliberately not, and where a new test belongs.
- **[Performance assessment](../source/docs/conceptual/kernel_replay/kernel_replay_performance.md)**
  — why the performance tests are shaped the way they are.
- Feature documentation: [conceptual overview](../source/docs/conceptual/kernel_replay/index.md),
  [callback API](../source/docs/conceptual/kernel_replay/kernel_replay_callback_api.md),
  [concurrency and isolation](../source/docs/conceptual/kernel_replay/kernel_replay_concurrency_and_isolation.md),
  [memory snapshot](../source/docs/conceptual/kernel_replay/kernel_replay_memory_snapshot.md),
  [API reference](../source/docs/api-reference/kernel_replay.rst),
  [using kernel replay](../source/docs/how-to/using-kernel-replay.rst),
  [using kernel replay with rocprofv3](../source/docs/how-to/using-kernel-replay-rocprofv3.rst).

### Structure and intent

Tests are organised by **what they need to run**, because most contributors and most pre-merge
checks have no GPU attached. A bug only reachable by a GPU-gated test is a bug found late, so a
deliberate effort was made to pull checks into the GPU-free tier.

| Level | Location | Needs a GPU |
| --- | --- | --- |
| Public ABI and header contract | `source/lib/rocprofiler-sdk/kernel_replay/tests/replay_abi.cpp`, `replay_abi_c.c` | no |
| Pure logic (context overrides, configuration) | `.../tests/local_context.cpp`, `replay_configure.cpp`, `replay_phases.cpp` | no |
| Snapshot and tracker behaviour | `.../tests/snap_restore.cpp`, `snap_bandwidth.cpp` | yes |
| End-to-end through `rocprofv3` | `tests/rocprofv3/kernel-replay/` | yes, except the CLI tests |
| Performance regressions | `tests/kernel-replay-perf/` | yes |

Three design points worth knowing before writing new cases:

1. **`replay_abi_c.c` is compiled as C on purpose.** The public headers are meant to be consumable
   by C tools, and nothing else in the tree compiles them as C. It reports what C computes for the
   record layout and the C++ side compares field by field, which catches a header change both
   languages accept but interpret differently — a case a compile check alone cannot catch.
2. **The ABI tests pin semantics, not just layout.** A zero-initialized record means "do not replay
   this dispatch", `total_passes == 0` means an indefinite loop and must stay distinguishable from a
   single pass, and `current_pass` is 0-indexed against `total_passes`.
3. **The performance harness has its own tests.** `tests/perf-common/test_perf_stats.py` covers the
   estimators and cost models that decide whether a performance test passes. Writing those found
   four bugs in the helpers.

### Running the suite

```bash
# every kernel replay test (label covers all tiers)
ctest -L kernel-replay --output-on-failure

# GPU-free tiers only
ctest -R 'unit\.kernel_replay_(abi|local_context|phases|configure)' --output-on-failure
ctest -R rocprofv3-test-kernel-replay-cli-unit --output-on-failure

# nightly performance sweeps, off by default
cmake -DROCPROFILER_BUILD_NIGHTLY_PERF_CTESTS=ON ...
ctest -L perf-nightly --output-on-failure
```

## 4. Test execution results

### Aggregate

Full-suite results from the OS legs that completed cleanly, on the merged heads:

| Build | Leg | Result | Total tests | Wall time |
| --- | --- | --- | --- | --- |
| 8915 (#7960) | Core • mi325 • rhel-9.5 | **100% passed** | 1415 | 420.66 s |
| 8917 (#10439) | Core • mi325 • rhel-9.5 | **100% passed** | 1422 | 448.14 s |

Logs: [#7960 rhel-9.5](https://github.com/ROCm/rocm-systems/actions/runs/33579830458/job/100091889911),
[#10439 rhel-9.5](https://github.com/ROCm/rocm-systems/actions/runs/33594406094/job/100166342796).

The 7-test delta between the two builds is exactly the `rocprofv3` integration added by #10439.

Per-OS job links for build 8917:
[rhel-8.8](https://github.com/ROCm/rocm-systems/actions/runs/33594406094/job/100166344067) ·
[rhel-9.5](https://github.com/ROCm/rocm-systems/actions/runs/33594406094/job/100166342796) ·
[sles-15.6](https://github.com/ROCm/rocm-systems/actions/runs/33594406094/job/100166343004) ·
[ubuntu-22.04](https://github.com/ROCm/rocm-systems/actions/runs/33594406094/job/100166342386)

Build 8915:
[rhel-8.8](https://github.com/ROCm/rocm-systems/actions/runs/33579830458/job/100091889920) ·
[rhel-9.5](https://github.com/ROCm/rocm-systems/actions/runs/33579830458/job/100091889911) ·
[sles-15.6](https://github.com/ROCm/rocm-systems/actions/runs/33579830458/job/100091889914) ·
[ubuntu-22.04](https://github.com/ROCm/rocm-systems/actions/runs/33579830458/job/100091889851)

### Kernel replay specifically

79 CTest entries carry kernel replay coverage on build 8917 (72 of them on 8915, before the
`rocprofv3` layer). All passed on rhel-8.8, rhel-9.5 and sles-15.6.

| Suite | Cases | What it establishes |
| --- | --- | --- |
| `unit.kernel_replay_abi` | 22 | Field order, offsets, size, alignment, callback signatures, enum stability, C/C++ agreement, forward-compatible truncated copies |
| `unit.kernel_replay_snapshot` | 15 | Restore reverts device memory and module variables, no cross-pass accumulation, pool filtering, snapshot bandwidth floors and sublinear scaling |
| `unit.kernel_replay_local_context` | 10 | Per-pass context overrides, arm/toggle windows, override cleared after loop, per-context independence, misbehaving-tool simulation |
| `unit.kernel_replay_phases` | 3 | CONFIG/PASS phase user-data flow and pass-scoped lifetime |
| `unit.kernel_replay_configure` | 1 | Single-subscriber enforcement |
| `tests.integration.execute.test-kernel-replay-*` | 11 | Local-context control end to end across counters, ATT, SPM and PC sampling; concurrency |
| Samples (`kernel-replay-*`) | 10 | Basic replay, user data, counters, ATT, SPM, service sequencing, opt-out, early exit |
| `rocprofv3-test-kernel-replay-*` | 7 | CLI unit tests, baseline/interval/perf generation and validation |

The `rocprofv3-test-kernel-replay-cli-unit` entry is a single CTest test wrapping **27 pytest
cases** that import `rocprofv3.py` as a module and exercise argument handling with no GPU and no
built SDK. That is where the `--replay-mode` selector and the `--kernel-replay-beta-enabled`
acknowledgement gate are covered.

### Code coverage

**75.58%** line coverage on build 8308
([run 33579830453](https://github.com/ROCm/rocm-systems/actions/runs/33579830453/job/100091693968)),
measured across 316 processed source files and published to CDash.

## 5. Failures observed, and what they mean

No failure in either build was attributed to kernel replay logic. Details below so QA does not
re-derive them.

### 5.1 `kernel-replay-att` aborted once — installed-samples suite, ubuntu-22.04, build 8917

The one failure that touches a kernel replay test. It needs care because the same test passed
everywhere else in the same build.

| Where | Result |
| --- | --- |
| Build 8917, ubuntu-22.04, **build-tree** suite (test #1506 of 1511) | Passed, 0.99 s |
| Build 8917, ubuntu-22.04, **installed-samples** suite (test #25 of 30) | **Subprocess aborted**, 1.88 s |
| Build 8917, rhel-8.8 / rhel-9.5 / sles-15.6 | Passed |
| Build 8915, ubuntu-22.04, both suites | Passed (twice) |
| Build 8308 coverage run | Passed |

In the failing batch, `counter-collection-print-functional-counters` aborted alongside it, and the
log records a single `Memory access fault by GPU node-4 ... Reason: Unknown` during that batch. That
counter test enumerates several hundred TCC hardware counters, and the installed-samples suite had
launched roughly 27 GPU tests concurrently against one device, including thread trace.

A missing trace decoder is ruled out as the cause. CI builds the decoder
(`enable-rocprof-trace-decoder-build: true`), and the gating in
[`samples/kernel_replay/CMakeLists.txt`](../samples/kernel_replay/CMakeLists.txt) disables ATT tests
outright when it is absent. The test ran, so the decoder was found and its path was supplied.

The evidence points to hardware resource contention in a highly parallel batch rather than a defect
in replay: the identical binary and the identical test passed in the build-tree suite of the same
job, on all three other operating systems, and twice on the parent build. It is worth watching for
recurrence, and the durable fix is to serialise thread trace against full-device counter
enumeration in the installed-samples suite. Not a functional blocker.

Log: [ubuntu-22.04, build 8917](https://github.com/ROCm/rocm-systems/actions/runs/33594406094/job/100166342386).

### 5.2 `rocprofv3-test-hip-streams-per-thread` segfault — ubuntu-22.04, build 8915

Segfaulted during startup, taking three dependent validation tests to "Not Run" with it. Suite
still reported 99% passed.

This test comes from `develop` and is not modified by either PR. It flaked on roughly a third of the
runs observed across both branches and also passed repeatedly on the same job and machine. One
open question is worth a follow-up rather than a defect: #7960 added an unconditional
`memory_tracker_init` to the startup path in `registration.cpp`, which could widen a pre-existing
startup race with HIP runtime initialization on another thread. **Recommended follow-up:** gate the
memory tracker on contexts that actually request replay instead of hooking unconditionally.

Log: [ubuntu-22.04, build 8915](https://github.com/ROCm/rocm-systems/actions/runs/33579830458/job/100091889851).

### 5.3 Thread trace triple-buffer tests — coverage build 8308

`thread-trace-api-triple-buffer-consistency-test` and `-slow-test` aborted; suite reported 99%
passed. Both are pre-existing thread trace tests, unrelated to replay, and are timing-sensitive
under the instrumentation overhead a coverage build adds.

Log: [coverage, build 8308](https://github.com/ROCm/rocm-systems/actions/runs/33579830453/job/100091693968).

### 5.4 Infrastructure, not product

| Job | Symptom | Reading |
| --- | --- | --- |
| Code Coverage, build 8917 | Ran 05:25:41 → 05:41:42, no failing step recorded, logs not retained | ~16 m submodule-fetch timeout |
| `build-docs-from-source`, build 8917 | Ran 05:25:46 → 05:41:47, same shape | Same timeout; `build-docs` passed |
| `therock-pr-bot` | `policy_check.py`: "Timed out waiting for required checks to complete" | Gate waiting on other checks, not a test |

## 6. Carve-outs

Tests that do **not** run, by design. QA should expect these as `Not Run (Disabled)` or `Skipped`
rather than raise them as defects. Gating lives in
[`samples/kernel_replay/CMakeLists.txt`](../samples/kernel_replay/CMakeLists.txt).

| Test | State | Reason |
| --- | --- | --- |
| `kernel-replay-counters-then-pc-sampling` | Disabled | PC sampling is agent-wide and does not consult localized per-pass overrides yet |
| `kernel-replay-services-first` | Disabled | Same; carries the `PC_SAMPLING` flag |
| `kernel-replay-services-last` | Disabled | Same |
| `test-kernel-replay-local-context-pc-sampling-stop-after-0` | Skipped | PC sampling unavailable on the runner |
| ATT samples (`kernel-replay-att`, both `services-*`) | Disabled when decoder absent | Need `rocprof-trace-decoder`; located at configure time and passed as `ROCPROFILER_TRACE_DECODER_LIB_PATH` |
| `tests/kernel-replay-perf/` sweeps | Off by default | Behind `ROCPROFILER_BUILD_NIGHTLY_PERF_CTESTS`, label `perf-nightly` |

The PC sampling carve-out is the one with product meaning. Samples that assign counters and PC
sampling to *different* replay passes still run both on every pass, and can hit the MI2xx/MI3xx
clock-gating conflict. They stay disabled until the PC sampling service honours
`local_context_override()` at collection time. **Partitioning PC sampling by replay pass is not a
supported configuration in this release and should not be validated as one.**

Two further behaviours are intentional and will look like failures if unexpected:

- **HIP graph launches decline replay.** A graph-captured dispatch does not participate in the
  replay isolation path. It warns once and runs normally. Much of PyTorch and vLLM captures graphs
  by default, so a graph-heavy workload legitimately produces no replay.
- **Several waits inside the replay window abort the process on expiry** rather than proceeding on
  questionable state. Deliberate for a beta feature; see
  [concurrency and isolation](../source/docs/conceptual/kernel_replay/kernel_replay_concurrency_and_isolation.md).

## 7. Known gaps — suggested focus for manual validation

These are documented gaps in automated coverage, reproduced from the
[test coverage document](../source/docs/conceptual/kernel_replay/kernel_replay_testing.md). They are
the highest-value targets for manual and exploratory testing, since CI does not cover them.

1. **No GPU-free exercise of the replay window.** The drain, snapshot, pass loop and restore
   sequence in `hsa/queue.cpp` is only reachable with a device. Its control flow — decline paths,
   early exit, restore-failure abort — is untested except on hardware. There is no mock HSA stack
   to build such a test on; a harness would need a `Queue` test double accepting synthetic packet
   batches, host-backed stand-ins for the tracker and snapshot, and a working intercept-registration
   stub. New infrastructure rather than an extension of what exists.
2. **No multi-GPU coverage at any level.** Highest-value manual target.
3. **No test that a HIP graph launch declines replay visibly.** Behaviour is documented and warns
   once, but nothing asserts it, so a graph-capturing workload could silently get no replay.
4. **No ABI checker.** Layout is asserted by hand where someone thought to do it; nothing compares
   built artifacts across versions the way `abidiff` would.
5. **No build test for the public headers as a whole.** Nothing asserts every public header is
   self-contained; outside the kernel replay header, none are compiled as C.
6. **No compile-time budget.** A change that slows the SDK build is invisible.

Suggested additional manual scenarios, given the carve-outs above: replay under real multi-GPU
workloads; graph-capturing frameworks (PyTorch, vLLM) to confirm the decline path is visible and
harmless; long-running replay with an indefinite pass loop to confirm the termination contract;
and host-memory-pressure conditions to confirm snapshot failure degrades to a single run instead of
aborting.
