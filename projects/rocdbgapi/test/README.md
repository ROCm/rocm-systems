# ROCdbgapi test suite

This directory holds the GoogleTest-based test suite for the
`amd-dbgapi` library.  Tests are split into two tiers:

```
test/
  unit/      OS-agnostic, no kernel driver, no GPU.  Drives internal
             seams (the `amd-dbgapi-internal` target) and the public
             API against in-process mocks.  Always runnable.
  feature/   End-to-end tests against the shipping public library.
             May require the kernel driver (KFD on Linux, KMD on
             Windows), HIP, and a real GPU.  Tests skip cleanly
             (`GTEST_SKIP`) when their prerequisites are missing, so
             the same binary works on a driver-less laptop, a
             driver-only CI runner, and a fully-equipped GPU host.
```

## Building

The suite is off by default.  Three CMake options gate it:

| Option                            | Default                       | Effect                       |
| --------------------------------- | ----------------------------- | ---------------------------- |
| `AMD_DBGAPI_BUILD_TESTS`          | `OFF`                         | Master switch — turns both tiers on. |
| `AMD_DBGAPI_BUILD_UNIT_TESTS`     | `${AMD_DBGAPI_BUILD_TESTS}`   | Build the `unit/` tier only. |
| `AMD_DBGAPI_BUILD_FEATURE_TESTS`  | `${AMD_DBGAPI_BUILD_TESTS}`   | Build the `feature/` tier only. |

Typical configure + build from the repo root:

```shell
cmake -S projects/rocdbgapi -B build-rocdbgapi \
      -DAMD_DBGAPI_BUILD_TESTS=ON
cmake --build build-rocdbgapi -j
```

GoogleTest + GMock are vendored under `third_party/` and built as part
of the suite; no system install is needed.

When `find_package(hip)` succeeds at configure time, the HIP idle
workload (`feature/workloads/idle_kernel.hip`) is built as
`dbgapi_test_idle_kernel`.  Without HIP the workload binary is empty
and the GPU-tier tests `GTEST_SKIP` at runtime.

## Running

From the build tree:

```shell
cd build-rocdbgapi
ctest --output-on-failure              # everything
ctest -L unit                          # OS-agnostic tier only
ctest -L feature                       # any feature test
ctest -L feature-driver                # subset needing the kernel driver
ctest -L feature-gpu                   # subset needing a real GPU
ctest -R dbgapi_feature_smoke_test     # one test by name regex
```

Individual binaries accept the standard GoogleTest flags
(`--gtest_filter`, `--gtest_list_tests`, `--gtest_repeat`, etc.):

```shell
./test/feature/dbgapi_feature_control_flow_test \
    --gtest_filter='ControlFlow.WaveStopProducesWaveStopEvent'
```

### CTest labels

Labels are applied automatically by the `add_dbgapi_*_test()` helpers
in `cmake/AddDbgapiTest.cmake`.  Conventions used across the suite:

| Label                  | Meaning                                              |
| ---------------------- | ---------------------------------------------------- |
| `dbgapi`               | Any rocdbgapi test (applied to every target).        |
| `unit`                 | Unit-tier test.                                      |
| `feature`              | Feature-tier test.                                   |
| `feature-driver`       | Needs the kernel driver (KFD on Linux, KMD on Win).  |
| `feature-kfd`          | Linux-only — requires `/dev/kfd`.                    |
| `feature-kmd`          | Windows-only — requires the kernel-mode driver.      |
| `feature-gpu`          | Needs a real GPU and (for the current tests) HIP.    |
| `feature-gpu-<arch>`   | Pinned to a specific GPU architecture (e.g. `gfx942`). |
| `os-linux`, `os-windows` | OS-specific tests.                                 |

CI uses these labels to route tests to runners with the right
hardware.  Locally, prefer label filters (`-L`) over `-R` for tier
selection.

## The skip envelope

A core property of the feature tier: **a missing prerequisite is a
skip, not a failure**.  Each test probes its needs at runtime and
emits `GTEST_SKIP() << "..."` when something is missing.  This keeps
one binary usable everywhere:

* No `/dev/kfd` → driver-tier tests skip.
* HIP not found at configure time → GPU-workload tests skip.
* Workload spawned but kernel never dispatches a wave → wave-inspection
  tests skip with a diagnostic, not a failure.
* `wave_stop` returns `WAVE_STOPPED` because the runtime already parked
  the wave → control-flow tests treat it as a skip path.

If you are debugging "why did my test skip?", run with
`GTEST_BRIEF=0` and read the `[  SKIPPED ]` line — every skip in this
suite carries a one-line reason.

## Adding a new test

Both tiers expose a small helper that wraps `add_executable` +
`add_test` and applies the labels and properties above.

### Unit test

```cmake
# test/unit/CMakeLists.txt
add_dbgapi_unit_test(my_new_unit_test
  SOURCES
    my_new_unit_test.cpp
  LIBS amd-dbgapi-internal)   # or amd-dbgapi for public-API tests
```

Unit tests link `GTest::gtest_main` automatically.  Use
`amd-dbgapi-internal` (the test seam set up in Commit 4) when you need
to reach internal symbols; otherwise link the shipping `amd-dbgapi`.

### Feature test

```cmake
# test/feature/CMakeLists.txt
add_dbgapi_feature_test(my_new_feature_test
  SOURCES
    my_new_feature_test.cpp
    support/capabilities.cpp        # if you probe driver/GPU presence
    support/spawn_hip_workload.cpp  # if you need a GPU workload
  REQUIRES_DRIVER                   # adds feature-driver + feature-kfd/kmd
  REQUIRES_GPU                      # adds feature-gpu
  LIBS amd-dbgapi)
```

Optional knobs:

* `REQUIRES_DRIVER` — keyword option; adds `feature-driver` plus the
  OS-specific `feature-kfd`/`feature-kmd` label.
* `REQUIRES_GPU [arch ...]` — keyword (with zero or more architecture
  names).  Adds `feature-gpu`, and one `feature-gpu-<arch>` label per
  named architecture.
* `LABELS extra-label ...` — extra CTest labels.
* `LIBS extra::link ...` — extra link dependencies.

### Support library (feature tier)

Common feature-test helpers live in `feature/support/`:

| Header                          | Provides                                                                  |
| ------------------------------- | ------------------------------------------------------------------------- |
| `support/capabilities.h`        | Driver / HIP / GPU capability probes used by `GTEST_SKIP` decisions.     |
| `support/dbgapi_fixture.h`      | Three-tier GoogleTest fixtures: no-driver, driver, GPU.                  |
| `support/minimal_callbacks.h`   | Minimal `amd_dbgapi_callbacks_s` for tests that never attach.            |
| `support/pid_callbacks.h`       | Callbacks that resolve an OS PID — needed for `process_attach`.          |
| `support/spawn_idle_child.h`    | Portable helper that forks a stub child for self-attach tests.           |
| `support/spawn_hip_workload.h`  | Spawns `dbgapi_test_idle_kernel` and waits for its `READY` handshake.    |

Reach for the highest-level fixture that fits — `gpu_fixture_t`
handles the HIP-availability and driver checks for you.

## Layout reference

```
test/
├── CMakeLists.txt              Top-level switch; wires in unit/ and feature/.
├── README.md                   You are here.
├── unit/
│   ├── CMakeLists.txt          One add_dbgapi_unit_test() per binary.
│   ├── *_test.cpp              One file per subsystem under test.
│   └── support/                Mocks (MockOsDriver), shared fixtures.
└── feature/
    ├── CMakeLists.txt          One add_dbgapi_feature_test() per binary.
    ├── *_test.cpp              One file per feature being exercised.
    ├── support/                Capability probes, fixtures, child spawners.
    └── workloads/
        └── idle_kernel.hip     Single-wave spinning HIP kernel used by
                                the GPU-tier tests.
```

## Troubleshooting

* **`ctest -L feature-kfd` skips everything** — KFD debug requires
  `/dev/kfd` to be present and accessible.  On a typical workstation
  the user must be in the `render` (and on some distros `video`)
  group.  Re-login after `usermod -aG render $USER`.
* **GPU tests skip with "attach / wave selection failed"** — the child
  spawned but the runtime could not enable debug.  Common causes:
  YAMA ptrace restrictions (the workload calls `prctl(PR_SET_DUMPABLE)`
  and `PTRACE_TRACEME` to mitigate this), or KFD being held open by
  another debugger.  Try `ps -ef | grep gdb` and check
  `/proc/sys/kernel/yama/ptrace_scope`.
* **`HSA_STATUS_ERROR_EXCEPTION ... code: 0x1016`** during
  `ResumeWithAbortExceptionTransitionsWave` is expected — the
  `EXCEPTION_WAVE_ABORT` resume mode contract is exactly that the
  queue moves to the error state.  The test itself still passes.
