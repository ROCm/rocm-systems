# RCCL Testing Guide

A practical reference for new hires. Covers every test suite, when to use it, which files to touch to add a test, and how to build and run each suite.

---

## Table of Contents

1. [Background and Philosophy](#1-background-and-philosophy)
2. [Repository Layout](#2-repository-layout)
3. [Quick-Reference Table](#3-quick-reference-table)
4. [Building the Tests](#4-building-the-tests)
5. [Suite A — `rccl-UnitTests` (Collective / TestBed)](#5-suite-a--rccl-unittests-collective--testbed)
6. [Suite B — `rccl-UnitTestsFixtures` (Header-Level, Release-Safe)](#6-suite-b--rccl-unittestsfixturesebug-header-level-release-safe)
7. [Suite C — `rccl-UnitTestsFixturesDebug` (Internal Symbols, Debug Only)](#7-suite-c--rccl-unittestsfixturesdebug-internal-symbols-debug-only)
8. [Suite D — `rccl-UnitTestsMock` (Link-Time Mocks, Debug Only)](#8-suite-d--rccl-unittestsmock-link-time-mocks-debug-only)
9. [Suite E — `rccl-UnitTestsMPI` (Multi-Process / Multi-GPU)](#9-suite-e--rccl-unittestsmpi-multi-process--multi-gpu)
10. [Suite F — `rccl-UnitTestsAltRsmi` (AltRsmi Special Linkage, Debug Only)](#10-suite-f--rccl-unittestsaltrsmi-altrsmi-special-linkage-debug-only)
11. [Suite G — `rccl-UnitTestsFixtures` device/ (GPU Kernel Tests)](#11-suite-g--rccl-unittestsfixturesdevice-gpu-kernel-tests)
12. [The Python Test Runner](#12-the-python-test-runner)
13. [Multi-Node Testing with `mnctl`](#13-multi-node-testing-with-mnctl)
14. [Test Infrastructure Reference](#14-test-infrastructure-reference)
15. [Decision Guide — Which Suite?](#15-decision-guide--which-suite)
16. [Day-to-Day Cheat Sheet](#16-day-to-day-cheat-sheet)

---

## 1. Background and Philosophy

RCCL is a multi-GPU, multi-node collective-communication library. Testing it requires multiple distinct strategies because:

- **Public API tests** run against the installed shared library — they work in both Release and Debug builds.
- **Internal symbol tests** access non-public functions inside `librccl.so`, which are only externally visible when compiled with `-DCMAKE_BUILD_TYPE=Debug` (Release hides them via `-fvisibility=hidden`).
- **Static-variable tests** — RCCL code frequently caches `getenv()` results in static variables that are set once per process and never updated. Standard GTest runs all tests in one process, so test order affects results. These tests need process isolation.
- **Collective-operation tests** need multiple communicating processes, one per GPU. A single process cannot exercise AllReduce, AllGather, and friends in any meaningful way. These tests use MPI.
- **GPU kernel tests** run device-side code directly from host tests, requiring a HIP kernel launch harness.
- **Mock/link-time override tests** replace real `librccl.so` symbols with fake implementations at link time. They must be isolated into their own binary to avoid poisoning tests that expect the real symbols.

---

## 2. Repository Layout

All RCCL test code lives under `projects/rccl/test/` (paths below are relative to the repository root `projects/rccl/`):

```
test/
├── CMakeLists.txt                  ← single source of truth for all binaries
│
├── ── Collective tests (Suite A) ──
│   AllReduceTests.cpp
│   AllGatherTests.cpp
│   AllToAllTests.cpp   … etc.
│   StandaloneTests.cpp
│   RegisterTests.cpp
│   _RecorderTests.cpp
│   proxy_trace/ProxyTraceUnitTests.cpp
│   latency_profiler/LatencyProfilerUnitTest.cpp
│
├── ── Fixture tests (Suites B & C) ──
│   BitOpsTests.cpp                 ← B: header-only maths utilities
│   CommTests.cpp                   ← B: comm.h header tests
│   EnqueueCountTests.cpp           ← B: public enqueue helpers
│   MiscTests.cpp                   ← B: misc header tests
│   device/TestOp128.cpp            ← B/G: GPU kernel tests (op128.h)
│   device/DeviceTestBase.hpp       ← G: GPU kernel test fixture
│   AllocTests.cpp                  ← C: ncclIbMalloc etc. (debug only)
│   ParamTests.cpp                  ← C: ncclGetParam (debug only)
│   ArgCheckTests.cpp               ← C: arg checking (debug only)
│   EnqueueTests.cpp                ← C: enqueue internals (debug only)
│   IpcsocketTests.cpp              ← C: IPC socket (debug only)
│   NetSocketTests.cpp              ← C: net socket (debug only)
│   ProxyTests.cpp                  ← C: proxy thread (debug only)
│   graph/XmlTests.cpp              ← C: XML graph parser (debug only)
│
├── ── Mock tests (Suite D) ──
│   RcclWrapTests.cpp               ← D: rocm library wrapping (mock)
│   TransportTests.cpp              ← D: transport setup (link-time mocks)
│
├── ── MPI tests (Suite E) ──
│   CommMPITests.cpp
│   RegistrationMPITests.cpp
│   ImplicitLaunchOrderMPITests.cpp
│   transport/
│     P2pMPITests.cpp
│     ShmMPITests.cpp
│     NetMPITests.cpp
│     NetIbMPITests.cpp
│     TransportMPIBase.cpp / .hpp
│
├── ── AltRsmi tests (Suite F) ──
│   AltRsmiTests.cpp
│
└── common/                         ← shared test infrastructure
    main.cpp                        ← entry point for Suite A
    main_fixtures.cpp               ← entry point for Suites B/C/D/F
    main_mpi.cpp                    ← entry point for Suite E
    TestBed.hpp / .cpp              ← TestBed collective harness (Suite A)
    TestBedChild.hpp / .cpp
    CallCollectiveForked.hpp / .cpp
    CollectiveArgs.hpp / .cpp
    PrepDataFuncs.hpp / .cpp
    StandaloneUtils.hpp / .cpp
    ProcessIsolatedTestRunner.hpp/cpp ← isolation framework (Suites C/D)
    MPITestBase.hpp                 ← GTest ↔ MPI adapter (Suite E)
    MPITestCore.hpp / .cpp          ← framework-agnostic MPI base
    MPIEnvironment.hpp / .cpp       ← global MPI init / GPU assignment
    MPIHelpers.hpp / .cpp           ← per-rank logging, TestLogAssertionContext
    MPIStandaloneTest.hpp           ← standalone (non-GTest) MPI base
    ResourceGuards.hpp              ← RAII guards (communicator, stream, …)
    DeviceBufferHelpers.hpp         ← MPI typed device buffers
    TestChecks.hpp / .cpp           ← TEST_INFO/WARN/… macros, HIP/NCCL checks
    EnvVars.hpp / .cpp              ← environment variable helpers
    ErrCode.hpp
    PtrUnion.hpp / .cpp
    TransportUtils.hpp              ← link-time mock helpers (Suite D)
    RcclMockFuncs.hpp

tools/scripts/test_runner/
├── test_runner.py                  ← Python orchestration entry point
├── configs/
│   ├── ci-precheckin.json          ← CI pipeline suite (single-node)
│   ├── mi300x_mellanox_ib.json     ← MI300X + InfiniBand full suite
│   ├── net_ib_transport.json       ← IB transport focused
│   ├── rccl_perf_tests.json        ← performance benchmarks
│   └── test_config_sample.json     ← annotated template for new configs
└── lib/
    test_executor.py
    test_config.py
    test_parser.py

docker/
├── mnctl/                          ← Python multi-node Docker orchestrator
│   ├── __main__.py
│   ├── README.md
│   └── … (docker_ops, orchestrate, ssh, utils, …)
├── Dockerfile.Multinode.Ubuntu
└── setup_multinode.sh
```

---

## 3. Quick-Reference Table

| Binary | Build type | Uses MPI | When to use |
|--------|-----------|----------|-------------|
| `rccl-UnitTests` | Release **or** Debug | No | Collective operations, full-stack functional tests |
| `rccl-UnitTestsFixtures` | Release **or** Debug | No | Header-only internals, public struct/utility tests, GPU kernel tests |
| `rccl-UnitTestsFixturesDebug` | **Debug only** | No | Internal symbols hidden in Release (`ncclIbMalloc`, enqueue, proxy, etc.) |
| `rccl-UnitTestsMock` | **Debug only** | No | Tests that replace real `librccl.so` symbols with mocks at link time |
| `rccl-UnitTestsMPI` | **Debug only** | **Yes** | Multi-process collective behaviour, transport-layer (P2P/SHM/NET/IB), multi-node |
| `rccl-UnitTestsAltRsmi` | **Debug only** | No | `alt_rsmi.cc` compiled with `ARSMI_TEST_BUILD` to expose statics |

---

## 4. Building the Tests

### 4.1 Prerequisites

| Tool | Minimum | Notes |
|------|---------|-------|
| ROCm | 6.4.0 | Fixture and fixture-debug binaries require `ROCM_VERSION >= 60400` |
| CMake | 3.16 | |
| OpenMPI (or MPICH) | any | Required for Suite E; default install path `/opt/ompi` |
| Python | 3.6+ | For `test_runner.py` and `mnctl`; stdlib only, no pip dependencies |

### 4.2 Using `install.sh`

`install.sh` is the canonical build wrapper. Tests are compiled when you pass `-t` (or `--tests_build`).

```bash
# ─── Release build, public tests only (Suites A and B)
./install.sh -t -l                    # -l = local GPU only (faster)

# ─── Debug build, all non-MPI suites (A, B, C, D, F, G)
./install.sh --debug -t -l

# ─── Debug build + MPI suites (A–G including Suite E)
./install.sh --debug -t -l --enable-mpi-tests

# ─── Debug + MPI, non-local GPU targets (for multi-arch CI)
./install.sh --debug -t --enable-mpi-tests

# ─── Pass extra CMake options
./install.sh --debug -t -l --cmake-options "-DFOO=ON -DBAR=OFF"

# ─── Build with code coverage instrumentation
./install.sh --debug -t -l --enable-code-coverage

# ─── Build, then immediately run the quick test subset
./install.sh --debug -t -l --run_tests_quick

# ─── Build, then run every test binary
./install.sh --debug -t -l --run_tests_all
```

**Build output locations:**

| Build type | Directory |
|-----------|-----------|
| Release | `build/release/` |
| Debug | `build/debug/` |

Test binaries are placed at `build/{release,debug}/test/rccl-UnitTests*` and installed to `build/{release,debug}/test/`.

> **`--enable-mpi-tests` requires `--debug`.**  
> MPI tests access internal RCCL symbols, so they cannot be built against a Release library.

### 4.3 Direct CMake (advanced)

```bash
mkdir -p build/debug && cd build/debug

cmake ../.. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTS=ON \
  -DENABLE_MPI_TESTS=ON \
  -DMPI_PATH=/opt/ompi \
  -DBUILD_LOCAL_GPU_TARGET_ONLY=ON \
  -DROCM_PATH=/opt/rocm

make -j$(nproc) rccl-UnitTests rccl-UnitTestsFixtures rccl-UnitTestsFixturesDebug \
                rccl-UnitTestsMock rccl-UnitTestsMPI rccl-UnitTestsAltRsmi
```

### 4.4 Custom RCCL library

If you have a pre-built RCCL, point the test runner at it to skip rebuilding:

```bash
export RCCL_LIB_PATH=/path/to/existing/build   # must contain librccl.so + test/
python3 tools/scripts/test_runner/test_runner.py --config tools/scripts/test_runner/configs/ci-precheckin.json
```

---

## 5. Suite A — `rccl-UnitTests` (Collective / TestBed)

### What it tests

End-to-end collective operations: AllReduce, AllGather, AllToAll, Broadcast, Gather, Reduce, ReduceScatter, Scatter, SendRecv, GroupCall, NonBlocking, plus the recorder, proxy-trace, and latency-profiler subsystems. These tests spawn child processes internally via `CallCollectiveForked` — no MPI is needed.

### Test framework: TestBed

`TestBed` (`test/common/TestBed.hpp`) is a custom harness that forks child worker processes, assigns GPUs, initialises RCCL communicators, runs the collective, and checks correctness. You drive it with `RunSimpleSweep()` or the lower-level `RunSimpleConfig()`.

**Reference:** [Google Test primer](https://google.github.io/googletest/primer.html) (test structure / `EXPECT_*`/`ASSERT_*`).

### Files to edit

| What you want to do | File to edit |
|---------------------|-------------|
| Add a test to an existing collective (e.g. AllReduce) | `test/AllReduceTests.cpp` |
| Add a brand-new collective or top-level feature | Create `test/MyFeatureTests.cpp`, add it to `TEST_SOURCE_FILES` in `test/CMakeLists.txt` |
| Add proxy-trace tests | `test/proxy_trace/ProxyTraceUnitTests.cpp` |
| Add latency-profiler tests | `test/latency_profiler/LatencyProfilerUnitTest.cpp` |
| Add a recorder test | `test/_RecorderTests.cpp` |

### Adding a test

```cpp
// test/AllReduceTests.cpp  (existing file)
#include "TestBed.hpp"
#include "CallCollectiveForked.hpp"

namespace RcclUnitTesting
{
  TEST(AllReduce, MyNewVariant)
  {
    TestBed testBed;

    std::vector<ncclFunc_t>     const funcTypes      = {ncclCollAllReduce};
    std::vector<ncclDataType_t> const dataTypes      = {ncclFloat32};
    std::vector<ncclRedOp_t>    const redOps         = {ncclSum};
    std::vector<int>            const roots          = {0};
    std::vector<int>            const numElements    = {1024};
    std::vector<bool>           const inPlace        = {false};
    std::vector<bool>           const managedMem     = {false};
    std::vector<bool>           const useHipGraph    = {false};

    testBed.RunSimpleSweep(funcTypes, dataTypes, redOps, roots, numElements,
                           inPlace, managedMem, useHipGraph);
    testBed.Finalize();
  }
}
```

If you create a new file, also add it to `test/CMakeLists.txt`:

```cmake
set(TEST_SOURCE_FILES
  ...
  MyFeatureTests.cpp   # ← add here
  ...
)
```

### Building and running

```bash
# Build (release is sufficient for Suite A)
./install.sh -t -l

# Run everything
./build/release/test/rccl-UnitTests

# Run one test
./build/release/test/rccl-UnitTests --gtest_filter="AllReduce.MyNewVariant"

# Run a pattern
./build/release/test/rccl-UnitTests --gtest_filter="AllReduce.*"

# Debug output on failure
NCCL_DEBUG=INFO ./build/release/test/rccl-UnitTests --gtest_filter="AllReduce.MyNewVariant"
```

---

## 6. Suite B — `rccl-UnitTestsFixtures` (Header-Level, Release-Safe)

### What it tests

Tests that depend **only on header-defined symbols** — `inline`, `static`, `constexpr`, and template functions — so they link against either Release or Debug builds of RCCL. Currently covers:

- `BitOpsTests.cpp` — `DIVUP`, `ROUNDUP`, `ALIGN_*`, bit-hash utilities (`bitops.h`)
- `CommTests.cpp` — `ncclTaskCollSorter` helpers (`comm.h`)
- `EnqueueCountTests.cpp` — enqueue send/recv count helpers (public API)
- `MiscTests.cpp` — miscellaneous header-level tests
- `device/TestOp128.cpp` — GPU kernel tests for `op128.h` (see [Suite G](#11-suite-g--rccl-unittestsfixturesdevice-gpu-kernel-tests))

> **Rule of thumb:** if your test only `#include`s headers (no `.cc`/`.cpp` sources that are compiled into `librccl.so`), it belongs here.

### Files to edit

| What you want to do | File to edit |
|---------------------|-------------|
| Test a header-only RCCL utility | Add to an existing `*Tests.cpp` or create a new one |
| New file | Add to `TEST_FIXTURE_SOURCE_FILES` in `test/CMakeLists.txt` |

```cmake
# test/CMakeLists.txt — inside the ROCM_VERSION >= 60400 block
set(TEST_FIXTURE_SOURCE_FILES
  BitOpsTests.cpp
  CommTests.cpp
  EnqueueCountTests.cpp
  MiscTests.cpp
  device/TestOp128.cpp
  MyNewHeaderTests.cpp   # ← add here
  common/main_fixtures.cpp
  common/EnvVars.cpp
  common/ProcessIsolatedTestRunner.cpp
  common/TestChecks.cpp
)
```

### Adding a test

```cpp
// test/MyNewHeaderTests.cpp
#include "gtest/gtest.h"
#include "bitops.h"   // or whichever header you're testing

namespace RcclUnitTesting
{
  TEST(MyUtils, SomeEdgeCase)
  {
    EXPECT_EQ(DIVUP(17, 8), 3);
  }
}
```

### Building and running

```bash
# Release is fine
./install.sh -t -l

./build/release/test/rccl-UnitTestsFixtures
./build/release/test/rccl-UnitTestsFixtures --gtest_filter="BitOps*"
```

---

## 7. Suite C — `rccl-UnitTestsFixturesDebug` (Internal Symbols, Debug Only)

### What it tests

Functions compiled into `librccl.so` that are hidden (`-fvisibility=hidden`) in Release builds but accessible in Debug builds:

| File | What it covers |
|------|---------------|
| `AllocTests.cpp` | `ncclIbMalloc`, `ncclCudaMemcpy` |
| `ParamTests.cpp` | `ncclGetParam`, environment-variable parameter lookup |
| `ArgCheckTests.cpp` | `CudaPtrCheck`, argument validation helpers |
| `EnqueueTests.cpp` | `ncclEnqueueCheck`, send/recv internal paths |
| `IpcsocketTests.cpp` | IPC socket utilities |
| `NetSocketTests.cpp` | Network socket utilities |
| `ProxyTests.cpp` | Proxy thread internals |
| `graph/XmlTests.cpp` | `graph/xml.cc` parser |

Many of these tests use `ProcessIsolatedTestRunner` because the code under test reads environment variables into static variables.

### Process isolation — why you need it here

RCCL internally does things like:

```cpp
void rcclSetP2pNetChunkSize(ncclComm* comm, int& chunkSize) {
  static int p2pNetChunkSize = RCCL_VALUE_UNSET; // set once, never again
  if (p2pNetChunkSize == RCCL_VALUE_UNSET) {
    const char* s = getenv("NCCL_P2P_NET_CHUNKSIZE");
    p2pNetChunkSize = s ? atoi(s) : calculateDefault();
  }
  chunkSize = p2pNetChunkSize;
}
```

If test 1 sets `NCCL_P2P_NET_CHUNKSIZE=12345` and test 2 clears it, the static variable keeps `12345` — test 2 sees stale state. `ProcessIsolatedTestRunner` runs each test in a `fork()`'d child process, so static variables reset between tests.

**Reference:**
- `test/common/ProcessIsolatedTestRunner.hpp` — API
- `test/common/ProcessIsolatedTestFramework.md` — full documentation

### Files to edit

Add your file to `TEST_FIXTURE_DEBUG_SOURCE_FILES` in `test/CMakeLists.txt`:

```cmake
# test/CMakeLists.txt — inside the Debug-only block
set(TEST_FIXTURE_DEBUG_SOURCE_FILES
  AllocTests.cpp
  ...
  MyInternalTests.cpp   # ← add here
  common/main_fixtures.cpp
  common/EnvVars.cpp
  common/ProcessIsolatedTestRunner.cpp
)
```

### Adding a test — plain GTest (no env-var interaction)

```cpp
// test/MyInternalTests.cpp
#include "gtest/gtest.h"
#include "alloc.h"                          // internal RCCL header

namespace RcclUnitTesting
{
  TEST(MyInternal, BasicBehaviour)
  {
    void* ptr = nullptr;
    EXPECT_EQ(ncclSuccess, ncclIbMalloc(&ptr, 1024));
    ASSERT_NE(ptr, nullptr);
    free(ptr);
  }
}
```

### Adding a test — with process isolation (env-var-sensitive code)

```cpp
// test/MyInternalTests.cpp
#include "gtest/gtest.h"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "graph/topo.h"    // internal RCCL header

namespace RcclUnitTesting
{
  // The outer TEST() is the GTest entry point; the isolated lambda is the
  // actual test body, running in a forked child process.
  TEST(MyInternal, EnvVarControlledBehaviour)
  {
    // Each call to RUN_ISOLATED_TEST_WITH_ENV forks a fresh child.
    RUN_ISOLATED_TEST_WITH_ENV(
      "BehaviourWithEnvSet",
      []() {
        // This runs in a child process — static variables are freshly initialised.
        const char* val = getenv("NCCL_MY_PARAM");
        ASSERT_STREQ(val, "42");
        // call the function that caches the env var in a static...
        EXPECT_TRUE(myInternalFunction());
      },
      {{"NCCL_MY_PARAM", "42"}}   // env vars set for this child
    );

    // A second isolated test: different env, independent static state.
    RUN_ISOLATED_TEST_WITH_ENV(
      "BehaviourWithEnvUnset",
      []() {
        EXPECT_TRUE(myInternalFunctionDefaultPath());
      },
      {}   // no env vars → default behaviour
    );
  }
}
```

Available macros (defined in `ProcessIsolatedTestRunner.hpp`):

| Macro | Use |
|-------|-----|
| `RUN_ISOLATED_TEST(name, lambda)` | Single test, inherit parent env |
| `RUN_ISOLATED_TEST_WITH_ENV(name, lambda, {{k,v},…})` | Single test, custom env |
| `RUN_ISOLATED_TESTS(TestConfig("A", lA), TestConfig("B", lB))` | Multiple tests in one `TEST()` |

### Building and running

```bash
# Debug build required
./install.sh --debug -t -l

./build/debug/test/rccl-UnitTestsFixturesDebug
./build/debug/test/rccl-UnitTestsFixturesDebug --gtest_filter="Alloc.*"

# With NCCL debug output on failure
NCCL_DEBUG=INFO ./build/debug/test/rccl-UnitTestsFixturesDebug \
  --gtest_filter="MyInternal.EnvVarControlledBehaviour"
```

---

## 8. Suite D — `rccl-UnitTestsMock` (Link-Time Mocks, Debug Only)

### What it tests

`RcclWrapTests.cpp` and `TransportTests.cpp`. These tests define their own implementations of RCCL-internal symbols (`bootstrapAllGather`, `bootstrapIntraNodeAllGather`, the `ncclBootstrap` struct, and `collNetTransport` function pointers) that override the real ones at link time. This lets tests drive transport-setup code paths without a real network fabric.

They must live in their own binary because link-time symbol overrides apply to every test in the same binary — if these were linked with `rccl-UnitTestsFixturesDebug`, the mocks would replace real symbols for all the other tests there too.

`test/common/TransportUtils.hpp` and `test/common/RcclMockFuncs.hpp` contain the shared mock helpers.

**Reference:** [Google Mock](https://google.github.io/googletest/gmock_for_dummies.html) (for function-pointer and interface mocking patterns).

### Files to edit

| What you want to do | File to edit |
|---------------------|-------------|
| Add a test using existing mocks | `test/RcclWrapTests.cpp` or `test/TransportTests.cpp` |
| Add new mock symbols | `test/common/TransportUtils.hpp` or `test/common/RcclMockFuncs.hpp` |
| New file with its own set of mocks | Create the file and add it to `TEST_MOCK_SOURCE_FILES` in `test/CMakeLists.txt` |

```cmake
# test/CMakeLists.txt — inside the Debug-only block
set(TEST_MOCK_SOURCE_FILES
  RcclWrapTests.cpp
  TransportTests.cpp
  MyMockTests.cpp        # ← add here
  common/main_fixtures.cpp
  common/EnvVars.cpp
  common/ProcessIsolatedTestRunner.cpp
)
```

### Adding a test

```cpp
// test/TransportTests.cpp  (existing file)
#include "gtest/gtest.h"
#include "TransportUtils.hpp"  // brings in mock symbol definitions

namespace RcclUnitTesting
{
  TEST(TransportTest, MyMockedScenario)
  {
    // mock helpers set up fake ncclComm, channels, and function pointers
    mockSetup();

    // exercise the real RCCL transport-setup logic against the mocks
    ncclResult_t result = collNetRecvSetup(/* ... */);
    EXPECT_EQ(result, ncclSuccess);

    mockTeardown();
  }
}
```

### Building and running

```bash
./install.sh --debug -t -l

./build/debug/test/rccl-UnitTestsMock
./build/debug/test/rccl-UnitTestsMock --gtest_filter="TransportTest.*"
```

---

## 9. Suite E — `rccl-UnitTestsMPI` (Multi-Process / Multi-GPU)

### What it tests

Features that require real multi-process communication: P2P transport, shared-memory transport, network (NIC) transport, InfiniBand, buffer registration (IPC, graph-capture), and collective order. Each test spawns N MPI ranks (one per GPU) using `mpirun`.

### Test framework: MPITestBase

`MPITestBase` (`test/common/MPITestBase.hpp`) is a Google Test adapter that inherits from both `::testing::Test` and `MPITestCore`. It handles:

- Automatic RCCL communicator setup and teardown per test
- HIP stream lifecycle
- `validateTestPrerequisites()` — skip the test with `GTEST_SKIP()` if not enough ranks or wrong topology
- `createTestCommunicator()` — create a test-specific `ncclComm_t`
- `getActiveCommunicator()` / `getActiveStream()` — access the communicators

**References:**
- `test/common/MPITestFramework.md` — complete framework documentation
- `test/common/ProcessIsolatedTestFramework.md` — process isolation reference
- [Google Test](https://google.github.io/googletest/primer.html)
- [Open MPI docs](https://www.open-mpi.org/doc/current/)
- [HIP programming guide](https://rocm.docs.amd.com/projects/HIP/en/latest/user_guide/programming_guide.html)

### Current MPI test files

| File | Tests |
|------|-------|
| `transport/P2pMPITests.cpp` | P2P transport send/receive |
| `transport/ShmMPITests.cpp` | Shared-memory transport |
| `transport/NetMPITests.cpp` | Network (NIC) transport |
| `transport/NetIbMPITests.cpp` | InfiniBand transport |
| `CommMPITests.cpp` | Communicator creation, traffic-class config |
| `RegistrationMPITests.cpp` | User-buffer and graph-capture registration |
| `ImplicitLaunchOrderMPITests.cpp` | Implicit launch ordering |

### Files to edit

**Add to an existing file** when your test is logically related to an existing suite. **Create a new file** for a distinct feature area, then add it to `MPI_TEST_SOURCE_FILES` in `test/CMakeLists.txt`:

```cmake
# test/CMakeLists.txt — inside the ENABLE_MPI_TESTS block
set(MPI_TEST_SOURCE_FILES
  common/main_mpi.cpp
  common/MPIHelpers.cpp
  common/MPITestCore.cpp
  common/MPIEnvironment.cpp
  common/TestChecks.cpp
  transport/TransportMPIBase.cpp
  transport/P2pMPITests.cpp
  transport/NetMPITests.cpp
  transport/ShmMPITests.cpp
  transport/NetIbMPITests.cpp
  ImplicitLaunchOrderMPITests.cpp
  CommMPITests.cpp
  RegistrationMPITests.cpp
  MyNewMPITests.cpp          # ← add here
)
```

### Adding a test

```cpp
// test/MyNewMPITests.cpp
#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "ResourceGuards.hpp"
#include "DeviceBufferHelpers.hpp"
#include "TestChecks.hpp"

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

// ── Simple test: inherit MPITestBase, get communicator management for free ──
class MyFeatureMPITest : public MPITestBase {};

TEST_F(MyFeatureMPITest, BasicAllReduce)
{
  // Skip if fewer than 2 GPUs/ranks are available.
  if (!validateTestPrerequisites(/*min_processes=*/2))
    return; // GTest marks this SKIPPED automatically

  // Creates an ncclComm_t shared across all ranks, stored in the base class.
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  ncclComm_t  comm   = getActiveCommunicator();
  hipStream_t stream = getActiveStream();
  int         rank   = MPIEnvironment::world_rank;
  int         size   = MPIEnvironment::world_size;

  constexpr size_t N = 1024;

  // RAII device buffers — hipFree on scope exit.
  DeviceBuffer<float> send_buf(N), recv_buf(N);
  send_buf.upload(static_cast<float>(rank + 1));   // rank 0 → 1.0, rank 1 → 2.0, …

  RCCL_TEST_CHECK(ncclAllReduce(send_buf.ptr, recv_buf.ptr,
                                N, ncclFloat, ncclSum,
                                comm, stream));
  HIP_TEST_CHECK(hipStreamSynchronize(stream));

  // Each rank should see sum = 1+2+…+size
  float expected = static_cast<float>(size * (size + 1) / 2);
  auto  result   = recv_buf.copyTo();
  for (size_t i = 0; i < N; ++i)
    EXPECT_FLOAT_EQ(result[i], expected) << "at index " << i;
}

// ── Single-node constraint ──
TEST_F(MyFeatureMPITest, P2POnSameNode)
{
  // kRequireSingleNode = 1: skip if ranks span multiple physical nodes.
  if (!validateTestPrerequisites(2, kNoProcessLimit, false, kRequireSingleNode))
    return;

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  // … your P2P test …
}

#endif // MPI_TESTS_ENABLED
```

**Key helpers:**

| Helper | What it does |
|--------|-------------|
| `validateTestPrerequisites(min, max, pow2, min_nodes, max_nodes)` | Returns false (and emits `GTEST_SKIP()`) if prerequisites aren't met |
| `createTestCommunicator()` | Creates `ncclComm_t` + `hipStream_t` for this test |
| `getActiveCommunicator()` | Returns the per-test `ncclComm_t` |
| `getActiveStream()` | Returns the per-test `hipStream_t` |
| `MPIEnvironment::world_rank` / `::world_size` | Global MPI rank and size |
| `kRequireSingleNode` / `kNoNodeLimit` | Node constraint constants |
| `RCCL_TEST_CHECK(expr)` | Asserts `ncclSuccess`; prints error and marks test failed otherwise |
| `HIP_TEST_CHECK(expr)` | Same for HIP calls |
| `TEST_INFO(fmt, …)` | Printf-style log — only visible when `NCCL_DEBUG=INFO` |
| `TEST_WARN(fmt, …)` | Log — only visible when `NCCL_DEBUG=WARN` |

**RAII guards** (from `ResourceGuards.hpp`):

```cpp
// Generic scope guard — runs cleanup lambda on scope exit
auto guard = makeScopeGuard([&]() { ncclCommDestroy(comm); });
guard.dismiss();  // call this if you want to keep the resource

// Typed guard for device memory
auto buf_guard = makeDeviceBufferAutoGuard(device_ptr);
```

### Logging in MPI tests

The `TEST_INFO`/`TEST_WARN`/`TEST_ABORT`/`TEST_TRACE` macros auto-prefix with rank and hostname and are gated by `NCCL_DEBUG`:

```
NCCL_DEBUG=WARN  → TEST_WARN
NCCL_DEBUG=INFO  → TEST_WARN + TEST_INFO          ← recommended for debugging
NCCL_DEBUG=ABORT → … + TEST_ABORT
NCCL_DEBUG=TRACE → all macros
```

**Per-rank log files** — set `RCCL_MPI_LOG_ALL_RANKS=1` to redirect each rank's stderr to `rccl_test_rank_<N>.log` in the current directory. Rank 0 also writes to the console.

**Asserting on RCCL debug output** inside a test:

```cpp
TEST_F(MyFeatureMPITest, VerifyRegistrationLog)
{
  if (!validateTestPrerequisites(2)) return;

  // Scoped log capture: sets NCCL_DEBUG_FILE to a temp file, restores on exit.
  MPIHelpers::TestLogAssertionOptions opts;
  opts.capture_nccl_debug_file = true;
  opts.read_per_rank_stderr_log = true;
  opts.isolate_new_output = true;
  MPIHelpers::TestLogAssertionContext log_ctx(opts);

  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  // … run the operation that should emit a specific debug line …

  // Read NCCL debug output captured to the temp file.
  std::string nccl_log = log_ctx.readNcclDebugLog();
  EXPECT_TRUE(nccl_log.find("IPC register buffer") != std::string::npos)
    << "Expected IPC registration message in log:\n" << nccl_log;
}
```

### Building and running

```bash
# Build — debug + MPI
./install.sh --debug -t -l --enable-mpi-tests

# Run all MPI tests on 4 ranks (= 4 GPUs)
mpirun -np 4 ./build/debug/test/rccl-UnitTestsMPI

# Run a specific test
mpirun -np 2 ./build/debug/test/rccl-UnitTestsMPI \
  --gtest_filter="P2pMPITest.SimpleSendRecv"

# Run with debug output (all ranks log to file)
NCCL_DEBUG=INFO RCCL_MPI_LOG_ALL_RANKS=1 \
  mpirun -np 4 ./build/debug/test/rccl-UnitTestsMPI \
  --gtest_filter="MyFeatureMPITest.BasicAllReduce"
# Then inspect: rccl_test_rank_2.log

# Multi-node (requires a hostfile)
RCCL_TEST_MPI_HOSTFILE=~/.my_hostfile \
  mpirun -np 16 --hostfile ~/.my_hostfile \
    --map-by slot \
    --mca plm_rsh_agent "ssh -p 2224 -o StrictHostKeyChecking=no -q" \
    --allow-run-as-root \
    ./build/debug/test/rccl-UnitTestsMPI \
    --gtest_filter="NetMPITest.*"
```

---

## 10. Suite F — `rccl-UnitTestsAltRsmi` (AltRsmi Special Linkage, Debug Only)

### What it tests

`src/misc/alt_rsmi.cc` provides an alternative ROCm SMI integration. The file uses internal linkage by default. `rccl-UnitTestsAltRsmi` compiles `alt_rsmi.cc` with `-DARSMI_TEST_BUILD`, which changes some `static` functions to `extern` so tests can call them directly.

### Files to edit

`test/AltRsmiTests.cpp`. If you create a second file add it to `TEST_ALTRSMI_SOURCE_FILES` in `test/CMakeLists.txt`.

### Building and running

```bash
./install.sh --debug -t -l

./build/debug/test/rccl-UnitTestsAltRsmi
./build/debug/test/rccl-UnitTestsAltRsmi --gtest_filter="AltRsmi.*"
```

---

## 11. Suite G — `rccl-UnitTestsFixtures` device/ (GPU Kernel Tests)

### What it tests

Device-side (`__global__`) RCCL kernels. Currently `test/device/TestOp128.cpp` tests `src/device/op128.h` (BytePack load/store primitives). These tests are compiled into **`rccl-UnitTestsFixtures`** (Suite B), not a separate binary.

### Test framework: DeviceTestBase

`test/device/DeviceTestBase.hpp` provides a GTest fixture that:
- Calls `hipSetDevice(0)` in `SetUp()`
- Provides `DeviceBuffer<T>` — a typed RAII wrapper for `hipMalloc`/`hipFree` with `copyFrom()`, `copyTo()`, `upload()`, `download()`, and `zero()` methods
- Provides `gridFor(n)` — computes the `dim3` grid for N elements
- Provides `syncAndCheck()` — calls `hipGetLastError()` + `hipDeviceSynchronize()` and asserts both succeed

**Reference:**
- [HIP programming guide](https://rocm.docs.amd.com/projects/HIP/en/latest/user_guide/programming_guide.html)
- [Google Test](https://google.github.io/googletest/primer.html)

### Files to edit

Create `test/device/MyKernelTests.cpp`. Add it to `TEST_FIXTURE_SOURCE_FILES` in `test/CMakeLists.txt` (the `ROCM_VERSION >= 60400` block) and add any extra include directories needed.

### Adding a test

```cpp
// test/device/MyKernelTests.cpp
#include "DeviceTestBase.hpp"
#include "my_device_header.h"   // the header containing the __device__ / __global__ code

namespace RcclUnitTesting
{

// GPU kernel under test
template<typename T>
__global__ void myKernel(const T* in, T* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = myDeviceOperation(in[i]);
}

class MyKernelTest : public DeviceTestBase {
protected:
  template<typename T>
  void RunTest(const std::vector<T>& h_in, const std::vector<T>& h_expected) {
    const int N = static_cast<int>(h_in.size());
    DeviceBuffer<T> d_in(N), d_out(N);
    d_in.copyFrom(h_in);

    myKernel<<<gridFor(N), kDefaultBlockSize>>>(d_in.ptr, d_out.ptr, N);
    syncAndCheck();   // asserts no HIP errors + device sync

    auto h_out = d_out.copyTo();
    for (int i = 0; i < N; ++i)
      EXPECT_EQ(h_out[i], h_expected[i]) << "at index " << i;
  }
};

TEST_F(MyKernelTest, FloatRoundtrip) {
  std::vector<float> in  = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> exp = {2.0f, 4.0f, 6.0f, 8.0f}; // doubling, for example
  RunTest(in, exp);
}

} // namespace RcclUnitTesting
```

### Building and running

```bash
./install.sh -t -l        # release is fine; or --debug

./build/release/test/rccl-UnitTestsFixtures --gtest_filter="MyKernelTest.*"
```

---

## 12. The Python Test Runner

The Python test runner (`tools/scripts/test_runner/test_runner.py`) is the preferred way to run tests in CI and for systematic test coverage sweeps. It reads a JSON config, builds RCCL if needed, dispatches each test individually with `--gtest_filter`, and reports a summary. It also supports per-test timeouts, env-variable injection, and automatic re-run of failures with an escalated debug environment.

### Quick-start

```bash
cd projects/rccl

# Run the CI precheckin suite (single-node, all collectives)
python3 tools/scripts/test_runner/test_runner.py \
  --config tools/scripts/test_runner/configs/ci-precheckin.json

# Run the full MI300X + InfiniBand suite
python3 tools/scripts/test_runner/test_runner.py \
  --config tools/scripts/test_runner/configs/mi300x_mellanox_ib.json

# Run a single named test from the config
python3 tools/scripts/test_runner/test_runner.py \
  --config tools/scripts/test_runner/configs/ci-precheckin.json \
  --test-name "AllReduce.OutOfPlace"

# Skip the build step (use an existing build)
python3 tools/scripts/test_runner/test_runner.py \
  --config tools/scripts/test_runner/configs/ci-precheckin.json \
  --no-build

# Automatically rerun failures with NCCL_DEBUG=INFO
python3 tools/scripts/test_runner/test_runner.py \
  --config tools/scripts/test_runner/configs/ci-precheckin.json \
  --rerun-failed

# Skip all MPI tests (single-node only)
python3 tools/scripts/test_runner/test_runner.py \
  --config tools/scripts/test_runner/configs/mi300x_mellanox_ib.json \
  --skip-mpi-check

# Use a pre-built RCCL (skips build automatically)
export RCCL_LIB_PATH=/path/to/existing/build
python3 tools/scripts/test_runner/test_runner.py \
  --config tools/scripts/test_runner/configs/ci-precheckin.json

# Generate an LLVM code-coverage report
python3 tools/scripts/test_runner/test_runner.py \
  --config tools/scripts/test_runner/configs/ci-precheckin.json \
  --coverage-report
```

### Adding your test to a config

Open or create a JSON config file. Each test entry inside a `test_configurations` block maps to one `--gtest_filter` invocation:

```jsonc
// tools/scripts/test_runner/configs/ci-precheckin.json  (excerpt)
{
  "test_configurations": {
    "unit_tests": {
      "is_gtest": true,
      "binary": "rccl-UnitTests",
      "num_ranks": 1,
      "num_nodes": 1,
      "num_gpus": 8,
      "timeout": 90,
      "env_variables": { "NCCL_DEBUG": "" },
      "rerun_env_variables": { "NCCL_DEBUG": "INFO" },
      "tests": [
        // ← add your test here:
        {
          "name": "AllReduce.MyNewVariant",
          "description": "My new AllReduce variant",
          "test_filter": "AllReduce.MyNewVariant",
          "timeout": 120
        }
      ]
    },
    "mpi_tests": {
      "is_gtest": true,
      "binary": "rccl-UnitTestsMPI",
      "num_ranks": 4,
      "num_nodes": 1,
      "num_gpus": 4,
      "timeout": 180,
      "env_variables": {
        "NCCL_DEBUG": "INFO",
        "RCCL_MPI_LOG_ALL_RANKS": "1"
      },
      "tests": [
        {
          "name": "MyFeatureMPITest.BasicAllReduce",
          "description": "Basic AllReduce across 4 ranks",
          "test_filter": "MyFeatureMPITest.BasicAllReduce"
        }
      ]
    }
  },
  "test_suites": [
    { "name": "Unit Tests",  "config": "unit_tests",  "enabled": true },
    { "name": "MPI Tests",   "config": "mpi_tests",   "enabled": true }
  ]
}
```

Use `"extends": "base_config_name"` to inherit env vars and other settings from another `test_configurations` entry.

---

## 13. Multi-Node Testing with `mnctl`

`docker/mnctl/` is a pure Python 3 tool (zero dependencies, no pip required) that builds ROCm Docker containers, deploys them to multiple nodes, wires up SSH, and gives you a ready-to-use MPI cluster:

```bash
cd projects/rccl/docker

# ── Single node: build + launch ──
python3 -m mnctl --run
docker exec -it rccl-mn bash

# ── Multi-node (reads ~/.mnctl_hostfile or SLURM_NODELIST automatically) ──
cat > ~/.mnctl_hostfile <<'EOF'
node-a slots=8
node-b slots=8
EOF

python3 -m mnctl --launch-all --ssh    # build, deploy, generate shared SSH keys

# Verify SSH between containers
python3 -m mnctl --verify

# Run rccl-tests or unit tests via MPI
docker exec -it rccl-mn bash
mpirun -np 16 \
  --hostfile ~/.mnctl_hostfile --map-by slot \
  --mca plm_rsh_agent "ssh -p 2224 -o StrictHostKeyChecking=no -q" \
  --allow-run-as-root \
  /workspace/build/debug/test/rccl-UnitTestsMPI \
  --gtest_filter="NetMPITest.*"

# ── Tear down ──
python3 -m mnctl --stop-all
```

**Key features:**
- SLURM auto-detection (reads `SLURM_NODELIST` if no hostfile is found)
- Shared-filesystem detection (skips rsync when nodes share a filesystem)
- SSH key auto-generation or bring-your-own key (`--ssh ~/.ssh/id_rsa`)
- Streaming output from remote containers

Full documentation: `docker/mnctl/README.md`.

---

## 14. Test Infrastructure Reference

### Headers you will use most

| Header | What it provides |
|--------|-----------------|
| `common/TestBed.hpp` | `TestBed` + `RunSimpleSweep()` for collective tests (Suite A) |
| `common/ProcessIsolatedTestRunner.hpp` | `RUN_ISOLATED_TEST*` macros for env-var-sensitive tests (Suites C/D) |
| `common/MPITestBase.hpp` | `MPITestBase` GTest fixture for Suite E |
| `common/MPITestCore.hpp` | Framework-agnostic MPI base (`validateTestPrerequisites`, `createTestCommunicator`, …) |
| `common/MPIHelpers.hpp` | `initializeMPI`, `setupGPU`, `TestLogAssertionContext`, `getRankLogFilePath` |
| `common/MPIEnvironment.hpp` | `MPIEnvironment::world_rank`, `::world_size` |
| `common/MPIStandaloneTest.hpp` | Non-GTest MPI base for benchmarks |
| `common/ResourceGuards.hpp` | `makeScopeGuard`, `makeDeviceBufferAutoGuard`, `NcclRegHandleGuard` |
| `common/DeviceBufferHelpers.hpp` | Typed `DeviceBuffer<T>` with MPI-aware helpers |
| `device/DeviceTestBase.hpp` | `DeviceTestBase`, `DeviceBuffer<T>`, `gridFor`, `syncAndCheck` |
| `common/TestChecks.hpp` | `RCCL_TEST_CHECK`, `HIP_TEST_CHECK`, `MPICHECK`, `TEST_INFO`, `TEST_WARN`, … |
| `common/EnvVars.hpp` | `SetEnvVar` / `UnsetEnvVar` RAII helpers |
| `common/TransportUtils.hpp` | Mock symbol helpers for Suite D |

### Documentation files in the repo

| File | Content |
|------|---------|
| `test/README.md` | Test suite overview |
| `test/common/ProcessIsolatedTestFramework.md` | Full `ProcessIsolatedTestRunner` documentation |
| `test/common/MPITestFramework.md` | Full MPI test framework documentation |
| `docker/mnctl/README.md` | `mnctl` multi-node Docker tool |
| `tools/scripts/test_runner/README.md` | Python test runner documentation |
| `tools/scripts/test_runner/configs/test_config_sample.json` | Annotated JSON config template |

### External links

| Technology | Documentation |
|-----------|--------------|
| Google Test | https://google.github.io/googletest/primer.html |
| Google Mock | https://google.github.io/googletest/gmock_for_dummies.html |
| Open MPI | https://www.open-mpi.org/doc/current/ |
| HIP programming | https://rocm.docs.amd.com/projects/HIP/en/latest/user_guide/programming_guide.html |
| RCCL API | https://rccl.readthedocs.io/en/latest/ |
| ROCm docs | https://rocm.docs.amd.com/ |

---

## 15. Decision Guide — Which Suite?

Use this flowchart when you need to add a test:

```
Does the test exercise a collective operation
(AllReduce, AllGather, Broadcast, …)?
│
├─ YES → Suite A (rccl-UnitTests) — use TestBed::RunSimpleSweep()
│
└─ NO
   │
   Does the test access symbols hidden in Release builds
   (internal functions, non-public struct members, …)?
   │
   ├─ NO
   │   Does the test run GPU kernels from a __global__ function?
   │   ├─ YES → Suite G (device/ under rccl-UnitTestsFixtures)
   │   │         use DeviceTestBase
   │   └─ NO  → Suite B (rccl-UnitTestsFixtures) — plain GTest
   │
   └─ YES (debug-only internals)
       │
       Does the test define fake/mock versions of RCCL symbols
       that will be linked in place of the real ones?
       │
       ├─ YES → Suite D (rccl-UnitTestsMock)
       │
       └─ NO
           │
           Does the test need alt_rsmi.cc compiled with
           ARSMI_TEST_BUILD to expose static variables?
           │
           ├─ YES → Suite F (rccl-UnitTestsAltRsmi)
           │
           └─ NO
               │
               Does the test require multiple GPUs/ranks
               communicating with each other?
               │
               ├─ YES → Suite E (rccl-UnitTestsMPI) — use MPITestBase
               │
               └─ NO
                   │
                   Does the code under test read env vars into
                   static variables (cached on first call)?
                   │
                   ├─ YES → Suite C + ProcessIsolatedTestRunner
                   └─ NO  → Suite C — plain GTest is fine
```

---

## 16. Day-to-Day Cheat Sheet

### Build commands

```bash
# Release (Suites A, B, G)
./install.sh -t -l

# Debug (Suites A–D, F, G)
./install.sh --debug -t -l

# Debug + MPI (all suites)
./install.sh --debug -t -l --enable-mpi-tests
```

### Run commands

```bash
# Suite A — all collective tests
./build/release/test/rccl-UnitTests

# Suite A — one test
./build/release/test/rccl-UnitTests --gtest_filter="AllReduce.OutOfPlace"

# Suite B — header-level fixture tests
./build/release/test/rccl-UnitTestsFixtures

# Suite C — internal-symbol fixture tests (debug only)
./build/debug/test/rccl-UnitTestsFixturesDebug

# Suite D — mock tests (debug only)
./build/debug/test/rccl-UnitTestsMock

# Suite E — MPI tests, 4 ranks
mpirun -np 4 ./build/debug/test/rccl-UnitTestsMPI

# Suite E — one MPI test, with per-rank logging
NCCL_DEBUG=INFO RCCL_MPI_LOG_ALL_RANKS=1 \
  mpirun -np 4 ./build/debug/test/rccl-UnitTestsMPI \
  --gtest_filter="P2pMPITest.SimpleSendRecv"

# Suite F — AltRsmi (debug only)
./build/debug/test/rccl-UnitTestsAltRsmi
```

### Python test runner

```bash
# CI precheckin
python3 tools/scripts/test_runner/test_runner.py \
  --config tools/scripts/test_runner/configs/ci-precheckin.json

# Full MI300X+IB suite, rerun failures with debug env
python3 tools/scripts/test_runner/test_runner.py \
  --config tools/scripts/test_runner/configs/mi300x_mellanox_ib.json \
  --rerun-failed --verbose

# Single test by name
python3 tools/scripts/test_runner/test_runner.py \
  --config tools/scripts/test_runner/configs/ci-precheckin.json \
  --test-name "AllReduce.OutOfPlace"

# Skip build (use existing)
python3 tools/scripts/test_runner/test_runner.py \
  --config tools/scripts/test_runner/configs/ci-precheckin.json \
  --no-build
```

### GTest flags

```bash
# List all tests without running
./rccl-UnitTests --gtest_list_tests

# Run tests matching a pattern
./rccl-UnitTests --gtest_filter="AllReduce*:Broadcast*"

# Exclude a pattern
./rccl-UnitTests --gtest_filter="-AllReduce.Channels"

# Stop on first failure
./rccl-UnitTests --gtest_fail_fast

# Repeat N times (for flakiness hunting)
./rccl-UnitTests --gtest_repeat=10 --gtest_filter="AllReduce.OutOfPlace"

# Also run disabled tests (DISABLED_ prefix)
./rccl-UnitTests --gtest_also_run_disabled_tests
```

### Useful environment variables

| Variable | Effect |
|----------|--------|
| `NCCL_DEBUG=INFO` | Enable RCCL info-level logging (also enables `TEST_INFO` in MPI tests) |
| `NCCL_DEBUG=WARN` | Enable `TEST_WARN` in MPI tests |
| `NCCL_DEBUG=TRACE` | Enable all `TEST_*` macros |
| `NCCL_DEBUG_SUBSYS=REG` | Limit NCCL debug output to the registration subsystem |
| `RCCL_MPI_LOG_ALL_RANKS=1` | Redirect each MPI rank's stderr to `rccl_test_rank_<N>.log` |
| `RCCL_TEST_MPI_HOSTFILE` | Path to hostfile for the Python test runner's MPI tests |
| `RCCL_LIB_PATH` or `RCCL_BUILD_DIR` | Pre-built RCCL directory; skips build in `test_runner.py` |
| `MPI_PATH` | Override MPI installation directory (default: `/opt/ompi`) |
| `ROCM_PATH` | Override ROCm installation directory (default: `/opt/rocm`) |
| `HSA_NO_SCRATCH_RECLAIM=1` | Suppress GPU scratch-memory reclaim; recommended for stability |
| `UT_POW2_GPUS=1` | Restrict tests to power-of-two GPU counts |

