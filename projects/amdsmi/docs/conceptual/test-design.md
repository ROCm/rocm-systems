---
myst:
  html_meta:
    "description lang=en": "AMD SMI test suite design: structure, conventions, and migration guide for C++ and Python tests."
    "keywords": "system, management, interface, test, unit, functional, gpu, cpu, nic, ifoe, amdsmitst, googletest, unittest, design"
---

# Test suite design

This document describes the design of the AMD SMI test suite: how tests are organized by
component and test type, the naming conventions used, how to build and run subsets of tests,
and the migration path from the previous flat file layout.

## Goals

The test suite redesign has four goals:

1. **Discoverability** — A developer looking for power-management tests should immediately know
   where to find them without searching the codebase.
2. **Scalability** — Adding tests for a new feature or component requires adding one file in the
   correct location; nothing else changes.
3. **Selective execution** — CI and developers can run only the tests relevant to a component or
   feature without maintaining manual filter lists.
4. **Test-type clarity** — Unit tests (no hardware) and functional tests (require hardware) are
   structurally separated so they can run in different environments.

## Component taxonomy

| Component | Abbreviation | Coverage |
| :--- | :--- | :--- |
| GPU | `gpu` | AMD Instinct GPU accelerators |
| CPU | `cpu` | AMD EPYC CPU sockets and cores |
| NIC / Switch | `nic` | Network interface cards and PCIe switches |
| Infinity Fabric over Ethernet | `ifoe` | IFoE links and endpoints |
| System | `system` | Multi-device topology, initialization, host platform |

`switch` devices are not a separate component; their tests live under `nic/`.

## Test type taxonomy

| Type | Directory | Hardware required | Framework |
| :--- | :--- | :--- | :--- |
| **Unit** | `unit/` | No — pure logic, static data, no device calls | C++: `TEST()` macro · Python: `unittest` |
| **Functional** | `functional/` | Yes — runs against a live device | C++: `TestBase` lifecycle · Python: `unittest` |

Performance benchmarks belong in `functional/` because they require a real device to produce
meaningful timing data.

## C++ test structure

### Directory layout

```text
tests/amd_smi_test/
├── main.cc                          # GTest entry point; registers all TestBase tests
├── test_base.{h,cc}                 # TestBase lifecycle (SetUp / Run / Close / DisplayResults)
├── test_common.{h,cc}               # Verbosity macros, enum-to-string helpers
├── test_utils.{h,cc}                # Additional enum helpers
├── amdsmitst.exclude                # Global ASIC blacklist for --gtest_filter
├── detect_asic_filter.sh            # ASIC detection and per-ASIC exclusion
│
├── unit/                            # No hardware required; pure TEST() macro tests
│   └── gpu/
│       ├── dynamic_metrics.cc       # Metric struct versioning and compatibility checks
│       ├── cper_read.cc             # CPER read path: synthetic edge cases (no fixtures)
│       ├── mock_cper.cc             # CPER parse/severity filtering vs mock_cper/ fixtures
│       └── mock_cper/               # Sanitized real CPER fixtures used by mock_cper.cc
│           ├── README.md            # Fixture provenance and scrubbing notes
│           ├── sanitize_cper.py     # Regenerates fixtures from raw captures
│           ├── cper_corrected.cper
│           ├── cper_fatal.cper
│           ├── cper_mixed.cper
│           └── cper_uncorrected.cper
│
└── functional/                      # Requires live hardware; uses TestBase lifecycle
    ├── gpu/
    │   ├── clock/
    │   │   ├── frequencies_read.{h,cc}
    │   │   └── frequencies_read_write.{h,cc}
    │   ├── events/
    │   │   └── evt_notif_read_write.{h,cc}
    │   ├── identity/
    │   │   ├── id_info_read.{h,cc}
    │   │   └── version_read.{h,cc}
    │   ├── memory/
    │   │   ├── mem_page_info_read.{h,cc}
    │   │   ├── mem_util_read.{h,cc}
    │   │   └── memory_read_write.{h,cc}
    │   ├── metrics/
    │   │   ├── gpu_busy_read.{h,cc}
    │   │   ├── gpu_cache_read.{h,cc}
    │   │   ├── gpu_metrics_read.{h,cc}
    │   │   ├── gpu_partition_metrics_read.{h,cc}
    │   │   ├── metrics_counter_read.{h,cc}
    │   │   └── process_info_read.{h,cc}
    │   ├── partition/
    │   │   ├── computepartition_read_write.{h,cc}
    │   │   ├── computepartition_memallocmode_read_write.{h,cc}
    │   │   └── memorypartition_read_write.{h,cc}
    │   ├── pci/
    │   │   └── pci_read_write.{h,cc}
    │   ├── perf/
    │   │   ├── overdrive_read.{h,cc}
    │   │   ├── overdrive_read_write.{h,cc}
    │   │   ├── perf_cntr_read_write.{h,cc}
    │   │   ├── perf_determinism.{h,cc}
    │   │   ├── perf_level_read.{h,cc}
    │   │   ├── perf_level_read_write.{h,cc}
    │   │   ├── volt_freq_curv_read.{h,cc}
    │   │   └── volt_read.{h,cc}
    │   ├── power/
    │   │   ├── power_cap_read_write.{h,cc}
    │   │   ├── power_read.{h,cc}
    │   │   └── power_read_write.{h,cc}
    │   ├── ras/
    │   │   └── err_cnt_read.{h,cc}
    │   ├── thermal/
    │   │   ├── fan_read.{h,cc}
    │   │   ├── fan_read_write.{h,cc}
    │   │   └── temp_read.{h,cc}
    │   └── xgmi/
    │       └── xgmi_read_write.{h,cc}
    ├── system/
    │   ├── cross_process_serialization.{h,cc}
    │   ├── hw_topology_read.{h,cc}
    │   ├── init_shutdown_refcount.{h,cc}
    │   ├── kfd_atfork_read.{h,cc}
    │   ├── mutual_exclusion.{h,cc}
    │   └── sys_info_read.{h,cc}
    ├── cpu/
    │   ├── clock/
    │   │   └── placeholder.cc       # Stub — CPU clock C++ tests added here
    │   └── power/
    │       └── placeholder.cc       # Stub — CPU power C++ tests added here
    ├── nic/
    │   ├── discovery/
    │   │   └── placeholder.cc       # Stub — NIC/switch discovery C++ tests added here
    │   └── identity/
    │       └── placeholder.cc       # Stub — NIC/switch identity C++ tests added here
    └── ifoe/
        ├── fabric/
        │   └── fabric_read.{h,cc}   # IFoE fabric link reads
        └── identity/
            └── ifoe_info_read.{h,cc} # IFoE endpoint info reads
```

### Component subdirectory depth

Each component groups tests into per-feature subdirectories (`<component>/<feature>/`, like `gpu/`).
A feature gets its own subdirectory even for a single test. A `placeholder.cc` holds a feature
directory until its first real test lands.

The names below are suggestions that mirror the Python suite's `test_<feature>.py` files. Sharing
names lets a feature line up across both suites. Adapt them as the APIs warrant.

| Component | Suggested feature subdirectories |
| :--- | :--- |
| `cpu/` | `clock/`, `dimm/`, `energy/`, `hsmp/`, `identity/`, `power/`, `thermal/` |
| `nic/` | `discovery/`, `identity/` |
| `ifoe/` | `fabric/`, `identity/` |

### Naming conventions

**Files**: `{feature}_{operation}.{h|cc}` where operation is `read`, `read_write`, or a descriptive
term such as `perf_determinism` or `dynamic_metrics`.

**Classes**: `Test{FeatureName}{Operation}` derived from `TestBase` for functional tests; plain
`TEST(Suite, Name)` for unit tests.

**GTest suites registered in `main.cc`**:

| Suite | Type | When used |
| :--- | :--- | :--- |
| `GpuFunctionalReadOnly` | functional | GPU tests that only read device state; no root required |
| `GpuFunctionalReadWrite` | functional | GPU tests that modify device state; root typically required |
| `GpuUnit` | unit | Pure unit tests under `unit/gpu/`; no device required |

The suite name scheme is `<Component><Type><Operation>`, making component, type, and operation
all independently filterable via `--gtest_filter` wildcards.

### Mocked unit tests and fixtures

Mocking is a *technique*, not a separate test level: a test that supplies canned input instead
of touching hardware is still a **unit test**. Mocked and non-mocked unit tests therefore live
side by side under `unit/<component>/`, organized by *what* they test rather than *how* they are
isolated.

The cper suite shows both styles:

- `unit/gpu/cper_read.cc` — builds CPER byte blobs in memory at runtime (no fixtures); covers
  read-path edge cases and error handling (zero-size file, empty ring, partial reads, buffer
  overflow, invalid args).
- `unit/gpu/mock_cper.cc` — drives the same API against committed `.cper` fixtures and validates
  record counting and severity-mask filtering on realistic records.

**To add a mocked unit test**, follow the `mock_cper` pattern:

1. Put the test at `unit/<component>/mock_<feature>.cc` and its fixtures in a sibling
   `unit/<component>/mock_<feature>/` folder — one folder per test, no shared catch-all.
2. Load fixtures relative to the per-test folder; the existing `MockDir()` helper finds it
   next to the installed binary or via the `AMDSMI_TEST_MOCK_DIR` build define.
3. Add an `install(DIRECTORY …)` line for the new fixture folder in `CMakeLists.txt`.

Fixtures should be static, sanitized blobs so the tests stay deterministic and **run on any
machine regardless of GPU** (see `mock_cper/README.md` for provenance).

### CMake integration

`tests/amd_smi_test/CMakeLists.txt` uses `file(GLOB_RECURSE ... CONFIGURE_DEPENDS)` to collect all
sources under `unit/` and `functional/` automatically. `CONFIGURE_DEPENDS` re-globs at build time,
so a new test file added to any subdirectory is picked up on the next build with no manual `cmake`
re-run:

```cmake
file(GLOB_RECURSE unitSources  CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cc)
file(GLOB_RECURSE functSources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/functional/*.cc)

add_executable(amdsmitst
    main.cc test_base.cc test_common.cc test_utils.cc
    ${unitSources} ${functSources}
)
```

`main.cc` include paths use the new subdirectory structure:

```cpp
// Before
#include "functional/power_read_write.h"

// After
#include "functional/gpu/power/power_read_write.h"
```

### Running C++ tests

All examples use `amdsmitst` directly with `--gtest_filter`. The binary is at
`<build>/tests/amd_smi_test/amdsmitst` or `/opt/rocm/share/amd_smi/tests/amdsmitst` after install.

The suite name scheme `<Component><Type><Operation>` makes every axis independently filterable:
- **Component**: `Gpu*`, `Cpu*`, `Nic*`, `Ifoe*`, `System*`
- **Type**: `*Unit*`, `*Functional*`
- **Operation**: `*ReadOnly*`, `*ReadWrite*`

```shell
# List all available tests
./amdsmitst --gtest_list_tests

# All tests
sudo ./amdsmitst
```

**By type:**

```shell
# Unit tests only — no hardware required
./amdsmitst --gtest_filter="*Unit*"

# All functional tests (read-only + read/write)
sudo ./amdsmitst --gtest_filter="*Functional*"

# Functional read-only only — no root required
./amdsmitst --gtest_filter="*FunctionalReadOnly*"

# Functional read/write only — root required
sudo ./amdsmitst --gtest_filter="*FunctionalReadWrite*"
```

**By component:**

```shell
# All GPU tests (unit + functional)
sudo ./amdsmitst --gtest_filter="Gpu*"

# GPU functional only
sudo ./amdsmitst --gtest_filter="GpuFunctional*"

# GPU unit only
./amdsmitst --gtest_filter="GpuUnit*"

# CPU tests (when added)
./amdsmitst --gtest_filter="Cpu*"
```

**By feature** (wildcard on test name, `:` is OR, `-` negates):

```shell
# Thermal (fan + temperature)
./amdsmitst --gtest_filter="*.*Fan*:*.*Temp*"

# Power
sudo ./amdsmitst --gtest_filter="*.*Power*"

# Clock / frequency
sudo ./amdsmitst --gtest_filter="*.*Freq*"

# Partition
sudo ./amdsmitst --gtest_filter="*.*Partition*"

# RAS / ECC
./amdsmitst --gtest_filter="*.*Err*"

# XGMI
sudo ./amdsmitst --gtest_filter="*.*XGMI*"
```

**Combining filters:**

```shell
# GPU functional read-only power tests
./amdsmitst --gtest_filter="GpuFunctionalReadOnly.*Power*"

# All functional except partition
sudo ./amdsmitst --gtest_filter="*Functional*:-*.*Partition*"
```

**Apply ASIC-specific exclusions** (as done in CI):

```shell
source amdsmitst.exclude
source detect_asic_filter.sh
./amdsmitst --gtest_filter="-${GTEST_EXCLUDE}"
```

## Python test structure

### Directory layout

```text
tests/python/
├── common/
│   ├── __init__.py
│   ├── common.py                      # Common base class, device enumeration, error mapping, runner machinery
│   └── runcmd.py                      # CLI subprocess wrapper
│
├── unit/                              # No hardware required — pure logic tests only
│   ├── __init__.py
│   ├── gpu/
│   │   ├── test_apu_metrics.py            # APU metrics interface helpers (unit conversions, N/A parity)
│   │   └── test_cli_metric_partition.py   # amd-smi metric --partition clock assembly (mock-based, stubs amdsmi)
│   └── system/
│       └── test_bdf.py                # BDF string parsing and formatting
│
├── functional/                        # Requires live hardware
│   ├── __init__.py
│   ├── gpu/
│   │   ├── test_clock.py              # clk_freq get/set, clock_info, PCI bandwidth get/set
│   │   ├── test_events.py             # gpu_event, gpu_counter
│   │   ├── test_identity.py           # asic_info, board_info, id, revision, vbios
│   │   ├── test_memory.py             # vram, bad_pages, reserved pages, UMA carveout, TTM
│   │   ├── test_metrics.py            # gpu_metrics, pm_metrics, partition metrics
│   │   ├── test_partition.py          # compute_partition, memory_partition, set, xgmi_plpd
│   │   ├── test_pci.py                # PCI bandwidth, throughput, replay counter
│   │   ├── test_overdrive.py          # perf_level, overdrive, od_volt, set_overdrive, set_perf_level
│   │   ├── test_power.py              # power_profile, power_cap get/set
│   │   ├── test_process.py            # process_info
│   │   ├── test_ras.py                # ras, ecc error counts
│   │   ├── test_system.py             # GPU system-level info
│   │   ├── test_thermal.py            # fan_rpms, fan_speed get/set
│   │   ├── test_xgmi.py               # xgmi_link, xcd_counter
│   │   └── test_benchmark.py          # per-API latency benchmarks with timing assertions
│   ├── cpu/
│   │   ├── test_clock.py              # clk_freq, clock_info, core_current_freq, fclk_mclk, soc_pstate
│   │   ├── test_dimm.py               # dimm_power, dimm_thermal, dimm_temp_range
│   │   ├── test_energy.py             # core_energy, socket_energy, energy_count
│   │   ├── test_hsmp.py               # hsmp_driver_version, hsmp_proto_ver, esmi_err_msg
│   │   ├── test_identity.py           # CPU socket identity
│   │   ├── test_power.py              # socket_power, power_cap get/set, boostlimit set
│   │   ├── test_thermal.py            # socket_temperature, prochot_status
│   │   └── test_benchmark.py          # per-API latency benchmarks with timing assertions
│   ├── nic/
│   │   ├── test_discovery.py          # NIC and switch BDF/device discovery (live enumeration)
│   │   └── test_identity.py           # NIC and switch BDF and device ID reads
│   ├── ifoe/
│   │   ├── test_discovery.py          # IFoE endpoint enumeration
│   │   └── test_identity.py           # IFoE endpoint BDF and device ID reads
│   └── system/
│       ├── test_init.py               # amdsmi init / shutdown lifecycle
│       └── test_topology.py           # socket, processor, and utilization count discovery
│
├── cli/                               # CLI tests — one module per command (command-first: a
│   │                                  # single command spans GPU/CPU/NIC; class Test<Command>)
│   ├── __init__.py
│   ├── base.py                        # TestCliBase: shared scaffolding + cached setUpClass (--json baseline)
│   ├── test_general.py                # help, invalid-args
│   ├── test_default.py                # bare amd-smi (default command)
│   ├── test_version.py
│   ├── test_list.py
│   ├── test_static.py                 # static (+ mem-carveout)
│   ├── test_firmware.py
│   ├── test_bad_pages.py
│   ├── test_metric.py
│   ├── test_monitor.py
│   ├── test_process.py
│   ├── test_event.py
│   ├── test_set.py
│   ├── test_reset.py
│   ├── test_topology.py
│   ├── test_xgmi.py
│   ├── test_partition.py
│   ├── test_node.py
│   ├── test_fabric.py
│   └── test_ras.py                    # ras (+ afid folder)
│
├── integration_test.py               # Runner: discovers and runs functional/ tests
├── cli_unit_test.py                  # Runner: discovers and runs cli/ tests
├── unit_tests.py                     # Runner: discovers and runs unit/ tests
├── CMakeLists.txt                    # Installs this tree into the python_unittest/ path
└── README.md                         # Prerequisites + pointer to this design doc
```

### Naming conventions

**Files**: `test_{feature}.py` within the appropriate component subdirectory.

**Classes**: One `unittest.TestCase` subclass per file, named `Test{Component}{Feature}`
(for example, `TestGpuPower`, `TestCpuClock`, `TestSystemInit`). Classes inherit from
`common.common.Common` for device enumeration and error-handling helpers.

**Methods**: `test_{operation}[_{qualifier}]` (for example, `test_get_power_cap`,
`test_set_power_cap_dry_run`).

### Running Python tests

Three top-level runner scripts install under `python_unittest/`, keeping the same path as before.
All support `-v`, `-b`, `-q`, `-k "pattern"` (include), and `-x "pattern"` (exclude — the inverse
of `-k`, skips tests whose id contains the pattern). Run from source by substituting
`tests/python/` for the install path.

**List all available tests** (no hardware, no execution):

```shell
/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py --list
/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py --list
/opt/rocm/share/amd_smi/tests/python_unittest/cli_unit_test.py --list
```

**All unit tests** (no hardware required):

```shell
/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py -v
/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py -b -v
/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py -k "bdf" -v
/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py -k "metric" -v
```

**All functional (integration) tests** (live hardware, root may be required):

```shell
/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -v
/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -b -v
/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -k "power" -v
```

**All CLI tests**:

```shell
/opt/rocm/share/amd_smi/tests/python_unittest/cli_unit_test.py -v
/opt/rocm/share/amd_smi/tests/python_unittest/cli_unit_test.py -k "gpu" -v
```

**Narrow to a single feature file** with `-k` on the matching runner. The individual
`test_*.py` files are *not* meant to be run directly — they are plain unittest modules that
the three runners discover, import, and execute. The runners own path resolution, verbosity,
filtering, and the amdsmi import, so a leaf file has no `sys.path` bootstrap of its own;
invoking one with `python test_power.py` simply raises `ModuleNotFoundError: common`, the same
as running a pytest test file directly. Always go through a runner with a `-k` filter:

```shell
# functional/gpu/test_power.py -> integration_test.py with a -k filter
/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -k "functional.gpu.test_power" -v
/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -k "test_power" -v

# unit/system/test_bdf.py -> unit_tests.py
/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py -k "unit.system.test_bdf" -v
```

**Equivalent matrix between Python and C++.**  
The two suites are structurally asymmetric: Python has **three independent runners** (`unit_tests.py`, `integration_test.py`, `cli_unit_test.py`),
while C++ is a **single `amdsmitst` binary** filtered with `--gtest_filter`. Some concepts map only
one way — **CLI tests are Python-only**, and the **read-only/read-write split is C++-only**.

| Intent | Python | C++ (`amdsmitst`) |
| :--- | :--- | :--- |
| List all tests | `--list` / `-l` on each runner (`unit_tests.py`, `integration_test.py`, `cli_unit_test.py`) | `--gtest_list_tests` |
| Unit only (no hardware) | `unit_tests.py -v` | `--gtest_filter="*Unit*"` |
| All functional | `integration_test.py -v` | `--gtest_filter="*Functional*"` |
| Functional read-only / read-write | Not distinguished — Python groups functional tests by component/feature, not by RO/RW | `--gtest_filter="*FunctionalReadOnly*"` / `"*FunctionalReadWrite*"` |
| CLI tests | `cli_unit_test.py -v` | _Python-only — no C++ equivalent_ |
| Feature filter (e.g. power) | `integration_test.py -k power -v` (use the runner that owns that test type) | `--gtest_filter="*.*Power*"` |
| Exclude / negate | `integration_test.py -x partition -v` (skip tests whose id contains `partition`) | `--gtest_filter="-*.*Partition*"` |
| Everything | `unit_tests.py -v && integration_test.py -v && cli_unit_test.py -v` | `./amdsmitst` |

### CMake integration

`tests/python/CMakeLists.txt` installs its own `tests/python/` tree into the legacy
`python_unittest/` install path. The source directory was consolidated into `tests/python/`,
but the install location is unchanged, so existing invocations keep working:

```cmake
install(
    DIRECTORY ./
    DESTINATION ${SHARE_INSTALL_PREFIX}/tests/python_unittest/
    COMPONENT ${TESTS_COMPONENT}
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING
    PATTERN "*.py"
)
```

The top-level `CMakeLists.txt` wires this in with `add_subdirectory("tests/python")`.

## Migration reference

### C++ file mapping

| Old path (`tests/amd_smi_test/`) | New path (`tests/amd_smi_test/`) |
| :--- | :--- |
| `functional/api_support_read.{h,cc}` | _Removed_ — amd-smi has no supported-function iterator API to exercise (support is reported per call via `AMDSMI_STATUS_NOT_SUPPORTED`), so the ported test had an empty `Run()`. |
| `functional/computepartition_read_write.{h,cc}` | `functional/gpu/partition/computepartition_read_write.{h,cc}` |
| `functional/dynamic_metrics_test.cc` | `unit/gpu/dynamic_metrics.cc` |
| `functional/cper_read.cc` | `unit/gpu/cper_read.cc` |
| `functional/mock_cper.cc` | `unit/gpu/mock_cper.cc` |
| `functional/mock_values/` | `unit/gpu/mock_cper/` |
| `functional/cross_process_serialization.{h,cc}` | `functional/system/cross_process_serialization.{h,cc}` |
| `functional/kfd_atfork_read.{h,cc}` | `functional/system/kfd_atfork_read.{h,cc}` |
| `functional/fabric_read.{h,cc}` | `functional/ifoe/fabric_read.{h,cc}` |
| `functional/ifoe_info_read.{h,cc}` | `functional/ifoe/ifoe_info_read.{h,cc}` |
| `functional/computepartition_memallocmode_read_write.{h,cc}` | `functional/gpu/partition/computepartition_memallocmode_read_write.{h,cc}` |
| `functional/err_cnt_read.{h,cc}` | `functional/gpu/ras/err_cnt_read.{h,cc}` |
| `functional/evt_notif_read_write.{h,cc}` | `functional/gpu/events/evt_notif_read_write.{h,cc}` |
| `functional/fan_read.{h,cc}` | `functional/gpu/thermal/fan_read.{h,cc}` |
| `functional/fan_read_write.{h,cc}` | `functional/gpu/thermal/fan_read_write.{h,cc}` |
| `functional/frequencies_read.{h,cc}` | `functional/gpu/clock/frequencies_read.{h,cc}` |
| `functional/frequencies_read_write.{h,cc}` | `functional/gpu/clock/frequencies_read_write.{h,cc}` |
| `functional/gpu_busy_read.{h,cc}` | `functional/gpu/metrics/gpu_busy_read.{h,cc}` |
| `functional/gpu_cache_read.{h,cc}` | `functional/gpu/metrics/gpu_cache_read.{h,cc}` |
| `functional/gpu_metrics_read.{h,cc}` | `functional/gpu/metrics/gpu_metrics_read.{h,cc}` |
| `functional/gpu_partition_metrics_read.{h,cc}` | `functional/gpu/metrics/gpu_partition_metrics_read.{h,cc}` |
| `functional/hw_topology_read.{h,cc}` | `functional/system/hw_topology_read.{h,cc}` |
| `functional/id_info_read.{h,cc}` | `functional/gpu/identity/id_info_read.{h,cc}` |
| `functional/init_shutdown_refcount.{h,cc}` | `functional/system/init_shutdown_refcount.{h,cc}` |
| `functional/mem_page_info_read.{h,cc}` | `functional/gpu/memory/mem_page_info_read.{h,cc}` |
| `functional/mem_util_read.{h,cc}` | `functional/gpu/memory/mem_util_read.{h,cc}` |
| `functional/memory_read_write.{h,cc}` | `functional/gpu/memory/memory_read_write.{h,cc}` |
| `functional/memorypartition_read_write.{h,cc}` | `functional/gpu/partition/memorypartition_read_write.{h,cc}` |
| `functional/metrics_counter_read.{h,cc}` | `functional/gpu/metrics/metrics_counter_read.{h,cc}` |
| `functional/mutual_exclusion.{h,cc}` | `functional/system/mutual_exclusion.{h,cc}` |
| `functional/overdrive_read.{h,cc}` | `functional/gpu/perf/overdrive_read.{h,cc}` |
| `functional/overdrive_read_write.{h,cc}` | `functional/gpu/perf/overdrive_read_write.{h,cc}` |
| `functional/pci_read_write.{h,cc}` | `functional/gpu/pci/pci_read_write.{h,cc}` |
| `functional/perf_cntr_read_write.{h,cc}` | `functional/gpu/perf/perf_cntr_read_write.{h,cc}` |
| `functional/perf_determinism.{h,cc}` | `functional/gpu/perf/perf_determinism.{h,cc}` |
| `functional/perf_level_read.{h,cc}` | `functional/gpu/perf/perf_level_read.{h,cc}` |
| `functional/perf_level_read_write.{h,cc}` | `functional/gpu/perf/perf_level_read_write.{h,cc}` |
| `functional/power_cap_read_write.{h,cc}` | `functional/gpu/power/power_cap_read_write.{h,cc}` |
| `functional/power_read.{h,cc}` | `functional/gpu/power/power_read.{h,cc}` |
| `functional/power_read_write.{h,cc}` | `functional/gpu/power/power_read_write.{h,cc}` |
| `functional/process_info_read.{h,cc}` | `functional/gpu/metrics/process_info_read.{h,cc}` |
| `functional/sys_info_read.{h,cc}` | `functional/system/sys_info_read.{h,cc}` |
| `functional/temp_read.{h,cc}` | `functional/gpu/thermal/temp_read.{h,cc}` |
| `functional/version_read.{h,cc}` | `functional/gpu/identity/version_read.{h,cc}` |
| `functional/volt_freq_curv_read.{h,cc}` | `functional/gpu/perf/volt_freq_curv_read.{h,cc}` |
| `functional/volt_read.{h,cc}` | `functional/gpu/perf/volt_read.{h,cc}` |
| `functional/xgmi_read_write.{h,cc}` | `functional/gpu/xgmi/xgmi_read_write.{h,cc}` |

### Python file mapping

One row per old file. Monolithic files were split into one `test_<feature>.py`
per feature under the matching component directory; representative examples are
shown in parentheses.

| Old file (`tests/python_unittest/`) | New location (`tests/python/`) |
| :--- | :--- |
| `unit_tests.py` | `unit/<component>/test_<feature>.py` (e.g. `unit/system/test_bdf.py`, `unit/gpu/test_apu_metrics.py`) |
| `integration_test.py` | `functional/<component>/test_<feature>.py` (e.g. `functional/system/test_init.py`, `functional/gpu/test_power.py`, `functional/nic/test_discovery.py`) |
| `partition_metric_unit_test.py` | `unit/gpu/test_cli_metric_partition.py` |
| `cli_unit_test.py` | `cli/test_<command>.py`, one per command (shared scaffolding in `cli/base.py`) |
| `perf_tests.py` | `functional/gpu/test_benchmark.py` |
| `perf_cputests.py` | `functional/cpu/test_benchmark.py` |
| `common.py` | `common/common.py` |
| `runcmd.py` | `common/runcmd.py` |
