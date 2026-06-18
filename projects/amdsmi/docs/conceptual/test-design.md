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
tests/cpp/
├── main.cc                          # GTest entry point; registers all TestBase tests
├── test_base.{h,cc}                 # TestBase lifecycle (SetUp / Run / Close / DisplayResults)
├── test_common.{h,cc}               # Verbosity macros, enum-to-string helpers
├── test_utils.{h,cc}                # Additional enum helpers
├── amdsmitst.exclude                # Global ASIC blacklist for --gtest_filter
├── detect_asic_filter.sh            # ASIC detection and per-ASIC exclusion
│
├── unit/                            # No hardware required; pure TEST() macro tests
│   └── gpu/
│       └── dynamic_metrics.cc       # Metric struct versioning and compatibility checks
│
└── functional/                      # Requires live hardware; uses TestBase lifecycle
    ├── gpu/
    │   ├── clock/
    │   │   ├── frequencies_read.{h,cc}
    │   │   └── frequencies_read_write.{h,cc}
    │   ├── events/
    │   │   └── evt_notif_read_write.{h,cc}
    │   ├── identity/
    │   │   ├── api_support_read.{h,cc}
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
    │   ├── hw_topology_read.{h,cc}
    │   ├── init_shutdown_refcount.{h,cc}
    │   ├── mutual_exclusion.{h,cc}
    │   └── sys_info_read.{h,cc}
    ├── cpu/
    │   └── placeholder.cc           # Stub — CPU C++ tests added here
    ├── nic/
    │   └── placeholder.cc           # Stub — NIC and switch C++ tests added here
    └── ifoe/
        └── placeholder.cc           # Stub — IFoE C++ tests added here
```

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

### CMake integration

`tests/cpp/CMakeLists.txt` uses `file(GLOB_RECURSE ...)` to collect all sources under
`unit/` and `functional/` automatically, so adding a new file to any subdirectory requires no
CMake change:

```cmake
file(GLOB_RECURSE unitSources  ${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cc)
file(GLOB_RECURSE functSources ${CMAKE_CURRENT_SOURCE_DIR}/functional/*.cc)

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
`<build>/tests/cpp/amdsmitst` or `/opt/rocm/share/amd_smi/tests/amdsmitst` after install.

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
│   ├── helpers.py                     # Common base class, device enumeration, error mapping
│   └── runcmd.py                      # CLI subprocess wrapper
│
├── unit/                              # No hardware required — pure logic tests only
│   ├── __init__.py
│   └── bdf/
│       └── test_bdf.py                # BDF string parsing, formatting, and error-code helpers
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
│   │   ├── test_perf.py               # perf_level, overdrive, od_volt, set_overdrive, set_perf_level
│   │   ├── test_power.py              # power_profile, power_cap get/set
│   │   ├── test_process.py            # process_info
│   │   ├── test_ras.py                # ras, ecc error counts
│   │   ├── test_system.py             # GPU system-level info
│   │   ├── test_thermal.py            # fan_rpms, fan_speed get/set
│   │   └── test_xgmi.py               # xgmi_link, xcd_counter
│   ├── cpu/
│   │   ├── test_clock.py              # clk_freq, clock_info, core_current_freq, fclk_mclk, soc_pstate
│   │   ├── test_dimm.py               # dimm_power, dimm_thermal, dimm_temp_range
│   │   ├── test_energy.py             # core_energy, socket_energy, energy_count
│   │   ├── test_hsmp.py               # hsmp_driver_version, hsmp_proto_ver, esmi_err_msg
│   │   ├── test_identity.py           # CPU socket identity
│   │   ├── test_power.py              # socket_power, power_cap get/set, boostlimit set
│   │   └── test_thermal.py            # socket_temperature, prochot_status
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
└── cli/
    ├── __init__.py
    ├── test_cli_common.py             # help, version, list, default, invalid commands
    ├── test_cli_gpu.py                # static, firmware, metric, process, event, bad_pages, set, reset
    ├── test_cli_cpu.py                # CPU-specific CLI commands
    └── test_cli_topology.py           # xgmi, topology, partition, ras, node
```

### Naming conventions

**Files**: `test_{feature}.py` within the appropriate component subdirectory.

**Classes**: One `unittest.TestCase` subclass per file, named `Test{Component}{Feature}`
(for example, `TestGpuPower`, `TestCpuClock`, `TestSystemInit`). Classes inherit from
`common.helpers.Common` for device enumeration and error-handling helpers.

**Methods**: `test_{operation}[_{qualifier}]` (for example, `test_get_power_cap`,
`test_set_power_cap_dry_run`).

### Running Python tests

Three top-level runner scripts install under `python_unittest/`, keeping the same path as before.
All support `-v`, `-b`, `-q`, and `-k "pattern"`. Run from source by substituting
`tests/python/` for the install path.

**List all available tests** (no hardware, no execution):

```shell
/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py --list
/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py --list
/opt/rocm/share/amd_smi/tests/python_unittest/cli_unit_test.py --list
```

**All unit tests** (no hardware required — currently only `bdf/`):

```shell
/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py -v
/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py -b -v
/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py -k "bdf" -v
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

**Run a single feature file directly** (individual test files are also executable):

```shell
/opt/rocm/share/amd_smi/tests/python_unittest/unit/gpu/test_clock.py -v
/opt/rocm/share/amd_smi/tests/python_unittest/unit/gpu/test_clock.py -k freq -v
/opt/rocm/share/amd_smi/tests/python_unittest/functional/gpu/test_power.py -v
```

**Equivalent matrix between Python and C++:**

| Intent | Python | C++ (`amdsmitst`) |
| :--- | :--- | :--- |
| List all tests | `unit_tests.py --list` / `-l` | `--gtest_list_tests` |
| Unit only (no hardware) | `unit_tests.py -v` | `--gtest_filter="*Unit*"` |
| All functional | `integration_test.py -v` | `--gtest_filter="*Functional*"` |
| Functional read-only | `integration_test.py -v` | `--gtest_filter="*FunctionalReadOnly*"` |
| Functional read/write | `integration_test.py -v` | `--gtest_filter="*FunctionalReadWrite*"` |
| CLI tests | `cli_unit_test.py -v` | N/A |
| Feature filter | `unit_tests.py -k power -v` | `--gtest_filter="*.*Power*"` |
| All tests | `unit_tests.py -v && integration_test.py -v` | `./amdsmitst` |

### CMake integration

`tests/python_unittest/CMakeLists.txt` installs the `tests/python/` source tree into the
`python_unittest/` install path, so the install location stays consistent with the original:

```cmake
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../python/
    DESTINATION ${SHARE_INSTALL_PREFIX}/tests/python_unittest/
    COMPONENT ${TESTS_COMPONENT}
    USE_SOURCE_PERMISSIONS
    FILES_MATCHING PATTERN "*.py"
)
```

## Migration reference

### C++ file mapping

| Old path (`tests/cpp/`) | New path (`tests/cpp/`) |
| :--- | :--- |
| `functional/api_support_read.{h,cc}` | `functional/gpu/identity/api_support_read.{h,cc}` |
| `functional/computepartition_read_write.{h,cc}` | `functional/gpu/partition/computepartition_read_write.{h,cc}` |
| `functional/dynamic_metrics_test.cc` | `unit/gpu/dynamic_metrics.cc` |
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

| Old file (`tests/python_unittest/`) | New file(s) (`tests/python/`) |
| :--- | :--- |
| `unit_tests.py` — BDF class | `unit/bdf/test_bdf.py` |
| `unit_tests.py` — GPU getters | `functional/gpu/test_{feature}.py` (one file per feature) |
| `unit_tests.py` — CPU getters | `functional/cpu/test_{feature}.py` (one file per feature) |
| `integration_test.py` — init/shutdown | `functional/system/test_init.py` |
| `integration_test.py` — GPU setters | `functional/gpu/test_{feature}.py` |
| `integration_test.py` — CPU setters | `functional/cpu/test_{feature}.py` |
| `integration_test.py` — topology | `functional/system/test_topology.py` |
| `integration_test.py` — NIC/switch | `functional/nic/test_discovery.py` |
| `cli_unit_test.py` | `cli/test_cli_{common,gpu,cpu,topology}.py` |
| `perf_tests.py` | *(deferred to future work)* |
| `perf_cputests.py` | *(deferred to future work)* |
| `common.py` | `common/helpers.py` |
| `runcmd.py` | `common/runcmd.py` |
