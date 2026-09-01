# Test Suite Map

Orientation diagram for everything under `tests/`. For the design rationale and
the full per-file reference, see
[../docs/conceptual/test-design.md](../docs/conceptual/test-design.md) and
[python/README.md](python/README.md).

## Three test families

```text
                        AMD SMI test estate
                                │
        ┌───────────────────────┼───────────────────────────┐
        │                       │                           │
   ┌────▼─────┐         ┌───────▼────────┐         ┌────────▼─────────┐
   │  C++     │         │    Python      │         │   Packaging /    │
   │ amdsmitst│         │  3 runners     │         │   Build / ABI    │
   │ (GTest)  │         │  (unittest)    │         │   (stdlib only)  │
   └────┬─────┘         └───────┬────────┘         └────────┬─────────┘
        │                       │                           │
 tests/amd_smi_test/     tests/python/            tests/abi_check/
                                                  tests/amdsmi_build/
                                                  tests/dme_integration/
                                                  tests/python/test_*_guard.py
                                                  tests/run_amdsmi_*.py
```

Only the first two families touch hardware. The third is pure logic plus
package-manager harnesses.

## C++ — one binary, filtered by suite name

```text
tests/amd_smi_test/
│
├── main.cc ─────────────► registers every *functional* test as
│                          TEST(<Component>Functional<Op>, <Feature><Case>)
│                          and drives the TestBase lifecycle
├── test_base.{h,cc} ────► SetUp → Run → Close → DisplayResults
├── test_common.{h,cc} ──► verbosity macros, enum→string
├── test_utils.{h,cc}
├── amdsmitst.exclude ───► BLACKLIST_ALL_ASICS + per-ASIC lists
├── detect_asic_filter.sh► reads KFD topology → sets $GTEST_EXCLUDE
├── check_test_conventions.py ──► pre-commit gate on layout/naming
│
├── unit/                  no hardware; plain TEST(), self-registering
│   ├── gpu/               dynamic_metrics, cper_read, mock_cper, wsl_backend
│   │   └── mock_cper/     committed .cper fixtures (AMDSMI_TEST_MOCK_DIR)
│   └── system/            lib_loader
│
└── functional/            live device; TestBase subclasses, .h + _test.cc pair
    ├── gpu/{clock,events,identity,memory,metrics,partition,
    │         pci,perf,power,ras,thermal,xgmi}/
    ├── system/            flat — no feature leaf
    ├── ifoe/{fabric,identity}/
    ├── cpu/{clock,power}/            placeholder_test.cc stubs
    ├── nic/{discovery,identity}/     placeholder_test.cc stubs
    └── wsl/smi/                      only when ENABLE_WSL_BACKEND
```

Nothing is listed by hand — CMake globs the tree:

```text
CMakeLists.txt (root)
  └─ add_subdirectory(tests/amd_smi_test)
        └─ file(GLOB_RECURSE ... CONFIGURE_DEPENDS unit/*.cc functional/*.cc)
              └─ add_executable(amdsmitst  main.cc test_*.cc  ${globbed})
                    ├─ links: libamd_smi, GTest::gtest, pthread
                    └─ install → <share>/amd_smi/tests/
```

Suite names are the only selection mechanism — `<Component><Type>[<Operation>]`:

```text
                 ┌──────────────── component ────────────────┐
   --gtest_filter= Gpu | Cpu | Nic | Ifoe | System | Wsl
                 └──────────┬────────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              │                            │
          ...Unit                    ...Functional
       (no hardware)                       │
                                ┌──────────┴──────────┐
                            ReadOnly              ReadWrite
                          (no root)              (root req'd)

  Currently registered:
    GpuUnit          GpuFunctionalReadOnly     GpuFunctionalReadWrite
    SystemUnit       SystemFunctionalReadOnly
                     IfoeFunctionalReadOnly
                     WslFunctionalReadOnly   (gated)
```

## Python — three runners over one shared engine

```text
tests/python/
│
├── unit_tests.py ───────┐
├── integration_test.py ─┼──► common/common.py :: run_test_dir()
├── cli_unit_test.py ────┘        │
│                                 ├─ parse -v/-q/-b/-k/-x/-l/-h
├── common/                       ├─ resolve amdsmi via
│   ├── common.py                 │    AMDSMI_PATH → ROCM_HOME → ROCM_PATH → /opt/rocm
│   └── runcmd.py                 ├─ unittest.discover("test_*.py") in own subtree
│                                 ├─ apply -k include / -x exclude on dotted id
│                                 ├─ require geteuid()==0
│                                 └─ GTestSummaryRunner → exit 0/1
│
├── unit/          ◄── unit_tests.py         no hardware
│   ├── gpu/       test_apu_metrics, test_cli_set_clk_limit, ...
│   └── system/    test_bdf, test_check_res, test_output_file_stdin
│
├── functional/    ◄── integration_test.py   live device + root
│   └── gpu/ cpu/ nic/ ifoe/ system/   test_<feature>.py
│
└── cli/           ◄── cli_unit_test.py      drives the installed amd-smi binary
    ├── base.py    TestCliBase — cached setUpClass, one --json baseline
    └── test_<command>.py   one module per CLI command (command-first)
```

Discovery is subtree-scoped, so each runner sees only its own tests:

```text
 unit_tests.py ──discovers──► unit/**/test_*.py
 integration_test.py ────────► functional/**/test_*.py
 cli_unit_test.py ───────────► cli/test_*.py
```

Leaf `test_*.py` files are **not** directly runnable — they have no `sys.path`
bootstrap. Always go through a runner with a `-k` filter.

The install target remaps the tree to the historical path:

```text
  tests/python/   ──CMake install──►  <share>/amd_smi/tests/python_unittest/
```

## Naming conventions, side by side

```text
  C++                                    Python
  ───────────────────────────────        ─────────────────────────────
  file   <feature>_<op>_test.cc          file   test_<feature>.py
  hdr    <feature>_<op>.h  (func only)   class  Test<Component><Feature>
  class  Test<Feature><Op> : TestBase    method test_<op>[_<qualifier>]
  suite  <Component><Type>[<Op>]         (suite == directory)
```

## Selection matrix

```text
  intent                 │ Python                       │ C++ (amdsmitst)
  ───────────────────────┼──────────────────────────────┼──────────────────────────
  list tests             │ <runner> -l                  │ --gtest_list_tests
  unit only              │ unit_tests.py                │ --gtest_filter="*Unit*"
  all functional         │ integration_test.py          │ "*Functional*"
  read-only / read-write │ ── not distinguished ──      │ "*FunctionalReadOnly*" /
                         │                              │ "*FunctionalReadWrite*"
  CLI                    │ cli_unit_test.py             │ ── none ──
  by feature             │ -k power                     │ "*.*Power*"
  exclude                │ -x partition                 │ "-*.*Partition*"
  ASIC exclusions        │ ── n/a ──                    │ source amdsmitst.exclude
                         │                              │ source detect_asic_filter.sh
                         │                              │ --gtest_filter="-$GTEST_EXCLUDE"
```

## Auxiliary suites (no GPU)

```text
tests/
├── abi_check/          abi_check.py + abi_check_test.py   header ABI diff vs develop
├── amdsmi_build/       run_amdsmi_build.py + tests        distro/pkg-mgr build driver
├── dme_integration/    metrics/services/submodules + tests
├── python/test_*_guard.py, test_packaging_scriptlets.py, test_abi_compat.py
│                       static assertions on CPack/DEBIAN/RPM templates
├── run_amdsmi_*.py     live package-manager harnesses (install/upgrade/remove/conflict)
└── api_summary.py      parses amdsmi.h + test logs → api_summary.{csv,txt}
```

## Where it all gets triggered

```text
 pre-commit ──► check_test_conventions.py   (layout + naming gate)
            └─► clang-format, ruff-format, gersemi, codespell

 CI (.github/workflows/)
   amdsmi-build.yml ──► run_amdsmi_build.py → build+install
                        └─► source amdsmitst.exclude; detect_asic_filter.sh
                            ./amdsmitst --gtest_filter="-$GTEST_EXCLUDE"
                            ./integration_test.py -v
                            ./unit_tests.py -v
                        └─► run_amdsmi_build.py summarize
   abi-compliance-check.yml ──► abi_check.py (major, then minor)
   amdsmi-python-versions.yml ─► run_amdsmi_python_versions_test.py (3.6.8 → latest)
   amdsmi-upgrade-downgrade.yml ► run_amdsmi_upgrade_downgrade_test.py
                                  run_amdsmi_component_removal_test.py
```

# API Summary Report
## Overview
The API summary report is generated from reading the amdsmi.h header file and the output from the python and C++ tests.  The python script, api_summary.py, will build a table from the available test log files.

## Pre-Requisites Before Running Summary Report
Run the python and C++ tests prior to running api_summary.py script.  The preferred way to run the tests is as follows:

<u>The python scripts are in the directory /opt/rocm/share/amd_smi/tests/python_unittest</u>
```
sudo unit_tests.py -v > _unit_test.log 2> _unit_test_err.log
sudo integration_test.py -v > _integration_test.log 2> _integration_test_err.log
```

<u>The C++ test is in the directory /opt/rocm/share/amd_smi/tests</u>

To run with ASIC-specific test exclusions (recommended):
```
cd /opt/rocm/share/amd_smi/tests
source amdsmitst.exclude
source detect_asic_filter.sh
sudo ./amdsmitst --gtest_filter="-${GTEST_EXCLUDE}" -v 1 > _amdsmitst.log 2> _amdsmitst_err.log
```

`detect_asic_filter.sh` reads the KFD topology to detect the installed ASIC
(e.g. `aldebaran`, `sienna_cichlid`) or falls back to `gfx_target_version` for
`ip discovery` nodes (e.g. `90400`, `90402`). It also detects SR-IOV
virtualization. The script sets `GTEST_EXCLUDE` by combining the global
blacklist (`BLACKLIST_ALL_ASICS`) with the device-specific filter from
`amdsmitst.exclude`.

To run without ASIC-specific exclusions (uses only the global blacklist):
```
cd /opt/rocm/share/amd_smi/tests
source amdsmitst.exclude
sudo ./amdsmitst --gtest_filter="-${BLACKLIST_ALL_ASICS}" -v 1 > _amdsmitst.log 2> _amdsmitst_err.log
```

## How to Run Summary Report
### Command Line Options

```
Header File:
  --amdsmi AMDSMI
    Path to header file, default=include/amd_smi/amdsmi.h
Log Files:
  --log_dir LOG_DIR
    Path to where logs exist, default=build
  --c_unit_test C_UNIT_TEST
    Filename for C unit_test output, default=_c_unit_test.log
  --c_integration C_INTEGRATION
    Filename for C integration_test output, default=_c_integration.log
  --py_unit_test PY_UNIT_TEST
    Filename for Python unit_test output, default=_py_unit_test.log
  --py_integration PY_INTEGRATION
    Filename for Python integration_test output, default=_py_integration.log
Output File:
  --output_dir OUTPUT_DIR
    Path to output file, default=.
```

Command line examples:
<details close>
  <summary>Click for example: <i><b>From amdsmi root directory with test output in build directory</i></b></summary>

~~~shell
api_summary.py
~~~
<i><b>With specifying summary output</i></b>
~~~shell
api_summary.py --output summary_dir
~~~
</details>

<details close>
  <summary>Click for example: <i><b>From amdsmi root directory with test output in current directory</i></b></summary>

~~~shell
api_summary.py --log_dir .
~~~
</details>

<details close>
  <summary>Click for example: <i><b>All input and output in current directory</i></b></summary>

~~~shell
api_summary.py --amdsmi ./amdsmi.h --log_dir . --output_dir .
~~~
</details>

<br> Output Files:
```
  api_summary.csv
  api_summary_table.txt
  api_summary_support.txt
```

<details close>
  <summary>Click for example: <i><b>api_summary.csv</i></b></summary>

~~~shell
API, Tested, c_unit_test, c_integration, py_unit_test, py_integration
amdsmi_init, 2, 0, 0, 1, 1
amdsmi_shut_down, 2, 0, 0, 1, 1
amdsmi_get_socket_handles, 2, 0, 0, 1, 1
amdsmi_get_cpu_handles, 1, 0, 0, 1, 0
amdsmi_get_socket_info, 1, 0, 0, 0, 1
amdsmi_get_processor_info, 1, 0, 0, 1, 0
amdsmi_get_processor_count_from_handles, 1, 0, 0, 1, 0
amdsmi_get_processor_handles_by_type, 1, 0, 0, 0, 1
amdsmi_get_processor_handles, 1, 0, 0, 1, 0
amdsmi_get_node_handle, 0, 0, 0, 0, 0
...
~~~
</details>

<details close>
  <summary>Click for example: <i><b>api_summary_table.txt</i></b></summary>

~~~shell
 API    Test(%)     Unit(%)     Func(%)
  C     64(34.2)    0( 0.0)   64(34.2)
 Py    117(62.6)  101(54.0)   27(14.4)
Total  132(70.6)  101(54.0)   82(43.9)
Num APIs: 187
~~~
</details>

<details close>
  <summary>Click for example: <i><b>api_summary_support.txt</i></b></summary>

~~~shell
API Not Supported: 3
	amdsmi_get_gpu_partition_metrics_info()
	amdsmi_set_gpu_accelerator_partition_profile()
	amdsmi_set_gpu_overdrive_level()
API Supported: 129
	amdsmi_clean_gpu_local_data()
	amdsmi_cpu_apb_disable()
	amdsmi_cpu_apb_enable()
	amdsmi_get_clk_freq()
	amdsmi_get_cpu_cclk_limit()
	amdsmi_get_cpu_core_boostlimit()
  ...
~~~
</details>
