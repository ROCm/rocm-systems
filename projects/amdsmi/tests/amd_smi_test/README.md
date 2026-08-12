# AMD SMI C++ tests (`amdsmitst`)

All C++ tests compile into a single GoogleTest binary, `amdsmitst`. Tests are
split into two tiers:

- **Unit** (`unit/`) — exercise the public `amdsmi.h` API surface, including
  invalid-parameter and per-enum / per-device cases. No root required.
- **Functional** (`functional/`) — run against live hardware. Read-only suites
  need no root; read-write suites mutate device state and typically require root.

For the full design (directory layout, naming rules, component taxonomy) see
[`docs/conceptual/test-design.md`](../../docs/conceptual/test-design.md). This
file is the practical "how to build and run" guide.

## Building

The suite is off by default; enable it with `BUILD_TESTS`:

```shell
cmake -B build -DBUILD_TESTS=ON
cmake --build build --target amdsmitst -j$(nproc)
```

The binary lands at `build/tests/amd_smi_test/amdsmitst`. An installed copy is
placed at `/opt/rocm/share/amd_smi/tests/amdsmitst`.

## Suite naming

GoogleTest suite names follow `<Component><Type>[<Operation>]`, so a single
`--gtest_filter` selects any slice:

| Suite pattern | Meaning |
| :--- | :--- |
| `<Component>Unit` | Unit tests (e.g. `GpuUnit`, `CpuUnit`, `NicUnit`, `SystemUnit`) |
| `<Component>FunctionalReadOnly` | Functional, reads only — no root needed |
| `<Component>FunctionalReadWrite` | Functional, mutates state — root usually needed |

`<Component>` is one of `Gpu`, `Cpu`, `Nic`, `Ifoe`, `System`. Individual test
names carry the feature (e.g. `GpuUnit.GetClkFreq_AllGpusAllTypes`), so you can
also filter by feature across suites.

## Running

Run from the build tree (or the install dir). List first, then filter:

```shell
cd build/tests/amd_smi_test

# List every test without running anything
./amdsmitst --gtest_list_tests

# Everything
sudo ./amdsmitst
```

### By tier

```shell
# Unit only — no root, minimal hardware dependence
./amdsmitst --gtest_filter="*Unit*"

# All functional (read-only + read-write) — root for the read-write suites
sudo ./amdsmitst --gtest_filter="*Functional*"

# Functional read-only only (safe, no root)
./amdsmitst --gtest_filter="*FunctionalReadOnly*"
```

### By component

```shell
./amdsmitst --gtest_filter="GpuUnit*"            # GPU unit tests
sudo ./amdsmitst --gtest_filter="Cpu*"           # all CPU tests
sudo ./amdsmitst --gtest_filter="GpuFunctional*" # GPU functional tests
```

### By feature (matches the test-name portion)

```shell
sudo ./amdsmitst --gtest_filter="*.*Power*"          # power
sudo ./amdsmitst --gtest_filter="*.*Freq*"           # clocks / frequency
./amdsmitst        --gtest_filter="*.*Temp*:*.*Fan*"  # thermal (OR with ':')
```

### Combining, negating, single tests

```shell
# GPU read-only power tests only
./amdsmitst --gtest_filter="GpuFunctionalReadOnly.*Power*"

# All functional except partition ('-' negates)
sudo ./amdsmitst --gtest_filter="*Functional*:-*.*Partition*"

# One exact test
./amdsmitst --gtest_filter="GpuUnit.GetClkFreq_AllGpusAllTypes"
```

## ASIC-specific exclusions (recommended for full runs)

Some tests do not apply to every ASIC. The helper scripts build a combined
exclusion filter for the detected hardware:

```shell
cd /opt/rocm/share/amd_smi/tests        # or build/tests/amd_smi_test
source amdsmitst.exclude
source detect_asic_filter.sh
sudo ./amdsmitst --gtest_filter="-${GTEST_EXCLUDE}" -v 1
```

`detect_asic_filter.sh` reads the KFD topology to detect the ASIC (and SR-IOV),
then sets `GTEST_EXCLUDE` from the global blacklist plus the device-specific
list in `amdsmitst.exclude`. To apply only the global blacklist, filter on
`-${BLACKLIST_ALL_ASICS}` instead.

## Controlling destructive writes

Read-write functional tests (and unit setter tests) modify device state. They
follow a store → change → verify → restore pattern, so a normal run is
non-destructive. To skip every mutating write entirely — e.g. on shared
hardware — set:

```shell
AMDSMI_TEST_DISALLOW_MUTATION=1 ./amdsmitst --gtest_filter="*Unit*"

# Also works across all test types:
AMDSMI_TEST_DISALLOW_MUTATION=1 ./amdsmitst
```

## Known test skips

Unconditional skips due to driver or library issues are tracked in
[`known_failures.md`](known_failures.md).

## Verbosity and logging

`-v 1` raises the verbosity level; combine with shell redirection to capture
logs (used by the API summary report — see [`tests/README.md`](../README.md)):

```shell
sudo ./amdsmitst -v 1 > _c_func_test.log 2> _c_func_test_err.log
```

Each test prints a framed section outline:

```text
###############################################################################
        #### TEST NAME ####
        #### TEST DESCRIPTION ####
        #### TEST SETUP ####
        #### TEST EXECUTION ####
        #### TEST RESULTS ####
        #### TEST CLEAN UP ####
```
