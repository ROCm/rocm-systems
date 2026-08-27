# AMD SMI C++ tests (`amdsmitst`)

All C++ tests compile into a single GoogleTest binary, `amdsmitst`. Tests are
split into three tiers:

- **Unit** (`unit/`) — pure logic and static data. No device, no `amdsmi_init`,
  no root. CPER parsing, metric struct versioning, the library loader.
- **Integration** (`integration/`) — the public `amdsmi.h` API surface against a
  live library: every API's invalid-input cases, plus getters driven with valid
  input and checked for valid output. No root required.
- **Functional** (`functional/`) — setters, and APIs that need setup from another
  API. Read-only suites need no root; read-write suites mutate device state,
  require root, and are opt-in (see "Controlling destructive writes").

Invalid-input cases run even when the matching device is absent. Positive cases
skip when there is no device to drive.

A positive case passes only on `AMDSMI_STATUS_SUCCESS`. When the API reports the
feature is absent here (`AMDSMI_STATUS_NOT_SUPPORTED`, a missing driver, ...) the
device counts as skipped, and a test whose every device was unsupported reports
SKIPPED rather than PASSED. A run against a library that answers "not supported"
for everything therefore reads as skipped, not green.

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
| `<Component>Unit` | Unit tests, no device (`GpuUnit`, `SystemUnit`) |
| `<Component>Integration` | API surface on a live library — no root needed |
| `<Component>FunctionalReadOnly` | Functional, reads only — no root needed |
| `<Component>FunctionalReadWrite` | Functional, mutates state — root and opt-in |

`<Component>` is one of `Gpu`, `Cpu`, `Nic`, `Ifoe`, `System`. Individual test
names carry the feature (e.g. `GpuIntegration.GetClkFreq_AllGpusAllTypes`), so
you can also filter by feature across suites.

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
# Unit only — no device needed at all
./amdsmitst --gtest_filter="*Unit*"

# Integration only — no root, needs a live library
./amdsmitst --gtest_filter="*Integration*"

# All functional (read-only + read-write) — root for the read-write suites
sudo ./amdsmitst --gtest_filter="*Functional*"

# Functional read-only only (safe, no root)
./amdsmitst --gtest_filter="*FunctionalReadOnly*"
```

### By component

```shell
./amdsmitst --gtest_filter="GpuIntegration*"     # GPU API-surface tests
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
./amdsmitst --gtest_filter="GpuIntegration.GetClkFreq_AllGpusAllTypes"
```

## ASIC-specific exclusions (recommended for full runs)

Some tests do not apply to every ASIC. The helper scripts build a combined
exclusion filter for the detected hardware:

```shell
cd /opt/rocm/share/amd_smi/tests
source amdsmitst.exclude
source detect_asic_filter.sh
sudo ./amdsmitst --gtest_filter="-${GTEST_EXCLUDE}" -v 1
```

`detect_asic_filter.sh` reads the KFD topology to detect the ASIC (and SR-IOV),
then sets `GTEST_EXCLUDE` from the global blacklist plus the device-specific
list in `amdsmitst.exclude`. To apply only the global blacklist, filter on
`-${BLACKLIST_ALL_ASICS}` instead.

Neither file is copied into the build tree. Running from a build directory,
source them from the source tree (`tests/amd_smi_test/`) instead.

## Controlling device writes

Every device write lives in a `*FunctionalReadWrite` suite behind
`AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED()`, which requires only the privilege the
write needs:

```shell
sudo ./amdsmitst --gtest_filter="*FunctionalReadWrite*"
```

| Condition | Effect when unmet |
|-----------|-------------------|
| `AMDSMI_NON_PRIVILEGED` is **not** set | skipped |
| process is root | skipped |

Each write test stores the original value, sets a different one, verifies the
readback against what it set, then restores the original and confirms the
restore took, so a run leaves the device as it found it.

The one setter the API cannot read back, `amdsmi_set_cpu_msr_floor_freq_limit`,
writes 0 to clear the floor, which is the default state, so it needs no restore.

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
