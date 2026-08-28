# AMD SMI Python Tests

Python `unittest`-based test suite for AMD SMI. Tests are split by **test type**
(unit vs. integration vs. functional vs. CLI) and by **component** (gpu, cpu, nic, ifoe, system),
each driven by one of three top-level runner scripts.

For the broader design rationale (C++ + Python), see the test design guide:
[../../docs/conceptual/test-design.md](../../docs/conceptual/test-design.md).

---

## Contents

- [Layout](#layout)
- [Prerequisites](#prerequisites)
- [The three runners](#the-three-runners)
- [Choosing a runner](#choosing-a-runner)
- [Command-line options](#command-line-options)
- [Environment variables](#environment-variables)
- [Examples](#examples)
- [Interpreting the output](#interpreting-the-output)
- [Execution flow](#execution-flow)
- [Troubleshooting](#troubleshooting)

---

## Layout

```text
tests/python/
├── run_tests.py            # runner: one tier, several, or all
├── unit_tests.py           # runner: discovers unit/         (no hardware)
├── integration_tests.py     # runner: discovers integration/  (live device)
├── functional_tests.py      # runner: discovers functional/   (live device + root)
├── cli_tests.py        # runner: discovers cli/          (drives amd-smi CLI)
├── common/                 # shared runner + helpers (common.py, api_test.py, runcmd.py)
├── unit/                   # no hardware: logic and mocked-import suites
│   ├── gpu/                #   apu_metrics, kfd_process_gpus, the cli_* mocked suites
│   └── system/             #   bdf, check_res, output_file_stdin
├── integration/            # live device: one suite per API area
│   ├── gpu/  cpu/  nic/  system/
│   └── test_api_coverage.py    #   every public API must be driven by a suite here
├── functional/             # live device + root: setters, lifecycles, benchmarks
│   ├── gpu/  cpu/  nic/  ifoe/  system/
└── cli/                    # exercise the installed `amd-smi` binary
```

Test files are always named `test_*.py`; each runner discovers only its own
subtree.

```mermaid
graph TD
    A[tests/python] --> R["run_tests.py<br/>--unit --integration --functional --cli"]
    A --> U[unit_tests.py]
    A --> I[integration_tests.py]
    A --> F[functional_tests.py]
    A --> C[cli_tests.py]
    R -->|discovers any combination| ALL[unit/ integration/<br/>functional/ cli/]
    U -->|discovers| UU[unit/**/test_*.py<br/>no hardware]
    I -->|discovers| II[integration/**/test_*.py<br/>live device]
    F -->|discovers| FF[functional/**/test_*.py<br/>live device + root]
    C -->|discovers| CC[cli/test_*.py<br/>amd-smi binary]
    I -.imports.-> CM[common/api_test.py]
    C -.imports.-> RC[common/runcmd.py]
```

---

## Prerequisites

- **Python 3.8+**
- The **`amdsmi` Python package installed** — every runner imports it at start-up
  (resolved via `AMDSMI_PATH` → `ROCM_HOME` → `ROCM_PATH` → `/opt/rocm`). See the
  [AMD SMI installation guide](https://rocm.docs.amd.com/projects/amdsmi/en/latest/).
- **Root privileges (`sudo`)** — all three runners enforce a root check and exit
  with an error if `geteuid() != 0`. Functional tests and the per-component unit
  API suites additionally need a live GPU/CPU/NIC; the remaining unit tests are
  logic-only but still go through the same runner. Running the API suites without
  root turns readable-only-as-root queries into `AMDSMI_STATUS_NO_PERM` failures.

---

## The tiers and their runners

| Runner | Discovers | Hardware | Purpose |
| :--- | :--- | :--- | :--- |
| `run_tests.py` | any tier(s) | Depends on the tiers chosen | Run one tier, several, or all of them in a single report |
| `unit_tests.py` | `unit/` | No | BDF parsing, formatting and CLI logic against stubbed imports |

`integration/<component>/` holds one suite per API area, and each API gets a single test that drives
it both ways via the driver in [common/api_test.py](common/api_test.py):

- **`reject()`** — one deliberately invalid argument per call; the library must refuse it. Every
  invalid value is rejected by the Python interface before the C entry point, so no device state can
  change even for a setter.
- **`expect()`** — valid arguments only. Prints the payload, checks it is structurally sound, and
  requires `AMDSMI_STATUS_SUCCESS`; a not-supported status is reported rather than failed.

Both repeat every call for each live processor handle and each enum value the other arguments take.
A suite skips when the platform has no processor of its kind, and an API skips when every
combination reports not-supported. Setters, stateful lifecycles and benchmarks stay in
`functional/`.

A test belongs in `unit/` when it runs without a device and in `integration/` when it
drives one. A unit test that replaces `amdsmi` or a CLI module in `sys.modules` must
inherit `common.common.ModuleIsolationMixin` and declare `ISOLATED_MODULES` (and
`ISOLATED_PATH` when it extends `sys.path`), so a stub never outlives its suite —
[unit/test_module_isolation.py](unit/test_module_isolation.py) finds those suites by
that marker and enforces the contract.

| `integration_tests.py` | `integration/` | Yes | Per-API argument rejection and payload validation |
| `functional_tests.py` | `functional/` | Yes | Setters, stateful lifecycles and benchmarks |
| `cli_tests.py` | `cli/` | Yes | Runs the installed `amd-smi` binary and checks its output |

They all share the same option set and the same GTest-style summary, because
they all delegate to `common.run_test_dir()`.

`integration/test_memory_partition_lifecycle.py` is a standalone script rather than a
suite: it reloads the driver, so it is run on its own with `sudo` and contributes no
tests to the runner.

### Invocation

From an installed package (recommended — avoids a shadowing `site-packages` copy):

```bash
sudo /opt/rocm/share/amd_smi/tests/python_unittest/run_tests.py -v
sudo /opt/rocm/share/amd_smi/tests/python_unittest/run_tests.py --unit --integration -v
sudo /opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py -v
```

From this source tree:

```bash
sudo ./run_tests.py -v                          # every tier
sudo ./run_tests.py --unit --integration -v     # two tiers together
sudo ./unit_tests.py -v                         # one tier on its own
sudo ./integration_tests.py -v
sudo ./functional_tests.py -v
sudo ./cli_tests.py -v
```

> The install target maps `tests/python/` to `.../tests/python_unittest/` so the
> historical installed path keeps working unchanged.

---

## Choosing a runner

```mermaid
flowchart TD
    Q{What are you testing?} --> C{Testing the<br/>amd-smi CLI?}
    C -->|Yes| CT[cli/]
    C -->|No| H{Needs a live device?}
    H -->|No| UT[unit/]
    H -->|Yes| S{Setter, stateful<br/>lifecycle or benchmark?}
    S -->|Yes| FT[functional/]
    S -->|No: an API's<br/>arguments and payload| IT[integration/]
```

---

## Command-line options

Every runner accepts the same flags (parsed in `common/common.py`):

| Option | Long form | Effect |
| :--- | :--- | :--- |
| `-v`, `-vv` | `--verbose` | Verbose: each test body prints its own per-call output; the runner's own dot line is suppressed to avoid duplication. **Recommended.** |
| `-q` | `--quiet` | Quiet: suppress the title/legend/per-test prints; dots only. |
| `-b` | `--buffer` | Buffer each test's stdout/stderr; show it only if that test fails. |
| `-k PAT` | `--keyword PAT` | Run only tests whose id contains `PAT` (substring; `-kPAT` joined form also works). |
| `-x PAT` | `--exclude PAT` | Skip tests whose id contains `PAT` (the inverse of `-k`). |
| `-l` | `--list` | List all discovered test ids to **stdout** and exit (nothing runs). |
| `-h` | `--help` | Show unittest help plus AMD SMI env-var/path guidance and examples. |

Notes:

- `-k` and `-x` match against the **full dotted test id**, e.g.
  `cli.test_event.TestEvent.test_command`, so you can filter by file, class, or
  method name fragment.
- `-l` output goes to stdout (test-run chatter goes to stderr), so
  `... -l | grep xgmi` works cleanly.
- `-h` does **not** require root and never runs any test.

---

## Environment variables

| Variable | Meaning |
| :--- | :--- |
| `AMDSMI_PATH` | Path to the `amd_smi` share dir; overrides `ROCM_HOME`/`ROCM_PATH`. |
| `ROCM_HOME` | ROCm install root, used when `AMDSMI_PATH` is unset. |
| `ROCM_PATH` | Fallback ROCm install root (default `/opt/rocm`). |
| `NO_COLOR` | If set, disables ANSI colors in the summary (see <https://no-color.org/>). |

The resolved path must contain the `amdsmi` package. If a system-installed
`amdsmi` shadows the intended build, the runner prints an explicit remediation
block and exits.

---

## Examples

```bash
# List every unit test without running anything
sudo ./unit_tests.py -l

# Run all unit tests, verbose (recommended)
sudo ./unit_tests.py -v

# Run only BDF-parsing tests
sudo ./unit_tests.py -k "bdf" -v

# Run all functional tests except the performance suites
sudo ./functional_tests.py -x performance -v

# Run only the CLI version-command test, buffering per-test output
sudo ./cli_tests.py -k "version" -b -v

# Point at a non-default build
sudo AMDSMI_PATH=/path/to/build/share/amd_smi ./functional_tests.py -v
```

### `-l` (list) output

```text
Available tests:
    unit.gpu.test_apu_metrics.TestAmdSmiApuMetrics.test_convert_apu_unit_scalar
    unit.gpu.test_vcn_busy_navi.TestVcnBusyNaviFallback.test_navi_vcn_busy_reads_sysfs
    unit.system.test_bdf.TestAmdSmiPythonBDF.test_parse_bdf
    unit.system.test_check_res.TestAmdSmiCheckRes.test_check_res
    ...
```

### `-v` (verbose) run

```text
AMD SMI Unit Tests

Running tests...

... (per-test output from each test body) ...

[----------] 12 tests ran. (34 ms total)
[  PASSED  ] 12 tests.
```

### A run with a skip and a failure

```text
======================================================================
Legend: . = pass, s = skipped, F = fail, E = error
======================================================================

[----------] 40 tests ran. (512 ms total)
[  PASSED  ] 38 tests.
[  FAILED  ] 1 test, listed below:
[  FAILED  ] TestPower.test_power_cap
[  SKIPPED ] 1 test, listed below:
[  SKIPPED ] TestOverdrive.test_overdrive_write
```

---

## Known failures

The API suites assert `AMDSMI_STATUS_SUCCESS` from every getter, which surfaces
defects that previously went unreported. On current hardware the following fail
for reasons in the library, not the tests — treat any *other* failure as a
regression:

| Status | Tests | Cause |
| :--- | :--- | :--- |
| `AMDSMI_STATUS_UNEXPECTED_DATA` | `test_get_clock_info`, `test_get_energy_count`, `test_get_gpu_activity`, `test_get_gpu_metrics_info`, `test_get_gpu_pci_bandwidth`, `test_get_gpu_xcd_counter`, `test_get_gpu_xgmi_link_status`, `test_get_link_metrics`, `test_get_pcie_info`, `test_get_temp_metric` (HBM sensors), `test_get_utilization_count`, `test_get_violation_status` | The library returns a payload it cannot parse |
| `AMDSMI_STATUS_NO_PERM` | `test_get_gpu_accelerator_partition_profile_config` | Appears only when a suite is invoked directly with `python3 -m unittest`; the runners require root, where it passes |

These are tracked as library defects rather than suppressed, so that a fix flips
the test green without anyone having to remember to unmark it.

---

## Interpreting the output

**Progress legend** (shown in non-verbose mode). While tests run, `unittest`
prints one character per test:

| Char | Meaning |
| :--- | :--- |
| `.` | pass |
| `s` | skipped (feature/hardware not applicable) |
| `F` | failure (an assertion did not hold) |
| `E` | error (an unexpected exception was raised) |

**GTest-style summary** (always printed at the end, to stderr):

- `[----------] N tests ran. (X ms total)` — total count and wall-clock time.
- `[  PASSED  ] N tests.` — count that passed.
- `[  FAILED  ] ...` — one line per failing test, listed by `Class.method`.
  Errors and unexpected `@expectedFailure` successes are counted here too.
- `[  SKIPPED ] ...` — one line per skipped test.

Colors mirror GTest (green pass / yellow skip / red fail / cyan separator) and
are automatically suppressed when output is not a TTY or when `NO_COLOR` is set.

**Exit code** — the process exits `0` when every test passed (skips are OK) and
`1` if any test failed or errored. This is what CI keys off of.

```mermaid
flowchart LR
    R[run tests] --> S{wasSuccessful?}
    S -->|no failures/errors| Z0[exit 0]
    S -->|any failure/error| Z1[exit 1]
```

---

## Execution flow

What `common.run_test_dir()` does for every runner:

```mermaid
sequenceDiagram
    participant U as User
    participant R as runner script
    participant C as common.run_test_dir
    participant L as unittest loader
    participant G as GTestSummaryRunner

    U->>R: sudo ./unit_tests.py -k pat -v
    R->>C: run_test_dir("unit", title, dir)
    alt -h / --help
        C-->>U: help + path guidance, exit 0
    end
    C->>L: discover(test_*.py) with -k filter
    opt -x pattern
        C->>C: drop excluded test ids
    end
    alt -l / --list
        C-->>U: print test ids, exit 0
    end
    C->>C: require root (geteuid==0) else exit 1
    C->>G: run(suite)
    G-->>U: dots / verbose lines
    G-->>U: GTest summary block
    C-->>U: exit 0 (pass) / 1 (fail)
```

---

## Troubleshooting

| Symptom | Cause / fix |
| :--- | :--- |
| `amdsmi path "..." does not exist` | Set `AMDSMI_PATH`, `ROCM_HOME`, or `ROCM_PATH` to a valid install. |
| `amdsmi loaded from '...' instead of expected path` | A system-installed `amdsmi` is shadowing the build. Run from the installed tests dir, set `AMDSMI_PATH`, or reinstall/uninstall the stale package (the error prints the exact commands). |
| `Please relaunch with elevated privileges.` then exit 1 | The runner requires root. Re-run with `sudo`. |
| Functional/CLI tests skip or fail | They need live hardware and the installed `amd-smi` binary on `PATH` (the CLI helper prepends `$ROCM_PATH/bin`). |

---

## Reference

- Test design guide: [../../docs/conceptual/test-design.md](../../docs/conceptual/test-design.md)
- Python `unittest`: <https://docs.python.org/3.8/library/unittest.html>
