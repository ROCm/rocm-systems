# AMD SMI Python Tests

Python `unittest`-based test suite for AMD SMI. Tests are split by **test type**
(unit vs. functional vs. CLI) and by **component** (gpu, cpu, nic, ifoe, system),
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
├── unit_tests.py           # runner: discovers unit/        (no hardware logic)
├── integration_test.py     # runner: discovers functional/  (live hardware)
├── cli_unit_test.py        # runner: discovers cli/         (drives amd-smi CLI)
├── common/                 # shared runner + helpers (common.py, runcmd.py)
├── unit/                   # pure-logic tests
│   ├── system/             #   test_bdf.py, test_check_res.py
│   └── gpu/                #   apu_metrics, ...
├── functional/             # hardware tests (require a live device + root)
│   ├── gpu/  cpu/  nic/  ifoe/  system/
└── cli/                    # exercise the installed `amd-smi` binary
```

Test files are always named `test_*.py`; each runner discovers only its own
subtree.

```mermaid
graph TD
    A[tests/python] --> U[unit_tests.py]
    A --> I[integration_test.py]
    A --> C[cli_unit_test.py]
    U -->|discovers| UU[unit/*/test_*.py<br/>no hardware]
    I -->|discovers| FF[functional/*/test_*.py<br/>live device]
    C -->|discovers| CC[cli/test_*.py<br/>amd-smi binary]
    U -.imports.-> CM[common/common.py]
    I -.imports.-> CM
    C -.imports.-> CM
    C -.imports.-> RC[common/runcmd.py]
```

---

## Prerequisites

- **Python 3.8+**
- The **`amdsmi` Python package installed** — every runner imports it at start-up
  (resolved via `AMDSMI_PATH` → `ROCM_HOME` → `ROCM_PATH` → `/opt/rocm`). See the
  [AMD SMI installation guide](https://rocm.docs.amd.com/projects/amdsmi/en/latest/).
- **Root privileges (`sudo`)** — all three runners enforce a root check and exit
  with an error if `geteuid() != 0`. Functional tests additionally need a live
  GPU/CPU/NIC. Unit tests are logic-only but still go through the same runner.

---

## The three runners

| Runner | Discovers | Hardware | Purpose |
| :--- | :--- | :--- | :--- |
| `unit_tests.py` | `unit/` | No | Pure logic: BDF parsing, formatting, static data |
| `integration_test.py` | `functional/` | Yes | API calls against a live device |
| `cli_unit_test.py` | `cli/` | Yes | Runs the installed `amd-smi` binary and checks its output |

All three share the same option set and the same GTest-style summary, because
they all delegate to `common.run_test_dir()`.

### Invocation

From an installed package (recommended — avoids a shadowing `site-packages` copy):

```bash
sudo /opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py -v
sudo /opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -v
sudo /opt/rocm/share/amd_smi/tests/python_unittest/cli_unit_test.py -v
```

From this source tree:

```bash
sudo ./unit_tests.py -v
sudo ./integration_test.py -v
sudo ./cli_unit_test.py -v
```

> The install target maps `tests/python/` to `.../tests/python_unittest/` so the
> historical installed path keeps working unchanged.

---

## Choosing a runner

```mermaid
flowchart TD
    Q{What are you testing?} --> L{Pure logic,<br/>no device?}
    L -->|Yes| UT[unit_tests.py]
    L -->|No| H{Testing the<br/>amd-smi CLI?}
    H -->|Yes| CT[cli_unit_test.py]
    H -->|No, library API| IT[integration_test.py]
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
sudo ./integration_test.py -x performance -v

# Run only the CLI version-command test, buffering per-test output
sudo ./cli_unit_test.py -k "version" -b -v

# Point at a non-default build
sudo AMDSMI_PATH=/path/to/build/share/amd_smi ./integration_test.py -v
```

### `-l` (list) output

```text
Available tests:
    system.test_bdf.TestBDF.test_invalid_bdfs
    system.test_bdf.TestBDF.test_valid_bdfs
    system.test_check_res.TestCheckRes.test_check_res
    gpu.test_apu_metrics.TestApuMetrics.test_metrics
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
