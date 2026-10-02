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
  ASIC exclusions        │ ── N/A ──                    │ source amdsmitst.exclude
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
├── api_summary.py      parses amdsmi.h + test logs → api_summary.{csv,txt}
├── run_api_coverage.sh one-shot: runs every suite traced, then both reports
├── api_coverage_report.py  api_summary.csv → KPI tables (subsystem + framework)
├── cli_surface.py      walks amdsmi_parser.py → total CLI command surface
├── api_coverage_baseline.csv  HAND-EDITED start/previous values
```

Generated run data lives in `./api-coverage-results-<timestamp>/`, never in the
source tree.

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

## Quick start: one-shot run

`run_api_coverage.sh` does everything in this section for you -- it runs each
suite with tracing enabled, writes the `_c_*`/`_py_*` logs, then chains
`api_summary.py` and `api_coverage_report.py`:

```
sudo tests/run_api_coverage.sh                  # -> ./api-coverage-results-<timestamp>/
sudo tests/run_api_coverage.sh -A               # no filtering at all
sudo tests/run_api_coverage.sh -v               # watch the suites run
sudo tests/run_api_coverage.sh -o ./kpi-logs    # pick the log dir
sudo tests/run_api_coverage.sh -n "after X"     # label the run in the trend
```

Root is required -- the Python runners `exit(1)` when `geteuid() != 0`.

| Flag | Meaning |
|------|---------|
| `-o, --out DIR` | Where logs and reports land (default `./api-coverage-results-<timestamp>`) |
| `-t, --tests-dir DIR` | Installed or build test dir holding `amdsmitst` |
| `-H, --history FILE` | Accumulate the trend in one file instead of inheriting it from the previous run dir |
| `-n, --notes TEXT` | Free-text note recorded with the run |
| `-U, --unit-filter P` | Positive GTest pattern for the Unit column (default `*Unit*`) |
| `-F, --func-filter P` | Positive GTest pattern for the Functional column |
| `-X, --extra-exclude L` | Extra exclusions, appended to the per-ASIC list |
| `-A, --no-exclude` | Run everything: drop the per-ASIC blacklist |
| `-v, --verbose` | Mirror suite output to the terminal as well as the log |
| `--no-cpp` / `--no-python` | Skip one language's suites |

Each run creates its own timestamped directory **in the current directory**, and
everything generated is written inside it. Nothing is written into the source
tree, so run data never ends up in git. Files are `chown`ed back to the invoking
user afterwards, so they stay readable after `sudo`.

### Watching a run

By default each suite's output goes only to its log, so a long run looks silent.
`-v` mirrors it to the terminal as well -- the log is still written in full, and
the suite's own exit code is still what gets recorded.

```
sudo tests/run_api_coverage.sh -v
```

Without `-v`, follow a log from a second terminal instead:

```
tail -f api-coverage-results-<timestamp>/_py_func_test.log
```

The Python functional log is the big one (over 1 MB), which is why terminal
output is opt-in rather than the default.

### Installed vs source tests

By default the runner uses the **installed** tests from `AMDSMI_PATH`
(default `/opt/rocm/share/amd_smi`). Those only pick up source changes after
`sudo make install`, which refreshes the library, `amdsmitst`, and every
`python_unittest/*.py`.

To run uninstalled source changes, point `-t` at the build tree:

```
sudo tests/run_api_coverage.sh -A -t build/tests/amd_smi_test
```

The runner falls back through `$TESTS_DIR/python_unittest` ->
`$TESTS_DIR/python` -> `<project>/tests/python` for the Python suites, and does
the same for `amdsmitst.exclude` / `detect_asic_filter.sh`, so a build tree that
holds only the binary still works.

### Outputs

| File | Contents |
|------|----------|
| `_api_coverage.md` | The report: run metadata, both KPI tables, untested APIs |
| `_api_coverage.csv` | Same content as the `.md` minus the untested list |
| `_api_coverage.tsv` | The `.csv` tables, tab-separated, for pasting into Excel |
| `_api_summary*.{csv,txt}` | Raw `api_summary.py` output |
| `_c_*.log`, `_py_*.log` | Raw suite logs |
| `api_coverage_trend.csv` | **Keep this one.** This run's row, after the rows inherited from the previous run |

**The leading `_` means disposable** -- delete or regenerate it freely. The one
generated file without the prefix is `api_coverage_trend.csv`, because deleting
it is the only way to lose the trend. Everything else lacking the prefix
(`run_api_coverage.sh`, `api_coverage_report.py`, `cli_surface.py`,
`api_coverage_baseline.csv`) is hand-written source that belongs in git.

**Nothing generated should be committed, prefix or not.** Run data records the
kernel, driver versions and GPU behind each number, and those numbers are
hardware-gated -- one machine's rows are not comparable with another's, and an
append-only file conflicts on the last line every time two people run it.

The hostname is never recorded. What explains a coverage number is the hardware
and driver configuration, which is already captured; "CPU/ESMI is 35%" is
explained by a missing HSMP driver, not by which box ran it. Use `-n` to
attribute a run on purpose:

```
sudo tests/run_api_coverage.sh -n "RX6800 rig, post-XYZ fix"
```

`_api_coverage.csv` and `_api_coverage.tsv` hold the same thing: the report's
tables, one block per table, each with its own title and header row and a blank
row between blocks. Markdown decoration is stripped so the cells hold plain
values, and both are written with a UTF-8 BOM so Excel renders the `→` arrows.

Use whichever matches how you get it into a spreadsheet:

| Want to | Use | Why |
|---------|-----|-----|
| Open the whole report as a sheet | `.csv` | Excel parses commas when it *opens* a file |
| Copy a block and paste it in | `.tsv` | Excel splits pasted text on tabs, **never** on commas |

Pasting `.csv` text into a sheet drops the whole line into one cell -- that is
Excel behaviour, not a broken file. Either paste the `.tsv` instead, or paste
the `.csv` and run Data -> Text to Columns -> Delimited -> Comma.

### How the trend carries across runs

There is no shared trend file. Each run inherits its rows from the first of
these that exists, then appends its own:

1. An explicit `-H FILE`.
2. The newest `api-coverage-results-*` directory **beside** the output dir.
3. The `api_coverage_trend.csv` already in the output dir itself.

With the default dated directories, rule 2 applies: the most recent directory
always holds the complete trend and can be zipped and sent on its own.

If you keep passing the same `-o DIR`, rule 3 applies and that directory
accumulates its own trend. Rule 2 is tried first so that re-running just the
report inside a dated directory replaces its row instead of adding a duplicate;
in a fixed `-o DIR` there are no siblings, so a re-run does append an extra row.

**When cleaning up, keep the newest run directory.** Deleting older ones costs
nothing -- the newest already carries their rows -- but deleting the newest, or
the `api_coverage_trend.csv` inside it, ends the trend. That file is the reason
it has no `_` prefix.

Consequences worth knowing:

- The first run in a fresh location has nothing to inherit, so **Trend** reads `-`.
- Move or rename a run directory out of the way and the next run starts a new trend.
- Selection is `max()` by directory name, so the timestamp must stay sortable.
- A directory named `api-coverage-results` (no timestamp) does not match the
  sibling pattern, so it trends only against itself.
- `-H FILE` keeps the trend in one file regardless of where you run.

Everything else in the run dir is regenerable, but at different cost. The raw
logs can only be recreated by re-running the suites; from `_api_summary.csv` the
report regenerates in under a second:

```
python3 tests/api_coverage_report.py --log_dir ./api-coverage-results-<timestamp>
```

### Where to insert data by hand

Exactly one file is hand-maintained. `tests/api_coverage_baseline.csv` is only
ever read, never written. It is laid out as the two report tables, keyed by
`section,row,metric` -- where `row` and `metric` are the row label and column
heading as they appear in the report -- with two value columns:

| Column | Purpose |
|--------|---------|
| `start` | The **S** in the `S -> Today -> Goal` columns |
| `previous` | Pins what the **Trend** column compares against. If any cell is set the whole file is used; otherwise Trend falls back to the last inherited trend row |

```
section,row,metric,start,previous
subsystem,GPU,tested,,
subsystem,Total,tested,148,
framework,Python,API,100,
framework,Python,CLI,13,
```

To snapshot the current run as the next comparison point, copy each row's
**Today** value out of the report's tables into the matching `previous` cell
here. `subsystem` rows take the *Tested* count; `framework` rows take the middle
value of the `S -> Today -> Goal` arrow.

Any cell may be left blank and renders as `-`. Goals never appear here; they are
always derived (see below). The file carries a single comment line, so a CSV
viewer still sees a normal table.

By contrast `api_coverage_trend.csv` is **generated**, and lives inside each run
directory rather than in the source tree. Edit it only to prune bad runs -- and
note the next run inherits whatever you leave behind.

### Goals are derived, not configured

No goal is typed in anywhere. Each column is measured against a **100% scale**,
and the goal is a percentage of it:

```
goal = scale x goal%
```

| Column | 100% scale | Goal |
|--------|-----------|------|
| API | APIs counted in `include/amd_smi/amdsmi.h` | 100% of it |
| Unit `*` | same as API (placeholder -- see below) | 80% of it |
| Functional `**` | same as API | 100% of it |
| CLI `***` | command surface from `tests/cli_surface.py` | 100% of it |

In the framework table the arrow ends at the **goal** in every row -- as a count
in the per-framework rows, as a percentage in the **Total** row:

```
| Gtest, C/C++ | 60 → 81 → 247    | 1 → 2 → 198   | ...   <- counts, ending at the goal
| Python       | 100 → 126 → 247  | 0 → 0 → 198   | ...
| **Total**    | 60% → 61% → 100% | 1% → 1% → 80% | ...   <- percentages, same goal
```

The generated report carries the full definition of each column as footnotes
`*` (Unit), `**` (Functional) and `***` (CLI), plus a legend table giving each
column's scale and goal -- so a pasted table explains itself.
`FRAMEWORK_GOALS` in `api_coverage_report.py` holds the percentages, and the
same legend is reproduced as the "Scale and goal per column" block in
`_api_coverage.csv` / `.tsv`.

**Unit's scale is a known placeholder.** It currently uses the full API count,
which is why it is the one column held below 100% -- an 80% goal against a scale
known to be too large. Not every API can be exercised without a device, so that
scale should shrink to a qualifying subset; when it does, only `FRAMEWORK_GOALS`
and the `scale_of` map in `api_coverage_report.py` need to change.

**Unit's numerator is also incomplete, independent of the scale.** Most unit
test files never emit the `### <api>(` marker at all (checked 2026-10-02: 1 of
21 C++ `unit/*.cc` files, 0 of 25 Python `unit/*.py` files), so an API can have
thorough unit-test assertions and still read untested here. Add
`DISPLAY_AMDSMI_API()`/the marker only where the test reaches the real
implementation -- not where the call is mocked out (e.g. Python tests that
`mock.patch.object(amdsmi_wrapper, ...)`) or where validation raises before ever
reaching it; marking a mocked call as tested would misreport coverage that was
never exercised.

So goals track the product automatically as APIs and CLI options are added.

`cli_surface.py` derives the CLI 100% mark by walking the real argparse tree in
`amdsmi_cli/amdsmi_parser.py`:

```
python3 tests/cli_surface.py            # full breakdown
python3 tests/cli_surface.py --total    # just the number, for scripting
```

```
base commands            19
feature options         298      (Device 84, CPU 48, CPU Core 9, ...)
format variants          12      (3 formats x 4 sinks)
valid combinations     3804      (19 + 298) x 12
invalid combinations     95      (19 x 5)
TOTAL CLI SURFACE      3899
```

Two things to know about that number:

- The parser registers `--cpu`/`--core` and the NIC/switch groups only when the
  matching device is present, so measuring on a GPU-only host hides 134 of the
  298 feature options. `cli_surface.py` forces the device probes on, otherwise
  the KPI denominator would change from machine to machine.
- The command and option counts are measured; the multipliers
  (`OUTPUT_FORMATS`, `OUTPUT_SINKS`, `INVALID_PER_COMMAND`) are policy -- they
  encode how thoroughly a command should be exercised. Tune them at the top of
  the file.

### Where each number comes from

Every cell in the report traces to one of four sources. **S, Today, Goal and
Trend are four different origins** -- they are easy to confuse because they sit
in the same row:

| Value | Source | Hand-edited? |
|-------|--------|--------------|
| **S** (the arrow's start) | `start` column of `tests/api_coverage_baseline.csv` | Yes |
| **Today** | Measured by this run, from `_api_summary.csv` and the suite logs | No |
| **Goal** (the arrow's end) | Derived: `scale x goal%` (see above) | No |
| **Trend** | Delta vs the previous run, from `api_coverage_trend.csv` | No |

So the arrow and the Trend column are **independent**: `S` comes from the
hand-maintained baseline and never changes on its own, while `Trend` comes from
the previous run's trend file and changes every run.

The trend file is read from the **previous** run directory. The copy written
into the current run directory is for the *next* run -- it is never read by the
run that creates it.

`Trend` resolves in this order:

1. The `previous` column of `api_coverage_baseline.csv`, if any cell is filled.
2. Otherwise the last row inherited from the previous run's trend file.

Which trend-file column feeds which cell:

| Table | Row | Compared against |
|-------|-----|------------------|
| Subsystem | GPU / System/Topo / CPU/ESMI / NIC / IFoE | `gpu_tested`, `systopo_tested`, `cpu_esmi_tested`, `nic_tested`, `ifoe_tested` |
| Subsystem | **Total** | `api_tested` |
| Framework | **Total** | `total_api` |
| Framework | Gtest C/C++, Python | *nothing -- these cells are intentionally blank* |

`delta()` renders `=` unchanged, `+N` / `-N` when it moves, and `-` when there
is no previous value -- which is why the first run in a fresh directory shows
`-` on every row.

**The per-framework rows carry no Trend.** The table has four metric columns
(API, Unit, Functional, CLI) but one Trend column, so a single cell cannot
describe all four; only **Total** gets one, on `total_api`. This is a known
limitation, not a bug.

### Subsystem buckets

Each API is bucketed by the `@ingroup` tag above its declaration in `amdsmi.h`
(the name alone is ambiguous -- `amdsmi_get_clk_freq` is a GPU API despite
having no `_gpu_` infix). Only NIC is name-based, because its APIs are split
across `tagNicInfo` and `tagProcDiscovery`.

| Bucket | Contents |
|--------|----------|
| GPU | everything not claimed below |
| System/Topo | init/shutdown, version, topology, error, node/tray |
| CPU/ESMI | every `tagEsmi*` group |
| NIC | any API whose name contains `_nic_` -- NIC and PCIe switch |
| IFoE | `tagFabric` -- Infinity Fabric over Ethernet |

IFoE is XGMI's counterpart (the GPU scale-up fabric), reported separately from
GPU because it is gated on its own kernel driver: `ifoe.ko` exposing
`/dev/cbl-cfg-*` plus a netlink family (see `src/ualoe_lib/`). Folded into GPU
it would silently depress that row on any host without the driver.

### Adjusting which tests run

GTest selection is `"<positive>-<negative>"`. The two halves come from different
places:

| Half | Source | Override |
|------|--------|----------|
| Positive | `*Unit*` / `*Functional*` in `run_api_coverage.sh` | `-U` / `-F` |
| Negative | `amdsmitst.exclude` -> `detect_asic_filter.sh` picks the list for the detected ASIC and exports `$GTEST_EXCLUDE` | edit `amdsmitst.exclude`, `-X` to append, `-A` to drop |

The defaults `*Unit*` and `*Functional*` together match every registered suite
(124 + 48 = 172 tests, no gap or overlap), so the only thing dropping tests is
the negative half. The split into two `amdsmitst` invocations is deliberate: it
keeps the Unit and Functional columns separate. Overlapping `-U`/`-F` patterns
would count an API in both columns.

```
sudo tests/run_api_coverage.sh -A                             # nothing excluded
sudo tests/run_api_coverage.sh -U 'GpuUnit*' -F 'GpuFunctional*'
sudo tests/run_api_coverage.sh -X 'GpuFunctionalReadWrite*'   # skip destructive
sudo tests/run_api_coverage.sh --no-python                    # C++ only
```

Each run echoes the filter it used, e.g.
`=== C++ unit ===   filter=*Unit*-GpuFunctionalReadWrite.TestPerfLevelReadWrite`.

### Trending across runs

Each run writes `api_coverage_trend.csv` into its own run directory, seeded
with the rows of the newest earlier `api-coverage-results-*` directory beside it
(override with `-H`, suppress with `--no_history`). See
[Where each number comes from](#where-each-number-comes-from) for which column
feeds which Trend cell, and [How the trend carries across
runs](#how-the-trend-carries-across-runs) for the inheritance rules.

Note the runner-to-column mapping: `unit_tests.py` drives `unit/` ->
`_py_unit_test.log`, and `integration_test.py` drives `functional/` ->
`_py_func_test.log`.

Coverage is hardware- **and driver**-dependent. CPU/ESMI needs an EPYC host with
HSMP, the NIC rows need a supported NIC, and IFoE needs `ifoe.ko`. On a consumer
GPU those suites skip and report 0%; the report's metadata block records each
driver's version (`amd-smi version` exposes `amd_hsmp_driver_version` and
`nic_driver_version`) so a low row is explainable rather than mysterious. For
this reason, only compare trend rows from comparable hardware.

### How an API is counted

`api_summary.py` scores an API tested only when a log line contains the literal
marker `### <api>(`. C++ emits it via `DISPLAY_AMDSMI_API()` before the call;
Python builds it into the `msg` string that `common.print()` / `check_ret()`
print -- both the success and exception paths print it, so a call is counted
either way.

The marker must have the API name immediately followed by `(`. These shapes are
silently invisible and make the API read untested:

```
### test amdsmi_get_lib_version()      <- extra word before the name
###Test amdsmi_get_device_id           <- no space, no parentheses
### amdsmi.amdsmi_get_utilization_count(   <- module prefix
### amdsmi_set_gpu_memory_partition idempotent(   <- text before the paren
### [sleeper] amdsmi_init(                <- bracketed tag before the name
Testing amdsmi_cpu_apb_disable on cpu 0   <- no marker at all
```

Unit tests are plain `TEST()`s with no `TestBase`, so the `VERB()` macro is
unavailable there -- gate on the global `GetTestVerbosity() >= 1` instead.

### Related tooling in progress

This report counts whether an API was invoked by *any* test -- it says nothing
about how much of that API's implementation actually ran. Two pieces of
upcoming work are relevant but distinct:

- Line/branch/function code coverage ([#9947](https://github.com/ROCm/rocm-systems/pull/9947)): a separate
  `tests/coverage/` tool (gcovr for C/C++, `coverage.py` for Python) measuring
  % of *implementation* lines/branches actually executed. An API can be
  "Tested" here (one invocation reaches it) while its coverage there is still
  low (most branches inside it are never exercised). The two numbers answer
  different questions and should be read together, not interchanged.
- Reusable unit-test fixtures ([#12494](https://github.com/ROCm/rocm-systems/pull/12494)): adds shared test helpers
  (e.g. an inotify-based `ScopedFileWatch` for "ran exactly once" assertions).
  `amdsmi_npm_limit_test.cc`/`rocm_smi_npm_test.cc` currently each define their
  own ad hoc `TempBoardDir`/`ScopedAmdSmiInit` helpers; once this lands, new
  unit tests needing similar fixtures should use the shared ones instead of
  duplicating them per file.

## Pre-Requisites Before Running Summary Report
Run the python and C++ tests prior to running api_summary.py script.  The preferred way to run the tests is as follows:

<u>The python scripts are in the directory /opt/rocm/share/amd_smi/tests/python_unittest</u>
```
sudo unit_tests.py -v > _py_unit_test.log 2> _py_unit_test_err.log
sudo integration_test.py -v > _py_intg_test.log 2> _py_intg_test_err.log
sudo functional_tests.py -v > _py_func_test.log 2> _py_func_test_err.log
```

<u>The C++ test is in the directory /opt/rocm/share/amd_smi/tests</u>

To run with ASIC-specific test exclusions (recommended):
```
cd /opt/rocm/share/amd_smi/tests
source amdsmitst.exclude
source detect_asic_filter.sh
sudo ./amdsmitst --gtest_filter="*Unit*.*-${GTEST_EXCLUDE}" -v 1 > _c_unit_test.log 2> _c_unit_test_err.log
sudo ./amdsmitst --gtest_filter="*Functional*.*-${GTEST_EXCLUDE}" -v 1 > _c_func_test.log 2> _c_func_test_err.log
sudo ./amdsmitst --gtest_filter="*Integration*.*-${GTEST_EXCLUDE}" -v 1 > _c_intg_test.log 2> _c_intg_test_err.log
```

`api_summary.py` reports the three C categories separately, so `amdsmitst` is run
once per category.  The positive pattern matches the GTest suite names (`GpuUnit`,
`GpuFunctionalReadOnly`, `GpuIntegration`, ...); everything after `-` is the
exclusion list.

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
sudo ./amdsmitst --gtest_filter="*Unit*.*-${BLACKLIST_ALL_ASICS}" -v 1 > _c_unit_test.log 2> _c_unit_test_err.log
sudo ./amdsmitst --gtest_filter="*Functional*.*-${BLACKLIST_ALL_ASICS}" -v 1 > _c_func_test.log 2> _c_func_test_err.log
sudo ./amdsmitst --gtest_filter="*Integration*.*-${BLACKLIST_ALL_ASICS}" -v 1 > _c_intg_test.log 2> _c_intg_test_err.log
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
  --c_func_test C_FUNC_TEST
    Filename for C functional test output, default=_c_func_test.log
  --c_intg_test C_INTG_TEST
    Filename for C integration test output, default=_c_intg_test.log
  --py_unit_test PY_UNIT_TEST
    Filename for Python unit test output, default=_py_unit_test.log
  --py_func_test PY_FUNC_TEST
    Filename for Python functional test output, default=_py_func_test.log
  --py_intg_test PY_INTG_TEST
    Filename for Python integration test output, default=_py_intg_test.log
Output File:
  --output_dir OUTPUT_DIR
    Path to output dir, will create if does not exist, default=build
```

Command line examples:
<details close>
  <summary>Click for example: <i><b>From amdsmi root directory with test output in build directory</i></b></summary>

~~~shell
api_summary.py
~~~
<i><b>With specifying summary output</i></b>
~~~shell
api_summary.py --output_dir summary_dir
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
API, Tested, c_unit_test, c_func_test, c_intg_test, py_unit_test, py_func_test, py_intg_test
amdsmi_alloc_fabric_telemetry, 2, 0, 1, 1, 0, 0, 0
amdsmi_clean_gpu_local_data, 3, 0, 0, 1, 0, 1, 1
amdsmi_cpu_apb_disable, 4, 0, 1, 1, 0, 1, 1
amdsmi_cpu_apb_enable, 4, 0, 1, 1, 0, 1, 1
amdsmi_fabric_telem_id_to_string, 1, 0, 0, 1, 0, 0, 0
amdsmi_first_online_core_on_cpu_socket, 2, 0, 0, 1, 0, 0, 1
amdsmi_free_fabric_telemetry, 2, 0, 1, 1, 0, 0, 0
amdsmi_get_afids_from_cper, 2, 0, 0, 1, 0, 0, 1
amdsmi_get_clk_freq, 3, 0, 1, 1, 0, 0, 1
amdsmi_get_clock_info, 2, 0, 0, 1, 0, 0, 1
...
~~~
</details>

<details close>
  <summary>Click for example: <i><b>api_summary_table.txt</i></b></summary>

~~~shell
 API   Test(  %  )  Unit(  %  )  Func(  %  )  Intg(  %  )
  C     245( 99.6)     0(  0.0)    82( 33.3)   243( 98.8)
 Py     224( 91.1)     0(  0.0)    41( 16.7)   224( 91.1)
Total   246(100.0)     0(  0.0)   105( 42.7)   246(100.0)
Num APIs: 246
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
