# amd-smi code-coverage tooling

Line/branch coverage for the amd-smi C/C++ library and Python (binding + CLI),
complementing the API-level metric produced by `tests/api_summary.py`.

- **C/C++** → [gcovr](https://gcovr.com/) reading clang's gcov data via `llvm-cov`.
- **Python** → [coverage.py](https://coverage.readthedocs.io/) (branch mode,
  including the `amd-smi` CLI subprocesses).

Everything is driven by [`run_coverage.sh`](run_coverage.sh); reports land in
`<repo>/coverage-report/`.

## Scope — what each tool measures

The two tiers are **complementary and do not overlap**: Python covers the
hand-written Python wrappers, C/C++ covers the compiled library. Neither sees
inside the other (coverage.py stops at the ctypes boundary).

**C/C++ (gcov/gcovr)** — the compiled amd-smi library, filtered to:

- `src/` — the amd-smi implementation (`amd_smi/*.cc`, the NIC `amdsmi_unified`
  sources).
- `rocm_smi/src/` — the bundled rocm-smi backend the library links against.

Excluded: test sources (`*_test.cc`), examples, GoogleTest, and the third-party
esmi library. Exercised by the `amdsmitst` GoogleTest suite (unit + functional)
running against the GPUs as root.

**Python (coverage.py)** — the hand-written amd-smi Python, two trees:

- Binding: `py-interface/` (installed as `amdsmi/`) — `amdsmi_interface.py`,
  `amdsmi_interface_utils.py`, `amdsmi_exception.py`, `__init__.py`.
- CLI: `amdsmi_cli/` — `amdsmi_cli.py`, `amdsmi_parser.py`, `amdsmi_commands.py`,
  `amdsmi_helpers.py`, `subcommands/*.py`.

Excluded: the generated `amdsmi_wrapper.py` (ctypes, 1:1 with the C header) and
test files. **Not** included: the compiled `libamd_smi.so` — when the binding
calls into the `.so` via ctypes, coverage.py counts the Python line making the
call but never the C/C++ implementation behind it (measure that with the C/C++
tier). Exercised by `unit_tests.py` (binding, mock-based), `integration_test.py`
(functional, against live GPUs) and `cli_unit_test.py` (spawns the real `amd-smi`
CLI). `UNIT_ONLY=1` runs only `unit_tests.py`.

## Prerequisites

### C/C++ tier

- **A compiler + a matching `gcov` reader.** The `.gcda`/`.gcno` data must be
  read by a `gcov` whose major version matches the compiler that produced it; a
  mismatch is silent and reports **0/0** (see “Choosing the gcov reader”). Use
  **either** toolchain — both are supported:

  - **GCC** (simplest — the `libgcov` runtime ships with gcc, nothing extra):

    ```bash
    sudo apt-get install -y gcc g++          # usually already present
    ```

    Reader: plain `gcov`, matched to your gcc (e.g. `gcov` 9 for gcc 9).

  - **clang** (needs the compiler-rt *profile* runtime, or the link fails with
    `cannot find libclang_rt.profile-x86_64.a`):

    ```bash
    sudo apt-get install -y clang-16 llvm-16 libclang-rt-16-dev
    ```

    Reader: `llvm-cov-16 gcov` (its major version must match the clang you build with).

  > On this box plain `clang`/`clang++` resolve to **clang 10**, so always build
  > with an explicit versioned driver (e.g. `clang-16`/`clang++-16`) and a
  > matching `GCOV_TOOL`. clang-16 and clang-10 have their profile runtime
  > installed; clang-14 does **not** (add `libclang-rt-14-dev` if you want 14).
- **gcovr** for the Python 3.8 interpreter:

  ```bash
  python3 -m pip install --user gcovr
  ```

### Python tier

- **coverage.py** for the *same* `python3` the tests run under (3.8 here; a stray
  `~/.local/bin/coverage` built for another Python fails with
  `No module named coverage`):

  ```bash
  python3 -m pip install --user coverage
  ```
- **The `amd-smi-lib` + `amd-smi-lib-tests` packages installed** — coverage.py
  measures the *installed* binding/CLI and needs the test suites present. Build
  with tests and install the package:

  ```bash
  cmake -S . -B build -DBUILD_TESTS=ON
  cmake --build build -j"$(nproc)"
  ( cd build && make package && sudo dpkg -i amd-smi-lib*.deb )   # DEB distros
  ```

## Quick start

```bash
# from projects/amdsmi -- run as your normal user, not under sudo
tests/coverage/run_coverage.sh all     # C/C++ + Python
tests/coverage/run_coverage.sh cpp     # C/C++ only
tests/coverage/run_coverage.sh py      # Python only
```

The tests themselves need root (hardware access, plus the Python runners'
`geteuid()==0` check), but the script escalates **only the test processes** with
`sudo` and keeps `gcovr`/`coverage` running as you (root can't import your
user-site tools), then hands the reports back.

**How the escalation is decided — and why a root container is fine:** the script
checks `id -u`. If you're **already root** (a Docker container running as root,
or you launched it with `sudo`), `id -u` is `0`, so it runs everything directly
with **no `sudo` call and no password prompt**. If you're a normal user, it
prepends `sudo` to just the test steps, prompting for your password once (unless
sudo is passwordless or already cached). So nothing here is too strict for a
root container — in that case it never shells out to `sudo` at all.

## Build + run in one command (opt-in)

By default the script only **runs** (against an existing instrumented build /
installed package). Two opt-in flags fold the build in so you don't have to
match `GCOV_TOOL` by hand or install to `/opt/rocm`:

```bash
# Build an instrumented build-cov with the given toolchain, then run.
# Auto-sets BUILD_DIR=build-cov and GCOV_TOOL to match the compiler.
tests/coverage/run_coverage.sh --build=clang-16 cpp
tests/coverage/run_coverage.sh --build=gcc cpp

# Build + stage-install to a user-writable prefix, then run BOTH tiers against
# the staged tree -- no `dpkg -i`, no /opt/rocm mutation, no stale-install skew.
tests/coverage/run_coverage.sh --build=clang-16 --stage all
```

- `--build[=gcc|clang|clang-NN]` — configures + builds `build-cov`
  (`-DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON`) and points `BUILD_DIR` +
  `GCOV_TOOL` at it. Default toolchain is `clang-16`. Builds only `amdsmitst`
  unless `--stage` is also set (which needs the full tree).
- `--stage[=DIR]` — `cmake --install`s the build into `DIR` (default
  `build-cov-stage`), which reproduces the install-tree layout the Python tests
  expect (`share/amd_smi/amdsmi`, `libexec/amdsmi_cli`, `bin/amd-smi`,
  `lib/libamd_smi.so`), and wires the Python tier there via `AMDSMI_PATH` /
  `LD_LIBRARY_PATH` / `PATH`. The Python tests then measure exactly what you
  built. **Root is still required to *run*** (the `geteuid()==0` gate); staging
  only removes the *install* root requirement. Auto-enabled when `--build` is
  combined with the Python tier (`all`/`py`), since a fresh build is the only
  correct thing to measure — pass `--installed` to opt out.
- `--installed` — Python tier measures the **system-installed** amd-smi package
  instead of staging. Opts out of the auto-stage above; may be stale relative to
  a fresh `--build`.

`install`-to-`/opt/rocm` is intentionally left out of the script (system-mutating,
needs `dpkg`); use `--stage` for a throwaway prefix, or the manual package
install in Prerequisites if you specifically want the system package.

## Summary table

Every run prints a combined C++/Python table at the very bottom and writes it to
`coverage-report/amdsmi_cc_summary.md` (Markdown) +
`coverage-report/amdsmi_cc_summary.txt` (text):

```
================  AMD SMI Code Coverage  ================
Date:         08-11-2026
C++ compiler: g++ 9.4.0
C compiler:   gcc 9.4.0

Metric    | C++                    | Python
--------- | ---------------------- | -----------------------
Lines     | 25.0% (6,026 / 24,068) | 51.6% (8,018 / 15,538)
Branches  | 15.5% (6,646 / 43,014) | 41.5% (2,972 / 7,163)
Functions | 32.1% (521 / 1,625)    | DNE

N/A   - tier not selected for this run.
ERROR - tier selected but no data (e.g. no .gcda; tests did not run).
DNE   - metric does not exist (coverage.py has no per-function metric).
```

- **Date** / **compiler** — the `C++ compiler` / `C compiler` lines record the
  drivers CMake used for `build-cov` (`CMAKE_CXX_COMPILER` / `CMAKE_C_COMPILER`),
  rendered as `<driver> <version>` (e.g. `g++ 9.4.0`, `clang++-16 16.0.6`).
  amd-smi has both `.cc` and `.c` sources, so both front-ends of the toolchain are
  used; they are normally the same version (e.g. `g++`/`gcc`, `clang++`/`clang`).
  When the C/C++ tier is not run the line collapses to a single `Compiler: N/A`.
  This matters because gcov's line/branch model is **compiler-specific** — GCC and
  Clang count "lines" and "branches" differently, so the numbers are only
  comparable **within the same compiler**, not across compilers (pick one as your
  canonical toolchain).
- **numbers** — the tier ran and produced data.
- **N/A** — the tier wasn't selected this run (e.g. C++ column under `py`).
- **ERROR** — the tier was selected but produced no data on our side (e.g. no
  `.gcda`; the C/C++ tests didn't run).
- **DNE** — the metric doesn't exist (Python `Functions`; coverage.py has no
  per-function metric).

## C/C++ coverage

### 1. Build with instrumentation

`ENABLE_COVERAGE` (top-level `CMakeLists.txt`) adds `--coverage -O0 -g` to the
compile and link flags. Use a dedicated build dir so it doesn't disturb your
normal build, and only build the test target (much faster than the full
package/install path). Build with **either** toolchain:

**GCC:**

```bash
CC=gcc CXX=g++ cmake -S . -B build-cov \
  -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON -DENABLE_COVERAGE=ON \
  -DBUILD_EXAMPLES=OFF
cmake --build build-cov --target amdsmitst -j"$(nproc)"
```

**clang-16:**

```bash
CC=clang-16 CXX=clang++-16 cmake -S . -B build-cov \
  -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON -DENABLE_COVERAGE=ON \
  -DBUILD_EXAMPLES=OFF
cmake --build build-cov --target amdsmitst -j"$(nproc)"
```

### 2. Run + report

Match the reader (`GCOV_TOOL`) to the compiler you built with — the script
default is `llvm-cov-14 gcov`:

```bash
# GCC build:
GCOV_TOOL="gcov" BUILD_DIR=build-cov tests/coverage/run_coverage.sh cpp

# clang-16 build:
GCOV_TOOL="llvm-cov-16 gcov" BUILD_DIR=build-cov tests/coverage/run_coverage.sh cpp
```

This runs `amdsmitst` (emitting `.gcda` next to the objects) and then `gcovr`.
Output:

- `coverage-report/cpp/index.html` — browsable HTML.
- `coverage-report/cpp-summary.txt` — text summary.

### Choosing the gcov reader (important)

The `.gcda`/`.gcno` files must be read by a `gcov` whose version matches the
**compiler that produced them**. Mismatch is silent: the reader prints
`Unexpected version: ... Invalid .gcno File!` and gcovr then reports **0/0**.

- Built with **GCC** → plain `gcov`, matched to your gcc: `GCOV_TOOL="gcov"`.
- Built with **clang N** → `llvm-cov-N gcov`: e.g. `GCOV_TOOL="llvm-cov-16 gcov"`.
- The script default is `llvm-cov-14 gcov`; override via `GCOV_TOOL` to match
  whatever you built with.

To confirm a reader matches, run it on a single note file:

```bash
llvm-cov-16 gcov -b <build>/…/some_file.cpp.gcno   # clang build — prints coverage = good
gcov         -b <build>/…/some_file.cpp.gcno       # GCC build
```

## Python coverage

No build step — coverage.py instruments at runtime against the **installed**
package:

```bash
tests/coverage/run_coverage.sh py
```

- Config: [`.coveragerc`](.coveragerc) (branch mode; omits the generated
  `amdsmi_wrapper.py` and the test files; maps installed copies back to the
  `py-interface/` and `amdsmi_cli/` source trees for reporting).
- Covers both the binding (`amdsmi_interface`) and the `amd-smi` CLI, including
  code that runs in spawned CLI **subprocesses** (via `COVERAGE_PROCESS_START` +
  a `sitecustomize` hook), then `coverage combine`.
- Output: `coverage-report/py/html/index.html` and `coverage-report/py-summary.txt`
  (the summary file also gets a **gcovr-style** `lines` + `branches` covered/total
  block appended, derived from `coverage json`, to match the C/C++ table).

## Environment knobs

| Var | Default | Meaning |
|-----|---------|---------|
| `BUILD_DIR` | `<repo>/build` | Build tree holding the instrumented `amdsmitst`. |
| `OUT_DIR` | `<repo>/coverage-report` | Where reports are written. |
| `GCOV_TOOL` | `llvm-cov-14 gcov` | gcov reader matching the build compiler. |
| `CLI_SRC` | `/opt/rocm/libexec/amdsmi_cli` | amd-smi CLI source dir for Python coverage. |
| `TEST_FILTER` | *(all)* | `--gtest_filter` passed to `amdsmitst`. |
| `UNIT_ONLY` | `0` | `1` = coverage from the **unit tests only** (C/C++ gtest filter `*Unit*`; Python skips `cli_unit_test.py`). Set `OUT_DIR` too so it doesn't overwrite the full-suite report. |
| `VERBOSE` | `0` | `1` (or the `--verbose`/`-v` flag) logs **per-test** Python output (`-v`: every test name + skip reason) instead of `-q`. C/C++ gtest already lists each `[RUN]`/`[OK]`/`[SKIPPED]`. |

Each test suite is also bracketed with `>>> START <suite>` / `<<< END <suite> (exit N)`
markers and the exact command, so you can see what ran and attribute any library
stderr (e.g. `Exception caught: …`) to the suite that produced it. Combine with
`--verbose` to see which individual tests ran or skipped:

```bash
tests/coverage/run_coverage.sh --verbose py
```

Unit-only vs full-suite measure different things: the full suite (default)
reports how much code the whole test suite exercises — for C/C++ that's mostly
the **functional** GPU tests hitting hardware, so unit-only will be much lower.
Example:

```bash
# unit-only, written to a separate dir
UNIT_ONLY=1 OUT_DIR=coverage-report-unit tests/coverage/run_coverage.sh all
```

The C/C++ run resets accumulated `.gcda` counters first, so each invocation's
report reflects only the tests it ran.

## Gotchas

- **Runners require root.** `unit_tests.py` / `cli_unit_test.py` /
  `integration_test.py` (via `common.run_test_dir`) `sys.exit(1)` when not
  `euid==0`, so without escalation they exit before running a single test and
  coverage reports **0%** with `No data was collected`. The script handles this
  automatically (see Quick start); if you invoke coverage manually, run the
  suites under `sudo`.
- **Match the Python version.** `coverage`/`gcovr` must be importable by the
  same `python3` the tests use (3.8 here). A `~/.local/bin/coverage` from a
  different Python silently fails with `No module named coverage`.
- **gcov version mismatch → 0/0.** See “Choosing the gcov reader” above.
- **Root-owned report dirs.** If you run the whole script as root (e.g. `sudo`
  or a root container), the report dirs are created root-owned; a later run as a
  normal user then hits `Permission denied`. Clean with
  `sudo rm -rf coverage-report` and re-run.

## Cleaning up generated files

Coverage leaves three kinds of artifacts; none are tracked by git (all live
under `coverage-report/`, `build*/`, or as loose `.coverage*` files).

The script can do this for you (each must be used **alone**, no other args):

- `tests/coverage/run_coverage.sh --clean` — removes the build/intermediate
  artifacts (`build-cov/`, `build-cov-stage/`, loose `.coverage*`) but **keeps
  the whole `coverage-report/`** (HTML, summaries, JSON) so you can still view it.
- `tests/coverage/run_coverage.sh --clean-all` — removes everything, including
  `coverage-report/`.

Both only ever touch coverage-owned paths and are sudo-aware for root-owned
artifacts — never a user `BUILD_DIR`. The manual commands below do the same by
hand, if you'd rather pick and choose:

```bash
# 1. Reports (HTML + text + coverage.py data + effective rc). Prepend sudo if a
#    root run left them root-owned.
rm -rf coverage-report            # or: sudo rm -rf coverage-report

# 2. C/C++ runtime data (.gcda) — clears the counts but keeps the instrumented
#    build, so the next `cpp` run just re-executes the binary (no recompile).
find build-cov -name '*.gcda' -delete
#    Also drop the compile-time notes (.gcno) for a full reset:
find build-cov \( -name '*.gcda' -o -name '*.gcno' \) -delete

# 3. Stray coverage.py data files, if a manual run wrote them outside OUT_DIR.
rm -f .coverage .coverage.*
```

To remove the instrumentation entirely, delete the dedicated coverage build dir
(the next report needs a fresh `-DENABLE_COVERAGE=ON` build):

```bash
rm -rf build-cov
```

Nuke everything in one line:

```bash
sudo rm -rf coverage-report build-cov build-cov-stage && rm -f .coverage .coverage.*
```

## Baseline (2026-08-11, full suite as root, gcc 9.4.0)

Snapshot from a full `run_coverage.sh --build gcc all` run; see
[Scope — what each tool measures](#scope--what-each-tool-measures) for exactly
which sources each tier covers and what drives them. The two percentages measure
**disjoint** code — the compiled library vs. the hand-written Python wrappers —
and are **not** additive. gcov's line/branch model is compiler-specific, so these
C/C++ figures are **gcc's**; a clang build reports different totals.

**C/C++** (`g++` / `gcc` 9.4.0):

| Metric | Coverage | Covered / Total |
|--------|----------|-----------------|
| Lines | 25.0% | 6,026 / 24,068 |
| Branches | 15.5% | 6,646 / 43,014 |
| Functions | 32.1% | 521 / 1,625 |

**Python** (coverage.py, compiler-independent):

| Metric | Coverage | Covered / Total |
|--------|----------|-----------------|
| Lines | 51.6% | 8,018 / 15,538 |
| Branches | 41.5% | 2,972 / 7,163 (956 partial) |
| Functions | N/A | coverage.py has no per-function metric |

Open the HTML reports (`coverage-report/cpp/index.html`,
`coverage-report/py/html/index.html`) for the per-file breakdown.
