---
name: rocprofsys
description: rocprofiler-systems / rocprof-sys project reference — structure, output formats, env vars, build/test/install
when_to_use: User asks about rocprofiler-systems / rocprof-sys project. Wants to configure/build/test/install/run/sample workload.
paths: "**/rocprofiler-systems/**,**/rocprof-sys/**,**/rocm-systems*/**"
---

# ROCm Systems Profiler -- Combined Skill

## 1. Project Overview

**rocprofiler-systems** (formerly Omnitrace) is a comprehensive profiling and
tracing tool for parallel applications (C, C++, Fortran, HIP, OpenCL, Python)
on CPU and GPU. C++17 codebase, CMake build system, part of the rocm-systems
monorepo (`projects/rocprofiler-systems/`).

### Output Formats

| Output Type | Environment Variable | Description |
| ------------- | --------------------- | ------------- |
| **Perfetto** | `ROCPROFSYS_TRACE=1` | Protobuf trace (viewable at ui.perfetto.dev) |
| **RocPD** | `ROCPROFSYS_USE_ROCPD=1` | SQLite database for offline analysis |

#### Perfetto Output Modes

| Mode | Variables | Status |
| ------ | ----------- | -------- |
| **Standard (Cached)** | `ROCPROFSYS_TRACE=1` | Current, recommended |
| **Legacy** | `ROCPROFSYS_TRACE=1` + `ROCPROFSYS_TRACE_LEGACY=true` | Legacy; has timing issues on short workloads (<0.5 s) |

Standard uses trace caching and is more robust. Legacy has a known bug where
counter tracks are not captured for workloads shorter than ~0.5 seconds.

### Project Structure

```text
rocm-systems/                          # Monorepo root
└── projects/
    └── rocprofiler-systems/           # Project root (source_dir)
        ├── CMakeLists.txt
        ├── CMakePresets.json
        ├── build/                     # Build directories
        │   ├── ci/
        │   ├── debug/
        │   ├── release/
        │   └── debug-optimized/
        ├── source/
        ├── external/
        ├── examples/
        ├── tests/
        ├── scripts/
        │   └── build-release.sh
        └── docs/
```

### CMake Presets

| Preset | Build Type | Use For | Testing |
| -------- | ----------- | --------- | --------- |
| `ci` | Release | CI builds | ON |
| `debug` | Debug | Development | ON |
| `debug-optimized` | RelWithDebInfo | Debug with perf | ON |
| `release` | Release | Production | OFF |
| `coverage` | Debug | Dev workflow w/ gcov coverage (see `rocprofsys-coverage`) | ON |

### Key CMake Options

| Option | Default | Description |
| -------- | --------- | ------------- |
| `ROCPROFSYS_BUILD_DYNINST` | OFF | Build Dyninst from source |
| `ROCPROFSYS_BUILD_TBB` | OFF | Build TBB from source |
| `ROCPROFSYS_BUILD_BOOST` | OFF | Build Boost from source |
| `ROCPROFSYS_BUILD_ELFUTILS` | OFF | Build elfutils from source |
| `ROCPROFSYS_BUILD_LIBIBERTY` | OFF | Build libiberty from source |
| `ROCPROFSYS_BUILD_TESTING` | ON | Enable tests |
| `ROCPROFSYS_USE_PYTHON` | ON | Enable Python support |
| `ROCPROFSYS_USE_MPI` | OFF | Enable full MPI support |
| `ROCPROFSYS_USE_PAPI` | ON | Enable PAPI hardware counters |

---

## 2. Resolving Paths Before Running Commands

The project lives inside the rocm-systems monorepo and may be checked out in
several git worktrees at once. **You resolve every path before each command
and use it explicitly.** Never reuse a path from memory or a previous task
without re-verifying it for the current one.

**`source_dir` resolution rules (in order):**

1. User named a path in this turn or a recent turn -- use that exactly.
2. A planning artifact (requirements.md / architecture.md / implementation.md)
   names the tree -- use that.
3. CWD is under the rocprofsys repo -- resolve via
   `git rev-parse --show-toplevel`, then confirm
   `projects/rocprofiler-systems/` exists under it.
4. Otherwise, **ask the user** which tree before running any command.

Also resolve, before acting:

| Item | How to resolve |
| --- | --- |
| `preset` | Match the task's target preset (debug / ci / release / debug-optimized). Do not silently switch presets to reuse an existing build. |
| Build dir | `build/<CMAKE_PRESET>` under `source_dir`, unless the user points at a non-preset build location. |
| Binary path | Under the resolved build dir, e.g. `build/<CMAKE_PRESET>/source/bin/<name>/<name>`. Verify it exists with `ls` before running it. |
| Output dir | `ROCPROFSYS_OUTPUT_PATH` if set by the task; otherwise the tool's default (current directory). |

---

## 3. Configure Phase

### Prerequisites

- CMake 3.21+
- GCC 7+ (or Clang if Dyninst is pre-installed)
- Ninja or Make

### Finding the Project

- Resolve via `git rev-parse --show-toplevel`, then look for
  `projects/rocprofiler-systems/` under it (see path resolution rules above).
- Must be the `projects/rocprofiler-systems/` directory, not the monorepo root.
- Must contain `CMakeLists.txt` and `CMakePresets.json`.

### Configuring

```bash
cd /path/to/projects/rocprofiler-systems
cmake --preset <CMAKE_PRESET>
```

Override individual options on top of presets with `-D` flags:

```bash
cmake --preset release \
  -DROCPROFSYS_BUILD_DYNINST=ON \
  -DROCPROFSYS_BUILD_TBB=ON \
  -DROCPROFSYS_BUILD_BOOST=ON \
  -DROCPROFSYS_BUILD_ELFUTILS=ON \
  -DROCPROFSYS_BUILD_LIBIBERTY=ON
```

Custom (no preset):

```bash
cmake -B build/my-custom-build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX=$HOME/rocprofsys-install \
  -DROCPROFSYS_BUILD_DYNINST=ON \
  -DROCPROFSYS_BUILD_TBB=ON \
  -DROCPROFSYS_BUILD_BOOST=ON \
  -DROCPROFSYS_USE_PYTHON=ON \
  -DROCPROFSYS_BUILD_TESTING=ON \
  -G Ninja
```

### Validation

- CMake exits 0 and prints "Build files have been written to..."
- Review enabled features and any warnings about missing optional deps.

---

## 4. Build Phase

### HARD RULE: never build a single CMake target in isolation

Always build the whole project. Do NOT pass `--target <name>` to
`cmake --build`, and do NOT name a single ninja/make target. Whole-project
builds catch cross-target breakage early; isolated target builds hide it.

### Building

```bash
cmake --build build/<CMAKE_PRESET> -j<N>
```

Never pass `--target`. Examples:

```bash
# Full build
cmake --build build/debug -j8

# Clean and rebuild
cmake --build build/debug -j8 --clean-first

# Verbose output (for debugging build issues)
cmake --build build/debug -j8 --verbose
```

### Parallelism Heuristic

~4 GB per parallel job. Safe default:
`SAFE_JOBS = min(nproc, total_mem_gb / 4)`.

### Running Tests

Never run `ctest`. Run the aggregated unit-test binary directly instead:

```bash
./build/<CMAKE_PRESET>/bin/rocprof-sys-unit-tests                        # all tests
./build/<CMAKE_PRESET>/bin/rocprof-sys-unit-tests --gtest_filter=<pattern> # single test
```

### Install

```bash
cmake --install build/<CMAKE_PRESET>
# Default prefix: /opt/rocprofiler-systems (may need sudo)
# Or override: cmake --install build/<CMAKE_PRESET> --prefix /custom/path
```

Verify:

```bash
ls <install-prefix>/bin/rocprof-sys-*
source <install-prefix>/share/rocprofiler-systems/setup-env.sh
```

### Build Artifacts Layout

```text
build/<CMAKE_PRESET>/
├── source/
│   ├── bin/
│   │   └── rocprof-sys-run/
│   │       └── rocprof-sys-run
│   └── lib/
│       └── rocprof-sys/
│           └── librocprof-sys.so
└── ...
```

### Common Build Errors

| Error | Cause | Fix |
| ------- | ------- | ----- |
| `c++: fatal error: Killed` | Out of memory | Reduce jobs: `-j2` or `-j4` |
| `fatal error: xyz.h: No such file` | Missing dependency | Reconfigure with `BUILD_<DEP>=ON` |
| `undefined reference to ...` | Dependency issue | Rebuild deps; clean rebuild |
| Error in `external/dyninst/` | External dep failure | Clean build or use system package |

---

## 5. Run / Sample Phase

### Listing available examples

```bash
ls <source_dir>/examples/
```

Check whether an example is built by looking for its binary under
`build/<CMAKE_PRESET>/` before running it.

### Querying available components / counters

```bash
build/<CMAKE_PRESET>/source/bin/rocprof-sys-avail/rocprof-sys-avail -H
```

### Running a workload

```bash
# Perfetto trace with GPU metrics
ROCPROFSYS_TRACE=1 \
ROCPROFSYS_AMD_SMI_METRICS="all" \
build/<CMAKE_PRESET>/source/bin/rocprof-sys-run/rocprof-sys-run -- ./my-app

# Legacy Perfetto mode
ROCPROFSYS_TRACE=1 \
ROCPROFSYS_TRACE_LEGACY=true \
ROCPROFSYS_AMD_SMI_METRICS="all" \
build/<CMAKE_PRESET>/source/bin/rocprof-sys-run/rocprof-sys-run -- ./my-app

# RocPD database output
ROCPROFSYS_USE_ROCPD=1 \
ROCPROFSYS_AMD_SMI_METRICS="all" \
build/<CMAKE_PRESET>/source/bin/rocprof-sys-run/rocprof-sys-run -- ./my-app

# NIC/AINIC metrics with fake devices
AMDSMI_FAKE_AINIC=1 \
ROCPROFSYS_TRACE=1 \
ROCPROFSYS_SAMPLING_AINICS="all" \
build/<CMAKE_PRESET>/source/bin/rocprof-sys-run/rocprof-sys-run -- ./my-app
```

### Sampling a workload

```bash
build/<CMAKE_PRESET>/source/bin/rocprof-sys-sample/rocprof-sys-sample -- ./my-app
```

Check the exit code (`echo $?`) and inspect the output directory
(`ROCPROFSYS_OUTPUT_PATH`, or the current directory by default) for the
generated trace/database.

---

## 6. Environment Variables Reference

### Runtime Variables

| Variable | Values | Description |
| ---------- | -------- | ------------- |
| `ROCPROFSYS_TRACE` | `0` / `1` | Enable Perfetto trace output |
| `ROCPROFSYS_TRACE_LEGACY` | `true` / `false` | Use legacy Perfetto implementation |
| `ROCPROFSYS_USE_ROCPD` | `0` / `1` | Enable RocPD SQLite output |
| `ROCPROFSYS_OUTPUT_PATH` | `<directory>` | Output directory for traces |
| `ROCPROFSYS_AMD_SMI_METRICS` | `all`, `power`, `temp`, ... | GPU metrics to collect |
| `ROCPROFSYS_SAMPLING_AINICS` | `all`, `none` | Enable NIC/AINIC sampling |
| `AMDSMI_FAKE_AINIC` | `0` / `1` | Simulate AINIC devices for testing |

### HIP Logging -- ALWAYS ask the user first

HIP logs help correlate what the workload actually did against what the
profiler recorded. They are **noisy and perturb timing**, so they are off by
default and never enabled silently.

**Hard rule:** before any run that *could* benefit from HIP logs (any
hard-validation run -- output assertion, regression check, profiler-vs-workload
diff, QA sign-off), the agent MUST ask the user whether to enable them:

- **Off** *(recommended for fast iteration)* -- no `AMD_LOG_*`, clean timing.
- **On -- curated mask** -- `AMD_LOG_LEVEL=3` + mask covering relevant
  categories (API/KERN/COPY/QUEUE/MEM).
- **On -- default mask** -- `AMD_LOG_LEVEL=3` + `AMD_LOG_MASK` unset (all
  categories, very noisy, only when nothing else surfaces the bug).

Never decide alone, even on QA sign-off. Routine runs (TDD red-to-green,
smoke, build verification, perf measurement) do not ask -- they always run
without HIP logs.

When ON, use:

| Var | Value | Why |
| ----- | ------- | ----- |
| `AMD_LOG_LEVEL` | `3` (INFO) | Captures API + kernel + copy events without DEBUG floods. Bump to `4` only if INFO is missing a category. |
| `AMD_LOG_MASK` | curated hex mask | Default `0x7FFFFFFF` enables everything and is unusable. Pick only categories with a counterpart in the profiler output being validated. |

Relevant mask categories (from ROCclr `amd_logging.hpp`; exact bits vary by
ROCm version -- read the installed header to resolve, or fall back to default
mask and filter downstream):

- `LOG_API` -- HIP API calls (matches profiler API track)
- `LOG_KERN` -- kernel launches (matches profiler kernel events)
- `LOG_COPY` -- memory copies (matches profiler memcpy/SDMA events)
- `LOG_QUEUE` -- stream/queue ops (matches profiler stream timeline)
- `LOG_MEM` -- alloc/free (matches profiler memory events)
- `LOG_AQL` -- only if validating AQL packet submission

Set the variables directly in the run's environment so the log output lands
next to the profiler artifacts:

```bash
AMD_LOG_LEVEL=3 \
AMD_LOG_MASK=<resolved hex> \
ROCPROFSYS_USE_ROCPD=1 \
ROCPROFSYS_AMD_SMI_METRICS="all" \
build/<CMAKE_PRESET>/source/bin/rocprof-sys-run/rocprof-sys-run -- ./my-app
```

---

## 7. Common Workflows

### Configure, build, run

```bash
cd /path/to/projects/rocprofiler-systems
cmake --preset debug
cmake --build build/debug -j8
ROCPROFSYS_TRACE=1 build/debug/source/bin/rocprof-sys-run/rocprof-sys-run -- ./my-app
```

### QA pipeline integration

When running a QA pass: configure/build/run/validate via the bash commands
above. Note the exact commands used in the qa-report for reproducibility.

### Clean rebuild

```bash
rm -rf build/<CMAKE_PRESET>
cmake --preset <CMAKE_PRESET>
cmake --build build/<CMAKE_PRESET> -j8
```

---

## Troubleshooting Quick Reference

| Issue | Solution |
| ------- | ---------- |
| "Cannot find Dyninst" | Configure with `-DROCPROFSYS_BUILD_DYNINST=ON` (and TBB/Boost/elfutils/libiberty) |
| "CMake version too old" | Install CMake 3.21+: `pip install --user cmake` |
| Out of memory during build | Reduce parallel jobs: `-j2` or `-j4` |
| Tests failing | Build with debug preset, run single test: `./build/debug/bin/rocprof-sys-unit-tests --gtest_filter=<name>` |
| Wrong project path | Must be `projects/rocprofiler-systems/`, not monorepo root |
| No counter tracks in Perfetto | Use standard mode (not legacy); check `ROCPROFSYS_AMD_SMI_METRICS` |
| Legacy mode: missing counters | Workload too short (<0.5 s); use standard mode |
| Example binary missing | Confirm it was built: `ls build/<CMAKE_PRESET>/examples/<name>/`; build the project first |
| Install permission denied | Use `sudo cmake --install` or reconfigure with user-writable prefix |

## Integration with Other Skills

| After This Skill | Use |
| ------------------ | ----- |
| Code changes made | `git-commit` |
| Feature added | `planning-feature` |
| Bug found | `planning-bugfix` |
| Ready to test | `testing-testplan` |
| Ready for PR | `git-prepare-pull-request` |
| Profiler crashes/errors | `debugging-rocprof-sys` |
| Verify PMC metrics | `verify-pmc-metrics` |
| AMD SMI API work | `library-amd-smi` |
