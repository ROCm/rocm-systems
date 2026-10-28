# Build and Install

How to configure, build, test, and install cpu-snap, plus the system
dependencies and the kernel `perf_event_paranoid` setting the perf backend
needs. This is the build-system reference; for what the library does and how
the pieces fit together see [`./architecture.md`](./architecture.md) and the
top-level [`../README.md`](../README.md).

## Layout at a glance

cpu-snap is one CMake project (`cpu_snap`, C++20) that produces three static
libraries from sibling directories under `src/`:

- `cpu_snap::core` (`src/core/`) - the shared wire format / chunk framing / reader.
- `cpu_snap::sampling` (`src/sampling/`) - the CPU callstack sampler.
- `cpu_snap::processing` (`src/processing/`) - offline processing of
  already-captured samples: `cpu_snap::processing::symbols` + `cpu_snap::processing::timeline`.

`sampling` and `processing` each link `core`. An interface target
`cpu_snap::all` links whichever of the three are enabled. Examples
(`examples/`), benchmarks (`bench/`), and tests (`tests/`) are optional
consumers built from the same top-level configure.

Minimum CMake is 3.21. The default `CMAKE_BUILD_TYPE` is `Release` when none is
set.

## Required dependencies

| Dependency | Needed by | Notes |
|---|---|---|
| C++20 compiler (GCC or Clang) | all | x86_64 Linux. GCC 13 `-Wmaybe-uninitialized` false positive in `std::variant` is suppressed automatically for `sampling`. |
| `Threads` (pthread) | `core` (transitive to all) | `find_package(Threads REQUIRED)`. `sampling` also links `rt` (`timer_create` on glibc < 2.34). |
| `libdw` / `libdwfl` (elfutils) | `cpu_snap::sampling` only | DWARF stack unwinding (`dwarf_unwinder`). Configure FATAL_ERRORs if missing. `core` builds without it. |
| GTest (`find_package(GTest)`) | tests only | If not found, the test tree is silently skipped (no failure). |

`addr2line` (binutils) is used at runtime by the offline `symbolizer` (it
shells out via `popen`); it is not a build dependency.

Install libdw for the sampling component:

| Distro | Package |
|---|---|
| Debian / Ubuntu | `libdw-dev` |
| Fedora / RHEL | `elfutils-devel` |
| Arch | `libelf` (provides libdw) |

The configure step probes libdw via `find_path(elfutils/libdwfl.h)` and
`find_library(dw)`; the same logic is re-exported to downstream consumers (see
[find_package](#consume-with-find_package)).

## CMake options

All options are cache variables you can pass with `-D`.

| Option | Default | Effect |
|---|---|---|
| `CPU_SNAP_BUILD_CORE` | `ON` | Build the shared core library. Force-enabled when sampling or processing is on. |
| `CPU_SNAP_BUILD_SAMPLING` | `ON` | Build the sampling component (pulls in libdw). |
| `CPU_SNAP_BUILD_PROCESSING` | `ON` | Build the processing component (symbol resolution + timeline shaping; no libdw). |
| `CPU_SNAP_BUILD_EXAMPLES` | `ON` | Build the demos under `examples/`. Each demo gates on the components it uses. |
| `CPU_SNAP_BUILD_TESTS` | `ON` | Build unit/integration tests and `enable_testing()`. Needs GTest. |
| `CPU_SNAP_BUILD_BENCH` | `ON` | Build the microbenchmarks under `bench/`. |
| `CPU_SNAP_INSTALL` | `ON` | Emit install rules + the `find_package(cpu_snap)` package config. |
| `CPU_SNAP_SANITIZER` | `none` | `none` / `asan` / `tsan` / `ubsan` / `asan+ubsan`. Sanitizers are mutually exclusive except `asan+ubsan`, which compose. Adds `-fno-omit-frame-pointer -g`. |
| `CPU_SNAP_COVERAGE` | `OFF` | gcov/clang-coverage instrumentation across all targets. Requires GCC or Clang; mutually exclusive with `CPU_SNAP_SANITIZER` (runtime libraries collide). |

Component selection is granular: a core-only configure still
builds the matching example and bench because those gate per-target. Example:

```bash
cmake -S . -B build-core \
    -DCPU_SNAP_BUILD_SAMPLING=OFF \
    -DCPU_SNAP_BUILD_TESTS=OFF
```

## Presets

`CMakePresets.json` defines configure/build/test presets so you do not have to
remember flag combinations. Each writes to its own binary dir.

| Preset | Build type | Sanitizer / coverage | Binary dir |
|---|---|---|---|
| `default` | Release | none | `build/` |
| `debug` | Debug | none | `build-debug/` |
| `asan` | Debug | `asan+ubsan` | `build-asan/` |
| `tsan` | Debug | `tsan` | `build-tsan/` |
| `ubsan` | Debug | `ubsan` | `build-ubsan/` |
| `coverage` | Debug | `CPU_SNAP_COVERAGE=ON` (bench off) | `build-coverage/` |

Matching `buildPresets` and `testPresets` exist for `default`, `asan`, `tsan`,
`ubsan`, and `coverage` (`debug` has a configure + build preset).

## Build

With presets:

```bash
cmake --preset default
cmake --build --preset default -j
```

Without presets:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`compile_commands.json` is exported and symlinked into the source root so
clangd/editors at the project root work without extra setup.

## Test

The test tree builds only when `CPU_SNAP_BUILD_TESTS=ON` and GTest is found.
Run via ctest:

```bash
ctest --preset default          # or: ctest --test-dir build --output-on-failure
```

Tests are organised by component under `tests/core/`, `tests/sampling/`,
and `tests/integration/`. Each component builds one test
executable; sampling splits into four targets where incompatible flags
(`-no-pie`, `-fomit-frame-pointer`) require separate binaries.

Important: many tests need no special privilege. Wire-format and decode tests
(framer, records, reader, etc.) feed synthetic input and do not touch
`perf_event_open`. Tests that drive real perf-backed sampling
(`test_stress*`, the streaming stress/equivalence tests, the bench smoke
wrapper) `GTEST_SKIP` when `perf_event_paranoid` is too high, so a restrictive
host degrades gracefully rather than failing. The signal-backend test
(`test_signal_sampler`) is CI-safe and needs no perf privilege.

### Sanitizers and coverage

Use the matching preset, then for coverage build the `coverage` target after
configuring with `CPU_SNAP_COVERAGE=ON`:

```bash
cmake --preset coverage
cmake --build --preset coverage -j
cmake --build build-coverage --target coverage   # runs ctest + gcovr/lcov, writes coverage_html/
```

The `coverage` target prefers `gcovr`, falling back to `lcov`+`genhtml`; if
neither is installed the target is not created. Auxiliary `format`,
`format-check`, and `tidy` targets exist when clang-format / clang-tidy are
found.

## Install

```bash
cmake --preset default
cmake --build build -j
sudo cmake --install build --prefix /usr/local
```

This installs the static libraries, the public headers
(`include/cpu_snap/...` for each enabled component), and the package config
(`cpu_snapConfig.cmake`, `cpu_snapConfigVersion.cmake`, `cpu_snapTargets.cmake`)
under `<libdir>/cmake/cpu_snap`. The version file uses `SameMajorVersion`
compatibility. Disable with `-DCPU_SNAP_INSTALL=OFF`.

## Consume with find_package

The installed package supports `core` and `sampling` components:

```cmake
find_package(cpu_snap REQUIRED)                       # umbrella: everything enabled at build time
find_package(cpu_snap REQUIRED COMPONENTS sampling)  # just the sampling component

target_link_libraries(myapp PRIVATE cpu_snap::sampling)
# or link whatever was enabled at build time:
target_link_libraries(myapp PRIVATE cpu_snap::all)
```

Importing an unknown component name makes the package config report
not-found with a clear message. When `sampling` is requested (explicitly or
via the default umbrella import), the config re-runs the libdw lookup and
exposes it as the `LibDw::LibDw` imported target, so consumers never call
`find_package(libdw)` or pass `-ldw` themselves. A `core`-only
import does not require libdw.

Public headers, once installed:

```cpp
#include "cpu_snap/sampling/sampler.hpp"
#include "cpu_snap/core/reader.hpp"
#include "cpu_snap/core/records.hpp"
#include "cpu_snap/core/types.hpp"
```

## perf_event_paranoid (perf backend only)

The default sampling capture path uses
`perf_event_open` for per-thread sampling, which the kernel gates on
`kernel.perf_event_paranoid`. Per-thread sampling needs the value `<= 2`, or
the `CAP_PERFMON` capability. Check and lower it temporarily:

```bash
cat /proc/sys/kernel/perf_event_paranoid     # need <= 2 for the perf backend
sudo sysctl kernel.perf_event_paranoid=1     # temporary; resets on reboot
```

The capability probe (`probe_sampler_capabilities()`) opens throw-away events
to detect what the kernel and CPU actually accept and needs no extra
privilege; the sampler exposes the result via `caps()`.

The signal backend is the privilege-free alternative: it samples with a
per-thread POSIX timer plus a realtime signal instead of `perf_event_open`, so
it needs no `perf_event_paranoid` change and no `CAP_PERFMON`. Select it at
sampler construction when the host kernel is locked down. See
[`./architecture.md`](./architecture.md) for the backend seam and the
perf-vs-signal trade-off.
