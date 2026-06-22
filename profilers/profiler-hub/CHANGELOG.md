# Changelog

All notable user-facing changes to the profiler-hub library are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Sections in each release entry:

- **Added** — new features, new public APIs, new build options
- **Changed** — changes to existing public behavior or APIs
- **Deprecated** — APIs that still work but will be removed
- **Removed** — APIs or build options that no longer exist
- **Fixed** — bug fixes visible to users
- **Security** — vulnerability fixes

Internal refactors and CI changes that have no user-visible impact do not need
a changelog entry. Release entries should be written from the perspective of a
downstream consumer of the library.

## [Unreleased]

### Added

- `libprofiler-hub.so` now ships with a SOVERSION (`libprofiler-hub.so.0` symlink and
  `libprofiler-hub.so.0.1.0` actual file) so consumers can pin to a specific ABI.
- New cache var `FMT_VERSION` (default `11.2.0`). When the system fmt is missing,
  the build fetches `fmtlib/fmt` at this version.

### Changed

- spdlog is now built with `SPDLOG_FMT_EXTERNAL=ON`. fmt is resolved as an
  independent dependency (via `find_package(fmt)` or FetchContent) rather than
  through spdlog's vendored copy. Internal includes switched from
  `<spdlog/fmt/bundled/core.h>` to `<fmt/core.h>`. Required to integrate
  profiler-hub into the TheRock super-project, which builds spdlog with
  `SPDLOG_FMT_EXTERNAL=ON` and rejects any duplicate fmt provider.
- Because fmt is now an external dependency, consumers of the installed
  `profiler-hub` CMake package (especially the static library) must have fmt
  discoverable; the package config calls `find_dependency(fmt)`.
- FetchContent fallback versions bumped to a compatible pair: spdlog `1.15.3`
  and fmt `11.2.0`. spdlog 1.14.x does not compile against fmt 11, so the
  external-fmt switch requires spdlog >= 1.15 when the system fmt is 11.x.
- `find_package(spdlog ...)`, `find_package(fmt ...)`, and the other system
  lookups keep their version variable as a minimum, so a system copy that is
  too old to satisfy the requirement falls back to FetchContent. A system
  spdlog is additionally accepted only when it was built with
  `SPDLOG_FMT_EXTERNAL`, to avoid linking two fmt copies into one binary.

### Removed

- Build options `PROFILER_HUB_USE_SYSTEM_SPDLOG`, `PROFILER_HUB_USE_SYSTEM_NLOHMANN_JSON`,
  `PROFILER_HUB_USE_SYSTEM_GTEST`, and `PROFILER_HUB_USE_SYSTEM_BENCHMARK`. These
  were always-on toggles that only suppressed the system `find_package` lookup;
  callers that need bundled builds can simply remove the system package or set
  `CMAKE_DISABLE_FIND_PACKAGE_<name>=ON`.

## [0.1.0] - 2026-05-05

Initial release.

### Added

- C++17 public API for storing and retrieving ROCm profiling data in the
  rocpd (SQLite) database format. Public types under `profiler-hub::` namespace:
  `storage_t`, `writer_t`, `reader_t`, `version_t`, plus the supporting
  type families in `writer_types`, `reader_types`, and `shared_types`.
- Schema versions 3.0.0 and 4.0.0, runtime-selectable.
- Both shared (`libprofiler-hub.so`) and static (`libprofiler-hub.a`) library
  variants built from a shared object set.
- CMake package config for downstream consumption:
  `find_package(profiler-hub REQUIRED)` resolves the namespaced
  `profiler-hub::profiler-hub` target, including a `Findprofiler-hub.cmake` module
  for non-CMake-config integrations.
- Build options: `PROFILER_HUB_BUILD_TESTS`, `PROFILER_HUB_BUILD_BENCHMARKS`,
  `PROFILER_HUB_ENABLE_LOGGING`, `PROFILER_HUB_ENABLE_COVERAGE`,
  `PROFILER_HUB_USE_SYSTEM_SPDLOG`, `PROFILER_HUB_USE_SYSTEM_GTEST`.
- System dependency support for SQLite3, spdlog, fmt, nlohmann_json,
  GoogleTest, and Google Benchmark, with FetchContent fallback for
  spdlog and GoogleTest when the system version is too old.
- Public install layout:
  - `<prefix>/lib/libprofiler-hub.{so,a}`
  - `<prefix>/include/profiler-hub/{reader,reader_types,shared_types,storage,version,writer,writer_types}.hpp`
  - `<prefix>/lib/cmake/profiler-hub/{profiler-hub-config,profiler-hub-config-version,profiler-hub-targets,Findprofiler-hub}.cmake`
- Cobertura code coverage reports via the `coverage-xml` CMake target.
- clang-tidy custom target using the bundled `.clang-tidy` configuration.

[Unreleased]: https://github.com/ROCm/rocm-systems/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ROCm/rocm-systems/releases/tag/v0.1.0
