# Changelog

All notable user-facing changes to the rocpdsna library are documented in this file.

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

- `librocpdsna.so` now ships with a SOVERSION (`librocpdsna.so.0` symlink and
  `librocpdsna.so.0.1.0` actual file) so consumers can pin to a specific ABI.

## [0.1.0] - 2026-04-23

Initial release.

### Added

- C++17 public API for storing and retrieving ROCm profiling data in the
  rocpd (SQLite) database format. Public types under `rocpdsna::` namespace:
  `storage_t`, `writer_t`, `reader_t`, `version_t`, plus the supporting
  type families in `writer_types`, `reader_types`, `shared_types`,
  `storage_types`.
- Schema versions 3.0.0 and 4.0.0, runtime-selectable.
- Both shared (`librocpdsna.so`) and static (`librocpdsna.a`) library
  variants built from a shared object set.
- CMake package config for downstream consumption:
  `find_package(rocpdsna REQUIRED)` resolves the namespaced
  `rocpdsna::rocpdsna` target, including a `Findrocpdsna.cmake` module
  for non-CMake-config integrations.
- Build options: `ROCPDSNA_BUILD_TESTS`, `ROCPDSNA_BUILD_BENCHMARKS`,
  `ROCPDSNA_ENABLE_LOGGING`, `ROCPDSNA_ENABLE_COVERAGE`,
  `ROCPDSNA_USE_SYSTEM_SPDLOG`, `ROCPDSNA_USE_SYSTEM_GTEST`.
- System dependency support for SQLite3, spdlog, fmt, nlohmann_json,
  GoogleTest, and Google Benchmark, with FetchContent fallback for
  spdlog and GoogleTest when the system version is too old.
- Public install layout:
  - `<prefix>/lib/librocpdsna.{so,a}`
  - `<prefix>/include/rocpdsna/{reader,reader_types,shared_types,storage,storage_types,writer,writer_types}.hpp`
  - `<prefix>/lib/cmake/rocpdsna/{rocpdsna-config,rocpdsna-config-version,rocpdsna-targets,Findrocpdsna}.cmake`
- Cobertura code coverage reports via the `coverage-xml` CMake target.
- clang-tidy custom target using the bundled `.clang-tidy` configuration.

[Unreleased]: https://github.com/ROCm/rocm-systems/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ROCm/rocm-systems/releases/tag/v0.1.0
