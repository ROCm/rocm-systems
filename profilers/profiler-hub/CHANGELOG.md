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

- New public C ABI (`c/profiler_hub.h`, `c/profiler_hub_types.h`):
  `ph_ctx_create`/`ph_ctx_free`, `ph_get_library_version`, `ph_get_schema_version`,
  `ph_get_track_list`, `ph_get_node`, `ph_get_track_events`, `ph_get_track_samples`,
  and stub declarations for a planned async task API (`ph_future_get`/`ph_future_wait`/
  `ph_future_cancel`/`ph_future_free`, not yet implemented).
- `reader_types.hpp`: `track_info_t` now carries `id`, `event_count`, and `agent_id`.
- `reader_types.hpp`: new `track_kind_t` enum, and `track_info_t` gains
  `category`, `queue_id`, `stream_id` fields. `reader_t::get_all_tracks()`
  now also returns per-agent+queue tracks for kernel dispatch, memory
  allocate, and memory copy, plus one merged per-host-stream track
  combining all three operation kinds per stream (optiq discovery parity,
  including its "one stream track per stream, not per event type"
  behavior; previously only thread-based `(nid,pid,tid)` tracks and
  per-agent PMC tracks were derived).
- `c/profiler_hub_types.h`: new `ph_track_category_t` enum;
  `ph_track_t` gains `category`, `queue_id`, `stream_id` fields (mirrors
  `reader_types::track_kind_t`). New `ph_process_t`/`ph_process_list_t`;
  `ph_node_t` gains a `process_list` field.
- `reader_t::get_events_for_track()` (and `ph_get_track_events()`) now
  return real events for the 4 optiq-parity category tracks
  (`kernel_dispatch_agent_queue`/`memory_allocate_agent_queue`/
  `memory_copy_agent_queue`/`stream`); previously always empty for these.
  Time-window filtering is supported for these categories too.

### Changed

- **Breaking:** public C++ headers moved from `<prefix>/include/profiler-hub/*.hpp`
  to `<prefix>/include/profiler-hub/cpp/*.hpp` (e.g.
  `#include <profiler-hub/storage.hpp>` becomes
  `#include <profiler-hub/cpp/storage.hpp>`). Groups the public C++ API
  headers under their own subdirectory, mirroring the language-scoped
  layout other public interfaces (e.g. a future C ABI) will use.
- `reader_types.hpp`: `counter_timeline_event_t::value` is now `double` (was `size_t`).
- `reader_types.hpp`: `timeline_event_t::display_name`/`category` are now
  `std::string_view` (were `std::string`).

### Fixed

- `ph_ctx::initialize_track_list()` no longer crashes on tracks with no
  associated process (e.g. the new per-agent+queue category tracks, which
  have no `pid`) — `node_info`/`process_info` are now null-checked the same
  way `thread_info` already was.

## [0.2.0] - 2026-09-02

### Added

- `libprofiler-hub.so` now ships with a SOVERSION (`libprofiler-hub.so.0` symlink and
  `libprofiler-hub.so.0.2.0` actual file) so consumers can pin to a specific ABI.
- New cache var `FMT_VERSION` (default `11.2.0`). When the system fmt is missing,
  the build fetches `fmtlib/fmt` at this version.
- `writer_t` now accepts `NIC` as an agent type and as a PMC `target_arch`,
  alongside `CPU` and `GPU` (RocPD schema v3.0.1). Anything else still throws
  `std::invalid_argument`.

### Changed

- profiler-hub now requires a C++20-compatible compiler for all build and consumer paths.
- RocPD schema target version is 3.0.1 (includes NIC agent support above).
- Schema SQL is obtained at configure time by cloning `rocprofiler-sdk-rocpd` from
  `rocm-systems` (`versions/3.0.1` by default) and embedding it as generated headers,
  instead of using local bundled `.sql` files or an installed `rocprofiler-sdk-rocpd`
  package at build time.
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

### Changed

- `libprofiler-hub.so` no longer exports the bundled `sqlite3_*` symbols (sealed via
  hidden visibility + `--exclude-libs`), preventing collisions with other SQLite versions.

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

[Unreleased]: https://github.com/ROCm/rocm-systems/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ROCm/rocm-systems/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ROCm/rocm-systems/releases/tag/v0.1.0
