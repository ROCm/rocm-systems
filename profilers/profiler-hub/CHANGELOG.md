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
- Track-based reading on `reader_t`: `get_interval_track()` returns a track's interval
  events ordered by start, each carrying a packing lane, a nesting level and a containment
  `parent_id`; `get_scalar_track()` returns a counter or sample track's
  `(timestamp, value)` samples ordered by timestamp; `get_track_stats()` returns
  count/extent statistics for one track over an optional window.
- `get_event_info(event_id_t)` on `reader_t` — one per-event detail call covering every
  event type. It returns a fixed common header (name, category, start, end) plus a generic
  `properties` bag of named, typed values, so a consumer can render an event's detail
  without switching on its type.
- Directed, typed flow edges and four selectors to retrieve them: `get_flows()` for every
  edge in the capture, `get_flows_for_event()` for the edges touching one event,
  `get_flows_for_chain()` for one causal chain, and `get_flows_in_window()` for the edges
  intersecting a time window on a set of tracks, with an optional cap that keeps the
  highest-latency edges and is stable across pans.
- Opaque handle types `track_id_t` and `event_id_t`. They are copyable, comparable and
  hashable, and they round-trip through the reader; their integer content is a
  profiler-hub-private database identity and is not part of the public contract.
- Reader type vocabulary for the above: `event_info_t`, `arg_t` / `arg_value_t`,
  `interval_entry_t`, `scalar_sample_t`, `track_stats_t`, `flow_edge_t`, `flow_kind_t`,
  `track_type_t`, `region_track_kind_t`, `nesting_model_t`.
- `get_call_stack()`, `get_source_context()` and `get_arguments()` gained `event_id_t`
  overloads, so a consumer holding only an opaque handle can reach the per-event data
  without reconstructing a typed event.

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
- All track kinds are described by a single `track_info_t`, discriminated by
  `track_type_t`, instead of one struct per track kind. A consumer enumerates tracks once
  and dispatches on the type field.
- `reader_t::get_all_tracks()` is now `reader_t::get_tracks()`, and
  `reader_t::get_data_time_range()` is now `reader_t::get_time_range()`.
- Tracks and events are addressed by opaque handle rather than by a raw database row id,
  so no reader signature exposes the underlying capture's schema. Code that passed row ids
  must carry the handle returned by `get_tracks()`, `get_interval_track()`,
  `get_scalar_track()` or a flow edge instead.
- Flow edges are derived from an event's `stack_id` / `parent_stack_id` rather than its
  `correlation_id`, which is empty on captures produced by current ROCm releases. Captures
  that previously yielded no flows now yield the parent/child and sibling edges implied by
  their call stacks.
- `timestamp_ns_t` is now `timestamp_t` in `reader_types`, `writer_types` and
  `shared_types`. The underlying type and nanosecond unit are unchanged.

### Removed

- Build options `PROFILER_HUB_USE_SYSTEM_SPDLOG`, `PROFILER_HUB_USE_SYSTEM_NLOHMANN_JSON`,
  `PROFILER_HUB_USE_SYSTEM_GTEST`, and `PROFILER_HUB_USE_SYSTEM_BENCHMARK`. These
  were always-on toggles that only suppressed the system `find_package` lookup;
  callers that need bundled builds can simply remove the system package or set
  `CMAKE_DISABLE_FIND_PACKAGE_<name>=ON`.
- The seven typed per-event detail accessors on `reader_t` — `get_event_details()`,
  `get_region_details()`, `get_kernel_dispatch_details()`, `get_memory_copy_details()`,
  `get_memory_alloc_details()`, `get_sample_details()` and `get_pmc_event_details()`.
  `get_event_info()` replaces all seven.

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

[Unreleased]: https://github.com/ROCm/rocm-systems/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ROCm/rocm-systems/releases/tag/v0.1.0
