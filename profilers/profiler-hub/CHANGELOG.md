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

- `ph_pmc_info_t.id` is `rocpd_info_pmc.id`, not the index of the counter in
  the PMC list. `reader_types::pmc_info_t` carries the same id.
- `reader_types::agent_info_t::id` is `rocpd_info_agent.id`, and
  `reader_types::process_info_t::id` is `rocpd_info_process.id`.
  `ph_agent_t.row_id` and `ph_process_t.row_id` carry the same values.
  `ph_agent_t.id` remains `type_index`, and `ph_process_t.id` remains the OS
  process id.
- `reader_types::thread_info_t::id` is `rocpd_info_thread.id`.
  `ph_thread_t.row_id` carries the same value. `ph_thread_t.thread_id` remains
  the OS thread id.
- `ph_stream_t` and `ph_queue_t` carry `nid` (`rocpd_info_node.id`) and `pid`
  (`rocpd_info_process.id`, not the OS process id).
- `reader_t::get_histogram()` and `ph_get_histogram()` return one histogram per
  track on a shared time grid. `max_buckets` is an upper bound: the grid never
  has more buckets than requested, and a short trace has fewer. Duration tracks
  count events that overlap each
  bucket and sum that overlap. PMC tracks average the samples in each bucket
  and hold the previous value across empty buckets between samples. Bucket 0
  starts at the earliest track timestamp.
- `ph_get_region_details()`, `ph_get_kernel_dispatch_details()`,
  `ph_get_memory_copy_details()`, and `ph_get_memory_alloc_details()` return
  the fields of `region_data_t`, `kernel_dispatch_data_t`, `memory_copy_data_t`,
  and `memory_alloc_data_t`. A missing row returns `PH_RESULT_SUCCESS` with
  `is_valid` 0. A mismatched event type returns `PH_RESULT_INVALID_ARGUMENT`.
  String pointers remain valid until `ph_ctx_free()`.
- `ph_get_call_stack()` returns the frames `reader_t::get_call_stack()` stores
  for one duration event. The event is identified by `ph_event_t::id` and
  `ph_event_t::type`. Index 0 is the top of the stack. An event with no call
  stack yields an empty list. String pointers remain valid until
  `ph_ctx_free()`.
- `ph_get_correlated_events()` returns the events `reader_t::get_correlated_events()`
  finds for one duration event. The source event is identified by
  `ph_event_t::id` and `ph_event_t::type`. The source event is excluded, and
  an event with no stack id yields an empty list. `ph_event_t::nesting_depth`
  is the depth of that event on its own track. The track is loaded when the
  depth is read. `ph_event_t::track_id` is `ph_track_t::id` of
  `timeline_event_t::track`, or of the track that supplied the depth when that
  pointer is null. A kernel, memory copy, or memory allocation uses its
  agent+queue track when that track exists, then its stream track, instead of
  the host thread track.
- `probe_file()` and the C equivalent `ph_probe_file()` detect a SQLite
  database, its `rocpd_metadata_<uuid>` table, and the schema version before a
  reader is opened. A trace that only has the `schema_version` tag is accepted
  when its value is a major number such as `3`, `major.minor` such as `3.0`, or
  `major.minor.patch` such as `3.0.1`. They do not create or modify the file.
  `ph_probe_file()` reports that outcome with `ph_result_t`. `ph_file_probe_t`
  carries the trace schema and schema version. `ph-c-client
  --probe` runs the C probe against one or more existing trace files. Schema 3
  is accepted. `ph_ctx_create()` returns `PH_RESULT_UNSUPPORTED_FILE`,
  `PH_RESULT_INVALID_FILE`, or `PH_RESULT_UNSUPPORTED_SCHEMA_VERSION` instead
  of opening an unsupported trace. Another file format is added by
  registering a detector, and another schema by registering the versions its
  reader accepts.
- `ph_get_node()` returns `stream_hardware_links`: one entry per distinct
  stream-to-hardware relationship in kernel-dispatch, memory-allocate, and
  memory-copy rows. `reader_t::get_stream_hardware_links()` returns the same
  relationship with the resolved track pointers. A missing agent, queue, or
  track has `is_valid` 0 (C) or a null track pointer (C++). Memory-copy links
  use the destination agent.
- `ph_get_node()` also returns the trace's threads, streams, queues, and PMC
  descriptions (`ph_thread_list_t`, `ph_stream_list_t`, `ph_queue_list_t`,
  `ph_pmc_info_list_t`). Optional numeric fields use `ph_optional_uint64_t`:
  `value` is 0 and `is_valid` is 0 when the trace did not record that field.
- New public C ABI (`c/profiler_hub.h`, `c/profiler_hub_types.h`):
  `ph_ctx_create`/`ph_ctx_free`, `ph_get_library_version`, `ph_get_schema_version`,
  `ph_get_track_list`, `ph_get_node`, `ph_get_track_events`, `ph_get_track_samples`,
  and the asynchronous operation handle `ph_future_t`.
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
- `ph_track_t`/`track_info_t` now carry `start_ts`/`end_ts` nanosecond
  timestamps spanning each track's events, computed for every track
  category (thread, pmc_agent, the 3 agent+queue categories, and stream —
  the latter combined across kernel-dispatch/memory-allocate/memory-copy).
- `track_info_t::value_range` and `ph_track_t::value_range` (`ph_value_range_t { min, max, is_valid }`)
  expose the smallest and largest sample value of PMC counter tracks. Other
  tracks, and counters with no non-NULL sample, have `value_range.is_valid == 0`.
- `ph_future_create`/`ph_future_wait`/`ph_future_cancel`/`ph_future_result`/
  `ph_future_free`: a future carries `on_progress` and `on_finished` callbacks and lets
  the caller wait for or cancel an asynchronous operation. New `ph_future_status_t`,
  `ph_progress_fn`, `ph_finished_fn` and `PH_RESULT_CANCELLED`.
- `ph_event_t` gains `id`, `type` (new `ph_event_type_t`) and `nesting_depth`. `id` is only
  unique together with `type`. `nesting_depth` is the number of events of the track, itself
  included, that are active when the event starts, relative to the whole track. A
  `ph_get_track_events()` request with a time window is now served from the
  whole-track events, so its events keep that nesting depth and come ordered by start.
- `ph_track_t::nesting_depth` is the largest `ph_event_t::nesting_depth` of the track (0 for
  PMC tracks).
- `ph_ctx_create()` takes a `ph_future_t` (null for the previous blocking behaviour).
  With a future it returns the context at once and reads the trace in the background,
  reporting progress and the end through the future's callbacks. Calls that return data
  wait for the read; `ph_ctx_free()` cancels a read that is still running.
- `ph_get_track_list()`, `ph_get_node()`, `ph_get_track_events()` and
  `ph_get_track_samples()` take a `ph_future_t` as their last argument (null for the
  previous blocking behaviour). With a future the call returns at once, the work runs on
  the context's thread pool, and the future ends with the `ph_result_t` the call would
  have returned or is cancelled; the out parameter may be read once it finished.
  `ph_ctx_free()` cancels the calls still waiting.
- `ph_ctx_create()` now reads the events of every track and the samples of every PMC
  track and keeps them in memory until `ph_ctx_free()`. Opening a trace takes longer
  and uses memory proportional to its event count; `ph_get_track_events()` and
  `ph_get_track_samples()` then never read the database.
- New `PH_TRACK_CATEGORY_THREAD_SAMPLE`/`track_kind_t::thread_sample`
  category: duration events explicitly tagged with a named track (via
  `writer_t::register_track_info()` + `trace_environment_t::track_name`)
  are now split out of the plain `thread` track into their own
  `thread_sample` track instead of being merged into it.

### Changed

- `storage_t::get_storage_version()` uses the same schema-version rules as
  `probe_file()`. All three split tags are required, otherwise
  `schema_version` must be `major`, `major.minor`, or `major.minor.patch`.
  A partial set of split tags is not a version.
- **Breaking:** `ph_process_t` now carries `ppid`, `init`, `fini`, `start`, `end`,
  `environment`, and `extdata` from the trace. The new numeric fields are
  `ph_optional_uint64_t`.
- **Breaking:** `ph_agent_t` now carries `extdata`. `absolute_index`,
  `logical_index`, and `uuid` are `ph_optional_uint64_t` instead of `uint32_t`,
  so a missing value is distinct from zero.
- **Breaking:** `ph_progress_fn` now takes a `ph_progress_description_t`
  (`const char*`) describing the work in progress. The pointer is valid only
  for the duration of the callback and may be null. Loading a trace reports
  the current phase (schema, catalog, track list, track load, node info).
- **Breaking:** public C++ headers moved from `<prefix>/include/profiler-hub/*.hpp`
  to `<prefix>/include/profiler-hub/cpp/*.hpp` (e.g.
  `#include <profiler-hub/storage.hpp>` becomes
  `#include <profiler-hub/cpp/storage.hpp>`). Groups the public C++ API
  headers under their own subdirectory, mirroring the language-scoped
  layout other public interfaces (e.g. a future C ABI) will use.
- `reader_types.hpp`: `counter_timeline_event_t::value` is now `double` (was `size_t`).
- `reader_types.hpp`: `timeline_event_t::display_name`/`category` are now
  `std::string_view` (were `std::string`).
- `ph_ctx_create()` is significantly faster when its internal connection
  pool has more than one connection: trace metadata (nodes/processes/
  threads/agents/tracks/code objects/kernel symbols/streams/queues/pmc
  info) is now built once and shared across all pooled connections,
  instead of every connection independently rebuilding its own copy. The
  independent categories are additionally fetched in parallel across the
  pool's own connections. Measured ~5x faster `ph_ctx_create()` on a
  5-connection pool against a 1.8GB trace (~780ms -> ~160ms
  steady-state).
- **Breaking (reader behavior):** `reader_t::get_all_tracks()` (and
  `ph_get_track_list()`) no longer derive `thread`/`pmc_agent` tracks from
  the trace's `rocpd_track` table. They're now derived directly from
  `GROUP BY` queries over the raw event/counter-sample tables, matching
  optiq's own discovery philosophy — a track only exists if it actually
  has events. A `writer_t::register_track_info()`-registered track with
  zero events is no longer reader-visible until it has data. Track ids are
  also now assigned densely in discovery order (`0,1,2,...`) instead of
  reusing the trace's own (sparse) `rocpd_track.id`. As a side effect,
  duplicate `rocpd_track` rows that happened to share the same
  `(nid,pid,tid)` (observed in real traces, previously exposed as
  separate, redundant tracks reporting identical event counts) now
  correctly collapse into a single track.

### Fixed

- `ph_ctx::initialize_track_list()` no longer crashes on tracks with no
  associated process (e.g. the new per-agent+queue category tracks, which
  have no `pid`) — `node_info`/`process_info` are now null-checked the same
  way `thread_info` already was.

### Fixed

- `storage_t::get_storage_version()` now reads the schema version actually
  stamped in the opened trace's `rocpd_metadata` table, instead of returning
  the profiler-hub library's own version (which can legitimately differ from
  the trace's schema version).
- Windows/MSVC build support. When SQLite is fetched, Windows downloads the
  official amalgamation zip instead of running `./configure`.

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
