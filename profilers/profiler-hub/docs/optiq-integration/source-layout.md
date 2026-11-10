# Profiler Hub Source Layout

This document lists every profiler-hub source file and what it is for. Paths
are relative to `profilers/profiler-hub/`. See
[reader-architecture.md](reader-architecture.md) for how the files work
together.

Each file has one role, the side it mainly serves:

| Role | Meaning |
|---|---|
| Reader | Reads a trace through the C or C++ reader API. |
| Writer | Writes a trace through `writer_t`. |
| Shared | Used by both, or by the C interface as a whole. |
| Cache | Keeps timeline or catalog data in memory or on disk. |
| Trim | Writes a time-window copy of a trace. |
| Build | CMake and code generation. |

## Directory tree

```text
profiler-hub/
  CMakeLists.txt
  include/
    c/                 public C API (reader, types, trim)
    cpp/               public C++ API (reader, writer, storage, types)
  source/
    *.cpp, *.hpp       context, C API, caches, catalog, queries, trim, writer core
    common/            connections, pools, threads, logging, helpers
    data_storage/      SQLite backend, schema-3 statements, schema SQL template
    formats/           format registry
      rocpd_v2/        legacy rocpd (.rpd, schema 2) reader and trim rules
      rocpd_v3/        rocpd schema 3 reader, histogram, tables, trim rules
    queries/           fluent SQL builders
    writers/           writer interfaces and schema-3 writers
  tests/
    unit/              Google Test unit tests
    benchmarks/        writer benchmarks
    find_package/      installed-package smoke consumer
  docs/                architecture documents and graphs
```

## Root

| File | Role | Purpose |
|---|---|---|
| `CMakeLists.txt` | Build | Configures the C++20 project, the object library, the schema options, the per-format switches `PROFILER_HUB_ENABLE_ROCPD_V2` / `_V3`, and install rules. |

## include/c

| File | Role | Purpose |
|---|---|---|
| `profiler_hub.h` | Reader | Public C reader API: probe a trace, open a context, query tracks, events, samples, details, tables, summaries, and text. |
| `profiler_hub_trim.h` | Trim | `ph_trim_trace`, which writes a time-window copy of an opened trace. |
| `profiler_hub_types.h` | Shared | C result codes, opaque handles, track categories, events, filters, and histogram structs. |

## include/cpp

| File | Role | Purpose |
|---|---|---|
| `CMakeLists.txt` | Build | Lists the public C++ headers exported by the library. |
| `reader.hpp` | Reader | `reader_t` and `probe_file`: catalog, timeline, detail, table, and summary queries. |
| `reader_types.hpp` | Reader | Reader enums and structs for events, filters, tracks, histograms, probe results, and `trace_format_t`. |
| `shared_types.hpp` | Shared | Call-stack, program-counter, and source-line types used by the writer and the reader. |
| `storage.hpp` | Shared | `storage_t`, which owns one SQLite trace path and UUID for a writer or reader. |
| `trimmer.hpp` | Trim | C++ include that re-exports the C trim API. |
| `version.hpp` | Shared | `version_t`, a major, minor, patch triple. |
| `writer.hpp` | Writer | `writer_t`: registers info rows and inserts regions, kernels, copies, allocations, and PMC events. |
| `writer_types.hpp` | Writer | Non-owning writer structs and ids for info tables and event inserts. |

## source: context and C API

| File | Role | Purpose |
|---|---|---|
| `profiler_hub_c_api.cpp` | Shared | Every C function. Checks arguments, runs work now or on a future, catches exceptions, and calls `ph_ctx`. |
| `ph_ctx.hpp`, `ph_ctx.cpp` | Reader | The opaque context. Opens each trace file through its format's registry entry, loads one or more traces, builds the published track list and global ids, owns caches and result buffers, and serves every query by asking each file's source. |
| `ph_future.hpp`, `ph_future.cpp` | Shared | The opaque `ph_future`: progress, cancel, and completion for one operation. |
| `pending_futures.hpp`, `pending_futures.cpp` | Reader | Futures that were submitted but not finished. Cancels them when the context is destroyed. |

## source: format-neutral reader

| File | Role | Purpose |
|---|---|---|
| `reader.cpp` | Reader | `reader_t` methods. Forwards each query to the format-specific implementation. Holds the catalog getters and default empty results. |
| `format_source.hpp` | Reader | `format_source`, one open trace file as the context sees it: the catalog and its loading, track reads, which track an event is on, per-event reads, table queries, summaries, and the histogram. A registry entry opens one per file. A read a format declines is checked first and throws if called anyway. |
| `reader_impl.hpp` | Reader | `reader_t::impl`, the abstract per-connection read backend of the rocpd formats: catalog build, timeline, details, correlation, tables, summaries, and histograms. |
| `reader_catalog.hpp`, `reader_catalog.cpp` | Reader | The shared in-memory catalog of strings, info rows, tracks, and lookup maps, and its schema-3 builders. |
| `populate_reader_catalog.hpp`, `populate_reader_catalog.cpp` | Reader | Fills one catalog in parallel across the pool's connections. |
| `file_probe.cpp`, `file_probe_failure.hpp` | Reader | Checks that the file exists, is a regular file and is readable, then asks each format registry entry in order whether the file is its own and returns that entry's probe result, without loading the file. The exception carries a failed probe result. |
| `event_query.hpp`, `event_query.cpp` | Reader | The resolved table query (time window, tracks, string and kernel-symbol ids) and case-insensitive term matching. |
| `event_partitions.hpp`, `event_partitions.cpp` | Reader | Splits an event id span into ranges for parallel reads and merges the results by start time. |
| `depth_tracker.hpp` | Reader | Gives each event the lowest free nesting level as events are added in start order. |
| `histogram_grid.hpp` | Reader | Histogram bucket bounds and per-track overlap counts. Shared by the formats and the timeline cache. |
| `track_window.hpp` | Reader | Turns a C time window into a reader filter. |
| `track_read_options.hpp` | Reader | Event-count threshold and part count for parallel whole-track reads. |
| `track_row_reader.hpp` | Reader | Track reads on one leased connection, returned on destruction. |
| `connection_source.hpp` | Reader | Interface a source hands to `cached_track_reader` for whole-track reads; it leases a `track_row_reader`. |
| `pooled_connection_source.hpp`, `pooled_connection_source.cpp` | Reader | The rocpd source's `connection_source`: it borrows readers from a `connection_pool`. |

## source: caches

| File | Role | Purpose |
|---|---|---|
| `cache_directory.hpp`, `cache_directory.cpp` | Cache | Names the per-trace cache folder: `<date>_<format>_<version>_<name>_<size>`. |
| `cached_track_reader.hpp`, `cached_track_reader.cpp` | Cache | In-memory per-track cache of sorted events, nesting depth, and samples. Uses the timeline files when they are attached. |
| `track_file_store.hpp`, `track_file_store.cpp` | Cache | The `timeline/` folder: per-track event, sample, and depth-index files, strings, and the stored histogram. |
| `catalog_aggregate_store.hpp`, `catalog_aggregate_store.cpp` | Cache | The `catalog/` folder: schema-2 aggregates (PHC1) and schema-3 derived tracks (PHD1). |

## source: trim

| File | Role | Purpose |
|---|---|---|
| `trace_trim.hpp`, `trace_trim.cpp` | Trim | Copies tables, views, and indexes into a new database, keeping only rows the trim rules allow. |
| `trim_schema.hpp` | Trim | Interface for per-table trim SQL and the copy-phase order. |

## source: storage and writer core

| File | Role | Purpose |
|---|---|---|
| `storage.cpp`, `storage_impl.hpp`, `storage_impl.cpp` | Shared | `storage_t`: path, UUID, lazy database creation, and the cached schema version. |
| `version_parser.hpp` | Shared | Parses `major[.minor[.patch]]` text into `version_t`. |
| `json_serializers.hpp`, `json_serializers.cpp` | Shared | Converts call stacks and source context to and from JSON text. |
| `writer.cpp` | Writer | `writer_t` methods. Forwards register and insert calls to the implementation. |
| `writer_impl.hpp`, `writer_impl.cpp` | Writer | Policy-based writer core that owns the info and event writers. Builds the schema-3 writer. |
| `autoincrementer.hpp` | Writer | Atomic counter that hands out the next primary key. |
| `primary_key_providers.hpp` | Writer | One autoincrementer per table the writer inserts into. |
| `entity_registry.hpp` | Writer | Per-entity registries of nodes, processes, agents, strings, and other info rows. |
| `entity_utility.hpp` | Writer | Thread-safe set or map of registered entities and their primary keys. |
| `insert_validator.hpp`, `insert_validator.cpp` | Writer | Rejects an insert that names an entity that was never registered. |
| `CMakeLists.txt` | Build | Adds the library sources, including the optional rocpd v2 and v3 readers. |

## source/common

| File | Role | Purpose |
|---|---|---|
| `connection.hpp`, `connection.cpp` | Reader | One reader connection: a `reader_t` built for the probed format. |
| `connection_pool.hpp`, `connection_pool.cpp` | Reader | Blocking pool of reader connections that share one catalog. |
| `helper_group.hpp` | Reader | Submits helper tasks and waits for them without deadlocking the same pool. |
| `natural_merge_sort.hpp` | Reader | Stable merge of already-sorted runs, used to join event partitions. |
| `thread_pool.hpp`, `thread_pool.cpp` | Shared | Fixed worker pool with a FIFO of cancellable tasks. |
| `debug.hpp`, `logger.hpp` | Shared | `LOG_*` macros (compiled out unless logging is on) and the spdlog singleton. |
| `directory.hpp` | Shared | Parent-directory, existence, and create-directory helpers. |
| `traits.hpp` | Shared | Compile-time traits for optionals, maps, string literals, and SQLite bind types. |
| `string_conversions.hpp` | Writer | Key and string helpers for writer entity validation. |
| `types.hpp` | Writer | Owning copies of writer ids used inside the entity registry. |

## source/data_storage

| File | Role | Purpose |
|---|---|---|
| `read_statements.hpp` | Reader | Schema-3 SELECT result structs and prepared read statements. |
| `schema_version.hpp` | Shared | Reads rocpd metadata version tags into a schema version. |
| `schema/rocpd_schema.in` | Build | CMake template that turns one schema SQL fragment into a constexpr string. |
| `schema_v3/insert_statements.hpp` | Writer | Prepared INSERT statements for every schema-3 table. |
| `backends/database_backend.hpp` | Shared | SQLite engine templated on a C API policy: open, execute, bind, flush. |
| `backends/sqlite_api_policy.hpp` | Shared | Production policy that calls the sqlite3 C API. |
| `backends/sqlite_backend.hpp`, `backends/sqlite_backend.cpp` | Shared | The production backend alias and its explicit instantiation. |
| `backends/sqlite_backend_impl.hpp` | Shared | Opens the database, applies schema SQL, and runs statements. |
| `CMakeLists.txt` | Build | Adds the backend source to the object library. |

## source/formats

| File | Role | Purpose |
|---|---|---|
| `trace_formats.hpp`, `trace_formats.cpp` | Shared | The format registry. One entry per format; an entry recognises its files, probes them, says which versions this build reads, and opens a source for a file, and maps a version to a reader factory and a trim-rule factory. The rocpd entry is always built; the format switches remove a version's reader and trim sources. |

### source/formats/rocpd

| File | Role | Purpose |
|---|---|---|
| `rocpd_probe.hpp`, `rocpd_probe.cpp` | Reader | Recognises a rocpd file by the SQLite header and probes its metadata tables for the schema version and GUIDs. |
| `rocpd_source.hpp`, `rocpd_source.cpp` | Reader | The rocpd source. Owns the catalog, the pool of eight connections and the pooled connection source. Loads the catalog (the parallel fill, then the v2 aggregates or v3 derived tracks when the cache missed) and answers each read through one leased connection. |

### source/formats/rocpd_v2 (legacy .rpd)

| File | Role | Purpose |
|---|---|---|
| `rocpd_v2_reader.hpp` | Reader | `rocpd_v2_reader` and the row types for API calls and GPU operations. |
| `rocpd_v2_reader.cpp` | Reader | Timeline, details, correlation, call stack, samples, histogram, counts, and topology indexes over `rocpd_api`, `rocpd_op`, `rocpd_api_ops`, and `rocpd_monitor`. |
| `rocpd_v2_catalog.cpp` | Reader | Builds the catalog. Synthesizes the node, processes, threads, agents, and queues the schema does not store. Loads the catalog cache when it matches. |
| `rocpd_v2_table_events.cpp` | Reader | Paged table rows and the SQL name-duration summary. |
| `rocpd_v2_trim_schema.cpp` | Trim | Trim rules: overlapping API, operation, and monitor rows, then their links. |

### source/formats/rocpd_v3

| File | Role | Purpose |
|---|---|---|
| `rocpd_v3_reader.hpp`, `rocpd_v3_reader.cpp` | Reader | Catalog categories, timeline, details, correlation, call stack, and counts over GUID-suffixed tables. Loads the derived-track cache when it matches. |
| `rocpd_v3_histogram.cpp` | Reader | SQL that buckets region, kernel, memory, and HIP durations. |
| `rocpd_v3_table_events.cpp` | Reader | Paged table rows and the SQL name-duration summary. |
| `rocpd_v3_trim_schema.cpp` | Trim | Trim rules by table prefix: intervals, samples, events, and child rows. |

## source/queries

| File | Role | Purpose |
|---|---|---|
| `query_builder_base.hpp`, `query_builder_base.cpp` | Shared | Base class that appends keywords and text to the SQL string. |
| `query_common.hpp` | Shared | `sort_order` and `join_type` enums. |
| `select/select_query_builders.hpp`, `select/select_query_builders.cpp` | Reader | SELECT, FROM, JOIN, WHERE, GROUP BY, ORDER BY, and LIMIT clause builders, with clause order checked at compile time. |
| `select/table_select_query.hpp` | Reader | Entry point for a fluent SELECT. |
| `insert/insert_query_builders.hpp` | Writer | Parenthesized INSERT value lists, including optional NULLs. |
| `insert/table_insert_query.hpp` | Writer | Fluent `INSERT INTO table (columns) VALUES ...` builder. |
| `CMakeLists.txt` | Build | Builds the `profiler-hub_queries` static library. |

## source/writers

| File | Role | Purpose |
|---|---|---|
| `writer_context.hpp` | Writer | Shared backend, entity registry, key providers, validator, and UUID. |
| `writer_policy_traits.hpp` | Writer | Compile-time check that a writer policy names every required writer type. |
| `common_insert_operations.hpp`, `info_registration_writer.hpp`, `kernel_dispatch_writer.hpp`, `memory_alloc_writer.hpp`, `memory_copy_writer.hpp`, `pmc_event_writer.hpp`, `region_writer.hpp` | Writer | Forward declarations specialized per schema tag. |
| `interfaces/api_writer_base.hpp` | Writer | CRTP base that downcasts to the concrete writer. |
| `interfaces/*_writer_interface.hpp` | Writer | CRTP interfaces for info registration, regions, kernel dispatches, memory copies, allocations, and PMC events. |
| `schema_v3/writer_policy.hpp` | Writer | Binds the active writer to the schema-3 statements and writer types. |
| `schema_v3/common_insert_operations.hpp` | Writer | Schema-3 inserts for event, string, argument, and sample rows. |
| `schema_v3/info_registration_writer.hpp` | Writer | Registers schema-3 node, process, agent, queue, code-object, and track rows. |
| `schema_v3/kernel_dispatch_writer.hpp`, `memory_alloc_writer.hpp`, `memory_copy_writer.hpp`, `pmc_event_writer.hpp`, `region_writer.hpp` | Writer | Insert one schema-3 event and its parent event row. |

## tests

| Path | Purpose |
|---|---|
| `tests/unit/` | Google Test coverage of writers, readers, the C API, SQL builders, pools, and the timeline cache. Built when `PROFILER_HUB_BUILD_TESTS` is on. |
| `tests/benchmarks/` | Google Benchmark timings of single-threaded and parallel writing. Built when `PROFILER_HUB_BUILD_BENCHMARKS` is on. |
| `tests/find_package/` | Standalone consumer that finds an installed profiler-hub and opens an in-memory `storage_t`. |

| Unit test | Covers |
|---|---|
| `c_api_test.cpp` | Context create, free, version, and track list. |
| `c_api_track_test.cpp` | Track events, time windows, and storage reuse. |
| `c_api_future_test.cpp` | Future create, attach, wait, and cancel. |
| `cached_track_reader_test.cpp` | In-memory whole-track cache, windows, and concurrent first reads. |
| `connection_pool_test.cpp` | Connection lease, release, blocking acquire, and the shared catalog. |
| `connection_source_test.cpp` | `with_reader` and pooled track-row readers. |
| `depth_tracker_test.cpp` | Nesting levels for sequential, nested, and abutting events. |
| `entity_utility_test.cpp` | Entity key lookup and missing-key errors. |
| `event_partitions_test.cpp` | Id-range planning and start-time merge. |
| `insert_query_builders_test.cpp` | INSERT value lists, including NULLs. |
| `json_serializer_test.cpp` | Call-stack and source-context JSON round trip. |
| `natural_merge_sort_test.cpp` | Stable merge of sorted runs. |
| `ph_ctx_test.cpp` | Context load of schema version, nodes, and tracks. |
| `populate_reader_catalog_test.cpp` | Parallel catalog fill against a standalone reader. |
| `reader_test.cpp` | Tracks, details, call stacks, and time range after a flush. |
| `reader_id_range_test.cpp` | Id-range visits match a full track query. |
| `sqlite_backend_impl_test.cpp` | Schema SQL and metadata version, UUID, and GUID substitution. |
| `table_insert_query_test.cpp`, `table_select_query_test.cpp` | Generated INSERT and SELECT SQL. |
| `thread_pool_test.cpp` | Task run, cancel, exceptions, and shutdown. |
| `track_file_store_test.cpp` | Timeline cache windows, including events that start before a checkpoint. |
| `track_window_test.cpp` | C window bounds to reader filters. |
| `trace_fixtures.hpp` | Temporary-trace and writer helpers shared by the tests. |
| `writer_test.cpp` | Register and insert round trip, validation errors, and flush. |

## Optiq integration files

These files are in the Optiq repository (`rocprofiler-visualizer`), not in
profiler-hub. They are the Optiq side of the paths in this branch.

| File | Purpose |
|---|---|
| `src/model/src/database/rocprofvis_db_profiler_hub.h`, `.cpp` | Optiq's `ProfilerHub` database class. Opens traces through the C API, converts tracks, slices, tables, flows, call stacks, and details into Optiq's data model, runs summaries, and deletes the cache folder on cleanup. |
| `src/model/src/database/rocprofvis_db_table_processor.h`, `.cpp` | Runs compound table queries. Detects the fixed kernel-summary GROUP and runs it as a SQL summary instead of loading rows. |
| `src/model/src/database/rocprofvis_db_systems.h` | Base `SystemDatabase`. Declares `SummarizeNameDurations` and `EventSummaryInSqlite` with defaults that keep the load-all path. |
| `src/model/src/database/rocprofvis_db_query_manager.h`, `.cpp` | Legacy SQLite databases. Wraps their table SQL in a `GROUP BY` for the summary. |
| `src/model/src/database/rocprofvis_db_trace_processor.h` | Chrome and Perfetto traces. Keeps summaries on the load-all path. |
| `src/model/src/database/rocprofvis_db_summary_cache.h`, `.cpp` | Reads and writes `summary.bin` in the trace cache folder. |
