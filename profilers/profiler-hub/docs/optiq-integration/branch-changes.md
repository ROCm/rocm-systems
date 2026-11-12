# Branch Changes

This document lists the profiler-hub changes that were merged into
`develop-optiq`, in merge order, and then the changes on the current branch,
`vs-profiler-hub-legacy-rocpd`, together with the matching Optiq changes:

1. `vs-profiler-hub-contribution`: commit `66db33a9ea` (#13223)
2. `vs-profiler-hub-trace-detection`: commit `ef8bca693d` (#13388)
3. `vs-profiler-hub-tables`: commit `559a1d81dc` (#14374)
4. `vs-profiler-hub-legacy-rocpd`: the commits on top of `559a1d81dc`

The three merged branches were deleted after merging, so their commits were
found on `develop-optiq` by author (all three are by vstempen) and by content.
Each one is a single squash commit. The probe commit is titled "Added database
probe", so I matched it to `trace-detection` by content. Please confirm that
match.

Commits by other authors on `develop-optiq` between these merges are not
listed. The Optiq side of the three merged branches is not in this git
history, so those rows are not included.

Profiler-hub paths are relative to `profilers/profiler-hub/`. Optiq paths are
relative to the `rocprofiler-visualizer` repository root. A blank justification means the reason was not found in the discussion.
Those three rows are also in
[branch-changes-open-justifications.docx](branch-changes-open-justifications.docx)
so the justification can be typed in directly.

## Merged branch vs-profiler-hub-contribution: commit 66db33a9ea (#13223)

- Parent: `6e1c2e692c`. Merged 2026-10-12. 18 files, +728 / -97.
- Commit motivation: "Trying to add missing features to profiler hub while
  integrating it to roc-optiq." The message lists one item only. The other
  reasons below come from the integration discussion.

| Change | Files affected | Justification |
|---|---|---|
| Progress description on `ph_future_t`. `ph_progress_fn` now receives a `ph_progress_description_t` (may be null). Trace loading reports its phase: schema, catalog, track list, track load, node info. This changes the callback signature (breaking). | `include/c/profiler_hub_types.h`, `source/ph_future.hpp`, `source/ph_future.cpp`, `source/ph_ctx.hpp`, `source/ph_ctx.cpp`, `tests/unit/c_api_future_test.cpp`, `examples/clients/ph-c-client/main.c` | Sending a progress message to the UI. |
| `ph_get_node()` returns `stream_hardware_links`. `reader_t::get_stream_hardware_links()` returns the same with resolved track pointers. A missing agent, queue, or track gives `is_valid` 0 (C) or a null track pointer (C++). Memory copies use the destination agent. | `include/c/profiler_hub_types.h`, `include/cpp/reader.hpp`, `include/cpp/reader_types.hpp`, `source/reader.cpp`, `source/reader_impl.hpp`, `source/reader_impl.cpp`, `source/reader_catalog.hpp`, `source/reader_catalog.cpp`, `source/data_storage/read_statements.hpp`, `source/ph_ctx.cpp` | Optiq builds the stream-to-hardware lookup itself (`GenerateInterdependencyTables`). The C API should return that relationship. |
| `ph_get_node()` also returns the threads, streams, queues, and PMC descriptions (`ph_thread_list_t`, `ph_stream_list_t`, `ph_queue_list_t`, `ph_pmc_info_list_t`). | `include/c/profiler_hub_types.h`, `include/c/profiler_hub.h`, `include/cpp/reader_types.hpp`, `source/populate_reader_catalog.cpp`, `source/reader_catalog.cpp`, `source/ph_ctx.cpp` | The catalog already had this information, but the C node only exposed agents and processes. Optiq needs the rest. |
| `ph_agent_t`: `absolute_index`, `logical_index`, and `uuid` change from `uint32_t` to `ph_optional_uint64_t`. Adds `extdata` (breaking). | `include/c/profiler_hub_types.h`, `include/cpp/reader_types.hpp`, `source/reader_catalog.cpp` | So a missing value is distinct from zero (CHANGELOG). |
| `ph_process_t` gains `ppid`, `init`, `fini`, `start`, `end`, `environment`, and `extdata` from the trace (breaking). | `include/c/profiler_hub_types.h`, `include/cpp/reader_types.hpp`, `source/data_storage/read_statements.hpp`, `source/reader_catalog.cpp` | `ph_process_t` only had the id and the command. The reader already had the rest of the process row, and Optiq needs it. |
| CHANGELOG entries for the above. | `CHANGELOG.md` | So the pull request describes the new API. |

## Merged branch vs-profiler-hub-trace-detection: commit ef8bca693d (#13388)

- Parent: `66db33a9ea`. Merged 2026-10-22. 30 files, +3603 / -252.
- Commit motivation: "Before client like roc-Optiq uses profiler hub to open
  database, it needs a tool to check if database is supported by profiler
  hub." The diff and CHANGELOG also add per-track histograms, event details,
  call stacks, correlated events, and the event id, type, and depth fields.
  The reasons for those rows come from the integration discussion.
- Names differ between the commit message and the code. The message names
  `probe_database()`, `ph_probe_database()`, `PH_RESULT_UNSUPPORTED_DATABASE`,
  `PH_RESULT_INVALID_DATABASE`, and `PH_RESULT_METADATA_MISSING`. The code
  uses `probe_file()`, `ph_probe_file()`, `PH_RESULT_UNSUPPORTED_FILE`, and
  `PH_RESULT_INVALID_FILE`, and keeps `PH_RESULT_UNSUPPORTED_SCHEMA_VERSION`.
  The rows below use the code names.

| Change | Files affected | Justification |
|---|---|---|
| Database probe. `probe_file()` and `ph_probe_file()` open the file read-only and report the schema and version without creating or modifying it. Checks in order: the file exists and is readable; the header is SQLite; exactly one `rocpd_metadata_<uuid>` table; the schema version is supported (schema 3). `ph_file_probe_t` carries the schema and version. | `source/file_probe.cpp` (new; its SQLite checks are now in `source/formats/rocpd/rocpd_probe.cpp`), `source/file_probe_failure.hpp` (new), `include/cpp/reader.hpp`, `include/cpp/reader_types.hpp`, `include/c/profiler_hub.h`, `include/c/profiler_hub_types.h`, `source/profiler_hub_c_api.cpp`, `source/reader.cpp`, `source/CMakeLists.txt` | Before roc-Optiq opens a database, it needs a check that profiler hub supports it. The probe rejects an unsupported trace before any reader is opened. |
| Schema-version parsing. Uses `schema_version_major`, `_minor`, and `_patch` when all three are present. Older traces with only `schema_version` are still accepted: `3` is 3.0.0, `3.0` is 3.0.0, `3.0.1` is 3.0.1. A partial set of split tags is not a version. `storage_t::get_storage_version()` uses the same rules. | `source/version_parser.hpp` (new), `source/data_storage/schema_version.hpp` (new), `source/storage_impl.cpp`, `source/data_storage/backends/sqlite_api_policy.hpp`, `source/data_storage/backends/sqlite_backend_impl.hpp` | Older traces that only have `schema_version` still open. |
| `ph_ctx_create()` runs the probe before the connection pool opens. A rejected file returns `PH_RESULT_UNSUPPORTED_FILE`, `PH_RESULT_INVALID_FILE`, or `PH_RESULT_UNSUPPORTED_SCHEMA_VERSION` instead of failing inside a reader. | `source/ph_ctx.cpp`, `source/ph_ctx.hpp`, `include/c/profiler_hub_types.h` | A rejected file returns one of the new result codes instead of failing later inside a reader. |
| Extensible probe. Another database format is added by registering an entry. Another schema is added by registering the versions its reader accepts. | `source/file_probe.cpp`, `source/formats/trace_formats.cpp` | Stated design goal in the commit message. |
| `ph-c-client --probe` runs the C probe on one or more trace files. | `examples/clients/ph-c-client/main.c` | A unit test that creates its own database does not exercise real traces. The example client runs the probe on existing files. |
| Per-track histograms. `ph_get_histogram()` and `reader_t::get_histogram()` return one histogram per track on a shared time grid. `max_buckets` is an upper bound. Duration tracks count the events that overlap each bucket and sum the overlap. PMC tracks average the samples in each bucket and hold the last value across empty buckets. Bucket 0 starts at the earliest track timestamp. | `source/histogram.cpp` (new), `include/c/profiler_hub.h`, `include/cpp/reader.hpp`, `source/reader.cpp`, `source/reader_impl.hpp`, `source/reader_impl.cpp`, `source/ph_ctx.hpp`, `source/ph_ctx.cpp`, `source/profiler_hub_c_api.cpp` | Optiq's `BuildHistogram` is per track. The hub had no equivalent, and the timeline uses the histogram to decide which time segments to load. |
| Per-track depth and in-memory track cache. `cached_track_reader` holds each track. Nesting depth is computed by `depth_tracker`. A track is loaded when its depth is first read. | `source/cached_track_reader.hpp`, `source/cached_track_reader.cpp`, `source/depth_tracker.hpp`, `tests/unit/depth_tracker_test.cpp`, `tests/unit/reader_test.cpp` | Nesting depth has to be computed by the library, and a depth of 0 is not a valid answer. The flow view reads it once to draw the arrow. The whole track is needed to compute it. |
| Event details: `ph_get_region_details()`, `ph_get_kernel_dispatch_details()`, `ph_get_memory_copy_details()`, `ph_get_memory_alloc_details()`. A missing row returns `PH_RESULT_SUCCESS` with `is_valid` 0. A mismatched event type returns `PH_RESULT_INVALID_ARGUMENT`. String pointers stay valid until `ph_ctx_free()`. | `include/c/profiler_hub.h`, `include/c/profiler_hub_types.h`, `include/cpp/reader_types.hpp`, `source/profiler_hub_c_api.cpp`, `source/ph_ctx.cpp`, `source/reader_impl.cpp` | The timeline reads a few fields per event. Optiq's detail view (`ReadExtEventInfo`) needs the rest: region, kernel dispatch, memory copy, and memory allocation. |
| Call stacks: `ph_get_call_stack()`. Index 0 is the top of the stack. An event with no call stack returns an empty list. | `include/c/profiler_hub.h`, `source/profiler_hub_c_api.cpp`, `source/ph_ctx.cpp`, `source/reader.cpp`, `source/reader_impl.cpp`, `source/json_serializers.cpp` | Optiq's stack view needs `get_call_stack` on the C API. The stack is the enclosing events on the same track. `rocpd_event.extdata` is empty on almost every trace. |
| Correlated events: `ph_get_correlated_events()`. The source event is excluded, and an event with no stack id returns an empty list. `ph_event_t` gains `id`, `type`, `nesting_depth`, and `track_id`. A kernel, memory copy, or allocation uses its agent+queue track, then its stream track, rather than the host thread track. | `include/c/profiler_hub.h`, `include/c/profiler_hub_types.h`, `source/profiler_hub_c_api.cpp`, `source/ph_ctx.hpp`, `source/ph_ctx.cpp`, `source/reader_impl.hpp`, `source/reader_impl.cpp` | Later fixed in `df63fb59d3` on the current branch: Optiq did not set `track_id`, so the call returned no endpoints. |
| Read statements and catalog: schema-3 SELECT statements and catalog fields for the new calls. | `source/data_storage/read_statements.hpp`, `source/reader_catalog.hpp`, `source/reader_catalog.cpp` | The histogram, details, call stack, and correlation queries read these rows. |
| CHANGELOG entries for the probe, histogram, details, call-stack, and correlated-event changes. | `CHANGELOG.md` | So the pull request describes the new API. |

## Merged branch vs-profiler-hub-tables: commit 559a1d81dc (#14374)

- Parent: `ef8bca693d`. Merged 2026-11-03. 37 files, +4816 / -569.
- Commit motivation: "Need to support extended events reader for table format
  data representation."
- File lists come from `git diff --stat` and symbol searches. They were not
  checked line by line, so treat them as a starting point.
- Later commits on the current branch moved `table_events.cpp` and
  `rocpd_trim_schema.cpp` into `source/formats/rocpd_v3/` as
  `rocpd_v3_table_events.cpp` and `rocpd_v3_trim_schema.cpp`. See the
  legacy-rocpd sections below.

| Change | Files affected | Justification |
|---|---|---|
| Expanded event filter. Adds a resolved event query (`event_query`) with time window, node, agent, process, thread, source, tracks, event types, and search terms. | `source/event_query.hpp` (new), `source/event_query.cpp` (new), `include/c/profiler_hub_types.h`, `include/c/profiler_hub.h`, `source/profiler_hub_c_api.cpp`, `source/ph_ctx.hpp`, `source/ph_ctx.cpp` | Commit motivation: an extended events reader for table-format data. |
| Paged table-event queries: `ph_event_query_create()`, `ph_event_query_read()`, `ph_event_query_id_span()`, `ph_event_query_free()`. | `source/table_events.cpp` (new), `include/c/profiler_hub.h`, `source/profiler_hub_c_api.cpp`, `source/ph_ctx.cpp`, `source/data_storage/read_statements.hpp` | Commit motivation: table-format data representation. |
| Region and region-sample events are reported separately. | `include/c/profiler_hub_types.h`, `source/data_storage/read_statements.hpp`, `source/ph_ctx.cpp` | Optiq treats a launch and a launch sample as different operations. Both were being read from `rocpd_region`. |
| Counter, sample, and memory-activity event reads. | `source/ph_ctx.cpp`, `source/cached_track_reader.hpp`, `source/cached_track_reader.cpp`, `include/c/profiler_hub.h`, `source/profiler_hub_c_api.cpp`, `source/data_storage/read_statements.hpp` | Optiq's counter, memory-activity, and SMI tables (`GetRocprofPerformanceCountersTableQuery`, `GetRocprofMemoryActivityTableQuery`, `GetRocprofSMIPerformanceCountersTableQuery`) had no hub equivalent. |
| Memory activity is derived in memory. | `source/reader_catalog.cpp`, `source/ph_ctx.cpp` | Optiq stores memory activity in `roc_optiq_*` tables. The hub must not write those tables, and a new trace does not have them. The series is computed while reading. |
| The populated sample table is selected. | `source/data_storage/read_statements.hpp` | Sample reads came back empty. A trace can have more than one sample table, and the read was using an empty one. |
| Timeline counter samples are sorted. The merge stores run bounds as offsets instead of iterators. | `source/ph_ctx.cpp`, `source/common/natural_merge_sort.hpp` | Counter samples have to be in time order. On a Debug MSVC build, storing the run bounds as iterators made a large track's sort look stuck, because destroying the iterator vector walks every registered iterator. |
| Neighboring samples are included in each timeline slice. | `source/ph_ctx.cpp`, `source/cached_track_reader.cpp` | |
| Each memory free is matched to its open allocation. | `source/ph_ctx.cpp`, `source/reader_catalog.cpp` | The memory-activity value is the running allocated size: an allocation adds its size and the matching free subtracts it. |
| Memory counters are added to the PMC catalog. | `source/reader_catalog.hpp`, `source/reader_catalog.cpp` | Memory activity is shown as a PMC counter. Its catalog entry needs units of bytes and a GPU target. |
| The histogram includes HIP events, and its sample join uses the populated sample table. | `source/histogram.cpp` | HIP events are part of the timeline, the same way Optiq shows them. The sample join had the same empty-table problem as the sample read. |
| Trim: `ph_trim_trace()` and its C++ wrapper. Copy logic in `trace_trim.cpp`, per-table rules in `trim_schema.hpp` and `rocpd_trim_schema.cpp`. Wired through `ph_ctx`, the format registry and the C API. | `include/c/profiler_hub_trim.h` (new), `include/cpp/trimmer.hpp` (new), `source/trace_trim.hpp` (new), `source/trace_trim.cpp` (new), `source/trim_schema.hpp` (new), `source/rocpd_trim_schema.cpp` (new), `source/ph_ctx.cpp`, `source/profiler_hub_c_api.cpp`, `CMakeLists.txt`, `source/CMakeLists.txt`, `include/cpp/CMakeLists.txt` | Optiq saves a time window with `SaveTrimmedData`. The hub had no equivalent, and the rules have to follow the schema rather than one set of table names. Trim is a separate API from the reader because the review asked for a writer API, a reader API, and a trimmer API. |
| Unsupported schema versions now return `PH_RESULT_UNSUPPORTED_FILE`. `PH_RESULT_UNSUPPORTED_SCHEMA_VERSION` is removed from `profiler_hub_types.h`. The probe is split into one check per metadata table (`probe_one_metadata()`). The CHANGELOG entry is updated to match. This conflicts with the probe commit's CHANGELOG, which listed `PH_RESULT_UNSUPPORTED_SCHEMA_VERSION` as a return value. The commit message does not mention this change. | `source/formats/rocpd/rocpd_probe.cpp`, `include/c/profiler_hub_types.h`, `CHANGELOG.md` | A review had already asked to drop the schema-version result code, and this puts that removal back. One check per metadata table is what lets a file contain more than one GUID. The probe used to reject a file with more than one `rocpd_metadata` table. |
| A connection takes the trace GUID and passes it to `storage_t`. | `source/common/connection.cpp`, `source/common/connection.hpp`, `source/common/connection_pool.cpp`, `source/common/connection_pool.hpp` | Schema-3 tables are named with a GUID. The connection has to open that GUID's tables. |
| Reader and reader-catalog methods for the table query and the new event kinds. | `include/cpp/reader.hpp`, `include/cpp/reader_types.hpp`, `source/reader.cpp`, `source/reader_impl.cpp`, `source/reader_impl.hpp`, `source/reader_catalog.hpp` | The C table query and the new event kinds go through the reader. |
| `ph-c-client` output for the table reads. | `examples/clients/ph-c-client/main.c` | So the example client can run the new table query. |
| Unit test for the cached track reader. | `tests/unit/cached_track_reader_test.cpp` | |

## Commit c5ca2dc357: Added legacy rocpd support

| Change | Files affected | Justification |
|---|---|---|
| Added `trace_format_t` and passed the probed format from the probe to each connection, the pool, and `ph_ctx`. | `include/cpp/reader_types.hpp`, `source/file_probe.cpp`, `source/formats/rocpd/rocpd_probe.cpp`, `source/common/connection.hpp`, `source/common/connection.cpp`, `source/common/connection_pool.hpp`, `source/common/connection_pool.cpp`, `source/ph_ctx.cpp` | Each file must open with the reader for its own schema. |
| Added a format registry that maps a schema and major version to a reader factory and a trim-rule factory. | `source/formats/trace_formats.hpp`, `source/formats/trace_formats.cpp`, `source/CMakeLists.txt` | Adding a format touches one place instead of every caller. |
| Made `reader_t::impl` an abstract base with one subclass per format. Shared catalog code stays in the base. | `source/reader_impl.hpp`, `source/reader.cpp`, `source/reader_catalog.hpp`, `source/reader_catalog.cpp` | Lets rocpd v2 and v3 readers live side by side behind one `reader_t`. |
| Moved the schema-3 reader, histogram, table rows, and trim rules into `formats/rocpd_v3` as `rocpd_v3_reader`. | `source/formats/rocpd_v3/rocpd_v3_reader.hpp`, `rocpd_v3_reader.cpp`, `rocpd_v3_histogram.cpp`, `rocpd_v3_table_events.cpp`, `rocpd_v3_trim_schema.cpp`, `source/reader.cpp`, `source/trace_trim.cpp`, `source/trim_schema.hpp` | Same as above. |
| Moved histogram bucket bounds and overlap counting into a shared header. | `source/histogram_grid.hpp` | Used by both format readers. |
| The probe now accepts schema 2 from an unsuffixed `rocpd_metadata` table. | `source/formats/rocpd/rocpd_probe.cpp`, `source/data_storage/schema_version.hpp` | Legacy `.rpd` files have one metadata table without a GUID suffix. |
| A standalone `reader_t` falls back to schema 3 when the probe cannot read the file. | `source/reader.cpp` | A writer database that is not flushed yet has no metadata to probe. |
| Added `rocpd_v2_reader` for legacy `.rpd` traces. | `source/formats/rocpd_v2/rocpd_v2_reader.hpp`, `rocpd_v2_reader.cpp`, `source/CMakeLists.txt` | Optiq opens legacy `.rpd` traces through profiler-hub. |
| The v2 catalog synthesizes one node and the processes, threads, agents, and queues from the event tables. | `source/formats/rocpd_v2/rocpd_v2_catalog.cpp` | Schema 2 has no node, process, thread, or agent tables. |
| Duplicate strings map to one canonical id. | `source/formats/rocpd_v2/rocpd_v2_catalog.cpp` | `rocpd_string` repeats the same text under several ids. |
| A queue's process is `MIN(rocpd_api.pid)` grouped by `gpuId` and `queueId`. | `source/formats/rocpd_v2/rocpd_v2_catalog.cpp` | A queue id is reused on every GPU, so the queue id alone does not identify the process. |
| Operation types whose text starts with `Copy` are memory copies. | `source/formats/rocpd_v2/rocpd_v2_reader.cpp`, `rocpd_v2_catalog.cpp` | Schema 2 stores the operation type as text, not as a category. |
| Opening a v2 trace creates indexes on the event tables. | `source/formats/rocpd_v2/rocpd_v2_reader.cpp` | Track and correlation queries scan the whole table without them. |
| v2 timeline events, details, samples, counts, and histogram over `rocpd_api`, `rocpd_op`, `rocpd_api_ops`, and `rocpd_monitor`. | `source/formats/rocpd_v2/rocpd_v2_reader.cpp` | Optiq's timeline and detail views need the same queries as v3. |
| v2 call stack and correlated events through `rocpd_api_ops`. | `source/formats/rocpd_v2/rocpd_v2_reader.cpp` | Optiq's flow and call-stack views. |
| v2 paged table rows. | `source/formats/rocpd_v2/rocpd_v2_table_events.cpp` | Optiq's event tables. |
| v2 trim rules. | `source/formats/rocpd_v2/rocpd_v2_trim_schema.cpp` | Trim must work on legacy traces too. |

## Commit 52e0c30d63: Added tracks and metadata filesystem storage

| Change | Files affected | Justification |
|---|---|---|
| Added `PROFILER_HUB_ENABLE_ROCPD_V2` and `PROFILER_HUB_ENABLE_ROCPD_V3` build switches. | `CMakeLists.txt`, `source/CMakeLists.txt`, `source/formats/trace_formats.cpp` | Each format can be left out of a build. |
| Cache folder named `<date>_<format>_<version>_<name>_<size>` from the newest source mtime, probed format and version, first file name, and total size. | `source/cache_directory.hpp`, `source/cache_directory.cpp` | One folder per trace. A changed trace gets a new folder instead of stale data. |
| Added `ph_ctx_create_files_cached`. | `include/c/profiler_hub.h`, `source/profiler_hub_c_api.cpp`, `source/ph_ctx.hpp`, `source/ph_ctx.cpp` | The application chooses the cache root, such as its config folder. |
| Added `ph_trace_cache_directory`. | `include/c/profiler_hub.h`, `source/profiler_hub_c_api.cpp` | The application can find the folder to delete it or to store its own files there. |
| Timeline cache: per-track event, sample, and depth-index files, strings, and the stored histogram. | `source/track_file_store.hpp`, `source/track_file_store.cpp` | Reopening a large trace should not read every row from SQLite again, and loaded tracks should not have to stay in memory. |
| On open, `ph_ctx` writes the timeline cache when it is missing and reads it when it is present. | `source/ph_ctx.hpp`, `source/ph_ctx.cpp` | Same as above. |
| Window reads come from the timeline files. Added `ph_release_track_events` and `ph_release_track_samples`. | `include/c/profiler_hub.h`, `source/profiler_hub_c_api.cpp`, `source/ph_ctx.cpp`, `source/cached_track_reader.hpp`, `source/cached_track_reader.cpp` | Optiq reads by time segment, so each window's memory can be freed after the segment is built. |
| The histogram is read from the timeline cache when it exists. | `source/ph_ctx.cpp`, `source/track_file_store.cpp` | Avoids rebuilding the histogram from SQLite on every open. |
| `nesting_depth_of` reads the depth from the timeline files. | `source/cached_track_reader.cpp`, `source/track_file_store.cpp` | Correlation needs the depth without loading the whole track. |
| Catalog cache: schema-2 aggregates (PHC1) and schema-3 derived tracks (PHD1). | `source/catalog_aggregate_store.hpp`, `source/catalog_aggregate_store.cpp`, `source/formats/rocpd_v2/rocpd_v2_catalog.cpp`, `source/formats/rocpd_v3/rocpd_v3_reader.cpp`, `source/ph_ctx.cpp` | Building the catalog is slow on large traces. |
| Added a `make_histogram_grid` overload for a track list. | `source/histogram_grid.hpp` | The stored histogram is built from the published tracks. Optiq uses it to skip time segments that have no events. |
| Added the timeline-cache unit test. | `tests/unit/track_file_store_test.cpp`, `tests/unit/CMakeLists.txt` | |

## Commit 5cef79b824: Added summary aggregation methods

| Change | Files affected | Justification |
|---|---|---|
| Added `ph_event_query_summarize_durations`, which returns count, min, max, and total duration per name. | `include/c/profiler_hub.h`, `include/cpp/reader.hpp`, `source/profiler_hub_c_api.cpp`, `source/ph_ctx.hpp`, `source/ph_ctx.cpp`, `source/reader.cpp` | Optiq's kernel summary loaded every row and grouped them in memory, which does not work on huge traces. |
| The base reader's summary returns "not supported". | `source/reader_impl.hpp` | A format without a SQL summary keeps Optiq's load-all path. |
| v2 summary as a SQL `GROUP BY`. | `source/formats/rocpd_v2/rocpd_v2_reader.hpp`, `rocpd_v2_table_events.cpp` | Same as the first row. |
| v3 summary as a SQL `GROUP BY`. | `source/formats/rocpd_v3/rocpd_v3_reader.hpp`, `rocpd_v3_table_events.cpp` | Same as the first row. |
| `ph_ctx` picks the source or sources the query covers and merges their rows. | `source/ph_ctx.cpp` | A trace can be several files or several GUIDs in one file. The summary has to cover the sources the query names and come back as one set of rows. |
| Comments say the cache folder also holds Optiq's `summary.bin`. | `include/c/profiler_hub.h`, `source/cache_directory.hpp` | Optiq stores its summary cache in the same folder. |

## Commit df63fb59d3: Fixed ph_get_correlated_events

| Change | Files affected | Justification |
|---|---|---|
| Added `source_for_event`. `get_correlated_events` and `get_call_stack` find the source from the event id instead of `track_id`. | `source/ph_ctx.hpp`, `source/ph_ctx.cpp` | Optiq did not set `track_id`, so the call failed and returned no endpoints. |
| A missing source returns an empty list. An endpoint that cannot be placed on a track is skipped. | `source/ph_ctx.cpp`, `include/c/profiler_hub.h` | One bad endpoint failed the whole call. |
| Added `has_local_events`. `nesting_depth_of` uses the timeline files only when the track is in them, and falls back to memory on error. | `source/track_file_store.hpp`, `source/track_file_store.cpp`, `source/cached_track_reader.cpp` | An endpoint on a track without cached records lost its depth. |

## Optiq changes

Optiq is not under version control in this workspace, so this list is
rebuilt from the work sessions. All files are under
`src/model/src/database/`.

| Change | Files affected | Justification |
|---|---|---|
| Profiler-hub traces open with `ph_ctx_create_files_cached`, using the config folder as the cache root. | `rocprofvis_db_profiler_hub.cpp` | Uses the timeline and catalog caches across opens. |
| `ReadTraceSlice` calls `ph_release_track_events` and `ph_release_track_samples` after each slice. | `rocprofvis_db_profiler_hub.cpp` | Optiq reads by time segment, so each window is freed after its slice. |
| `ProfilerHub::Cleanup` deletes the trace's cache folder, found with `ph_trace_cache_directory` and the same file list used to open it. | `rocprofvis_db_profiler_hub.h`, `rocprofvis_db_profiler_hub.cpp` | Cleanup removes cached data, as it does for other database types. |
| `ProcessTrack` writes the track's process into the `Queue` row before copying queue properties. | `rocprofvis_db_profiler_hub.cpp` | A queue id is shared by every GPU, and the catalog keeps one row per queue id, so queues showed the wrong process. |
| Added a SQL name-duration summary path: `IsNameDurationSummary`, `ExecuteNameDurationSummary`, and the `SummarizeNameDurations` virtual, with `QueryManager` and `ProfilerHub` versions. | `rocprofvis_db_table_processor.h`, `rocprofvis_db_table_processor.cpp`, `rocprofvis_db_systems.h`, `rocprofvis_db_query_manager.h`, `rocprofvis_db_query_manager.cpp`, `rocprofvis_db_profiler_hub.h`, `rocprofvis_db_profiler_hub.cpp` | Loading every row for the kernel summary does not work on huge traces. |
| Added the `EventSummaryInSqlite()` gate. `GoogleTraceProcessor` returns false. A "not supported" result falls back to load-all. | `rocprofvis_db_systems.h`, `rocprofvis_db_query_manager.h`, `rocprofvis_db_profiler_hub.h`, `rocprofvis_db_trace_processor.h`, `rocprofvis_db_table_processor.cpp` | Sources that are not SQLite must keep the post-load aggregation. The controller's behavior must not change. |
| Summary results are cached in `summary.bin` in the trace cache folder. | `rocprofvis_db_summary_cache.h`, `rocprofvis_db_summary_cache.cpp`, `rocprofvis_db_query_manager.cpp`, `rocprofvis_db_profiler_hub.cpp` | The aggregation takes a long time but its output is small. |
| `ReadFlowTraceInfo` and `ReadStackTraceInfo` zero-initialize `ph_event_t`, `ph_event_list_t`, and `ph_call_stack_t`. | `rocprofvis_db_profiler_hub.cpp` | `track_id` was passed to profiler-hub uninitialized. |
