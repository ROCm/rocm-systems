# Write rocpd directly from the native tool

Plan to make the rocprofiler-compute native C++ tool write its counter data into a
rocpd (SQLite) database via the profiler-hub library, instead of emitting CSV that
the Python layer later parses and injects.

The work is split in two parts that can land independently:

- Part 1 (C++): native tool stops writing CSV and writes its own rocpd via profiler-hub.
- Part 2 (Python): merge each native-tool rocpd into the SDK-tool rocpd, per PID.

ISA / code-object collection is out of scope and stays as the existing JSON output.

---

## Context

During a profiling run rocprofiler-compute produces data from two in-process
libraries, both loaded via `LD_PRELOAD` into the target application
(`src/rocprof_compute_profile/profiler_rocprofiler_sdk.py:33-43`):

1. The rocprofiler-sdk "SDK tool", run with `ROCPROF_OUTPUT_FORMAT=rocpd`, writes a
   rocpd SQLite database (kernel dispatches, agents, kernel symbols, an
   `rocpd_info_pmc` table, and an empty `rocpd_pmc_event` table). The SDK tool is
   told NOT to collect counters (`ROCPROF_COUNTER_COLLECTION=0`).
2. The rocprofiler-compute "native tool" (`src/lib/rocprofiler_compute_tool`) collects
   the hardware counters and currently writes a CSV
   (`<pid>_native_counter_collection.csv`) via
   `CsvCountersWriter::write_counters` (`src/lib/rocprofiler_compute_tool/counters_writer.cpp:8`).

The Python layer then reads each CSV and injects rows into the SDK rocpd's
`rocpd_pmc_event_<guid>` table, remapping `dispatch_id` to `event_id`
(`src/utils/rocpd_data.py:103-167`, called from `src/utils/utils_profile.py:210-235`).

### Key schema facts (verified)

From the bundled schema (`profilers/profiler-hub/source/data_storage/schema/rocpd_tables.sql`):

- `rocpd_pmc_event.event_id` is a FK to `rocpd_event.id` (rocpd_tables.sql:231,236).
- `rocpd_kernel_dispatch` has its own `id`, a `dispatch_id` column, and an `event_id`
  FK to `rocpd_event.id` (rocpd_tables.sql:281,295).
- The counters view joins pmc to dispatch on the shared event id:
  `rocpd_kernel_dispatch K ON K.event_id = PMC_E.event_id`
  (`source/data_storage/schema/data_views.sql:407`).
- In a real SDK rocpd, every `rocpd_pmc_event.event_id` equals a
  `rocpd_kernel_dispatch.event_id` (verified: all rows join cleanly in
  `workloads/rocflop_profile/MI300X_A1/pmc_perf_0_287205.db`).
- `rocpd_info_pmc` IS populated by the SDK tool (15 rows in that workload), with
  columns `name`, `symbol`, `event_code`, `target_arch`, etc. `rocpd_pmc_event.pmc_id`
  is the `rocpd_info_pmc.id`.

### profiler-hub facts (verified)

- Public API: `storage_t(db_path, uuid)` (`include/storage.hpp:19`) plus `writer_t`
  (`include/writer.hpp`). Input structs in `include/writer_types.hpp`.
- `writer_t::insert_pmc_event_data` ALWAYS mints a fresh `rocpd_event` row per call
  (`source/writers/schema_v3/pmc_event_writer.hpp:36-58`), and so does
  `insert_kernel_dispatch_data`. So a pmc event and a dispatch CANNOT share an
  `event_id` through this API; the intended dispatch linkage is
  `event_data_t.correlation_id`, not a shared PK.
- profiler-hub is a standalone library; nothing else in the monorepo consumes it yet
  (no references under `projects/rocprofiler-systems/` on develop, sna-develop, or
  rocprofiler-compute-develop; the README claim is aspirational). rocprofiler-compute
  will be the first consumer.
- Build targets: `profiler-hub` (shared) and `profiler-hub-static`. Default schema is
  bundled SQL (`USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD=0`), no rocprofiler-sdk-rocpd
  dependency required. `cmake/Findprofiler-hub.cmake` supports both installed and
  add_subdirectory consumption.

### Design consequence

Because profiler-hub mints a fresh event per pmc sample, the native rocpd will NOT
have pmc/dispatch sharing an event_id internally. We therefore carry the native
`dispatch_id` on the pmc event's `correlation_id`, and the native tool writes ONLY
the counter-related tables (no kernel_dispatch / kernel_symbol / code_object). Part 2
recovers `dispatch_id` from `correlation_id` and remaps to the SDK event_id.

---

## Design

### Part 1: native tool writes rocpd

Data available per counter sample in `tool_data->counter_records`
(`src/lib/rocprofiler_compute_tool/sdk_callbacks.h`, `counter_info_record_t`):
`dispatch_id, agent_id, kernel_id, LDS_memory_size, counter_id, counter_name, counter_value`.

New writer `RocpdCountersWriter : public CountersWriter` (the existing abstraction in
`counters_writer.h`), selected as the default `g_counters_writer`
(`src/lib/rocprofiler_compute_tool/rocprofiler_compute_tool.cpp:30`). The CSV writer
stays in the tree but is no longer the default. No env-var gating: the native tool
always writes rocpd. (`ROCPROF_OUTPUT_FORMAT` is read only by the SDK tool, never by
the native tool, so nothing needs removing.)

`RocpdCountersWriter::write_counters(tool_data_t*)` does:

1. `storage_t storage(db_path, uuid)` where `db_path` ends in
   `_native_counter_collection.db` (generated alongside the existing CSV name in
   `create_tool_data`, `rocprofiler_compute_tool.cpp:197`) and `uuid` is a generated
   guid string.
2. `writer_t writer(std::move(storage))`.
3. Register FK-target info rows once:
   - `register_node_info` (node_id = 0, a machine_id).
   - `register_process_info` (pid = getpid(), node_id = 0).
   - `register_agent_info` per distinct `agent_id` (needs
     `agent_unique_id_t{agent_type, type_index}`).
   - `register_pmc_info` per distinct counter, populating `rocpd_info_pmc`:
     `unique_id{name = counter_name, agent}`, `symbol = counter_name`,
     `event_code = counter_id`, `target_arch`, node/process.
4. Per counter record: `insert_pmc_event_data(pmc_event_data_t{ event:{correlation_id =
   dispatch_id}, value = counter_value, sample:{...} }, pmc_info_unique_id_t{name =
   counter_name, agent})`.

Not written by the native tool: kernel_dispatch, kernel_symbol, code_object, LDS, grid,
kernel name. All of that comes from the SDK rocpd at merge time.

#### Open design items for Part 1

1. Carrier for dispatch_id: `event_data_t.correlation_id` (decided).
2. Agent + counter metadata capture (confirmed required). The callbacks today capture
   only `agent_id.handle` (sdk_callbacks.cpp:31,305) and the counter name
   (sdk_callbacks.cpp:245), discarding everything else. profiler-hub needs more:
   - `register_agent_info` needs `agent_unique_id_t{agent_type, type_index}`,
     `logical_index`, `name`, etc.; `register_pmc_info` needs `target_arch`. None of
     this is in `rocprofiler_dispatch_counting_service_data_t`. The full data is in
     `rocprofiler_agent_t` (`/rocm/include/rocprofiler-sdk/agent.h:204-262`: `name`,
     `model_name`, `node_id`, `logical_node_id`, `logical_node_type_id`, `uuid`,
     `type`, `gfx_target_version`), reachable via `rocprofiler_query_available_agents`
     (`agent.h:296`), which is NOT currently wrapped in `SdkWrapper`.
   - `register_pmc_info` also wants `description`, `block`, `expression`,
     `is_constant`, `is_derived`. These are already queried as
     `rocprofiler_counter_info_v0_t` (`/rocm/include/rocprofiler-sdk/counters.h:42-51`)
     at sdk_callbacks.cpp:222-228 but only `name` is kept. (v0 has no `event_code`;
     the SDK rocpd leaves that column null, so we mirror that.)
3. target_arch string value: confirm what the SDK tool stores in
   `rocpd_info_pmc.target_arch` so the Part 2 pmc_id join matches (Part 2 detail).

### Part 2: Python merges native rocpd into SDK rocpd (per PID)

Replace the CSV read + row-by-row insert in
`src/utils/rocpd_data.py:update_rocpd_pmc_events` with a set-based merge that runs
entirely inside SQLite. For each PID, given the SDK db and the matching native db:

```sql
ATTACH DATABASE :native_db AS native;

INSERT INTO rocpd_pmc_event_{sdk_guid} (guid, event_id, pmc_id, value)
SELECT
    sdk_kd.guid,
    sdk_kd.event_id,                      -- remapped to SDK event id space
    sdk_pmc.id,                           -- remapped to SDK info_pmc id space
    n_pmc.value
FROM native.rocpd_pmc_event_{native_guid}  AS n_pmc
JOIN native.rocpd_event_{native_guid}      AS n_ev
     ON n_ev.id = n_pmc.event_id
JOIN rocpd_kernel_dispatch_{sdk_guid}      AS sdk_kd
     ON sdk_kd.dispatch_id = n_ev.correlation_id     -- dispatch_id carrier
JOIN native.rocpd_info_pmc_{native_guid}   AS n_info
     ON n_info.id = n_pmc.pmc_id
JOIN rocpd_info_pmc_{sdk_guid}             AS sdk_pmc
     ON sdk_pmc.symbol = n_info.symbol
     AND sdk_pmc.target_arch IS n_info.target_arch;

DETACH DATABASE native;
```

Two id spaces are remapped by the joins:

- `event_id`: native pmc -> native event -> `correlation_id` (= dispatch_id) ->
  SDK kernel_dispatch.dispatch_id -> SDK kernel_dispatch.event_id.
- `pmc_id`: native info_pmc -> SDK info_pmc on a stable key (`symbol`, plus
  `target_arch`).

Loop the attach/insert/detach over all N PIDs (one native db + one SDK db per PID).
Cross-guid merging into a single combined rocpd is a separate, later PR and is out of
scope here.

#### Open design items for Part 2

1. pmc_id join key: `symbol` (+ `target_arch`). Confirm uniqueness in real dbs.
2. Pairing native db to SDK db per PID (both filenames embed the PID).
3. Decide the fate of `convert_dbs_to_csv` and any CSV-era code paths that become dead.

---

## Implementation

### Part 1 (C++)

1. Add `RocpdCountersWriter` (new
   `src/lib/rocprofiler_compute_tool/rocpd_counters_writer.{h,cpp}`) implementing
   `CountersWriter::write_counters` per the Design section.
2. Generate the `.db` output filename in `create_tool_data`
   (`rocprofiler_compute_tool.cpp:197`) and store it on `tool_data` (extend
   `tool_data_t` if a separate field from the CSV name is wanted).
3. Switch the default `g_counters_writer`
   (`rocprofiler_compute_tool.cpp:30`) to `RocpdCountersWriter`. Keep
   `CsvCountersWriter` and the `test_knobs::set_csv_writer` hook for tests.
4. Capture agent + counter metadata in the callbacks (confirmed required, open item 2):
   - Wrap `rocprofiler_query_available_agents` in `SdkWrapper`
     (`src/lib/rocprofiler_compute_tool/sdk_wrapper.h` / `.cpp`).
   - Call it once (tool_init or first dispatch) and build a `tool_data` map
     `agent_handle -> {type, logical_node_type_id, name, model_name, node_id,
     gfx_target_version}`.
   - Extend the counter capture at sdk_callbacks.cpp:245 to store full
     `rocprofiler_counter_info_v0_t` fields per counter (replace/augment
     `counter_id_name_map` with `handle -> counter_info` in `tool_data`).
   - `counter_info_record_t` (`sdk_callbacks.h`) stays as-is; the rocpd writer resolves
     agent + counter metadata from the two maps via the record's `agent_id` /
     `counter_id` handles at write time.
5. Build wiring: see Building section.

### Part 2 (Python)

1. Rewrite `update_rocpd_pmc_events` (`src/utils/rocpd_data.py:103`) to do the
   ATTACH + INSERT...SELECT merge instead of reading CSV and inserting rows.
2. Update the caller (`src/utils/utils_profile.py:210-235`) to pass the per-PID native
   db path instead of the CSV path, and to pair native db to SDK db per PID.
3. Remove now-dead CSV plumbing (the native counter CSV read via
   `src/utils/utils_profile_csv.py`) once the merge path is in.

---

## Building and testing

Two test phases.

### Phase 1: prototype against the already-installed profiler-hub

profiler-hub is already built and installed (e.g. under `/opt/rocm`). Use it directly
to validate the C++ writer and the Python merge before doing the ExternalProject work.

Reference install commands (for the record):

```bash
cd profilers/profiler-hub
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DPROFILER_HUB_BUILD_TESTS=OFF -DPROFILER_HUB_BUILD_BENCHMARKS=OFF
cmake --build build -j"$(nproc)"
cmake --install build --prefix /opt/rocm
```

For prototyping, link the native tool against the installed library via
`find_package(profiler-hub REQUIRED)` +
`target_link_libraries(rocprofiler-compute-tool PRIVATE profiler-hub::profiler-hub)`
(temporary; replaced in Phase 2).

Phase 1 validation:

- Build the native tool; run a small workload (e.g. `tests/vcopy`) with the native tool
  preloaded; confirm a `<pid>_native_counter_collection.db` is produced with
  `rocpd_info_pmc`, `rocpd_event` (carrying `correlation_id`), and `rocpd_pmc_event`
  populated.
- Run the Python Part 2 merge against a real SDK rocpd (e.g.
  `workloads/rocflop_profile/...`) and confirm the SDK `rocpd_pmc_event` table fills and
  the counters view joins cleanly (counts match expected dispatch x counter rows).
- Run unit tests only (not the CPU-heavy analyze / integration suites).

### Phase 2: ExternalProject-based source build

Replace the prototype `find_package` with an in-repo source build so the monorepo
builds profiler-hub itself and the native tool links it statically into the single
`librocprofiler-compute-tool.so`.

**Placement: both the sparse-checkout guard and `ExternalProject_Add` go in
`src/lib/CMakeLists.txt`, NOT the top-level `CMakeLists.txt`.** There are two configure
entry points that build the native tool, and only `src/lib` is common to both:

- Full / install build (TheRock CI): top-level `CMakeLists.txt` reaches the tool via
  `add_subdirectory(src/lib)` (`CMakeLists.txt:266`).
- Runtime fallback build: the Python layer configures `src/lib` *directly* as the source
  root (`cmake -S <root>/lib -B <root>/lib/_build`, `src/utils/native_tool_finder.py:67-79`),
  invoked from `__build_collector` when no installed `librocprofiler-compute-tool.so` is
  found. The top-level `CMakeLists.txt` never runs on this path.

  `src/lib/CMakeLists.txt:2` is its own standalone `project(rocprofiler-compute-tool ...)`,
  so it configures cleanly on its own. Putting the wiring in top-level only would leave the
  runtime build with no profiler-hub. Putting it in `src/lib` covers both paths. The guard
  locates the source via `git rev-parse --show-toplevel`, so it resolves the monorepo root
  the same way regardless of which configure root is active.

- In `src/lib/CMakeLists.txt`, ensure the source is on disk before `ExternalProject_Add`
  runs. Full checkouts (TheRock CI) already have `profilers/profiler-hub`, so the guard is
  a no-op; sparse checkouts get the path materialized at configure time. This only fetches
  the source; the build still happens via `ExternalProject_Add` (not `add_subdirectory`),
  so dependency isolation is preserved. Requires cone-mode sparse checkout (git >= 2.27
  default).

  ```cmake
  set(PROFILER_HUB_PATH profilers/profiler-hub)

  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --show-toplevel
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE REPO_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(
    COMMAND ${GIT_EXECUTABLE} config --get core.sparseCheckout
    WORKING_DIRECTORY ${REPO_ROOT}
    OUTPUT_VARIABLE SPARSE_ENABLED OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

  if(SPARSE_ENABLED STREQUAL "true" AND NOT EXISTS "${REPO_ROOT}/${PROFILER_HUB_PATH}/CMakeLists.txt")
    message(STATUS "Adding profiler-hub to sparse-checkout...")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} sparse-checkout add ${PROFILER_HUB_PATH}
      WORKING_DIRECTORY ${REPO_ROOT} RESULT_VARIABLE rc)
    if(NOT rc EQUAL 0)
      message(FATAL_ERROR "Failed to expand sparse-checkout for profiler-hub")
    endif()
  endif()
  # source now guaranteed on disk; ExternalProject_Add below builds it in isolation.
  ```
- In `src/lib/CMakeLists.txt`, add an `ExternalProject_Add` (or equivalent) that builds
  profiler-hub from `profilers/profiler-hub` with `PROFILER_HUB_BUILD_TESTS=OFF`,
  `PROFILER_HUB_BUILD_BENCHMARKS=OFF`, default bundled schema, producing
  `libprofiler-hub.a`.
- Link `profiler-hub-static` into `rocprofiler-compute-tool`
  (`src/lib/rocprofiler_compute_tool/CMakeLists.txt`).
- Watch for dependency target collisions: profiler-hub pulls its own fmt, spdlog,
  nlohmann_json, sqlite3 (via `find_package(... QUIET)` then FetchContent), while
  `src/lib/external/` already vendors fmt, json, and googletest. The ExternalProject
  (separate CMake invocation) is chosen specifically to isolate profiler-hub's
  dependency resolution from the native tool's `external/` tree and avoid duplicate
  target definitions.

Phase 2 validation: repeat the Phase 1 functional checks against the
ExternalProject-built static library, plus a clean-from-scratch build to confirm the
ExternalProject wiring and dependency isolation hold.
