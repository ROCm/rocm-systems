# Profiler Hub Reader Execution Flow

This document shows the execution path of each reader call, from Optiq down
to SQLite or the on-disk cache. The text explanation of the same paths is in
[reader-architecture.md](reader-architecture.md). Function names are the real
names in the code. `_GUID` stands for the GUID suffix of a schema-3 table.

The diagrams are Mermaid. They render on GitHub and in any Mermaid viewer.

## 1. End-to-end flow

This diagram shows every path in one picture. Solid arrows are calls. Dotted
arrows are reads from a cache or writes to it.

```mermaid
flowchart TB
  subgraph optiq ["Optiq: src/model/src/database"]
    o_open["ProfilerHub::ReadTraceMetadata<br/>HubYamlFiles expands a YAML project"]
    o_slice["ReadTraceSlice<br/>ReadTracePMCSlice<br/>BuildHistogram"]
    o_flow["ReadFlowTraceInfo"]
    o_stack["ReadStackTraceInfo"]
    o_det["ReadExtEventInfo"]
    o_table["Table read<br/>TableProcessor"]
    o_isSum{"IsNameDurationSummary<br/>and EventSummaryInSqlite?"}
    o_sum["ExecuteNameDurationSummary<br/>SummarizeNameDurations"]
    o_sumcache{"SummaryCacheFind<br/>summary.bin hit?"}
    o_store["SummaryCacheStore"]
    o_loadall["Load all rows,<br/>aggregate in memory"]
    o_trim["SaveTrimmedData"]
    o_clean["ProfilerHub::Cleanup"]
  end

  subgraph capi ["C API: profiler_hub_c_api.cpp"]
    c_create["ph_ctx_create_files_cached"]
    c_dir["ph_trace_cache_directory"]
    c_events["ph_get_track_events<br/>ph_get_track_samples"]
    c_release["ph_release_track_events<br/>ph_release_track_samples"]
    c_hist["ph_get_histogram"]
    c_corr["ph_get_correlated_events"]
    c_cs["ph_get_call_stack"]
    c_det["ph_get_*_details"]
    c_qcreate["ph_event_query_create"]
    c_qread["ph_event_query_read<br/>ph_event_query_id_span"]
    c_qsum["ph_event_query_summarize_durations"]
    c_trim["ph_trim_trace"]
    run_op{"run_operation:<br/>future passed?"}
    pool_task["context thread pool task"]
    inline_call["guard_call on caller thread"]
    ready["ph_ctx::wait_until_ready"]
  end

  subgraph ctx ["ph_ctx: ph_ctx.cpp"]
    accept["accept_traces<br/>probe_file per path"]
    readtrace["read_trace"]
    place["place_of<br/>global id to source and local id"]
    sfe["source_for_event<br/>track_of on each source"]
    pick["pick source for query"]
    trimctx["ph_ctx::trim_trace"]
  end

  subgraph caches ["Caches"]
    ctr["cached_track_reader<br/>in-memory tracks"]
    tfs["track_file_store"]
    cas["catalog_aggregate_store"]
    namedir["cache_directory<br/>date_format_version_name_size folder"]
  end

  subgraph disk ["Cache folder: config/date_format_version_name_size/"]
    d_tl[("timeline/<br/>timeline.meta, strings.bin,<br/>e_id.bin, i_id.bin, s_id.bin")]
    d_cat[("catalog/<br/>aggregates-src.bin<br/>derived-src.bin")]
    d_sum[("summary.bin")]
  end

  subgraph fmt ["Format reader"]
    registry["trace_formats registry"]
    source["format_source<br/>rocpd_source"]
    pool["connection_pool<br/>8 reader_t connections"]
    v2["rocpd_v2_reader"]
    v3["rocpd_v3_reader"]
    trimmer["trim_trace_files<br/>trace_trim.cpp"]
  end

  subgraph db ["SQLite"]
    sql2[("schema 2 .rpd<br/>rocpd_api, rocpd_op,<br/>rocpd_api_ops, rocpd_monitor,<br/>rocpd_string")]
    sql3[("schema 3<br/>rocpd_event_GUID, rocpd_region_GUID,<br/>rocpd_kernel_dispatch_GUID,<br/>rocpd_memory_*_GUID, rocpd_sample_GUID,<br/>rocpd_info_*_GUID")]
    out[("trimmed copy")]
  end

  o_open --> c_create --> accept --> registry
  accept --> readtrace
  o_slice --> c_events
  o_slice --> c_release
  o_slice --> c_hist
  o_flow --> c_corr
  o_stack --> c_cs
  o_det --> c_det
  o_table --> o_isSum
  o_isSum -->|"no"| o_loadall --> c_qcreate
  o_isSum -->|"yes"| o_sum --> o_sumcache
  o_sumcache -->|"hit"| o_table
  o_sumcache -->|"miss"| c_qcreate
  o_sumcache -.->|"read"| d_sum
  c_qcreate --> c_qsum
  c_qsum -->|"NOT_SUPPORTED"| o_loadall
  c_qsum -->|"rows"| o_store
  o_store -.->|"append"| d_sum
  o_loadall --> c_qread
  o_trim --> c_trim
  o_clean --> c_dir --> namedir

  c_events --> run_op
  c_hist --> run_op
  c_corr --> run_op
  c_cs --> run_op
  c_det --> run_op
  run_op -->|"yes"| pool_task --> ready
  run_op -->|"no"| inline_call --> ready
  c_qread --> ready
  c_qsum --> ready

  ready --> place
  ready --> sfe
  ready --> pick
  c_trim --> trimctx --> registry
  registry -->|"trim"| trimmer

  readtrace --> source
  readtrace -.->|"load or save"| cas
  readtrace -.->|"prepare, write or read"| tfs
  readtrace --> namedir
  cas -.-> d_cat
  tfs -.-> d_tl

  place -->|"timeline cache attached"| tfs
  place -->|"no cache folder"| ctr
  ctr -->|"first read of track"| source
  sfe --> source
  sfe -->|"nesting depth"| ctr
  ctr -->|"has_local_events"| tfs
  pick --> source

  source --> pool
  pool --> v2 --> sql2
  pool --> v3 --> sql3
  registry -->|"open"| source
  registry --> v2
  registry --> v3
  trimmer --> sql2
  trimmer --> sql3
  trimmer --> out
```

## 2. Opening a trace

```mermaid
sequenceDiagram
  autonumber
  participant O as Optiq ReadTraceMetadata
  participant C as C API
  participant X as ph_ctx
  participant P as probe_file
  participant R as format registry
  participant SRC as rocpd_source
  participant CP as connection_pool
  participant RD as reader rocpd_v2 or v3
  participant CA as catalog_aggregate_store
  participant TF as track_file_store
  participant S as SQLite

  O->>O: HubYamlFiles(path) gives the path list
  O->>C: ph_ctx_create_files_cached(paths, config folder, future)
  C->>X: accept_traces(paths), on the caller thread
  loop each path
    X->>P: probe_file
    P->>R: first entry that recognises the file
    R->>S: header, sqlite_master, rocpd_metadata or rocpd_metadata_GUID
    R-->>P: schema family and version
    P-->>X: probe result
  end
  alt future passed
    C-->>O: PH_RESULT_SUCCESS, load continues on the thread pool
  end
  X->>X: read_trace
  X->>R: open_sources: open_format_source for each source
  R->>SRC: the rocpd entry opens a rocpd_source
  SRC->>CP: one pool of 8 connections
  CP->>R: create reader for each connection
  X->>SRC: load_catalog
  SRC->>RD: populate_reader_catalog
  RD->>S: create topology indexes (written into the trace file)
  RD->>S: strings, nodes, processes, threads, agents (in order)
  RD->>CA: load catalog cache
  alt catalog cache matches path, size, mtime
    CA-->>RD: tracks and aggregates
  else miss
    RD->>S: GROUP BY scans for tracks, queues, counters (in parallel)
    SRC->>CA: save_catalog_aggregates (v2) or save_derived_tracks (v3)
  end
  X->>X: initialize_track_list: global_of_local, m_track_by_id
  X->>TF: prepare_track_store(cache folder)
  alt timeline.meta matches
    TF-->>X: ready
  else mismatch or missing
    TF->>TF: clear timeline folder for writing
  end
  loop load_all_tracks, every track
    alt cache ready
      TF-->>X: depth and event count
    else cache being written
      X->>RD: read whole track
      RD->>S: track SELECT
      X->>TF: write e_id.bin or s_id.bin, i_id.bin
    else no cache folder
      X->>RD: read whole track into cached_track_reader
      RD->>S: track SELECT
    end
  end
  X->>SRC: finish_load, once every track is loaded
  X->>X: attach_track_store, initialize_node_* lists
  X-->>O: future finished, or the call returns
```

## 3. Timeline slice and release

```mermaid
sequenceDiagram
  autonumber
  participant O as Optiq ReadTraceSlice
  participant C as C API
  participant X as ph_ctx
  participant CT as cached_track_reader
  participant TF as track_file_store
  participant RD as format source and reader
  participant S as SQLite

  O->>C: ph_get_track_events(track id, window, future)
  C->>C: run_operation
  C->>X: wait_until_ready
  X->>X: place_of(global id) gives source and local id
  alt timeline files attached and track stored
    X->>TF: events(local id, window)
    TF->>TF: seek nearest checkpoint in e_id.bin
    TF->>TF: read records, names from strings.bin
    TF-->>X: window, held in the context
  else no cache folder
    X->>CT: events(local id, window)
    alt track not loaded yet
      CT->>RD: read track (id ranges in parallel for large threads)
      RD->>S: track SELECT
    end
    CT-->>X: whole track or a window copy
  end
  X-->>O: ph_event_list_t
  O->>O: build slice
  O->>C: ph_release_track_events(list)
  C->>X: free the window (cache) or clear the struct (memory)

  O->>C: ph_get_histogram(max buckets)
  alt stored grid matches
    X->>TF: 500-bucket histogram from timeline.meta
  else other grid
    X->>TF: rebuild from record files
  else no cache folder
    X->>RD: histogram SQL per source
    RD->>S: bucketed duration query
  end
```

Samples (`ReadTracePMCSlice`) follow the same path with `ph_get_track_samples`, `s_id.bin`,
`rocpd_monitor` (schema 2), or the PMC sample statement (schema 3), and are
freed with `ph_release_track_samples`.

## 4. Correlated events and call stack

```mermaid
sequenceDiagram
  autonumber
  participant O as Optiq ReadFlowTraceInfo
  participant C as C API
  participant X as ph_ctx
  participant RD as format source and reader
  participant S as SQLite
  participant CT as cached_track_reader
  participant TF as track_file_store

  O->>O: zero-initialize ph_event_t and ph_event_list_t
  O->>C: ph_get_correlated_events(event id, type)
  C->>X: wait_until_ready
  X->>X: source_for_event(track, id, type)
  alt one source
    X->>X: source 0
  else several sources
    loop each source
      X->>RD: track_of(id, type)
    end
  end
  alt no source has the event
    X-->>O: empty list, success
  end
  X->>RD: correlated events
  alt schema 2
    RD->>S: rocpd_api_ops from API to op, or op to API
  else schema 3
    RD->>S: rocpd_event_GUID joined on shared stack_id
  end
  loop each result
    X->>X: place on a published track
    alt cannot place
      X->>X: skip
    else placed
      X->>CT: nesting_depth_of
      alt has_local_events in timeline files
        CT->>TF: depth from i_id.bin
      else not stored or read error
        CT->>CT: depth from in-memory track
      end
    end
  end
  X-->>O: endpoints

  O->>C: ph_get_call_stack(event id, type)
  X->>X: source_for_event
  alt schema 2
    RD->>S: enclosing rocpd_api calls on the same pid, tid
  else schema 3
    RD->>S: regions on the event's track, line_info from rocpd_event_GUID
  end
```

## 5. Event details

```mermaid
sequenceDiagram
  autonumber
  participant O as Optiq ReadExtEventInfo
  participant C as C API get_event_details
  participant X as ph_ctx
  participant RD as format source and reader
  participant S as SQLite

  O->>C: ph_get_region_details, kernel_dispatch, memory_copy, memory_alloc
  C->>C: is_duration_event_type, run_operation
  C->>X: wait_until_ready
  alt event track_id is known
    X->>X: place_of(track_id)
  else unknown
    X->>RD: search each source
    X->>X: write track_id back into the event
  end
  X->>RD: details from that source
  alt schema 2
    RD->>S: rocpd_api, rocpd_op, rocpd_api_ops, rocpd_kernelapi, rocpd_copyapi
  else schema 3
    RD->>S: rocpd_region_GUID or kernel_dispatch or memory tables, plus rocpd_event_GUID
  end
  X-->>O: details struct
```

## 6. Table reads and summaries

```mermaid
sequenceDiagram
  autonumber
  participant T as Optiq TableProcessor
  participant D as Optiq ProfilerHub or QueryManager
  participant F as summary.bin
  participant C as C API
  participant X as ph_ctx
  participant RD as format source and reader
  participant S as SQLite

  T->>T: IsNameDurationSummary(commands)
  alt summary and EventSummaryInSqlite
    T->>D: ExecuteNameDurationSummary, SummarizeNameDurations(requests)
    loop each request
      D->>F: SummaryCacheFind(folder, key)
      alt hit
        F-->>D: rows
      else miss
        alt ProfilerHub
          D->>C: ph_event_query_create(filter)
          C->>X: copy filter, match search terms in the catalog
          D->>C: ph_event_query_summarize_durations
          C->>X: pick source, map global to local track ids
          X->>RD: summarize
          RD->>S: COUNT, MIN, MAX, SUM of duration GROUP BY name
          RD-->>D: one callback per name
        else QueryManager (legacy SQLite)
          D->>S: table SQL wrapped in GROUP BY
        end
        D->>F: SummaryCacheStore(folder, key, rows)
      end
    end
    D-->>T: merged rows
    alt result is NotSupported
      T->>T: fall through to load-all
    end
  end
  alt load-all path (non-SQLite sources, or fallback)
    T->>C: ph_event_query_create
    T->>C: ph_event_query_id_span
    C->>RD: MIN(id), MAX(id)
    loop pages
      T->>C: ph_event_query_read(after, capacity)
      RD->>S: next page by id, ORDER BY id LIMIT capacity
    end
    T->>T: aggregate in memory
  end
```

## 7. Trim

```mermaid
sequenceDiagram
  autonumber
  participant O as Optiq SaveTrimmedData
  participant C as C API
  participant X as ph_ctx
  participant TT as trim_trace_files
  participant R as format registry
  participant S as source SQLite
  participant OUT as output SQLite

  O->>C: ph_trim_trace(ctx, window, output path)
  C->>X: trim_trace
  X->>X: sources of more than one format: return PH_RESULT_UNSUPPORTED_FILE
  X->>X: convert window to each source's clock
  X->>R: trim_in_format(schema, version, sources, windows)
  R->>TT: the entry's trim, or PH_RESULT_UNSUPPORTED_FILE when it has none
  TT->>TT: reject output equal to an input, delete old output
  TT->>R: trim rules for the format
  TT->>S: read sqlite_master
  TT->>TT: give each table a role and WHERE clause
  TT->>OUT: create tables
  loop phases: copy, interval, sample, event, event child
    TT->>OUT: ATTACH DATABASE source AS oldDb
    TT->>OUT: INSERT INTO t SELECT * FROM oldDb.t WHERE rule
    alt cancelled
      TT->>OUT: roll back phase
    end
  end
  TT->>OUT: recreate views and indexes
  TT-->>O: result
```

## 8. Cache cleanup

```mermaid
sequenceDiagram
  autonumber
  participant O as Optiq ProfilerHub::Cleanup
  participant C as C API
  participant H as cache_directory
  participant FS as file system

  O->>O: HubYamlFiles(path) gives the same path list as the open
  O->>C: ph_trace_cache_directory(config folder, paths)
  C->>H: probe format and version, newest mtime, name, total size
  H-->>O: config/date_format_version_name_size
  O->>FS: delete folder (timeline, catalog, summary.bin)
```
