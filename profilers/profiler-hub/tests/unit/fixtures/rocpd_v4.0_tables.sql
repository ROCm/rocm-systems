-- =============================================================================
-- rocpd Schema Version 4.0  (canonical v4)
-- =============================================================================
-- Source: PR-347 (rocm-systems) generated rocpd_tables.sql build output.
--
-- NOTE: PR-347 also produced a separate, later 4.1.0 schema variant not used
-- by this test suite. This file is the top-level v4.0 schema with the
-- rocpd_timestamp spine.
--
-- KEY CHANGES vs v3:
--
-- 1. rocpd_track gains: ppid, agent_id, queue_id, stream_id.
--    Now the universal FK anchor for ALL event types.
--    Track topology:
--      CPU thread:  (nid, pid, tid)
--      GPU queue:   (nid, pid, agent_id, queue_id)
--      GPU stream:  (nid, pid, agent_id, stream_id)
--
-- 2. NEW: rocpd_timestamp table — a normalized timestamp spine.
--    All start/end times on event tables are now start_id/end_id FKs
--    into rocpd_timestamp rather than inline BIGINT columns.
--    rocpd_timestamp.phase: 0=instantaneous, 1=start/enter, 2=end/exit.
--
-- 3. ALL event tables use track_id FK (no inline identity columns).
--
-- 4. rocpd_event gains stack_depth column; loses call_stack/line_info JSONB blobs.
--    New relational tables: rocpd_line_info, rocpd_call_stack.
--    New info tables: rocpd_info_category, rocpd_info_address_range,
--                     rocpd_info_source_code, rocpd_info_pc.
--
-- 5. rocpd_memory_allocate gains running_total_bytes column.
--
-- 6. rocpd_sample uses timestamp_id (FK → rocpd_timestamp) instead of inline timestamp.
--
-- Implication for profiler-hub reader API:
--   get_interval_track(track_id) requires a JOIN through rocpd_timestamp:
--     SELECT ts_start.value, ts_end.value, r.id
--     FROM rocpd_region r
--     JOIN rocpd_timestamp ts_start ON ts_start.id = r.start_id
--     JOIN rocpd_timestamp ts_end   ON ts_end.id   = r.end_id
--     WHERE r.track_id = ?
--   get_scalar_track(track_id) similarly joins through timestamp_id.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Utility / identity tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS
    "rocpd_metadata{{uuid}}" (
        "id"    INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "tag"   TEXT NOT NULL,
        "value" TEXT NOT NULL
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_string{{uuid}}` (
        "id"     INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"   TEXT DEFAULT "{{guid}}" NOT NULL,
        "string" TEXT NOT NULL UNIQUE ON CONFLICT ABORT
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_info_node{{uuid}}` (
        "id"            INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"          TEXT DEFAULT "{{guid}}" NOT NULL,
        "hash"          BIGINT NOT NULL UNIQUE,
        "machine_id"    TEXT NOT NULL UNIQUE,
        "name"          TEXT,
        "system_name"   TEXT,
        "hostname"      TEXT,
        "release"       TEXT,
        "version"       TEXT,
        "hardware_name" TEXT,
        "domain_name"   TEXT
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_info_process{{uuid}}` (
        "id"          INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"        TEXT DEFAULT "{{guid}}" NOT NULL,
        "nid"         INTEGER NOT NULL,
        "ppid"        INTEGER,
        "pid"         INTEGER NOT NULL,
        "name"        TEXT,
        "init"        BIGINT,
        "fini"        BIGINT,
        "start"       BIGINT,
        "end"         BIGINT,
        "command"     TEXT,
        "environment" JSONB DEFAULT "{}" NOT NULL,
        "extdata"     JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node{{uuid}}` (id) ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_info_thread{{uuid}}` (
        "id"      INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"    TEXT DEFAULT "{{guid}}" NOT NULL,
        "nid"     INTEGER NOT NULL,
        "ppid"    INTEGER,
        "pid"     INTEGER NOT NULL,
        "tid"     INTEGER NOT NULL,
        "name"    TEXT,
        "start"   BIGINT,
        "end"     BIGINT,
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node{{uuid}}`    (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid) REFERENCES `rocpd_info_process{{uuid}}` (id) ON UPDATE CASCADE
    );

-- NEW in v4: Category table (replaces string FK in rocpd_event.category_id)
CREATE TABLE IF NOT EXISTS
    `rocpd_info_category{{uuid}}` (
        "id"      INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"    TEXT DEFAULT "{{guid}}" NOT NULL,
        "name"    TEXT NOT NULL,
        "extdata" JSONB DEFAULT "{}" NOT NULL
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_info_agent{{uuid}}` (
        "id"             INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"           TEXT DEFAULT "{{guid}}" NOT NULL,
        "nid"            INTEGER NOT NULL,
        "pid"            INTEGER NOT NULL,
        "type"           TEXT CHECK ("type" IN ('CPU', 'GPU')),
        "absolute_index" INTEGER,
        "logical_index"  INTEGER,
        "type_index"     INTEGER,
        "uuid"           INTEGER,
        "name"           TEXT,
        "generic_name"   TEXT,         -- replaces user_name from v3
        "model_name"     TEXT,
        "vendor_name"    TEXT,
        "product_name"   TEXT,
        "extdata"        JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node{{uuid}}`    (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid) REFERENCES `rocpd_info_process{{uuid}}` (id) ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_info_queue{{uuid}}` (
        "id"      INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"    TEXT DEFAULT "{{guid}}" NOT NULL,
        "nid"     INTEGER NOT NULL,
        "pid"     INTEGER NOT NULL,
        "name"    TEXT,
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node{{uuid}}`    (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid) REFERENCES `rocpd_info_process{{uuid}}` (id) ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_info_stream{{uuid}}` (
        "id"      INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"    TEXT DEFAULT "{{guid}}" NOT NULL,
        "nid"     INTEGER NOT NULL,
        "pid"     INTEGER NOT NULL,
        "name"    TEXT,
        "extdata" JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node{{uuid}}`    (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid) REFERENCES `rocpd_info_process{{uuid}}` (id) ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_info_pmc{{uuid}}` (
        "id"               INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"             TEXT DEFAULT "{{guid}}" NOT NULL,
        "nid"              INTEGER NOT NULL,
        "pid"              INTEGER NOT NULL,
        "agent_id"         INTEGER,
        "target_arch"      TEXT CHECK ("target_arch" IN ('CPU', 'GPU')),
        "event_code"       INT,
        "instance_id"      INTEGER,
        "name"             TEXT NOT NULL,
        "symbol"           TEXT NOT NULL,
        "qualifier"        TEXT,          -- NEW in v4
        "description"      TEXT,
        "long_description" TEXT DEFAULT "",
        "component"        TEXT,
        "units"            TEXT DEFAULT "",
        "value_type"       TEXT CHECK ("value_type" IN ('ABS', 'ACCUM', 'RELATIVE')),
        "block"            TEXT,
        "expression"       TEXT,
        "is_constant"      INTEGER,
        "is_derived"       INTEGER,
        "extdata"          JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid)      REFERENCES `rocpd_info_node{{uuid}}`    (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid)      REFERENCES `rocpd_info_process{{uuid}}` (id) ON UPDATE CASCADE,
        FOREIGN KEY (agent_id) REFERENCES `rocpd_info_agent{{uuid}}`   (id) ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_info_code_object{{uuid}}` (
        "id"           INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"         TEXT DEFAULT "{{guid}}" NOT NULL,
        "nid"          INTEGER NOT NULL,
        "pid"          INTEGER NOT NULL,
        "agent_id"     INTEGER,
        "uri"          TEXT,
        "load_base"    BIGINT,
        "load_size"    BIGINT,
        "load_delta"   BIGINT,
        "storage_type" TEXT CHECK ("storage_type" IN ('FILE', 'MEMORY')),
        "extdata"      JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid)      REFERENCES `rocpd_info_node{{uuid}}`    (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid)      REFERENCES `rocpd_info_process{{uuid}}` (id) ON UPDATE CASCADE,
        FOREIGN KEY (agent_id) REFERENCES `rocpd_info_agent{{uuid}}`   (id) ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_info_kernel_symbol{{uuid}}` (
        "id"                        INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"                      TEXT DEFAULT "{{guid}}" NOT NULL,
        "nid"                       INTEGER NOT NULL,
        "pid"                       INTEGER NOT NULL,
        "code_object_id"            INTEGER NOT NULL,
        "kernel_name"               TEXT,
        "display_name"              TEXT,
        "kernel_object"             INTEGER,
        "kernarg_segment_size"      INTEGER,
        "kernarg_segment_alignment" INTEGER,
        "group_segment_size"        INTEGER,
        "private_segment_size"      INTEGER,
        "sgpr_count"                INTEGER,
        "arch_vgpr_count"           INTEGER,
        "accum_vgpr_count"          INTEGER,
        "extdata"                   JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid)            REFERENCES `rocpd_info_node{{uuid}}`        (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid)            REFERENCES `rocpd_info_process{{uuid}}`     (id) ON UPDATE CASCADE,
        FOREIGN KEY (code_object_id) REFERENCES `rocpd_info_code_object{{uuid}}` (id) ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_info_address_range{{uuid}}` (
        "id"           INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"         TEXT DEFAULT "{{guid}}" NOT NULL,
        "nid"          INTEGER NOT NULL,
        "pid"          INTEGER NOT NULL,
        "address_base" BIGINT,
        "address_low"  BIGINT CHECK ("address_low" >= "address_base"),
        "address_high" BIGINT CHECK ("address_high" >= "address_low"),
        "extdata"      JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid) REFERENCES `rocpd_info_node{{uuid}}`    (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid) REFERENCES `rocpd_info_process{{uuid}}` (id) ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_info_source_code{{uuid}}` (
        "id"           INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"         TEXT DEFAULT "{{guid}}" NOT NULL,
        "nid"          INTEGER NOT NULL,
        "pid"          INTEGER NOT NULL,
        "address_id"   INTEGER,
        "file"         TEXT,
        "line_number"  INTEGER,
        "lines"        JSONB DEFAULT "[]" NOT NULL,
        "instructions" JSONB DEFAULT "[]" NOT NULL,
        "extdata"      JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid)        REFERENCES `rocpd_info_node{{uuid}}`          (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid)        REFERENCES `rocpd_info_process{{uuid}}`       (id) ON UPDATE CASCADE,
        FOREIGN KEY (address_id) REFERENCES `rocpd_info_address_range{{uuid}}` (id) ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_info_pc{{uuid}}` (
        "id"         INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"       TEXT DEFAULT "{{guid}}" NOT NULL,
        "nid"        INTEGER NOT NULL,
        "pid"        INTEGER NOT NULL,
        "function"   TEXT NOT NULL,
        "address_id" INTEGER,
        "file"       TEXT,
        "line"       INTEGER,
        "extdata"    JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid)        REFERENCES `rocpd_info_node{{uuid}}`          (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid)        REFERENCES `rocpd_info_process{{uuid}}`       (id) ON UPDATE CASCADE,
        FOREIGN KEY (address_id) REFERENCES `rocpd_info_address_range{{uuid}}` (id) ON UPDATE CASCADE
    );

-- ---------------------------------------------------------------------------
-- rocpd_track  (v4: universal FK anchor for all event types)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS
    `rocpd_track{{uuid}}` (
        "id"        INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"      TEXT DEFAULT "{{guid}}" NOT NULL,
        "nid"       INTEGER NOT NULL,
        "ppid"      INTEGER,
        "pid"       INTEGER,
        "tid"       INTEGER,
        "agent_id"  INTEGER,
        "queue_id"  INTEGER,
        "stream_id" INTEGER,
        "name_id"   INTEGER,
        "extdata"   JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (nid)       REFERENCES `rocpd_info_node{{uuid}}`    (id) ON UPDATE CASCADE,
        FOREIGN KEY (pid)       REFERENCES `rocpd_info_process{{uuid}}` (id) ON UPDATE CASCADE,
        FOREIGN KEY (tid)       REFERENCES `rocpd_info_thread{{uuid}}`  (id) ON UPDATE CASCADE,
        FOREIGN KEY (agent_id)  REFERENCES `rocpd_info_agent{{uuid}}`   (id) ON UPDATE CASCADE,
        FOREIGN KEY (queue_id)  REFERENCES `rocpd_info_queue{{uuid}}`   (id) ON UPDATE CASCADE,
        FOREIGN KEY (stream_id) REFERENCES `rocpd_info_stream{{uuid}}`  (id) ON UPDATE CASCADE,
        FOREIGN KEY (name_id)   REFERENCES `rocpd_string{{uuid}}`       (id) ON UPDATE CASCADE
    );

-- ---------------------------------------------------------------------------
-- NEW in v4: rocpd_timestamp — normalized timestamp spine
-- All start/end columns on event tables are start_id/end_id FKs into this table.
-- phase: 0=instantaneous, 1=start/enter/load, 2=end/exit/unload
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS
    `rocpd_timestamp{{uuid}}` (
        "id"       INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"     TEXT DEFAULT "{{guid}}" NOT NULL,
        "value"    BIGINT NOT NULL,
        "phase"    INTEGER CHECK ("phase" IN (0, 1, 2)),
        "track_id" INTEGER,  -- NULL if associated with more than one track
        FOREIGN KEY (track_id) REFERENCES `rocpd_track{{uuid}}` (id) ON UPDATE CASCADE
    );

-- ---------------------------------------------------------------------------
-- Event / arg / stack tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS
    `rocpd_event{{uuid}}` (
        "id"              INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"            TEXT DEFAULT "{{guid}}" NOT NULL,
        "category_id"     INTEGER,
        "stack_id"        INTEGER,
        "parent_stack_id" INTEGER,
        "stack_depth"     INTEGER,   -- NEW in v4
        "correlation_id"  INTEGER,
        "extdata"         JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (category_id) REFERENCES `rocpd_info_category{{uuid}}` (id) ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_arg{{uuid}}` (
        "id"       INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"     TEXT DEFAULT "{{guid}}" NOT NULL,
        "event_id" INTEGER NOT NULL,
        "position" INTEGER NOT NULL,
        "type"     TEXT NOT NULL,
        "name"     TEXT NOT NULL,
        "value"    TEXT,
        "extdata"  JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (event_id) REFERENCES `rocpd_event{{uuid}}` (id) ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_line_info{{uuid}}` (
        "id"             INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"           TEXT DEFAULT "{{guid}}" NOT NULL,
        "event_id"       INTEGER NOT NULL,
        "source_code_id" INTEGER,
        "pc_id"          INTEGER,
        "extdata"        JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (event_id)       REFERENCES `rocpd_event{{uuid}}`            (id) ON UPDATE CASCADE,
        FOREIGN KEY (source_code_id) REFERENCES `rocpd_info_source_code{{uuid}}` (id) ON UPDATE CASCADE,
        FOREIGN KEY (pc_id)          REFERENCES `rocpd_info_pc{{uuid}}`          (id) ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_call_stack{{uuid}}` (
        "id"       INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"     TEXT DEFAULT "{{guid}}" NOT NULL,
        "event_id" INTEGER NOT NULL,
        "pc_id"    INTEGER,
        "depth"    INTEGER NOT NULL,
        "extdata"  JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (event_id) REFERENCES `rocpd_event{{uuid}}`  (id) ON UPDATE CASCADE,
        FOREIGN KEY (pc_id)    REFERENCES `rocpd_info_pc{{uuid}}` (id) ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_pmc_event{{uuid}}` (
        "id"       INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"     TEXT DEFAULT "{{guid}}" NOT NULL,
        "event_id" INTEGER,
        "pmc_id"   INTEGER NOT NULL,
        "value"    REAL DEFAULT 0.0,
        "extdata"  JSONB DEFAULT "{}",
        FOREIGN KEY (pmc_id)   REFERENCES `rocpd_info_pmc{{uuid}}` (id) ON UPDATE CASCADE,
        FOREIGN KEY (event_id) REFERENCES `rocpd_event{{uuid}}`    (id) ON UPDATE CASCADE
    );

-- ---------------------------------------------------------------------------
-- Event tables (all use track_id + timestamp_id/start_id/end_id FKs)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS
    `rocpd_region{{uuid}}` (
        "id"       INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"     TEXT DEFAULT "{{guid}}" NOT NULL,
        "track_id" INTEGER NOT NULL,
        "name_id"  INTEGER NOT NULL,
        "start_id" INTEGER NOT NULL,
        "end_id"   INTEGER NOT NULL,
        "event_id" INTEGER,
        "extdata"  JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (track_id) REFERENCES `rocpd_track{{uuid}}`     (id) ON UPDATE CASCADE,
        FOREIGN KEY (name_id)  REFERENCES `rocpd_string{{uuid}}`    (id) ON UPDATE CASCADE,
        FOREIGN KEY (start_id) REFERENCES `rocpd_timestamp{{uuid}}` (id) ON UPDATE CASCADE,
        FOREIGN KEY (end_id)   REFERENCES `rocpd_timestamp{{uuid}}` (id) ON UPDATE CASCADE,
        FOREIGN KEY (event_id) REFERENCES `rocpd_event{{uuid}}`     (id) ON UPDATE CASCADE
    );

-- Read pattern: SELECT ts.value, s.id
--               FROM rocpd_sample s
--               JOIN rocpd_timestamp ts ON ts.id = s.timestamp_id
--               WHERE s.track_id = ?
CREATE TABLE IF NOT EXISTS
    `rocpd_sample{{uuid}}` (
        "id"           INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"         TEXT DEFAULT "{{guid}}" NOT NULL,
        "track_id"     INTEGER NOT NULL,
        "name_id"      INTEGER NOT NULL,
        "timestamp_id" INTEGER NOT NULL,
        "event_id"     INTEGER,
        "extdata"      JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (track_id)     REFERENCES `rocpd_track{{uuid}}`     (id) ON UPDATE CASCADE,
        FOREIGN KEY (name_id)      REFERENCES `rocpd_string{{uuid}}`    (id) ON UPDATE CASCADE,
        FOREIGN KEY (timestamp_id) REFERENCES `rocpd_timestamp{{uuid}}` (id) ON UPDATE CASCADE,
        FOREIGN KEY (event_id)     REFERENCES `rocpd_event{{uuid}}`     (id) ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_kernel_dispatch{{uuid}}` (
        "id"                   INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"                 TEXT DEFAULT "{{guid}}" NOT NULL,
        "track_id"             INTEGER NOT NULL,
        "kernel_id"            INTEGER NOT NULL,
        "dispatch_id"          INTEGER NOT NULL,
        "start_id"             INTEGER NOT NULL,
        "end_id"               INTEGER NOT NULL,
        "private_segment_size" INTEGER,
        "group_segment_size"   INTEGER,
        "workgroup_size_x"     INTEGER NOT NULL,
        "workgroup_size_y"     INTEGER NOT NULL,
        "workgroup_size_z"     INTEGER NOT NULL,
        "grid_size_x"          INTEGER NOT NULL,
        "grid_size_y"          INTEGER NOT NULL,
        "grid_size_z"          INTEGER NOT NULL,
        "region_name_id"       INTEGER,
        "event_id"             INTEGER,
        "extdata"              JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (track_id)       REFERENCES `rocpd_track{{uuid}}`             (id) ON UPDATE CASCADE,
        FOREIGN KEY (kernel_id)      REFERENCES `rocpd_info_kernel_symbol{{uuid}}` (id) ON UPDATE CASCADE,
        FOREIGN KEY (start_id)       REFERENCES `rocpd_timestamp{{uuid}}`         (id) ON UPDATE CASCADE,
        FOREIGN KEY (end_id)         REFERENCES `rocpd_timestamp{{uuid}}`         (id) ON UPDATE CASCADE,
        FOREIGN KEY (region_name_id) REFERENCES `rocpd_string{{uuid}}`            (id) ON UPDATE CASCADE,
        FOREIGN KEY (event_id)       REFERENCES `rocpd_event{{uuid}}`             (id) ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_memory_copy{{uuid}}` (
        "id"             INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"           TEXT DEFAULT "{{guid}}" NOT NULL,
        "track_id"       INTEGER NOT NULL,
        "start_id"       INTEGER NOT NULL,
        "end_id"         INTEGER NOT NULL,
        "name_id"        INTEGER NOT NULL,
        "dst_agent_id"   INTEGER,
        "dst_address"    INTEGER,
        "src_agent_id"   INTEGER,
        "src_address"    INTEGER,
        "size"           INTEGER NOT NULL,
        "region_name_id" INTEGER,
        "event_id"       INTEGER,
        "extdata"        JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (track_id)       REFERENCES `rocpd_track{{uuid}}`     (id) ON UPDATE CASCADE,
        FOREIGN KEY (start_id)       REFERENCES `rocpd_timestamp{{uuid}}` (id) ON UPDATE CASCADE,
        FOREIGN KEY (end_id)         REFERENCES `rocpd_timestamp{{uuid}}` (id) ON UPDATE CASCADE,
        FOREIGN KEY (name_id)        REFERENCES `rocpd_string{{uuid}}`    (id) ON UPDATE CASCADE,
        FOREIGN KEY (dst_agent_id)   REFERENCES `rocpd_info_agent{{uuid}}` (id) ON UPDATE CASCADE,
        FOREIGN KEY (src_agent_id)   REFERENCES `rocpd_info_agent{{uuid}}` (id) ON UPDATE CASCADE,
        FOREIGN KEY (region_name_id) REFERENCES `rocpd_string{{uuid}}`    (id) ON UPDATE CASCADE,
        FOREIGN KEY (event_id)       REFERENCES `rocpd_event{{uuid}}`     (id) ON UPDATE CASCADE
    );

CREATE TABLE IF NOT EXISTS
    `rocpd_memory_allocate{{uuid}}` (
        "id"                  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        "guid"                TEXT DEFAULT "{{guid}}" NOT NULL,
        "track_id"            INTEGER NOT NULL,
        "type"                TEXT CHECK ("type" IN ('ALLOC', 'FREE', 'REALLOC', 'RECLAIM')),
        "level"               TEXT CHECK ("level" IN ('REAL', 'VIRTUAL', 'SCRATCH')),
        "start_id"            INTEGER NOT NULL,
        "end_id"              INTEGER NOT NULL,
        "name_id"             INTEGER NOT NULL,
        "address"             INTEGER,
        "size"                INTEGER NOT NULL,
        "running_total_bytes" INTEGER,            -- NEW in v4
        "region_name_id"      INTEGER,
        "event_id"            INTEGER,
        "extdata"             JSONB DEFAULT "{}" NOT NULL,
        FOREIGN KEY (track_id)       REFERENCES `rocpd_track{{uuid}}`     (id) ON UPDATE CASCADE,
        FOREIGN KEY (start_id)       REFERENCES `rocpd_timestamp{{uuid}}` (id) ON UPDATE CASCADE,
        FOREIGN KEY (end_id)         REFERENCES `rocpd_timestamp{{uuid}}` (id) ON UPDATE CASCADE,
        FOREIGN KEY (name_id)        REFERENCES `rocpd_string{{uuid}}`    (id) ON UPDATE CASCADE,
        FOREIGN KEY (region_name_id) REFERENCES `rocpd_string{{uuid}}`    (id) ON UPDATE CASCADE,
        FOREIGN KEY (event_id)       REFERENCES `rocpd_event{{uuid}}`     (id) ON UPDATE CASCADE
    );
