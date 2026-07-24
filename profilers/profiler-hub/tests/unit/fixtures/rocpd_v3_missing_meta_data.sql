-- =============================================================================
-- Synthetic v3 missing-metadata naming-fallback fixture (profiler-hub task 048)
-- =============================================================================
-- WHY THIS EXISTS:
--   The 043 coverage recon (gap 11) found the Tier-3 "missing metadata" naming
--   fallbacks in synthesize_derived_tracks() (source/reader_impl.cpp) are never
--   lit by any committed v3 fixture, because every real capture / synthetic
--   fixture names its streams and threads. Specifically dark:
--     * reader_impl.cpp:715  stream track  -> "Stream <id>"  (stream_info absent
--                            OR its name empty)
--     * reader_impl.cpp:772  cpu_thread    -> "Thread"       (thread_info ENTIRELY
--                            absent: region.tid matches no rocpd_info_thread row)
--     * reader_impl.cpp:768  cpu_thread    -> "Thread <tid>" (thread_info present
--                            but its name is NULL/empty)      [cheap sibling]
--     * reader_impl.cpp:252  get_all_agents drops an agent with NULL type_index
--                            ("Corrupted database detected" continue) [cheap]
--
--   This file builds a tiny v3 database whose stream has no name, one thread is
--   entirely absent, one thread has a NULL name, and one agent has a NULL
--   type_index -- so each fallback fires and tests assert the EXACT fallback
--   string / dropped-agent count (behavior, not merely line touches).
--
-- HOW IT IS BUILT (see tests/unit/CMakeLists.txt):
--   {{uuid}} -> "_" + <hex uuid>, {{guid}} -> <hex uuid> (mirrors get_schema_query()),
--   then the canonical v3 schema (source/data_storage/schema/rocpd_tables.sql) and
--   this file are piped through the sqlite3 CLI. The db has NO rocpd_timestamp
--   table, so the reader selects the v3 backend. FK constraints are not enforced by
--   the CLI, so a region.tid that references no rocpd_info_thread row is accepted --
--   that dangling tid is exactly what lights the "Thread" (no thread_info) fallback.
--
-- TRACK / AGENT MATRIX (what the reader returns):
--   stream (from rocpd_memory_copy stream_id=7, distinct nid,pid,stream_id):
--     * stream_id=7 whose rocpd_info_stream row has name=NULL -> name "Stream 7"
--   cpu_thread (synthesized from rocpd_region, distinct nid,pid,tid,is_sample;
--     both regions are non-sample -> main tracks):
--     * tid=555 with NO rocpd_info_thread row        -> name "Thread"
--     * tid=8   -> thread row id=8, tid=99001, name NULL -> name "Thread 99001"
--   dma (incidental, from the same memory_copy; queue_id/dst_agent_id NULL) -- not
--     asserted, harmless.
--   agents (get_all_agents):
--     * id=1 valid (type_index=0) -> kept
--     * id=2 with type_index=NULL -> dropped (corrupted-database continue)
-- =============================================================================

-- The canonical schema sets `PRAGMA foreign_keys = ON`. One region below (tid=555)
-- deliberately references a thread that does NOT exist in rocpd_info_thread -- the
-- partial/corrupt-capture state the bare-"Thread" fallback (reader_impl.cpp:772)
-- exists to defend against. Disable FK enforcement for THIS fixture's inserts so
-- that dangling reference is accepted (build-time only; the reader opens the db
-- read-only and never runs an FK check). This PRAGMA runs after the schema's ON,
-- so it wins for the data below. Every other reference here is valid.
PRAGMA foreign_keys = OFF;

-- Bare alias views (the v3 reader joins these three by bare name) ----------------
CREATE VIEW rocpd_event AS SELECT * FROM "rocpd_event{{uuid}}";
CREATE VIEW rocpd_string AS SELECT * FROM "rocpd_string{{uuid}}";
CREATE VIEW rocpd_sample AS SELECT * FROM "rocpd_sample{{uuid}}";

-- Identity spine ------------------------------------------------------------
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 730048, 'synthetic-machine-v3-missing-meta', 'Linux', 'synth-v3-missing-meta-host');

INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 4242, 'synthetic-missing-meta-app');

-- One thread with a NULL name (lights the "Thread <tid>" fallback for tid=99001).
-- There is deliberately NO thread row for the region below whose tid=555, so that
-- region's track has thread_info entirely absent and lights the bare "Thread".
INSERT INTO "rocpd_info_thread{{uuid}}" (id, nid, pid, tid, name)
VALUES (8, 1, 1, 99001, NULL);

-- Two GPU agents: one valid, one with a NULL type_index that get_all_agents must
-- drop ("Corrupted database detected" continue).
INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 1, 'GPU', 0, 0,    'Synthetic GPU 0'),
       (2, 1, 1, 'GPU', 1, NULL, 'Synthetic GPU corrupt');

-- One stream with a NULL name (lights the "Stream <id>" fallback for stream_id=7).
INSERT INTO "rocpd_info_stream{{uuid}}" (id, nid, pid, name)
VALUES (7, 1, 1, NULL);

-- Strings (memory_copy + region name_id targets).
INSERT INTO "rocpd_string{{uuid}}" (id, string)
VALUES (1, 'copyHtoD'),
       (2, 'RegionOnUnknownThread'),
       (3, 'RegionOnUnnamedThread');

-- Stream source: a single memory copy carrying stream_id=7. distinct_stream_tracks
-- discovers (nid,pid,stream_id)=(1,1,7); the stream's rocpd_info_stream row has a
-- NULL name -> "Stream 7". queue_id / dst_agent_id NULL (incidental dma track).
-- cols: (id, nid, pid, start, "end", name_id, dst_agent_id, size, queue_id, stream_id, event_id)
INSERT INTO "rocpd_memory_copy{{uuid}}"
    (id, nid, pid, start, "end", name_id, dst_agent_id, size, queue_id, stream_id, event_id)
VALUES (1, 1, 1, 1000, 1100, 1, NULL, 1024, NULL, 7, NULL);

-- Regions (two cpu_thread main tracks; no rocpd_sample -> is_sample=0). Row-id
-- order != start order is irrelevant here (no interval assertions).
--   tid=555 -> no thread row      -> "Thread"
--   tid=8   -> thread id=8 (NULL name, tid=99001) -> "Thread 99001"
-- cols: (id, nid, pid, tid, start, "end", name_id, event_id)
INSERT INTO "rocpd_region{{uuid}}" (id, nid, pid, tid, start, "end", name_id, event_id)
VALUES (1, 1, 1, 555, 2000, 2100, 2, NULL),
       (2, 1, 1, 8,   3000, 3100, 3, NULL);
