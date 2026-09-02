-- =============================================================================
-- Synthetic v3 missing-metadata naming-fallback fixture
-- =============================================================================
-- WHY THIS EXISTS:
--   Coverage recon found the Tier-3 "missing metadata" naming
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

-- The canonical schema sets `PRAGMA foreign_keys = ON`. Region tid=555 below
-- deliberately references a thread absent from rocpd_info_thread, to light the
-- bare-"Thread" fallback (reader_impl.cpp:772). Override FK enforcement here
-- (build-time only; this PRAGMA runs after the schema's ON and wins) so that
-- dangling reference is accepted; every other reference in this file is valid.
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

INSERT INTO "rocpd_info_thread{{uuid}}" (id, nid, pid, tid, name)
VALUES (8, 1, 1, 99001, NULL);

INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 1, 'GPU', 0, 0,    'Synthetic GPU 0'),
       (2, 1, 1, 'GPU', 1, NULL, 'Synthetic GPU corrupt');

INSERT INTO "rocpd_info_stream{{uuid}}" (id, nid, pid, name)
VALUES (7, 1, 1, NULL);

INSERT INTO "rocpd_string{{uuid}}" (id, string)
VALUES (1, 'copyHtoD'),
       (2, 'RegionOnUnknownThread'),
       (3, 'RegionOnUnnamedThread');

-- Incidental: also produces an unasserted dma track (queue_id/dst_agent_id NULL).
INSERT INTO "rocpd_memory_copy{{uuid}}"
    (id, nid, pid, start, "end", name_id, dst_agent_id, size, queue_id, stream_id, event_id)
VALUES (1, 1, 1, 1000, 1100, 1, NULL, 1024, NULL, 7, NULL);

-- Regions (two cpu_thread main tracks; no rocpd_sample -> is_sample=0).
-- Row-id order != start order is irrelevant here (no interval assertions).
INSERT INTO "rocpd_region{{uuid}}" (id, nid, pid, tid, start, "end", name_id, event_id)
VALUES (1, 1, 1, 555, 2000, 2100, 2, NULL),
       (2, 1, 1, 8,   3000, 3100, 3, NULL);
