-- =============================================================================
-- Synthetic v3 track-shapes fixture data
-- =============================================================================
-- WHY THIS EXISTS:
--   The 043 coverage recon found that the per-track_type x schema switch arms in
--   get_track_stats() and get_interval_track() (source/reader_impl.cpp) are only
--   partially exercised by the committed v3 fixtures. Specifically the following
--   v3 arms were never lit by any fixture:
--     * dma  queue-only        (memory_copy queue_id set, dst_agent_id NULL)
--     * dma  agent-only        (memory_copy queue_id NULL, dst_agent_id set)
--     * dma  queue+agent STATS  (v3_dma_agent lights the interval arm but never
--                                calls get_track_stats, so the stats arm was dark)
--     * memory queue+agent     (memory_allocate agent_id set, queue_id set)
--     * memory queue-only      (memory_allocate agent_id NULL, queue_id set)
--     * memory neither         (memory_allocate agent_id NULL, queue_id NULL)
--     * cpu_thread SAMPLE      (regions whose events carry a rocpd_sample ->
--                               region_is_sample=1 -> the _sample query variant)
--   (dma "neither" and memory "agent-only" are already covered by v3_edge.)
--
--   This file builds a tiny v3 database with exactly one track of each missing
--   shape, every row hand-chosen so tests assert on KNOWN min_ts/max_ts/count and
--   the exact interval start order -- locking behavior, not just touching lines.
--
-- TRACK MATRIX (what get_tracks() returns):
--   dma (from rocpd_memory_copy, distinct nid,pid,queue_id,dst_agent_id):
--     * qa      queue_id=1, dst_agent_id=1  -> starts {1000,1200} end 1300  count 2
--     * q_only  queue_id=2, dst_agent_id=NULL -> starts {2000,2200} end 2300 count 2
--     * a_only  queue_id=NULL, dst_agent_id=2 -> start {3000} end 3100      count 1
--   memory (from rocpd_memory_allocate, distinct nid,agent_id,queue_id,pid):
--     * qa      agent_id=1, queue_id=1   -> starts {4000,4200} end 4300     count 2
--     * q_only  agent_id=NULL, queue_id=2 -> starts {5000,5200} end 5300    count 2
--     * neither agent_id=NULL, queue_id=NULL -> start {6000} end 6100       count 1
--   cpu_thread (synthesized from rocpd_region; both regions' events carry a
--     rocpd_sample so is_sample=1 -> a single SAMPLE track):
--     * sample  tid=1 -> starts {7000,7200} end 7500 count 2
--   memory_activity (synthesized separately, distinct nid,pid,agent from
--     memory_allocate, NULL agent skipped): agent 1 only. Present but not queried
--     by these tests -- harmless.
--   No kernel_dispatch / stream_id rows, so NO gpu_queue and NO stream tracks.
-- =============================================================================

-- Bare alias views (the v3 reader joins these three by bare name) ----------------
CREATE VIEW rocpd_event AS SELECT * FROM "rocpd_event{{uuid}}";
CREATE VIEW rocpd_string AS SELECT * FROM "rocpd_string{{uuid}}";
CREATE VIEW rocpd_sample AS SELECT * FROM "rocpd_sample{{uuid}}";

-- Identity spine ------------------------------------------------------------
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 730073, 'synthetic-machine-v3-shapes', 'Linux', 'synth-v3-shapes-host');

INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 4242, 'synthetic-track-shapes-app');

INSERT INTO "rocpd_info_thread{{uuid}}" (id, nid, pid, tid)
VALUES (1, 1, 1, 1001);

-- Two GPU agents (both carry type_index; get_all_agents drops NULL type_index).
INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 1, 'GPU', 0, 0, 'Synthetic GPU 0'),
       (2, 1, 1, 'GPU', 1, 1, 'Synthetic GPU 1');

INSERT INTO "rocpd_info_queue{{uuid}}" (id, nid, pid, name)
VALUES (1, 1, 1, 'Queue-A'),
       (2, 1, 1, 'Queue-B');

INSERT INTO "rocpd_string{{uuid}}" (id, string)
VALUES (1, 'copyHtoD'),
       (2, 'RegionAlpha'),
       (3, 'RegionBeta');

-- rocpd_track: only exists as the FK target for the region samples below --
-- no pmc sample (not a counter) and no rocpd_region of its own (not a
-- cpu_thread) -> correctly ignored by track discovery.
INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, tid, name_id)
VALUES (1, 1, 1, 1, NULL);

-- Events (region sample carriers; stack_id irrelevant here, no flows asserted) ---
INSERT INTO "rocpd_event{{uuid}}" (id, stack_id)
VALUES (101, NULL),
       (102, NULL);

-- Regions (cpu_thread SAMPLE track for tid=1) --------------------------------
-- Row-id order != start order (region 2 inserted before region 1), proving
-- ORDER BY start:
--   get_interval_track -> [region 1 (7000), region 2 (7200)]
--   count(DISTINCT r.id)=2, MIN(start)=7000, MAX(end)=7500.
INSERT INTO "rocpd_region{{uuid}}" (id, nid, pid, tid, start, "end", name_id, event_id)
VALUES (2, 1, 1, 1, 7200, 7300, 3, 102),
       (1, 1, 1, 1, 7000, 7500, 2, 101);

-- One sample per region event (interval DISTINCT r.id and stats COUNT(DISTINCT
-- r.id) must agree at 2); track_id references the intentionally-ignored track 1.
INSERT INTO "rocpd_sample{{uuid}}" (id, track_id, timestamp, event_id)
VALUES (1, 1, 7000, 101),
       (2, 1, 7200, 102);

-- Memory copies (dma tracks) ------------------------------------------------
INSERT INTO "rocpd_memory_copy{{uuid}}"
    (id, nid, pid, start, "end", name_id, dst_agent_id, size, queue_id, event_id)
VALUES (1, 1, 1, 1200, 1300, 1, 1,    1024, 1,    NULL),   -- dma qa
       (2, 1, 1, 1000, 1100, 1, 1,    1024, 1,    NULL),   -- dma qa (earlier start)
       (3, 1, 1, 2200, 2300, 1, NULL, 1024, 2,    NULL),   -- dma q_only
       (4, 1, 1, 2000, 2100, 1, NULL, 1024, 2,    NULL),   -- dma q_only (earlier)
       (5, 1, 1, 3000, 3100, 1, 2,    1024, NULL, NULL);   -- dma a_only

-- Memory allocations (memory tracks) ----------------------------------------
INSERT INTO "rocpd_memory_allocate{{uuid}}"
    (id, nid, pid, agent_id, type, level, start, "end", size, queue_id, event_id)
VALUES (1, 1, 1, 1,    'ALLOC', 'REAL', 4200, 4300, 4096, 1,    NULL),  -- memory qa
       (2, 1, 1, 1,    'ALLOC', 'REAL', 4000, 4100, 4096, 1,    NULL),  -- memory qa (earlier)
       (3, 1, 1, NULL, 'ALLOC', 'REAL', 5200, 5300, 2048, 2,    NULL),  -- memory q_only
       (4, 1, 1, NULL, 'ALLOC', 'REAL', 5000, 5100, 2048, 2,    NULL),  -- memory q_only (earlier)
       (5, 1, 1, NULL, 'ALLOC', 'REAL', 6000, 6100, 1024, NULL, NULL);  -- memory neither
