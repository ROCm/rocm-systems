-- =============================================================================
-- Synthetic v3 dma-by-destination-agent fixture
-- =============================================================================
-- WHY THIS EXISTS:
--   v3 dma track identity is keyed on (nid, pid, queue_id, dst_agent_id) -- by
--   DESTINATION AGENT -- to match Optiq's shipping GetRocprofMemoryCopyTrackQuery
--   swimlane grouping (GROUP BY nid, dst_agent_id, queue_id, pid; NO stream_id).
--   stream_id is deliberately NOT part of the dma identity: stream-level grouping
--   of memory copies lives on the separate `stream` track type instead.
--
--   No in-tree fixture exercised the by-agent partition: rocpd.db has two copies
--   on distinct agents but a single stream, and rocpd_v3_edge_data.sql carries no
--   dst_agent_id at all (all NULL -> a single NULL-agent lane, which covers the
--   NULL branch). The by-destination-agent oracle -- the crux of this change --
--   was only demonstrable against roc-optiq's rocpd-transpose.db, which is not in
--   this repo's test tree. This file reproduces that DB's essential shape as a
--   self-contained, hand-reviewable fixture built from the canonical v3 schema.
--
-- CROSSED PARTITION (the oracle):
--   48 memory_copy events, all on ONE queue (queue_id = 1), fully crossing two
--   destination agents with two streams, 12 events per cell:
--       id  1..12  -> dst_agent 1, stream 1   (copyStreamX)
--       id 13..24  -> dst_agent 1, stream 2   (copyStreamY)
--       id 25..36  -> dst_agent 2, stream 1   (copyStreamX)
--       id 37..48  -> dst_agent 2, stream 2   (copyStreamY)
--   Keyed by (nid, pid, queue_id, dst_agent_id) this yields exactly 2 dma tracks:
--       agent 1 = 24 events (12 stream-X + 12 stream-Y)
--       agent 2 = 24 events (12 stream-X + 12 stream-Y)
--   Each agent-track therefore SPANS BOTH streams -- the membership test asserts a
--   track's copies include both copyStreamX and copyStreamY names, proving the
--   partition is by destination agent, NOT by stream (the old key would have given
--   2 stream tracks of 24, each spanning both agents -- the exact inverse).
--   queue_id is non-null so this exercises the queue+dst_agent ("qa") interval
--   variant, the path real Optiq captures take.
-- =============================================================================

-- Bare alias views (v3 reader joins rocpd_event/rocpd_string/rocpd_sample bare).
CREATE VIEW rocpd_event AS SELECT * FROM "rocpd_event{{uuid}}";
CREATE VIEW rocpd_string AS SELECT * FROM "rocpd_string{{uuid}}";
CREATE VIEW rocpd_sample AS SELECT * FROM "rocpd_sample{{uuid}}";

-- Identity spine ------------------------------------------------------------
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 333333, 'synthetic-machine-dma', 'Linux', 'synth-dma-host');

INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 4343, 'synthetic-dma-app');

INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 1, 'GPU', 0, 0, 'Synthetic GPU 0'),
       (2, 1, 1, 'GPU', 1, 1, 'Synthetic GPU 1');

INSERT INTO "rocpd_info_queue{{uuid}}" (id, nid, pid, name)
VALUES (1, 1, 1, 'Queue-A');

-- Two streams (crossed against both agents; NOT part of the dma identity).
INSERT INTO "rocpd_info_stream{{uuid}}" (id, nid, pid, name)
VALUES (1, 1, 1, 'Stream-X'),
       (2, 1, 1, 'Stream-Y');

-- Distinct copy names per stream so display_name reveals which stream a copy came
-- from within an agent-track (the membership assertion keys on these).
INSERT INTO "rocpd_string{{uuid}}" (id, string)
VALUES (1, 'copyStreamX'),
       (2, 'copyStreamY');

-- Memory copies (dma tracks) ------------------------------------------------
-- Generated crossed pattern (see CROSSED PARTITION above): the arithmetic below
-- assigns name_id/dst_agent_id/stream_id for ids 1-48.
WITH RECURSIVE gen(i) AS (
    SELECT 1
    UNION ALL
    SELECT i + 1 FROM gen WHERE i < 48
)
INSERT INTO "rocpd_memory_copy{{uuid}}"
    (id, nid, pid, start, "end", name_id, dst_agent_id, size, queue_id, stream_id, event_id)
SELECT
    i,
    1,
    1,
    i * 1000,
    i * 1000 + 100,
    ((i - 1) / 12) % 2 + 1,        -- name_id: stream 1->copyStreamX, 2->copyStreamY
    (i - 1) / 24 + 1,              -- dst_agent_id: 1..24 -> agent 1, 25..48 -> agent 2
    1024,
    1,                             -- queue_id (non-null -> qa variant)
    ((i - 1) / 12) % 2 + 1,        -- stream_id: crossed, NOT part of dma identity
    NULL
FROM gen;
