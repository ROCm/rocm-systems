-- =============================================================================
-- Synthetic v3 memory_activity fixture data
-- =============================================================================
-- WHY THIS EXISTS:
--   Covers get_scalar_track(memory_activity track) on the v3 backend: cumulative
--   per-agent running sum over rocpd_memory_allocate with FREE agent_id recovery
--   via address self-join, and REALLOC/RECLAIM no-op semantics.
--
-- DATA SHAPE:
--   * 2 GPU agents: agent 1 (nid=1, pid=1), agent 2 (nid=1, pid=1)
--   * 5 rocpd_memory_allocate rows:
--       row 1: ALLOC  agent=1  start=1000  size=4096  address=0x1000
--       row 2: ALLOC  agent=2  start=2000  size=8192  address=0x2000
--       row 3: FREE   agent=NULL start=3000 size=0 address=0x1000
--                     (agent_id NULL; recovered from addr_map -> agent=1, size=4096)
--       row 4: REALLOC agent=1  start=4000  size=2048  address=0x3000  (no-op)
--       row 5: ALLOC  agent=1  start=5000  size=2048  address=0x4000
--
--   Expected memory_activity series for agent 1 (3 scalar samples: ALLOC, FREE, ALLOC):
--       ts=1000: cumsum=4096   (ALLOC 4096)
--       ts=3000: cumsum=0      (FREE recovered 4096 -> 4096-4096=0)
--       ts=5000: cumsum=2048   (ALLOC 2048)
--
--   Expected memory_activity series for agent 2 (1 scalar sample):
--       ts=2000: cumsum=8192
--
-- COVERAGE:
--   * 2 memory_activity tracks discovered (agent 1, agent 2)
--   * Agent 1: running sum includes ALLOC, FREE (address-recovered), and ALLOC
--   * Agent 2: non-interference (REALLOC row and agent-1 rows don't affect sum)
--   * REALLOC is a no-op (row 4 not emitted in agent 1's scalar series)
--   * FREE with NULL agent_id recovered via address self-join
--   * Non-interference: get_scalar_track for agent 2 ignores agent 1 rows
--   * Non-interference with memory (interval) and counter tracks: none present
-- =============================================================================

-- Bare alias views ----------------------------------------------------------
CREATE VIEW rocpd_event AS SELECT * FROM "rocpd_event{{uuid}}";
CREATE VIEW rocpd_string AS SELECT * FROM "rocpd_string{{uuid}}";
CREATE VIEW rocpd_sample AS SELECT * FROM "rocpd_sample{{uuid}}";

-- Identity spine ------------------------------------------------------------
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 333333, 'synthetic-machine-ma', 'Linux', 'ma-host');

INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 1, 'synthetic-mem-activity-app');

INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 1, 'GPU', 0, 0, 'GPU-0'),
       (2, 1, 1, 'GPU', 1, 1, 'GPU-1');

-- Memory allocate rows ------------------------------------------------------
-- row 3 has agent_id=NULL to trigger FREE address self-join recovery.
INSERT INTO "rocpd_memory_allocate{{uuid}}"
    (id, nid, pid, agent_id, type, level, start, "end", size, address)
VALUES
    (1, 1, 1, 1,    'ALLOC',   'REAL',    1000, 1100, 4096, 4096),
    (2, 1, 1, 2,    'ALLOC',   'REAL',    2000, 2100, 8192, 8192),
    (3, 1, 1, NULL, 'FREE',    'REAL',    3000, 3100,    0, 4096),
    (4, 1, 1, 1,    'REALLOC', 'REAL',    4000, 4100, 2048, 12288),
    (5, 1, 1, 1,    'ALLOC',   'REAL',    5000, 5100, 2048, 16384);
