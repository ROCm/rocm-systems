-- =============================================================================
-- Synthetic v4.0 memory_activity fixture data
-- =============================================================================
-- WHY THIS EXISTS:
--   Covers get_scalar_track(memory_activity track) on the v4.0 backend. In v4,
--   agent_id comes from rocpd_track (always non-null per track), so the FREE
--   address self-join recovery path is not needed. Running sum still computed
--   in C++ over ordered rows.
--
-- DATA SHAPE (mirrors v3 fixture for cross-schema regression):
--   * 2 GPU agents: agent 1 (track_id=1), agent 2 (track_id=2)
--   * 5 rocpd_memory_allocate rows (same logical sequence as v3 fixture):
--       row 1: ALLOC  track_id=1 (agent=1)  ts=1000  size=4096
--       row 2: ALLOC  track_id=2 (agent=2)  ts=2000  size=8192
--       row 3: FREE   track_id=1 (agent=1)  ts=3000  size=4096
--       row 4: REALLOC track_id=1 (agent=1) ts=4000  size=2048  (no-op)
--       row 5: ALLOC  track_id=1 (agent=1)  ts=5000  size=2048
--
--   Expected memory_activity for agent 1:
--       ts=1000: cumsum=4096
--       ts=3000: cumsum=0      (FREE: 4096-4096=0)
--       ts=5000: cumsum=2048
--
--   Expected memory_activity for agent 2:
--       ts=2000: cumsum=8192
--
-- COVERAGE: same as v3 fixture except FREE recovery uses v4 track_id path.
--   Rows inserted out of start order to prove ORDER BY ts_s.value.
-- =============================================================================

-- Identity spine ------------------------------------------------------------
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 444444, 'synthetic-machine-v4-ma', 'Linux', 'v4-ma-host');

INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 1, 'synthetic-v4-mem-activity-app');

INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 1, 'GPU', 0, 0, 'GPU-0'),
       (2, 1, 1, 'GPU', 1, 1, 'GPU-1');

INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, agent_id)
VALUES (1, 1, 1, 1),
       (2, 1, 1, 2);

-- rocpd_string: provides name_id values (required NOT NULL in v4 memory_allocate).
INSERT INTO "rocpd_string{{uuid}}" (id, string)
VALUES (1, 'ALLOC'), (2, 'FREE'), (3, 'REALLOC');

-- Timestamp spine: 5 rows x 2 timestamps each = 10 rows.
-- Inserted out of value order to prove ORDER BY ts_s.value:
--   row 5 timestamps inserted first (ids 1,2 -> value 5000/5100)
--   row 1 timestamps inserted second (ids 3,4 -> value 1000/1100)
INSERT INTO "rocpd_timestamp{{uuid}}" (id, value, track_id)
VALUES (1,  5000, 1),  -- row 5 start
       (2,  5100, 1),  -- row 5 end
       (3,  1000, 1),  -- row 1 start
       (4,  1100, 1),  -- row 1 end
       (5,  2000, 2),  -- row 2 start
       (6,  2100, 2),  -- row 2 end
       (7,  3000, 1),  -- row 3 start
       (8,  3100, 1),  -- row 3 end
       (9,  4000, 1),  -- row 4 start
       (10, 4100, 1);  -- row 4 end

-- Memory allocate rows (inserted out of start order; ORDER BY ts_s.value enforced).
INSERT INTO "rocpd_memory_allocate{{uuid}}"
    (id, track_id, type, level, start_id, end_id, name_id, address, size)
VALUES
    (5, 1, 'ALLOC',   'REAL', 1,  2,  1, 16384, 2048),
    (1, 1, 'ALLOC',   'REAL', 3,  4,  1,  4096, 4096),
    (2, 2, 'ALLOC',   'REAL', 5,  6,  1,  8192, 8192),
    (3, 1, 'FREE',    'REAL', 7,  8,  2,  4096, 4096),
    (4, 1, 'REALLOC', 'REAL', 9, 10,  3, 12288, 2048);
