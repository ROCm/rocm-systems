-- =============================================================================
-- Synthetic v4.0 kernel_dispatch_pmc fixture data
-- =============================================================================
-- WHY THIS EXISTS:
--   The real v4.0 fixture (rocpd_v4.db) has no rocpd_pmc_event rows joined to
--   rocpd_kernel_dispatch. This synthetic fixture covers the v4 backend for the
--   kernel_dispatch_pmc track type.
--
-- DATA SHAPE (mirrors the v3 kd_pmc fixture for cross-schema regression):
--   * 1 GPU agent (id=1, nid=1, pid=100)
--   * 2 PMC types: pmc 1 = "SQ_WAVES", pmc 2 = "GRBM_COUNT"
--   * 1 kernel symbol: vecAdd(float*, int)
--   * 1 rocpd_track (track_id=1, nid=1, pid=100, agent_id=1) for gpu_queue use
--   * 3 kernel_dispatch rows via track_id=1 + timestamp spine (2 ids each):
--       kd 1: ts=1000..1200, pmc_id=1
--       kd 2: ts=2000..2300, pmc_id=1
--       kd 3: ts=3000..3100, pmc_id=2
--   * 2 kernel_dispatch_pmc tracks synthesized by distinct_kd_pmc_tracks():
--       track A: (nid=1, pid=100, agent_id=1, pmc_id=1) -> kd 1, kd 2 (2 events)
--       track B: (nid=1, pid=100, agent_id=1, pmc_id=2) -> kd 3 (1 event)
-- =============================================================================

-- Identity spine ------------------------------------------------------------
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 333333, 'synthetic-machine-v4-kd-pmc', 'Linux', 'v4-kd-pmc-host');

INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 100, 'synthetic-v4-kd-pmc-app');

INSERT INTO "rocpd_info_thread{{uuid}}" (id, nid, pid, tid)
VALUES (1, 1, 100, 100);

INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 100, 'GPU', 0, 0, 'Synthetic GPU v4');

INSERT INTO "rocpd_info_pmc{{uuid}}" (id, nid, pid, agent_id, name, symbol)
VALUES (1, 1, 100, 1, 'SQ_WAVES',   'SQ_WAVES'),
       (2, 1, 100, 1, 'GRBM_COUNT', 'GRBM_COUNT');

INSERT INTO "rocpd_info_code_object{{uuid}}" (id, nid, pid, agent_id)
VALUES (1, 1, 100, 1);

INSERT INTO "rocpd_info_kernel_symbol{{uuid}}" (id, nid, pid, code_object_id, kernel_name, display_name)
VALUES (1, 1, 100, 1, 'vecAdd', 'vecAdd(float*, int)');

-- agent_id=1 so distinct_kd_pmc_tracks' JOIN rocpd_track carries the GPU agent.
INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, agent_id)
VALUES (1, 1, 1, 1);

INSERT INTO "rocpd_event{{uuid}}" (id) VALUES (1), (2), (3);

-- Timestamp spine: 3 dispatches x 2 timestamps each = 6 rows
-- Inserted out of value order to prove ORDER BY ts_s.value:
--   kd 2 timestamps inserted first (ids 1,2) but have values 2000/2300
--   kd 1 timestamps inserted second (ids 3,4) but have values 1000/1200
INSERT INTO "rocpd_timestamp{{uuid}}" (id, value, track_id)
VALUES (1, 2000, 1),   -- kd 2 start
       (2, 2300, 1),   -- kd 2 end
       (3, 1000, 1),   -- kd 1 start
       (4, 1200, 1),   -- kd 1 end
       (5, 3000, 1),   -- kd 3 start
       (6, 3100, 1);   -- kd 3 end

-- kernel_dispatch rows (inserted out of start order to prove ORDER BY ts_s.value):
--   kd 2 first (ts=2000), kd 1 second (ts=1000), kd 3 last (ts=3000)
INSERT INTO "rocpd_kernel_dispatch{{uuid}}"
    (id, track_id, kernel_id, dispatch_id, start_id, end_id,
     workgroup_size_x, workgroup_size_y, workgroup_size_z,
     grid_size_x, grid_size_y, grid_size_z, event_id)
VALUES (2, 1, 1, 2, 1, 2, 64, 1, 1, 1024, 1, 1, 1),
       (1, 1, 1, 1, 3, 4, 64, 1, 1,  512, 1, 1, 2),
       (3, 1, 1, 3, 5, 6, 64, 1, 1,  256, 1, 1, 3);

INSERT INTO "rocpd_pmc_event{{uuid}}" (id, event_id, pmc_id, value)
VALUES (1, 2, 1, 12345.0),
       (2, 1, 1, 23456.0),
       (3, 3, 2, 9999.0);

-- Schema-version metadata (not read by the reader; presence of rocpd_timestamp
-- is the actual v4 detection signal)
INSERT INTO "rocpd_metadata{{uuid}}" (tag, value)
VALUES ('schema_version', '4.0.0');
