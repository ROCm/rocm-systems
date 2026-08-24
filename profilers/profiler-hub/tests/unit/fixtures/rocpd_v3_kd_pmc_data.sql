-- =============================================================================
-- Synthetic v3 kernel_dispatch_pmc fixture data
-- =============================================================================
-- WHY THIS EXISTS:
--   The two real-capture v3 fixtures (rocpd.db, rocpd_v3_edge.db) contain
--   rocpd_pmc_event rows only as scalar (sample) PMC backing -- they are never
--   joined to a rocpd_kernel_dispatch row. The kernel_dispatch_pmc track type
--   requires rocpd_pmc_event.event_id == rocpd_kernel_dispatch.event_id, which
--   neither real capture provides. This synthetic fixture covers that gap.
--
-- DATA SHAPE (all values chosen so tests assert on real numbers):
--   * 1 GPU agent (id=1, nid=1, pid=100, agent_id=1)
--   * 2 PMC types:
--       pmc 1 = "SQ_WAVES"  (nid=1, pid=100, agent_id=1)
--       pmc 2 = "GRBM_COUNT" (nid=1, pid=100, agent_id=1)
--   * 1 kernel symbol: vecAdd (display_name = "vecAdd(float*, int)")
--   * 3 kernel_dispatch rows, each with a pmc_event row:
--       kd 1: event_id=1, pmc_id=1 (SQ_WAVES),   start=1000, end=1200
--       kd 2: event_id=2, pmc_id=1 (SQ_WAVES),   start=2000, end=2300
--       kd 3: event_id=3, pmc_id=2 (GRBM_COUNT), start=3000, end=3100
--   * 2 kernel_dispatch_pmc tracks synthesized:
--       track A: (nid=1, agent_id=1, pmc_id=1, pid=100) -> kd 1, kd 2 (2 events)
--       track B: (nid=1, agent_id=1, pmc_id=2, pid=100) -> kd 3 (1 event)
--   * ORDER BY start proof: tracks returned by get_interval_track in ascending
--     start order even if row-id order differs (kd 2 inserted before kd 1).
-- =============================================================================

-- Bare alias views used by the v3 reader (see other v3 fixtures) ------------
CREATE VIEW rocpd_event AS SELECT * FROM "rocpd_event{{uuid}}";
CREATE VIEW rocpd_string AS SELECT * FROM "rocpd_string{{uuid}}";
CREATE VIEW rocpd_sample AS SELECT * FROM "rocpd_sample{{uuid}}";

-- Identity spine ------------------------------------------------------------
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 222222, 'synthetic-machine-kd-pmc', 'Linux', 'kd-pmc-host');

INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 100, 'synthetic-kd-pmc-app');

INSERT INTO "rocpd_info_thread{{uuid}}" (id, nid, pid, tid)
VALUES (1, 1, 1, 100);

INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 1, 'GPU', 0, 0, 'Synthetic GPU');

INSERT INTO "rocpd_info_pmc{{uuid}}" (id, nid, pid, agent_id, name, symbol)
VALUES (1, 1, 1, 1, 'SQ_WAVES',   'SQ_WAVES'),
       (2, 1, 1, 1, 'GRBM_COUNT', 'GRBM_COUNT');

INSERT INTO "rocpd_info_code_object{{uuid}}" (id, nid, pid, agent_id)
VALUES (1, 1, 1, 1);

INSERT INTO "rocpd_info_kernel_symbol{{uuid}}" (id, nid, pid, code_object_id, kernel_name, display_name)
VALUES (1, 1, 1, 1, 'vecAdd', 'vecAdd(float*, int)');

-- Queue and stream required by v3 rocpd_kernel_dispatch NOT NULL FKs
INSERT INTO "rocpd_info_queue{{uuid}}" (id, nid, pid, name)
VALUES (1, 1, 1, 'Queue-0');

INSERT INTO "rocpd_info_stream{{uuid}}" (id, nid, pid, name)
VALUES (1, 1, 1, 'Stream-0');

-- v3 track table: only used for counter classification; kernel_dispatch_pmc
-- tracks are synthesized from rocpd_pmc_event JOIN rocpd_kernel_dispatch.
-- No rocpd_track rows needed for this fixture.

-- Events (event_id links pmc_event -> kernel_dispatch; row-id != start order)
-- kd 2 gets event 1, kd 1 gets event 2, kd 3 gets event 3 -- different order
-- to prove ORDER BY start is enforced and not ORDER BY event_id or row_id.
INSERT INTO "rocpd_event{{uuid}}" (id) VALUES (1), (2), (3);

-- Inserted out of start order to prove ORDER BY start:
--   get_interval_track(pmc_1 track) -> [kd_pmc 1 (start=1000), kd_pmc 2 (start=2000)]
INSERT INTO "rocpd_kernel_dispatch{{uuid}}"
    (id, nid, pid, agent_id, kernel_id, dispatch_id, queue_id, stream_id,
     start, "end", workgroup_size_x, workgroup_size_y, workgroup_size_z,
     grid_size_x, grid_size_y, grid_size_z, event_id)
VALUES (2, 1, 1, 1, 1, 2, 1, 1, 2000, 2300, 64, 1, 1, 1024, 1, 1, 1),
       (1, 1, 1, 1, 1, 1, 1, 1, 1000, 1200, 64, 1, 1,  512, 1, 1, 2),
       (3, 1, 1, 1, 1, 3, 1, 1, 3000, 3100, 64, 1, 1,  256, 1, 1, 3);

INSERT INTO "rocpd_pmc_event{{uuid}}" (id, event_id, pmc_id, value)
VALUES (1, 2, 1, 12345.0),
       (2, 1, 1, 23456.0),
       (3, 3, 2, 9999.0);
