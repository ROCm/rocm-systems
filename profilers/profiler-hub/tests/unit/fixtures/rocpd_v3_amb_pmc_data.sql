-- =============================================================================
-- Synthetic v3 ambiguous-pmc fixture data
-- =============================================================================
-- WHY THIS EXISTS:
--   The main v3 fixture (rocpd.db) uses pmc_id 2356 (of 2358 PMCs) as its lone
--   ambiguous case. This fixture isolates one ambiguous pmc (pmc 1: 2 rows/event_id)
--   and one clean pmc (pmc 2: 1 row/event_id) in a minimal, self-contained DB.
--   The main rocpd.db is still used for the pmc_id 2356 / count-54 assertions.
--
-- DATA SHAPE:
--   * 1 CPU agent (id=1)
--   * 2 PMC types: pmc id=1 = "FAULT_COUNT", pmc id=2 = "CLEAN_COUNT"
--   * 2 events; event 1 has 2 pmc_event rows for pmc_id=1; event 2 has 1 for pmc_id=2
--   Expected: get_all_pmc_info() returns pmc 1 with ambiguous=true,
--             pmc 2 with ambiguous=false.
-- =============================================================================

-- Views for legacy unversioned table references used in some v3 reader queries.
-- Required pattern: see rocpd_v3_mem_activity_data.sql.
CREATE VIEW rocpd_event AS SELECT * FROM "rocpd_event{{uuid}}";
CREATE VIEW rocpd_string AS SELECT * FROM "rocpd_string{{uuid}}";
CREATE VIEW rocpd_sample AS SELECT * FROM "rocpd_sample{{uuid}}";

-- Identity spine --
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 888888, 'synthetic-machine-v3-amb-pmc', 'Linux', 'v3-amb-pmc-host');

INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 100, 'synthetic-v3-amb-pmc-app');

INSERT INTO "rocpd_info_thread{{uuid}}" (id, nid, pid, tid)
VALUES (1, 1, 1, 100);

INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 1, 'CPU', 0, 0, 'Synthetic CPU v3');

INSERT INTO "rocpd_info_pmc{{uuid}}" (id, nid, pid, agent_id, name, symbol)
VALUES (1, 1, 1, 1, 'FAULT_COUNT', 'FAULT_COUNT'),
       (2, 1, 1, 1, 'CLEAN_COUNT', 'CLEAN_COUNT');

INSERT INTO "rocpd_event{{uuid}}" (id) VALUES (1), (2);

INSERT INTO "rocpd_pmc_event{{uuid}}" (id, event_id, pmc_id, value)
VALUES (1, 1, 1, 142942.0),
       (2, 1, 1, 220332032.0),
       (3, 2, 2, 999.0);
