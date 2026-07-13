-- =============================================================================
-- Synthetic v3 ambiguous-pmc fixture data (profiler-hub task 014)
-- =============================================================================
-- WHY THIS EXISTS:
--   The main v3 fixture (rocpd.db) has 2358 PMCs and pmc_id 2356 as the lone
--   ambiguous case. This synthetic fixture is a minimal, self-contained v3 DB
--   with exactly one ambiguous pmc_id (pmc 1: 2 rocpd_pmc_event rows per event_id)
--   and one clean pmc_id (pmc 2: 1 row per event_id), so the ambiguity detection
--   can be verified against clean value counts independently of the large fixture.
--   The main rocpd.db is still used for the pmc_id 2356 / count-54 assertions.
--
-- HOW IT IS BUILT (see tests/unit/CMakeLists.txt):
--   {{uuid}} / {{guid}} substituted against the canonical v3 schema, then fed
--   through the sqlite3 CLI (same mechanism as v3 edge/clique/kd_pmc fixtures).
--   No rocpd_timestamp table is inserted, so the reader selects the v3 backend.
--
-- DATA SHAPE:
--   nid/pid columns in child tables are FK row-ids, NOT the raw nid/pid values:
--     rocpd_info_node.id=1  (nid)
--     rocpd_info_process.id=1 (pid FK), rocpd_info_process.pid=100 (actual pid)
--   * 1 CPU agent (id=1)
--   * 2 PMC types:
--       pmc id=1 = "FAULT_COUNT" -- AMBIGUOUS: 2 rocpd_pmc_event rows per event_id
--       pmc id=2 = "CLEAN_COUNT" -- CLEAN: 1 rocpd_pmc_event row per event_id
--   * 2 events; event 1 has 2 pmc_event rows for pmc_id=1; event 2 has 1 for pmc_id=2
--   Expected: get_all_pmc_info() returns pmc 1 with ambiguous=true,
--             pmc 2 with ambiguous=false.
-- =============================================================================

-- Views for legacy unversioned table references used in some v3 reader queries.
-- Required pattern: see rocpd_v3_mem_activity_data.sql.
CREATE VIEW rocpd_event AS SELECT * FROM "rocpd_event{{uuid}}";
CREATE VIEW rocpd_string AS SELECT * FROM "rocpd_string{{uuid}}";
CREATE VIEW rocpd_sample AS SELECT * FROM "rocpd_sample{{uuid}}";

-- Identity spine (FK note: nid=1 refs rocpd_info_node.id=1; pid=1 refs .id=1) --
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 888888, 'synthetic-machine-v3-amb-pmc', 'Linux', 'v3-amb-pmc-host');

INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 100, 'synthetic-v3-amb-pmc-app');

INSERT INTO "rocpd_info_thread{{uuid}}" (id, nid, pid, tid)
VALUES (1, 1, 1, 100);

INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 1, 'CPU', 0, 0, 'Synthetic CPU v3');

-- pmc.nid = rocpd_info_node.id = 1; pmc.pid = rocpd_info_process.id = 1
INSERT INTO "rocpd_info_pmc{{uuid}}" (id, nid, pid, agent_id, name, symbol)
VALUES (1, 1, 1, 1, 'FAULT_COUNT', 'FAULT_COUNT'),
       (2, 1, 1, 1, 'CLEAN_COUNT', 'CLEAN_COUNT');

INSERT INTO "rocpd_event{{uuid}}" (id) VALUES (1), (2);

-- PMC events:
--   event 1: TWO rows for pmc_id=1 (the ambiguous collision)
--   event 2: ONE row for pmc_id=2 (clean)
INSERT INTO "rocpd_pmc_event{{uuid}}" (id, event_id, pmc_id, value)
VALUES (1, 1, 1, 142942.0),    -- pmc 1, quantity A
       (2, 1, 1, 220332032.0), -- pmc 1, quantity B (same event_id!)
       (3, 2, 2, 999.0);       -- pmc 2, clean
