-- =============================================================================
-- Synthetic v4.0 ambiguous-pmc fixture data (profiler-hub task 014)
-- =============================================================================
-- WHY THIS EXISTS:
--   The real v4.0 fixture has no ambiguous (pmc_id, event_id) pairs. This
--   synthetic fixture creates a minimal DB with exactly one ambiguous pmc_id
--   (pmc 1: two rocpd_pmc_event rows for the same event_id, two different
--   physical quantities) and one clean pmc_id (pmc 2: one row per event_id),
--   so both branches of the ambiguity flag can be asserted on the v4 backend.
--
-- HOW IT IS BUILT (see tests/unit/CMakeLists.txt):
--   {{uuid}} / {{guid}} substituted, then fed through rocpd_v4.0_tables.sql +
--   this file via the sqlite3 CLI. The presence of rocpd_timestamp makes the
--   reader select the v4 backend.
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
--   * No rocpd_sample rows needed — ambiguity detection operates only on
--     rocpd_pmc_event, and pmc_info_statement queries only rocpd_info_pmc.
--   Expected: get_all_pmc_info() returns pmc 1 with ambiguous=true,
--             pmc 2 with ambiguous=false.
-- =============================================================================

-- Identity spine (FK note: nid=1 refs rocpd_info_node.id=1; pid=1 refs .id=1) --
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 777777, 'synthetic-machine-v4-amb-pmc', 'Linux', 'v4-amb-pmc-host');

INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 100, 'synthetic-v4-amb-pmc-app');

INSERT INTO "rocpd_info_thread{{uuid}}" (id, nid, pid, tid)
VALUES (1, 1, 1, 100);

INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 1, 'CPU', 0, 0, 'Synthetic CPU v4');

-- pmc.nid = rocpd_info_node.id = 1; pmc.pid = rocpd_info_process.id = 1
INSERT INTO "rocpd_info_pmc{{uuid}}" (id, nid, pid, agent_id, name, symbol)
VALUES (1, 1, 1, 1, 'FAULT_COUNT', 'FAULT_COUNT'),
       (2, 1, 1, 1, 'CLEAN_COUNT', 'CLEAN_COUNT');

INSERT INTO "rocpd_event{{uuid}}" (id) VALUES (1), (2);

-- Timestamp spine (v4 detection signal: presence of rocpd_timestamp).
-- One stub row is sufficient for reader backend selection; no samples reference it.
INSERT INTO "rocpd_timestamp{{uuid}}" (id, value)
VALUES (1, 1000);

-- PMC events:
--   event 1: TWO rows for pmc_id=1 (the ambiguous collision)
--   event 2: ONE row for pmc_id=2 (clean)
INSERT INTO "rocpd_pmc_event{{uuid}}" (id, event_id, pmc_id, value)
VALUES (1, 1, 1, 142942.0),    -- pmc 1, quantity A (lo band)
       (2, 1, 1, 220332032.0), -- pmc 1, quantity B (hi band) -- same event_id!
       (3, 2, 2, 999.0);       -- pmc 2, clean

INSERT INTO "rocpd_metadata{{uuid}}" (tag, value)
VALUES ('schema_version', '4.0.0');
