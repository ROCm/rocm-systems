-- =============================================================================
-- Synthetic v4.0 ambiguous-pmc fixture data
-- =============================================================================
-- WHY THIS EXISTS:
--   The real v4.0 fixture has no ambiguous (pmc_id, event_id) pairs. This
--   fixture creates a minimal DB with exactly one ambiguous pmc_id (pmc 1: two
--   rocpd_pmc_event rows for the same event_id) and one clean pmc_id (pmc 2:
--   one row per event_id), so both branches of the ambiguity flag can be
--   asserted on the v4 backend.
--
-- DATA SHAPE:
--   * 1 CPU agent (id=1)
--   * 2 PMC types: pmc id=1 = "FAULT_COUNT", pmc id=2 = "CLEAN_COUNT"
--   * 2 events; event 1 has 2 pmc_event rows for pmc_id=1; event 2 has 1 for pmc_id=2
--   * No rocpd_sample rows needed — ambiguity detection operates only on
--     rocpd_pmc_event, and pmc_info_statement queries only rocpd_info_pmc.
--   Expected: get_all_pmc_info() returns pmc 1 with ambiguous=true,
--             pmc 2 with ambiguous=false.
-- =============================================================================

-- Identity spine --
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 777777, 'synthetic-machine-v4-amb-pmc', 'Linux', 'v4-amb-pmc-host');

INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 100, 'synthetic-v4-amb-pmc-app');

INSERT INTO "rocpd_info_thread{{uuid}}" (id, nid, pid, tid)
VALUES (1, 1, 1, 100);

INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 1, 'CPU', 0, 0, 'Synthetic CPU v4');

INSERT INTO "rocpd_info_pmc{{uuid}}" (id, nid, pid, agent_id, name, symbol)
VALUES (1, 1, 1, 1, 'FAULT_COUNT', 'FAULT_COUNT'),
       (2, 1, 1, 1, 'CLEAN_COUNT', 'CLEAN_COUNT');

INSERT INTO "rocpd_event{{uuid}}" (id) VALUES (1), (2);

-- Timestamp spine (v4 detection signal: presence of rocpd_timestamp).
-- One stub row is sufficient for reader backend selection; no samples reference it.
INSERT INTO "rocpd_timestamp{{uuid}}" (id, value)
VALUES (1, 1000);

INSERT INTO "rocpd_pmc_event{{uuid}}" (id, event_id, pmc_id, value)
VALUES (1, 1, 1, 142942.0),
       (2, 1, 1, 220332032.0),
       (3, 2, 2, 999.0);

INSERT INTO "rocpd_metadata{{uuid}}" (tag, value)
VALUES ('schema_version', '4.0.0');
