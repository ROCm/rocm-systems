-- =============================================================================
-- Synthetic v4.0 summary / GROUP-BY-name fixture data
-- =============================================================================
-- WHY THIS EXISTS:
--   Cross-schema companion to rocpd_v3_summary_data.sql: exercises the v4 backend
--   of get_kernel_summary / get_region_summary, where durations are computed over
--   the rocpd_timestamp spine (start_id/end_id) instead of inline start/"end".
--   Same by-construction oracle as the v3 fixture so both backends are asserted
--   against identical expected aggregates.
--
-- ORACLE (duration = ts_end.value - ts_start.value); identical to the v3 fixture.
--   Symbols 1 and 3 share display_name "kA(int)" so their GROUP BY rows merge by
--   name in fold_summary_rows (kd1,kd2 on symbol 1; kd4 on symbol 3):
--   KERNELS: kA "kA(int)" kd1=100, kd2=300, kd4=200 -> count 3, total 600, min 100, max 300, avg 200
--            kB "kB(float)" kd3=50          -> count 1, total 50,  min 50,  max 50,  avg 50
--   REGIONS: rX r1=200, r2=200 -> count 2, total 400, min 200, max 200, avg 200
--            rY r3=600         -> count 1, total 600, min 600, max 600, avg 600
--   WINDOWED kernels [1500,5000] -> kA count 2 (kd2,kd4), kB count 1.
--   WINDOWED regions [0,750]     -> rX count 1 (r1), rY absent.
--   FAR-FUTURE window            -> empty list.
-- =============================================================================

-- Identity spine.
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 740050, 'synthetic-machine-v4-summary', 'Linux', 'v4-summary-host');
INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 100, 'synthetic-v4-summary-app');
INSERT INTO "rocpd_info_thread{{uuid}}" (id, nid, pid, tid)
VALUES (1, 1, 100, 100);
INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 100, 'GPU', 0, 0, 'Synthetic GPU v4');
INSERT INTO "rocpd_info_code_object{{uuid}}" (id, nid, pid, agent_id)
VALUES (1, 1, 100, 1);

INSERT INTO "rocpd_info_kernel_symbol{{uuid}}" (id, nid, pid, code_object_id, kernel_name, display_name)
VALUES (1, 1, 100, 1, 'kA', 'kA(int)'),
       (2, 1, 100, 1, 'kB', 'kB(float)'),
       (3, 1, 100, 1, 'kA_alias', 'kA(int)');

INSERT INTO "rocpd_string{{uuid}}" (id, string)
VALUES (1, 'rX'),
       (2, 'rY');

-- One track carries every row (summary aggregates the whole table, not per-track).
INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, agent_id)
VALUES (1, 1, 1, 1);

INSERT INTO "rocpd_event{{uuid}}" (id) VALUES (1), (2), (3), (4), (5), (6), (7);

-- Timestamp spine: kernels (ids 1-6, 13-14) then regions (ids 7-12).
--   kd1 [1000,1100], kd2 [2000,2300], kd3 [3000,3050], kd4 [4000,4200],
--   r1  [500,700],   r2  [800,1000],  r3  [1000,1600].
INSERT INTO "rocpd_timestamp{{uuid}}" (id, value, track_id)
VALUES (1, 1000, 1), (2, 1100, 1),
       (3, 2000, 1), (4, 2300, 1),
       (5, 3000, 1), (6, 3050, 1),
       (7,  500, 1), (8,  700, 1),
       (9,  800, 1), (10, 1000, 1),
       (11, 1000, 1), (12, 1600, 1),
       (13, 4000, 1), (14, 4200, 1);

INSERT INTO "rocpd_region{{uuid}}" (id, track_id, name_id, start_id, end_id, event_id)
VALUES (1, 1, 1,  7,  8, 4),
       (2, 1, 1,  9, 10, 5),
       (3, 1, 2, 11, 12, 6);

INSERT INTO "rocpd_kernel_dispatch{{uuid}}"
    (id, track_id, kernel_id, dispatch_id, start_id, end_id,
     workgroup_size_x, workgroup_size_y, workgroup_size_z,
     grid_size_x, grid_size_y, grid_size_z, event_id)
VALUES (1, 1, 1, 1, 1, 2, 64, 1, 1, 256, 1, 1, 1),
       (2, 1, 1, 2, 3, 4, 64, 1, 1, 256, 1, 1, 2),
       (3, 1, 2, 3, 5, 6, 64, 1, 1, 256, 1, 1, 3),
       (4, 1, 3, 4, 13, 14, 64, 1, 1, 256, 1, 1, 7);

-- Schema-version metadata (presence of rocpd_timestamp is the real v4 signal).
INSERT INTO "rocpd_metadata{{uuid}}" (tag, value)
VALUES ('schema_version', '4.0.0');
