-- =============================================================================
-- Synthetic v3 summary / GROUP-BY-name fixture data
-- =============================================================================
-- WHY THIS EXISTS:
--   get_kernel_summary / get_region_summary aggregate events GROUP BY name into
--   one event_summary_t per distinct name {count, total/avg/min/max duration}.
--   This fixture has TWO distinct kernel names and TWO distinct region names with
--   hand-chosen, varied durations so each aggregate is a by-construction oracle,
--   and it is built so a windowed call drops a subset of rows (proving the filter).
--
-- ORACLE (duration = "end" - start):
--   KERNELS (kernel_id -> kernel_symbol display_name). TWO distinct kernel_symbol
--   ids (1 and 3) share display_name "kA(int)" so GROUP BY kernel_id yields two
--   rows that MUST be merged by name in C++ (exercises fold_summary_rows' merge):
--     kA "kA(int)"   : kd1[1000,1100]=100, kd2[2000,2300]=300 (symbol 1)
--                    + kd4[4000,4200]=200 (symbol 3, display "kA(int)")
--         -> count 3, total 600, min 100, max 300, avg 200
--     kB "kB(float)" : kd3[3000,3050]=50
--         -> count 1, total 50,  min 50,  max 50,  avg 50
--   REGIONS (name_id -> rocpd_string):
--     rX : r1[500,700]=200, r2[800,1000]=200
--         -> count 2, total 400, min 200, max 200, avg 200
--     rY : r3[1000,1600]=600
--         -> count 1, total 600, min 600, max 600, avg 600
--   WINDOWED kernels [1500,5000] (keep start<=5000 AND "end">=1500):
--     kd1 ends 1100 < 1500 -> dropped; kA -> count 2 (kd2+kd4), kB -> count 1.
--   WINDOWED regions [0,750] (keep start<=750 AND "end">=0):
--     r2 starts 800 > 750 -> dropped, r3 starts 1000 > 750 -> dropped;
--     rX -> count 1 (r1 only), rY absent.
--   FAR-FUTURE window [1000000000, 1000000001]: no rows match -> empty list.
-- =============================================================================

-- Bare alias views (the v3 reader joins these by bare name).
CREATE VIEW rocpd_event AS SELECT * FROM "rocpd_event{{uuid}}";
CREATE VIEW rocpd_string AS SELECT * FROM "rocpd_string{{uuid}}";
CREATE VIEW rocpd_sample AS SELECT * FROM "rocpd_sample{{uuid}}";

-- Minimal identity spine.
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 730050, 'synthetic-machine-summary', 'Linux', 'synth-summary-host');
INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 5555, 'synthetic-summary-app');
INSERT INTO "rocpd_info_thread{{uuid}}" (id, nid, pid, tid)
VALUES (1, 1, 1, 1001);
INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 1, 'GPU', 0, 0, 'Synthetic GPU 0');
INSERT INTO "rocpd_info_queue{{uuid}}" (id, nid, pid, name)
VALUES (1, 1, 1, 'Queue-A');
INSERT INTO "rocpd_info_stream{{uuid}}" (id, nid, pid, name)
VALUES (1, 1, 1, 'Stream-X');
INSERT INTO "rocpd_info_code_object{{uuid}}" (id, nid, pid, agent_id)
VALUES (1, 1, 1, 1);

INSERT INTO "rocpd_info_kernel_symbol{{uuid}}" (id, nid, pid, code_object_id, kernel_name, display_name)
VALUES (1, 1, 1, 1, 'kA', 'kA(int)'),
       (2, 1, 1, 1, 'kB', 'kB(float)'),
       (3, 1, 1, 1, 'kA_alias', 'kA(int)');

INSERT INTO "rocpd_string{{uuid}}" (id, string)
VALUES (1, 'rX'),
       (2, 'rY');

INSERT INTO "rocpd_event{{uuid}}" (id)
VALUES (1), (2), (3), (4), (5), (6), (7);

INSERT INTO "rocpd_region{{uuid}}" (id, nid, pid, tid, start, "end", name_id, event_id)
VALUES (1, 1, 1, 1,  500,  700, 1, 1),
       (2, 1, 1, 1,  800, 1000, 1, 2),
       (3, 1, 1, 1, 1000, 1600, 2, 3);

INSERT INTO "rocpd_kernel_dispatch{{uuid}}"
    (id, nid, pid, agent_id, kernel_id, dispatch_id, queue_id, stream_id,
     start, "end", workgroup_size_x, workgroup_size_y, workgroup_size_z,
     grid_size_x, grid_size_y, grid_size_z, event_id)
VALUES (1, 1, 1, 1, 1, 1, 1, 1, 1000, 1100, 64, 1, 1, 256, 1, 1, 4),
       (2, 1, 1, 1, 1, 2, 1, 1, 2000, 2300, 64, 1, 1, 256, 1, 1, 5),
       (3, 1, 1, 1, 2, 3, 1, 1, 3000, 3050, 64, 1, 1, 256, 1, 1, 6),
       (4, 1, 1, 1, 3, 4, 1, 1, 4000, 4200, 64, 1, 1, 256, 1, 1, 7);
