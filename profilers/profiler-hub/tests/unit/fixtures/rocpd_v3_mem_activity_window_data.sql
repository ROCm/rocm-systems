-- =============================================================================
-- Synthetic v3 memory_activity time-window fixture
-- =============================================================================
-- WHY THIS EXISTS:
--   Coverage recon found that the time-window `continue` filters
--   inside the memory_activity branch of get_scalar_track() (source/reader_impl.cpp
--   ~2515-2520 ALLOC, ~2548-2553 FREE) never execute: every committed test queries
--   a memory_activity scalar track WITHOUT a time_window, so all rows pass and the
--   four `continue` guards are dark.
--
--   This fixture builds a single memory_activity track (one agent) whose ALLOC and
--   FREE rows deliberately STRADDLE a chosen window [3000, 5000]. Rows sit both
--   BEFORE and AFTER the window on both event kinds, so all four filters fire:
--     * ALLOC start < window.start  -> reader_impl.cpp 2516/2517 continue
--     * ALLOC start > window.end    -> reader_impl.cpp 2519/2520 continue
--     * FREE  start < window.start  -> reader_impl.cpp 2549/2550 continue
--     * FREE  start > window.end    -> reader_impl.cpp 2552/2553 continue
--   while ALLOC/FREE/ALLOC rows INSIDE the window are emitted, letting the test
--   assert the exact in-window subset (boundary-inclusive) and running-sum values.
--
--   Kept as a SEPARATE fixture (not folded into rocpd_v3_mem_activity_data.sql)
--   because that fixture's tests assert exact track/sample counts that extra rows
--   would break. The window filter code is schema-agnostic (pure C++ over
--   mem_activity_raw_track), so covering it once on v3 lights the lines for v4 too.
--
-- DATA SHAPE (single agent 1; FREE rows carry explicit agent_id=1 + size, so the
--   running sum is deterministic without address recovery -- recovery is already
--   covered by rocpd_v3_mem_activity_data.sql):
--   rocpd_memory_allocate rows, ORDER BY start (= reader processing order):
--     r1 ALLOC start=1000 size=100  -> BEFORE window, running 0   -> 100  (skip: <start)
--     r2 FREE  start=2000 size=100  -> BEFORE window, running 100 -> 0    (skip: <start)
--     r3 ALLOC start=3000 size=500  -> IN window (== start),  running 0   -> 500  EMIT
--     r4 FREE  start=4000 size=200  -> IN window,             running 500 -> 300  EMIT
--     r5 ALLOC start=5000 size=700  -> IN window (== end),    running 300 -> 1000 EMIT
--     r6 ALLOC start=6000 size=999  -> AFTER window,          running 1000-> 1999 (skip: >end)
--     r7 FREE  start=7000 size=999  -> AFTER window,          running 1999-> 1000 (skip: >end)
--   Windowed [3000,5000] scalar series (3 samples): {3000:500, 4000:300, 5000:1000}.
--   Unwindowed scalar series (7 samples): {1000:100, 2000:0, 3000:500, 4000:300,
--     5000:1000, 6000:1999, 7000:1000}.  windowed(3) < unwindowed(7) proves the filter.
-- =============================================================================

-- Bare alias views (v3 reader joins these by bare name) ----------------------
CREATE VIEW rocpd_event AS SELECT * FROM "rocpd_event{{uuid}}";
CREATE VIEW rocpd_string AS SELECT * FROM "rocpd_string{{uuid}}";
CREATE VIEW rocpd_sample AS SELECT * FROM "rocpd_sample{{uuid}}";

-- Identity spine ------------------------------------------------------------
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 730045, 'synthetic-machine-ma-window', 'Linux', 'ma-window-host');

INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 1, 'synthetic-mem-activity-window-app');

-- Single GPU agent (carries type_index; get_all_agents drops NULL type_index).
INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 1, 'GPU', 0, 0, 'GPU-0');

-- Memory allocate rows (ORDER BY start = processing order) -------------------
-- FREE rows carry explicit agent_id=1 and size; address NULL (no self-join needed).
INSERT INTO "rocpd_memory_allocate{{uuid}}"
    (id, nid, pid, agent_id, type, level, start, "end", size, address)
VALUES
    (1, 1, 1, 1, 'ALLOC', 'REAL', 1000, 1100, 100, NULL),
    (2, 1, 1, 1, 'FREE',  'REAL', 2000, 2100, 100, NULL),
    (3, 1, 1, 1, 'ALLOC', 'REAL', 3000, 3100, 500, NULL),
    (4, 1, 1, 1, 'FREE',  'REAL', 4000, 4100, 200, NULL),
    (5, 1, 1, 1, 'ALLOC', 'REAL', 5000, 5100, 700, NULL),
    (6, 1, 1, 1, 'ALLOC', 'REAL', 6000, 6100, 999, NULL),
    (7, 1, 1, 1, 'FREE',  'REAL', 7000, 7100, 999, NULL);
