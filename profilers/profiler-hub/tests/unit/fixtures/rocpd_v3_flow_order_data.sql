-- =============================================================================
-- Synthetic v3 flow-ordering / tie-break fixture data (profiler-hub task 049)
-- =============================================================================
-- WHY THIS EXISTS:
--   The 043 coverage recon (gap 14/15) found two ordering branches of the flow
--   builder unlit by any committed fixture, because the existing clique fixture
--   (rocpd_v3_clique_data.sql) gives every clique two endpoints with DISTINCT
--   start timestamps and every same-source leg a DISTINCT arrow latency. Dark:
--     * reader_impl.cpp:2841-2842  get_flows() equal-start deterministic tie-break
--         (source_start == dest_start AND neither parent branch fires) -> the
--         direction is decided purely by handle order: src = key.first (the lower
--         event_id_t handle), dst = key.second.  [PRIMARY target, TESTABLE]
--     * reader_impl.cpp:3033        get_flows_in_window() decimation sort final
--         tie-break `return a.dest < b.dest`, reached only when two flows share
--         BOTH equal arrow-span latency AND equal source. The clique fixture's
--         only same-source flows (region1's three legs) have distinct latencies,
--         so line 3033 never executes there.  [TESTABLE via a natural fixture]
--
--   (reader_impl.cpp:2830-2831, the source_parent==stack_id parent-lineage branch,
--   is NOT targeted by this fixture: the clique join forces both endpoints to share
--   stack_id, so `source_parent == stack_id` reduces to "the source event is its
--   own parent" -- a self-referential (1-cycle) call-stack node that no correct
--   tracer emits. It is DEFENSIVE-only; see 049-result.md for the full reasoning.)
--
--   This file builds a tiny v3 database whose cliques are hand-chosen so each of
--   the two testable branches fires, and the tests assert the EXACT resulting flow
--   direction / decimation survivor (behavior, not merely line touches).
--
-- HOW IT IS BUILT (see tests/unit/CMakeLists.txt): identical mechanism to the edge
--   / clique fixtures -- canonical v3 schema (source/data_storage/schema/
--   rocpd_tables.sql) + this data, {{uuid}}/{{guid}} substituted, piped through
--   sqlite3. No rocpd_timestamp table -> reader selects the v3 backend.
--
-- FLOW ORACLE (what get_flows() must return; parent_stack_id is NULL throughout so
-- neither parent-lineage branch ever fires):
--   stack 1000 = { region 1 (start 5000), region 2 (start 5000) }  EQUAL START
--     -> region->region : (region 1 -> region 2)   generic, flow_id 1000
--        Equal starts -> tie-break by handle order: region 1 (row id 1) mints the
--        lower handle than region 2, so src = region 1, dst = region 2. This is the
--        ONLY thing that decides direction here (starts are identical), so it is a
--        clean 2841-2842 witness.
--   stack 2000 = { region 3 (start 6000, end 6500), kd 1 (start 6100),
--                  mc 1 (start 6200) }
--     -> region->kernel_dispatch : (region 3 -> kd 1)  launch_to_dispatch, fid 2000
--     -> region->memory_copy     : (region 3 -> mc 1)  copy_submit_to_exec, fid 2000
--        Both children start BEFORE region 3 ends (6100,6200 < 6500), so both arrow
--        latencies (dst.start - src.end) clamp to 0 -> EQUAL latency, and both share
--        source region 3 -> the get_flows_in_window decimation sort reaches line 3033
--        (a.dest < b.dest). kd handle < mc handle (event_type kernel_dispatch < memory_copy),
--        so with max_edges=1 the survivor is region 3 -> kd 1.
--   stack 3000 = { kd 2 (start 7000), kd 3 (start 7000) }  EQUAL START
--     -> kernel_dispatch sibling : (kd 2 -> kd 3)   stream_dependency, flow_id 3000
--        Second 2841-2842 witness on the same-type sibling path (src = kd 2, the
--        lower handle).
--   => get_flows() returns 4 directed edges total.
-- =============================================================================

-- Bare alias views (the v3 reader joins these by bare name; see edge fixture).
CREATE VIEW rocpd_event AS SELECT * FROM "rocpd_event{{uuid}}";
CREATE VIEW rocpd_string AS SELECT * FROM "rocpd_string{{uuid}}";
CREATE VIEW rocpd_sample AS SELECT * FROM "rocpd_sample{{uuid}}";

-- Minimal identity spine (FKs referenced by the type-table rows below).
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 730049, 'synthetic-machine-flow-order', 'Linux', 'synth-flow-order-host');
INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 4444, 'synthetic-flow-order-app');
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
VALUES (1, 1, 1, 1, 'vecAdd', 'vecAdd(int*)');

INSERT INTO "rocpd_string{{uuid}}" (id, string)
VALUES (1, 'RegionEqualStartLo'),
       (2, 'RegionEqualStartHi'),
       (3, 'RegionEnclosing'),
       (4, 'copyHtoD');

-- Events: stack_id defines the cliques; parent_stack_id left NULL throughout so the
-- parent-lineage branches (2823 / 2828) never fire and direction is decided purely by
-- the start-ts / equal-start tie-break ladder.
INSERT INTO "rocpd_event{{uuid}}" (id, stack_id)
VALUES (1, 1000),   -- region 1  (equal-start pair)
       (2, 1000),   -- region 2  (equal-start pair)
       (3, 2000),   -- region 3  (enclosing; zero-latency source)
       (4, 2000),   -- kd 1      (child of region 3)
       (5, 2000),   -- mc 1      (child of region 3)
       (6, 3000),   -- kd 2      (equal-start sibling pair)
       (7, 3000);   -- kd 3      (equal-start sibling pair)

-- Regions: 1 & 2 share start 5000 (equal-start region->region pair, stack 1000);
-- region 3 (stack 2000) encloses its two GPU children so their latencies clamp to 0.
INSERT INTO "rocpd_region{{uuid}}" (id, nid, pid, tid, start, "end", name_id, event_id)
VALUES (1, 1, 1, 1, 5000, 5100, 1, 1),
       (2, 1, 1, 1, 5000, 5100, 2, 2),
       (3, 1, 1, 1, 6000, 6500, 3, 3);

-- Kernel dispatches: kd 1 (stack 2000, starts 6100 -- inside region 3's [6000,6500]);
-- kd 2 & kd 3 (stack 3000) share start 7000 (equal-start sibling pair).
INSERT INTO "rocpd_kernel_dispatch{{uuid}}"
    (id, nid, pid, agent_id, kernel_id, dispatch_id, queue_id, stream_id,
     start, "end", workgroup_size_x, workgroup_size_y, workgroup_size_z,
     grid_size_x, grid_size_y, grid_size_z, event_id)
VALUES (1, 1, 1, 1, 1, 1, 1, 1, 6100, 6200, 64, 1, 1, 256, 1, 1, 4),
       (2, 1, 1, 1, 1, 2, 1, 1, 7000, 7100, 64, 1, 1, 256, 1, 1, 6),
       (3, 1, 1, 1, 1, 3, 1, 1, 7000, 7100, 64, 1, 1, 256, 1, 1, 7);

-- Memory copy: mc 1 (stack 2000, starts 6200 -- inside region 3's [6000,6500]).
INSERT INTO "rocpd_memory_copy{{uuid}}"
    (id, nid, pid, start, "end", name_id, size, queue_id, stream_id, event_id)
VALUES (1, 1, 1, 6200, 6300, 4, 1024, NULL, 1, 5);
