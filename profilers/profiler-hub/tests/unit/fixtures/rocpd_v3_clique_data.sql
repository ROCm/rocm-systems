-- =============================================================================
-- Synthetic v3 flow-clique fixture data (profiler-hub task 005B-5-fix-1)
-- =============================================================================
-- WHY THIS EXISTS:
--   get_flows() now emits the FULL stack-clique: region->region and same-type
--   sibling edges (kernel_dispatch->kernel_dispatch, memory_copy->memory_copy,
--   memory_allocate->memory_allocate) in addition to the original
--   region->{kernel_dispatch, memory_copy, memory_allocate} edges. It also tags
--   every endpoint with its event_type, because the opaque ids are per-type-table
--   row ids and COLLIDE across types (a region, a kernel_dispatch, a memory_copy
--   and a memory_allocate can all carry id = 1).
--
--   The existing rocpd_v3_edge fixture is a FLAT clique (each stack has exactly
--   one region + one GPU event), so it can prove neither the new edge categories
--   nor the collision. This dedicated fixture is authored specifically as a
--   by-construction oracle for those two properties, with zero overlap with the
--   edge fixture's precise track/interval/scalar assertions.
--
-- HOW IT IS BUILT (see tests/unit/CMakeLists.txt): identical mechanism to the
--   edge fixture -- canonical v3 schema (source/data_storage/schema/rocpd_tables.sql)
--   + this data, {{uuid}}/{{guid}} substituted, piped through sqlite3. No
--   rocpd_timestamp table -> reader selects the v3 backend.
--
-- FLOW ORACLE (what get_flows() must return, keyed on stack_id cliques):
--   stack 1000 = { region 1, kd 1, mc 1, ma 1 }  (one of each type)
--     -> region->kernel_dispatch : (region 1 -> kd 1)
--     -> region->memory_copy     : (region 1 -> mc 1)
--     -> region->memory_allocate : (region 1 -> ma 1)
--   stack 2000 = { region 2, region 3 }
--     -> region->region          : (2 -> 3), (3 -> 2)
--   stack 3000 = { kd 2, kd 3 }
--     -> kernel_dispatch sibling : (2 -> 3), (3 -> 2)
--   stack 4000 = { mc 2, mc 3 }
--     -> memory_copy sibling     : (2 -> 3), (3 -> 2)
--   stack 5000 = { ma 2, ma 3 }
--     -> memory_allocate sibling : (2 -> 3), (3 -> 2)
--   stack 0    = { region 4 }    -> excluded (stack_id == 0)
--   => 11 flows total; by category rkd=1, rmc=1, rma=1, rr=2, kdkd=2, mcmc=2,
--      mama=2. Endpoint ids collide (region 1 / kd 1 / mc 1 / ma 1 all appear),
--      so type tags are the ONLY disambiguator -- the point of the fix.
-- =============================================================================

-- Bare alias views (the v3 reader joins these by bare name; see edge fixture).
CREATE VIEW rocpd_event AS SELECT * FROM "rocpd_event{{uuid}}";
CREATE VIEW rocpd_string AS SELECT * FROM "rocpd_string{{uuid}}";
CREATE VIEW rocpd_sample AS SELECT * FROM "rocpd_sample{{uuid}}";

-- Minimal identity spine (FKs referenced by the type-table rows below).
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 333333, 'synthetic-machine-clique', 'Linux', 'synth-clique-host');
INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 4343, 'synthetic-clique-app');
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
VALUES (1, 'Region1'),
       (2, 'Region2'),
       (3, 'Region3'),
       (4, 'Region4'),
       (5, 'copyHtoD');

-- Events: stack_id defines the cliques (see oracle above). One stack-0 region
-- is the negative guard (excluded from every flow set).
INSERT INTO "rocpd_event{{uuid}}" (id, stack_id)
VALUES (1, 1000),   -- region 1
       (2, 1000),   -- kd 1
       (3, 1000),   -- mc 1
       (4, 1000),   -- ma 1
       (5, 2000),   -- region 2
       (6, 2000),   -- region 3
       (7, 3000),   -- kd 2
       (8, 3000),   -- kd 3
       (9, 4000),   -- mc 2
       (10, 4000),  -- mc 3
       (11, 5000),  -- ma 2
       (12, 5000),  -- ma 3
       (13, 0);     -- region 4 (stack 0 -> excluded)

-- Regions: id 1 (stack1000), 2 & 3 (stack2000), 4 (stack0, excluded).
INSERT INTO "rocpd_region{{uuid}}" (id, nid, pid, tid, start, "end", name_id, event_id)
VALUES (1, 1, 1, 1, 1000, 1100, 1, 1),
       (2, 1, 1, 1, 2000, 2100, 2, 5),
       (3, 1, 1, 1, 2050, 2150, 3, 6),
       (4, 1, 1, 1, 9000, 9100, 4, 13);

-- Kernel dispatches: id 1 (stack1000), 2 & 3 (stack3000 siblings).
INSERT INTO "rocpd_kernel_dispatch{{uuid}}"
    (id, nid, pid, agent_id, kernel_id, dispatch_id, queue_id, stream_id,
     start, "end", workgroup_size_x, workgroup_size_y, workgroup_size_z,
     grid_size_x, grid_size_y, grid_size_z, event_id)
VALUES (1, 1, 1, 1, 1, 1, 1, 1, 1200, 1300, 64, 1, 1, 256, 1, 1, 2),
       (2, 1, 1, 1, 1, 2, 1, 1, 3000, 3100, 64, 1, 1, 256, 1, 1, 7),
       (3, 1, 1, 1, 1, 3, 1, 1, 3050, 3150, 64, 1, 1, 256, 1, 1, 8);

-- Memory copies: id 1 (stack1000), 2 & 3 (stack4000 siblings).
INSERT INTO "rocpd_memory_copy{{uuid}}"
    (id, nid, pid, start, "end", name_id, size, queue_id, stream_id, event_id)
VALUES (1, 1, 1, 1400, 1500, 5, 1024, NULL, 1, 3),
       (2, 1, 1, 4000, 4100, 5, 2048, NULL, 1, 9),
       (3, 1, 1, 4050, 4150, 5,  512, NULL, 1, 10);

-- Memory allocates: id 1 (stack1000), 2 & 3 (stack5000 siblings).
INSERT INTO "rocpd_memory_allocate{{uuid}}"
    (id, nid, pid, agent_id, type, level, start, "end", size, stream_id, event_id)
VALUES (1, 1, 1, 1, 'ALLOC', 'REAL', 1600, 1700, 4096, 1, 4),
       (2, 1, 1, 1, 'ALLOC', 'REAL', 5000, 5100, 8192, 1, 11),
       (3, 1, 1, 1, 'ALLOC', 'REAL', 5050, 5150, 2048, 1, 12);
