-- =============================================================================
-- Synthetic v3 edge-matrix fixture data
-- =============================================================================
-- WHY THIS EXISTS:
--   The bundled real-capture v3 fixture (tests/unit/rocpd.db) is a trivial
--   bit_extract workload: it exercises the common shapes but leaves the edges of
--   the track-scoped read API (get_interval_track / get_scalar_track / get_flows
--   + the track_info_t identity fields) untested. In particular it has NO counter
--   track that carries a thread id, only ONE gpu queue, only ONE dma lane, and no
--   deliberately out-of-order timestamps. A real blob also cannot be a
--   by-construction oracle: you cannot assert an exact value you did not author.
--
-- TRACK MATRIX (what get_tracks() must return):
--   COUNTER tracks come from rocpd_track (a track is a counter iff a PMC-backed
--   rocpd_sample references it -- i.e. a sample row whose event_id joins
--   rocpd_pmc_event; reader_impl.cpp v3 classification). Non-counter rocpd_track
--   rows are NOT tracks: cpu_thread/region tracks are synthesized from rocpd_region
--   instead (v3 rocpd_track is an unreliable grab-bag). So tracks 1, 4, 5 below are
--   deliberately IGNORED by discovery, proving the skip path.
--   From rocpd_track rows -> counters (4):
--     track 2 = counter, pid set, tid NULL    -> thread_info NULL, agent_info NULL
--     track 3 = counter, pid + tid set        -> thread_info SET  (the edge #147
--                                                guards: a v3 counter CAN carry a
--                                                tid; agent_info still NULL in v3)
--     track 6 = counter, pid NULL, tid NULL   -> process_info NULL, thread_info NULL
--                                                (re-homes the NULL-pid/NULL-tid
--                                                schema-branch coverage that used to
--                                                live on a cpu_thread rocpd_track row)
--     track 8 = counter, pmc_id 99 has empty name in rocpd_info_pmc -> the empty-name
--                                                guard (!nit->second.empty()) in
--                                                reader_impl.cpp prevents overwriting
--                                                rocpd_track.name -> fallback fires (F7)
--   From rocpd_track rows -> IGNORED (not returned):
--     track 1 (pid+tid), track 4 (pid, no tid), track 5 (no pid) -- non-counter,
--       no sample ref.
--     track 7 (pid, no tid) -- HAS a rocpd_sample (sample 7) but NO rocpd_pmc_event,
--       so it is a non-PMC sample track: NOT a counter, and (no rocpd_region row) not
--       a cpu_thread track either. Regression guard for the pmc_event join in
--       distinct_sample_track_ids() -- a bare "DISTINCT track_id FROM rocpd_sample"
--       would over-include it as an empty counter (rocpd-transpose.db 21-vs-18 bug).
--   Synthesized from rocpd_region:
--     1 cpu_thread track (the sole (nid,pid,tid)=(1,1,1) thread; all regions are
--       "main" -- none of their events carry a sample -- so a single main track).
--   Synthesized (not rocpd_track rows):
--     2 gpu_queue tracks  (distinct nid,pid,agent_id,queue_id in kernel_dispatch)
--     2 dma tracks        (distinct nid,pid,queue_id,stream_id in memory_copy)
--     2 stream tracks     (distinct nid,pid,stream_id across kernel_dispatch +
--                          memory_copy + memory_allocate; ADDITIVE to the queue/dma
--                          tracks -- the same events also appear there):
--       stream 1 (1,1,1): 3 kernel_dispatch + 2 memory_copy + 1 memory_allocate = 6
--       stream 2 (1,1,2): 0 kernel_dispatch + 1 memory_copy + 0 memory_allocate = 1
--   => by type: cpu_thread=1, counter=4, gpu_queue=2, dma=2, stream=2.
-- =============================================================================

-- Bare alias views (the v3 reader joins these three by bare name) ----------
CREATE VIEW rocpd_event AS SELECT * FROM "rocpd_event{{uuid}}";
CREATE VIEW rocpd_string AS SELECT * FROM "rocpd_string{{uuid}}";
CREATE VIEW rocpd_sample AS SELECT * FROM "rocpd_sample{{uuid}}";

-- Identity spine ------------------------------------------------------------
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 222222, 'synthetic-machine-v3', 'Linux', 'synth-v3-host');

-- rocpd_info_process below carries the OS pid (4242) as data; child rows'
-- `pid` columns are FKs to this row's id (1), not the OS pid.
INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 4242, 'synthetic-edge-app');

-- Thread row id=1 (OS tid 1001 is plain data here); rocpd_track.tid /
-- rocpd_region.tid FK to this row's id (1), not the OS tid.
INSERT INTO "rocpd_info_thread{{uuid}}" (id, nid, pid, tid)
VALUES (1, 1, 1, 1001);

-- GPU agent carries type_index (get_all_agents drops NULL type_index, reader_impl.cpp:206).
INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 1, 'GPU', 0, 0, 'Synthetic GPU 0');

-- Q9: gpu_queue display name = queue identity (rocpd_info_queue.name).
INSERT INTO "rocpd_info_queue{{uuid}}" (id, nid, pid, name)
VALUES (1, 1, 1, 'Queue-A'),
       (2, 1, 1, 'Queue-B');

INSERT INTO "rocpd_info_stream{{uuid}}" (id, nid, pid, name)
VALUES (1, 1, 1, 'Stream-X'),
       (2, 1, 1, 'Stream-Y');

-- Q9: three distinct PMCs give the three counter tracks different display names.
INSERT INTO "rocpd_info_pmc{{uuid}}" (id, nid, pid, agent_id, name, symbol)
VALUES (1,  1, 1, 1, 'GRBM_COUNT', 'GRBM_COUNT'),
       (2,  1, 1, 1, 'SQ_WAVES',   'SQ_WAVES'),
       (3,  1, 1, 1, 'CPU_CYCLES', 'CPU_CYCLES'),
       (99, 1, 1, 1, '',           '');           -- empty name -> display-name fallback

INSERT INTO "rocpd_info_code_object{{uuid}}" (id, nid, pid, agent_id)
VALUES (1, 1, 1, 1);

INSERT INTO "rocpd_info_kernel_symbol{{uuid}}" (id, nid, pid, code_object_id, kernel_name, display_name)
VALUES (1, 1, 1, 1, 'vecAdd', 'vecAdd(int*)');

-- Strings (rocpd_track / rocpd_region / rocpd_memory_copy display names) --------
INSERT INTO "rocpd_string{{uuid}}" (id, string)
VALUES (1, 'CPU Thread 1001'),
       (2, 'RegionAlpha'),
       (3, 'RegionBeta'),
       (4, 'RegionGamma'),
       (5, 'RegionDelta'),
       (6, 'copyHtoD'),
       (7, 'FallbackCounter');  -- track 8 rocpd_track.name; pmc_id 99 absent from rocpd_info_pmc

-- Tracks (id, nid, pid, tid, name_id) -- counter classification is by sample
-- reference, NOT by any column here. See TRACK MATRIX above for per-id detail.
INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, tid, name_id)
VALUES (1, 1, 1,    1,    1);
INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, tid, name_id)
VALUES (2, 1, 1,    NULL, NULL);
INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, tid, name_id)
VALUES (3, 1, 1,    1,    NULL);
INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, tid, name_id)
VALUES (4, 1, 1,    NULL, 1);
INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, tid, name_id)
VALUES (5, 1, NULL, NULL, NULL);
INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, tid, name_id)
VALUES (6, 1, NULL, NULL, NULL);
INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, tid, name_id)
VALUES (7, 1, 1,    NULL, NULL);
INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, tid, name_id)
VALUES (8, 1, 1,    NULL, 7);

-- Events --------------------------------------------------------------------
-- Flows key on stack_id (Q4): a region flows to a GPU-side event sharing the
-- same non-zero stack_id. stack_id = 0 / NULL is excluded from flows.
--   ev1 stack 100 = RegionAlpha  <-> ev4 stack 100 = kernel_dispatch (flow)
--   ev2 stack 200 = RegionBeta   <-> ev5 stack 200 = memory_copy     (flow)
--   ev6 stack 400 = RegionDelta  <-> ev7 stack 400 = memory_allocate (flow)
--   ev3 stack 0   = RegionGamma  -> excluded (stack_id == 0)
--   ev8..ev12     = sample events, stack_id NULL (never in a flow)
INSERT INTO "rocpd_event{{uuid}}" (id, stack_id)
VALUES (1, 100),
       (2, 200),
       (3, 0),
       (4, 100),
       (5, 200),
       (6, 400),
       (7, 400),
       (8, NULL),
       (9, NULL),
       (10, NULL),
       (11, NULL),
       (12, NULL),
       (13, NULL),   -- sample event for the NULL-pid counter (track 6)
       (14, NULL),   -- sample event for the non-pmc sample track (track 7); no pmc_event
       (15, NULL);   -- sample event for the fallback counter (track 8); pmc_id 99 absent

-- Regions (cpu_thread interval track for track 1, tid=1) ----------------------
-- Row-id order deliberately != start order so ORDER BY start is proven:
--   get_interval_track(track 1) must return, by ascending start:
--     region 2 (start 1000, RegionAlpha, outer)   level 0
--     region 3 (start 2000, RegionBeta,  nested)  level 1
--     region 1 (start 3000, RegionGamma, nested)  level 1
--     region 4 (start 6000, RegionDelta, separate)level 0
INSERT INTO "rocpd_region{{uuid}}" (id, nid, pid, tid, start, "end", name_id, event_id)
VALUES (1, 1, 1, 1, 3000, 3500, 4, 3),   -- RegionGamma (stack 0 -> no flow)
       (2, 1, 1, 1, 1000, 5000, 2, 1),   -- RegionAlpha outer (stack 100 -> kd flow)
       (3, 1, 1, 1, 2000, 2500, 3, 2),   -- RegionBeta nested (stack 200 -> mc flow)
       (4, 1, 1, 1, 6000, 6500, 5, 6);   -- RegionDelta (stack 400 -> ma flow)

-- Kernel dispatches (gpu_queue tracks) --------------------------------------
-- Two distinct (nid,pid,agent_id,queue_id) -> 2 gpu_queue tracks.
-- On queue 1, two dispatches with start out of row order prove ORDER BY start:
--   get_interval_track(gpu_queue Queue-A) -> [kd 3 (1200), kd 1 (1600)]
-- kd 2 is the sole dispatch on queue 2 (a second gpu_queue track).
-- Only kd 1 carries an event (stack 100) so only it participates in a flow.
INSERT INTO "rocpd_kernel_dispatch{{uuid}}"
    (id, nid, pid, agent_id, kernel_id, dispatch_id, queue_id, stream_id,
     start, "end", workgroup_size_x, workgroup_size_y, workgroup_size_z,
     grid_size_x, grid_size_y, grid_size_z, event_id)
VALUES (1, 1, 1, 1, 1, 1, 1, 1, 1600, 1700, 64, 1, 1, 256, 1, 1, 4),
       (2, 1, 1, 1, 1, 2, 2, 1, 1400, 1500, 64, 1, 1, 256, 1, 1, NULL),
       (3, 1, 1, 1, 1, 3, 1, 1, 1200, 1300, 64, 1, 1, 256, 1, 1, NULL);

-- Memory copies (dma tracks) ------------------------------------------------
-- Two distinct (nid,pid,queue_id,stream_id); queue_id NULL is a distinct group
-- value (Q2), so both go through the "queue IS NULL AND stream = ?" branch:
--   (1,1,NULL,1) and (1,1,NULL,2) -> 2 dma tracks.
-- On stream 1, two copies out of row order prove ORDER BY start:
--   get_interval_track(dma Stream-X) -> [mc 3 (2100), mc 1 (2200)]
-- Only mc 1 carries an event (stack 200) so only it participates in a flow.
INSERT INTO "rocpd_memory_copy{{uuid}}"
    (id, nid, pid, start, "end", name_id, size, queue_id, stream_id, event_id)
VALUES (1, 1, 1, 2200, 2300, 6, 1024, NULL, 1, 5),
       (2, 1, 1, 2400, 2500, 6, 2048, NULL, 2, NULL),
       (3, 1, 1, 2100, 2150, 6,  512, NULL, 1, NULL);

-- Memory allocate (flow target + stream-track member; not its own track type) --
-- stream_id = 1 makes this the sole memory_allocate contribution to a stream track,
-- exercising the third UNION leg of the stream aggregation (op_kind memory_allocate).
-- No real capture available to the project has a memory_allocate row with a stream, so
-- this synthetic row is the only coverage of that leg.
INSERT INTO "rocpd_memory_allocate{{uuid}}"
    (id, nid, pid, agent_id, type, level, start, "end", size, stream_id, event_id)
VALUES (1, 1, 1, 1, 'ALLOC', 'REAL', 6100, 6200, 4096, 1, 7);

-- Arguments (rocpd_arg) keyed on the shared rocpd_event row --------------------
-- Args attach to rocpd_event.id (rocpd_arg.event_id -> rocpd_event.id), the same
-- row kernel_dispatch/memory_copy/memory_allocate carry via their event_id column.
-- The bundled bit_extract capture (rocpd.db) only has args on region events, so
-- this is the sole coverage that get_event_info folds args for the other three
-- detail types. Keyed by:
--   event 4 -> kernel_dispatch id 1 (2 args)
--   event 5 -> memory_copy     id 1 (1 arg)
--   event 7 -> memory_allocate id 1 (1 arg)
INSERT INTO "rocpd_arg{{uuid}}" (id, event_id, position, type, name, value)
VALUES (1, 4, 0, 'const char*', 'kernel_name', 'vecAdd'),
       (2, 4, 1, 'unsigned int', 'grid', '256'),
       (3, 5, 0, 'size_t', 'bytes', '1024'),
       (4, 7, 0, 'size_t', 'alloc_bytes', '4096');

-- Counter samples + PMC values ----------------------------------------------
-- Track 2 (counter, no tid) -- pmc 1 GRBM_COUNT. Row-id order != timestamp
-- order proves get_scalar_track()'s "ORDER BY timestamp":
--   sample 1 -> ts 3000 -> value 30.5
--   sample 2 -> ts 1000 -> value 10.5
--   sample 3 -> ts 2000 -> value 20.5
--   => get_scalar_track(track 2) = [(2,1000,10.5),(3,2000,20.5),(1,3000,30.5)]
-- Track 3 (counter, WITH tid) -- pmc 2 SQ_WAVES:
--   sample 4 -> ts 500  -> value 5.0
--   sample 5 -> ts 1500 -> value 15.0
--   => get_scalar_track(track 3) = [(4,500,5.0),(5,1500,15.0)]
INSERT INTO "rocpd_pmc_event{{uuid}}" (id, event_id, pmc_id, value)
VALUES (1, 8,  1,  30.5),
       (2, 9,  1,  10.5),
       (3, 10, 1,  20.5),
       (4, 11, 2,  5.0),
       (5, 12, 2,  15.0),
       (6, 13, 3,  42.0),
       (7, 15, 99, 1.0);   -- pmc_id 99 has empty name -> display-name fallback fires

-- Track 7 (sample 7, event 14) is deliberately NOT pmc-backed: there is no
-- rocpd_pmc_event row for event 14, so this sample track must NOT classify as a
-- counter (regression guard for the distinct_sample_track_ids() pmc_event join).
INSERT INTO "rocpd_sample{{uuid}}" (id, track_id, timestamp, event_id)
VALUES (1, 2, 3000, 8),
       (2, 2, 1000, 9),
       (3, 2, 2000, 10),
       (4, 3, 500,  11),
       (5, 3, 1500, 12),
       (6, 6, 700,  13),
       (7, 7, 800,  14),
       (8, 8, 900,  15);   -- track 8: pmc_id 99 has empty name -> name falls back to track name
