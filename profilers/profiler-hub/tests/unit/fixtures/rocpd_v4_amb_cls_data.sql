-- =============================================================================
-- Synthetic v4.0 ambiguous-classification fixture (profiler-hub task 018)
-- =============================================================================
-- WHY THIS EXISTS:
--   Forces the track-classification ambiguity: a single rocpd_track row (id=1)
--   appears in BOTH the counter discovery set (rocpd_sample JOIN rocpd_pmc_event)
--   AND the memory-allocate discovery set (rocpd_memory_allocate). This is the
--   schema state that build_v4_tracks() detects and logs; it tests that
--   track_info_t::ambiguous_classification is set true on the affected track.
--
-- HOW IT IS BUILT (see tests/unit/CMakeLists.txt):
--   {{uuid}} / {{guid}} substituted, then fed through rocpd_v4.0_tables.sql +
--   this file via the sqlite3 CLI. rocpd_timestamp present = v4 backend selected.
--
-- DATA SHAPE:
--   * 1 rocpd_track row (id=1) -- the ambiguous track
--   * 1 rocpd_sample row referencing track_id=1
--   * 1 rocpd_pmc_event row for the sample's event_id  (→ track 1 enters counter set)
--   * 1 rocpd_memory_allocate row referencing track_id=1  (→ track 1 enters memory set)
--   * 1 rocpd_info_pmc row (id=1, "COUNTER_X") so counter_track_names resolves it
--   Expected: get_tracks() returns exactly 1 track, type=counter
--             (counter takes precedence), ambiguous_classification=true.
-- =============================================================================

-- Identity spine --
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 888888, 'synthetic-machine-v4-amb-cls', 'Linux', 'v4-amb-cls-host');

INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 200, 'synthetic-v4-amb-cls-app');

INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 1, 'GPU', 0, 0, 'Synthetic GPU v4');

INSERT INTO "rocpd_info_pmc{{uuid}}" (id, nid, pid, agent_id, name, symbol)
VALUES (1, 1, 1, 1, 'COUNTER_X', 'COUNTER_X');

-- The ambiguous track: referenced by both rocpd_sample and rocpd_memory_allocate --
INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, agent_id)
VALUES (1, 1, 1, 1);

-- Timestamp spine (v4 detection signal) --
INSERT INTO "rocpd_timestamp{{uuid}}" (id, value) VALUES (1, 1000), (2, 2000), (3, 3000);

-- String table: name for the sample row --
INSERT INTO "rocpd_string{{uuid}}" (id, string) VALUES (1, 'counter-sample'), (2, 'memalloc-name');

-- Event row shared by sample and pmc_event --
INSERT INTO "rocpd_event{{uuid}}" (id) VALUES (1);

-- rocpd_sample: puts track_id=1 into the counter discovery set (via pmc_event join) --
INSERT INTO "rocpd_sample{{uuid}}" (id, track_id, name_id, timestamp_id, event_id)
VALUES (1, 1, 1, 1, 1);

-- rocpd_pmc_event: the pmc join that qualifies the sample as a PMC sample --
INSERT INTO "rocpd_pmc_event{{uuid}}" (id, event_id, pmc_id, value)
VALUES (1, 1, 1, 42.0);

-- rocpd_memory_allocate: puts track_id=1 into the memory discovery set --
INSERT INTO "rocpd_memory_allocate{{uuid}}" (id, track_id, type, level, start_id, end_id, name_id, size)
VALUES (1, 1, 'ALLOC', 'REAL', 2, 3, 2, 1024);

INSERT INTO "rocpd_metadata{{uuid}}" (tag, value)
VALUES ('schema_version', '4.0.0');
