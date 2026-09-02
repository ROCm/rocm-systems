-- =============================================================================
-- Synthetic v4.0 counter/scalar fixture data
-- =============================================================================
-- WHY THIS EXISTS:
--   No real captured v4.0 database available to the project contains any
--   rocpd_sample rows, so the v4.0 scalar/counter read path
--   (get_scalar_track / get_event_info + counter-track classification)
--   cannot be positively exercised by a real fixture. This file, together with
--   the canonical schema in rocpd_v4.0_tables.sql, deterministically constructs
--   a tiny v4.0 database that DOES contain counter samples, purely to close that
--   coverage gap. It is intentionally minimal and hand-reviewable.
--
-- DATA SHAPE (deliberately chosen so tests assert on real values, not "no crash"):
--   * 2 tracks:
--       track 1 = COUNTER  (referenced by rocpd_sample rows; agent_id set -> Q10
--                            agent_info populated on v4.0; PMC name = "GRBM_COUNT")
--       track 2 = CPU_THREAD (no sample rows -> not a counter; used to prove
--                             get_scalar_track() returns empty for a non-counter
--                             track, Q7 misroute-returns-empty)
--   * 3 samples on the counter track, inserted so that row-id order != timestamp
--     order, proving get_scalar_track()'s "ORDER BY timestamp" contract:
--       sample.id=1 -> ts 3000 -> value 30.5
--       sample.id=2 -> ts 1000 -> value 10.5
--       sample.id=3 -> ts 2000 -> value 20.5
--     => get_scalar_track(track 1) must return, in order:
--          (opaque_id=2, ts=1000, value=10.5)
--          (opaque_id=3, ts=2000, value=20.5)
--          (opaque_id=1, ts=3000, value=30.5)
-- =============================================================================

-- Identity spine ------------------------------------------------------------
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 111111, 'synthetic-machine-0001', 'Linux', 'synth-host');

INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 4242, 'synthetic-counter-app');

-- track 1 thread (tid=1) and track 2 thread (tid=2)
INSERT INTO "rocpd_info_thread{{uuid}}" (id, nid, pid, tid)
VALUES (1, 1, 4242, 4242),
       (2, 1, 4242, 4243);

-- GPU agent referenced by the counter track (Q10: v4 counter tracks carry agent_id).
-- type_index is required (get_all_agents, reader_impl.cpp:206 drops rows missing
-- it), else this agent -- and the counter track's agent_id -- would be unresolved.
INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 4242, 'GPU', 0, 0, 'Synthetic GPU');

-- Performance counter definition; its name is the counter track's display name (Q9).
-- PMC 99 has an intentionally empty name: the empty-name guard in reader_impl.cpp
-- (!nit->second.empty()) prevents it from overwriting rocpd_track.name -> fallback.
INSERT INTO "rocpd_info_pmc{{uuid}}" (id, nid, pid, agent_id, name, symbol)
VALUES (1,  1, 4242, 1, 'GRBM_COUNT', 'GRBM_COUNT'),
       (99, 1, 4242, 1, '',           '');           -- empty name -> display-name fallback

INSERT INTO "rocpd_string{{uuid}}" (id, string)
VALUES (1, 'GRBM_COUNT'),
       (2, 'FallbackCounterV4');  -- track 3 rocpd_track.name; pmc_id 99 absent

-- Tracks --------------------------------------------------------------------
-- track 1: COUNTER (agent_id set, referenced by samples below)
-- track 2: CPU_THREAD (tid only, never referenced by a sample)
-- track 3: COUNTER (fallback path -- see PMC 99 above)
INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, tid, agent_id)
VALUES (1, 1, 1, 1, 1);
INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, tid)
VALUES (2, 1, 1, 2);
INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, name_id)
VALUES (3, 1, 1, 2);    -- name_id=2 -> 'FallbackCounterV4'; agent_id NULL

-- Events (one per sample; scalar path joins rocpd_pmc_event on event_id) ----
INSERT INTO "rocpd_event{{uuid}}" (id) VALUES (1), (2), (3), (4);

-- Timestamp spine (row-id order deliberately != value order) ----------------
INSERT INTO "rocpd_timestamp{{uuid}}" (id, value, track_id)
VALUES (1, 3000, 1),
       (2, 1000, 1),
       (3, 2000, 1),
       (4, 500,  3);  -- track 3 fallback counter

-- PMC event values (paired with events) -------------------------------------
INSERT INTO "rocpd_pmc_event{{uuid}}" (id, event_id, pmc_id, value)
VALUES (1, 1, 1,  30.5),
       (2, 2, 1,  10.5),
       (3, 3, 1,  20.5),
       (4, 4, 99, 7.0);   -- pmc_id 99 absent from rocpd_info_pmc -> fallback

-- Counter samples on track 1 + track 3 -------------------------------------
INSERT INTO "rocpd_sample{{uuid}}" (id, track_id, name_id, timestamp_id, event_id)
VALUES (1, 1, 1, 1, 1),
       (2, 1, 1, 2, 2),
       (3, 1, 1, 3, 3),
       (4, 3, 2, 4, 4);  -- track 3: pmc_id 99 absent -> name falls back to track name

-- Schema-version metadata (self-documenting; not read by the reader) ---------
INSERT INTO "rocpd_metadata{{uuid}}" (tag, value)
VALUES ('schema_version', '4.0.0');
