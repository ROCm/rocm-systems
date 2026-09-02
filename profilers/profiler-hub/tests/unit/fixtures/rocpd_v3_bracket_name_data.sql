-- =============================================================================
-- Synthetic v3 bracket-in-name counter fixture data
-- Coverage cell A3 x delimiter-in-name.
-- =============================================================================
-- WHY THIS EXISTS:
--   ranked_pmc_resolver strips the agent ordinal " [N]" from a counter track's
--   name_id string by cutting at the FIRST " [". This is wrong when the metric
--   base name itself contains " [" (here "TCC_HIT [sum]"): the strip cuts at the
--   wrong bracket ("TCC_HIT [sum] [0]" -> "TCC_HIT"), matching neither pmc name,
--   so the name-match rank collapses and co-sampled pmcs tie at rank-key 1 --
--   only the pmc_id tiebreaker (ORDER BY ..., pe.pmc_id) then decides.
--
--   This fixture gives the CORRECT pmc (TCC_HIT [sum]) the LOWER pmc_id (1) than
--   its co-sampled sibling (TCC_MISS [sum], id=2), so the tiebreaker still lands
--   on the right pmc here; had the ids been reversed, the resolver would mis-rank.
--
-- DATA SHAPE:
--   * 1 GPU agent (id=1, type_index=0) -- matches the " [0]" ordinal in the name
--   * 2 PMC types, both with " [" in the base name:
--       pmc id=1 = "TCC_HIT [sum]"   (the track's own metric; LOWER id)
--       pmc id=2 = "TCC_MISS [sum]"  (a co-sampled sibling; higher id)
--   * 1 counter track (id=10), name_id -> string "TCC_HIT [sum] [0]"
--   * 1 event (id=1) co-sampling BOTH pmcs (the AMD-SMI fan-out)
--   * 1 sample on the track referencing that event -> track classified as counter
--   Expected: the track resolves to pmc id=1 "TCC_HIT [sum]" via the pmc_id
--             tiebreaker, and its Q9 display name is "TCC_HIT [sum]".
-- =============================================================================

-- Views for legacy unversioned table references used in some v3 reader queries.
CREATE VIEW rocpd_event AS SELECT * FROM "rocpd_event{{uuid}}";
CREATE VIEW rocpd_string AS SELECT * FROM "rocpd_string{{uuid}}";
CREATE VIEW rocpd_sample AS SELECT * FROM "rocpd_sample{{uuid}}";

-- Identity spine --
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 777777, 'synthetic-machine-v3-bracket', 'Linux', 'v3-bracket-host');

INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 100, 'synthetic-v3-bracket-app');

INSERT INTO "rocpd_info_thread{{uuid}}" (id, nid, pid, tid)
VALUES (1, 1, 1, 100);

-- GPU agent with type_index 0, matching the " [0]" ordinal in the track name.
INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 1, 'GPU', 0, 0, 'Synthetic GPU v3 bracket');

INSERT INTO "rocpd_info_pmc{{uuid}}" (id, nid, pid, agent_id, name, symbol)
VALUES (1, 1, 1, 1, 'TCC_HIT [sum]', 'TCC_HIT'),
       (2, 1, 1, 1, 'TCC_MISS [sum]', 'TCC_MISS');

INSERT INTO "rocpd_string{{uuid}}" (id, string) VALUES (500, 'TCC_HIT [sum] [0]');

INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, name_id) VALUES (10, 1, 1, 500);

INSERT INTO "rocpd_event{{uuid}}" (id) VALUES (1);

INSERT INTO "rocpd_sample{{uuid}}" (id, track_id, timestamp, event_id)
VALUES (1, 10, 1000, 1);

INSERT INTO "rocpd_pmc_event{{uuid}}" (id, event_id, pmc_id, value)
VALUES (1, 1, 1, 42.0),
       (2, 1, 2, 7.0);
