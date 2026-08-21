-- =============================================================================
-- Synthetic v3 bracket-in-name counter fixture data
-- Coverage cell A3 x delimiter-in-name.
-- =============================================================================
-- WHY THIS EXISTS:
--   ranked_pmc_resolver strips the agent ordinal " [N]" from a v3 counter track's
--   name_id string by cutting at the FIRST " [":
--       instr(str.string, ' [') > 0
--         ? substr(str.string, 1, instr(str.string, ' [') - 1)
--         : str.string
--   That is correct only when the metric base name contains no " [" of its own
--   (e.g. "device_busy_gfx [0]" -> "device_busy_gfx"). When the metric name itself
--   contains " [" (here "TCC_HIT [sum]"), the strip cuts at the WRONG bracket:
--   "TCC_HIT [sum] [0]" -> "TCC_HIT", which matches NEITHER pmc name, so the
--   name-match rank collapses and every co-sampled pmc ties at rank-key 1. This is
--   the LATENT degradation: the name key no longer selects the
--   pmc; only the deterministic pmc_id tiebreaker (ORDER BY ... , pe.pmc_id) does.
--
--   This fixture is crafted so the CORRECT pmc (TCC_HIT [sum]) has the LOWER pmc_id
--   (id=1) than its co-sampled sibling (TCC_MISS [sum], id=2). The tiebreaker
--   therefore still resolves the track to the right pmc -- the degradation is
--   non-fatal on this shape. The paired test asserts that GREEN result and documents
--   that the outcome now rides on pmc_id ordering, not the (defeated) name match: had
--   the correct pmc carried the higher id, the resolver would mis-rank.
--
-- HOW IT IS BUILT (see tests/unit/CMakeLists.txt):
--   {{uuid}} / {{guid}} substituted against the canonical v3 schema, then fed
--   through the sqlite3 CLI (same mechanism as the v3 amb-pmc fixture). No
--   rocpd_timestamp table is inserted, so the reader selects the v3 backend.
--
-- DATA SHAPE:
--   nid/pid columns in child tables are FK row-ids, NOT raw nid/pid values.
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

-- Identity spine (FK note: nid=1 refs rocpd_info_node.id=1; pid=1 refs .id=1) --
INSERT INTO "rocpd_info_node{{uuid}}" (id, hash, machine_id, system_name, hostname)
VALUES (1, 777777, 'synthetic-machine-v3-bracket', 'Linux', 'v3-bracket-host');

INSERT INTO "rocpd_info_process{{uuid}}" (id, nid, pid, command)
VALUES (1, 1, 100, 'synthetic-v3-bracket-app');

INSERT INTO "rocpd_info_thread{{uuid}}" (id, nid, pid, tid)
VALUES (1, 1, 1, 100);

-- GPU agent with type_index 0, matching the " [0]" ordinal in the track name.
INSERT INTO "rocpd_info_agent{{uuid}}" (id, nid, pid, type, absolute_index, type_index, name)
VALUES (1, 1, 1, 'GPU', 0, 0, 'Synthetic GPU v3 bracket');

-- Both metric base names contain " [" -- the exact shape that defeats the ordinal
-- strip. The correct pmc (TCC_HIT [sum]) is id=1 (lower) so the pmc_id tiebreaker
-- still lands on it.
INSERT INTO "rocpd_info_pmc{{uuid}}" (id, nid, pid, agent_id, name, symbol)
VALUES (1, 1, 1, 1, 'TCC_HIT [sum]', 'TCC_HIT'),
       (2, 1, 1, 1, 'TCC_MISS [sum]', 'TCC_MISS');

-- The track's name_id string: metric base "TCC_HIT [sum]" + agent ordinal " [0]".
INSERT INTO "rocpd_string{{uuid}}" (id, string) VALUES (500, 'TCC_HIT [sum] [0]');

-- One counter track pointing at that name string.
INSERT INTO "rocpd_track{{uuid}}" (id, nid, pid, name_id) VALUES (10, 1, 1, 500);

-- One event co-sampling both pmcs under a single sample (the AMD-SMI poll fan-out).
INSERT INTO "rocpd_event{{uuid}}" (id) VALUES (1);

-- One sample on the track referencing that event -> distinct_sample_track_ids
-- classifies track 10 as a counter track (a sample joins a pmc_event on event_id).
INSERT INTO "rocpd_sample{{uuid}}" (id, track_id, timestamp, event_id)
VALUES (1, 10, 1000, 1);

-- Fan-out: the single event carries a pmc_event for BOTH pmcs, forcing the resolver
-- to choose between them for track 10.
INSERT INTO "rocpd_pmc_event{{uuid}}" (id, event_id, pmc_id, value)
VALUES (1, 1, 1, 42.0),   -- TCC_HIT [sum]  (the track's own metric)
       (2, 1, 2, 7.0);    -- TCC_MISS [sum] (co-sampled sibling)
