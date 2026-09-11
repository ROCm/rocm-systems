# Feature: PMC/Counter Event Reading Per Track

## Goal
Expose per-track PMC/counter event data (e.g. `device_busy_gfx`, `device_power`,
`device_temp`) through the profiler-hub C API. Today `rocpd_pmc_event` is never
queried at all by the reader/C API — only `pmc_info` (counter metadata) is
read. Investigation showed `rocpd_pmc_event.event_id` joins cleanly to
`rocpd_sample.event_id`, and `rocpd_sample.track_id` correctly separates
different counters into different tracks (confirmed: track 16 = only
`device_busy_gfx` values, track 346 = only `device_power`, etc. — no mixing).
The existing `pmc_events` SQL view in the DB is misleading (joins through
`rocpd_kernel_dispatch` instead, losing track_id) but the raw tables already
support correct per-track separation.

## Changelog Summary
### Added
- C API support for reading PMC/counter event values (timestamp + value) for
  a specific track, via `ph_get_pmc_events_for_track()`.

## Analysis
- Schema is fixed (can't change), but `rocpd_pmc_event -> rocpd_sample (on
  event_id) -> track_id` already gives correct per-track attribution.
- Existing `timeline_event_t`/`get_events_for_track()` infra (region/kernel
  dispatch/memory events) doesn't fit: those are named duration events
  (start/end), PMC events are instantaneous (timestamp + numeric value).
  Needs its own result/reader type, not a reuse of `timeline_event_t`.
- `pmc_info` metadata + `m_pmc_info_utility` (id -> `pmc_info_ptr_t`) already
  cached in `reader_impl` (`get_all_pmc_infos()`), reusable for name/units
  resolution.
- Out of scope (follow-up, not bundled here): filtering `ph_get_track_list()`
  to drop the ~397 tracks with zero samples. Separate concern from adding the
  read path itself.

## PR Strategy
Single PR (~200-250 lines across schema query, reader, C API, example).

## Tasks
- [ ] Add `pmc_event_result` to `source/data_storage/schema_v3/result_types.hpp`
      (id, track_id, timestamp, pmc_id, value)
- [ ] Add PMC event read statement (base + track_filtered) in
      `source/data_storage/read_statements.hpp`, joining
      `rocpd_pmc_event JOIN rocpd_sample ON sample.event_id = pmc_event.event_id`
- [ ] Add `reader_types::pmc_event_t` / `pmc_event_list_t` (timestamp, value,
      `pmc_info_ptr_t`) to `include/reader_types.hpp`
- [ ] Add `reader_t::impl::get_pmc_events_for_track()` to
      `source/reader_impl.hpp`/`.cpp`, resolving counter info via
      `m_pmc_info_utility`
- [ ] Add public `reader_t::get_pmc_events_for_track()` wrapper to
      `include/reader.hpp`
- [ ] Add `ph_pmc_event_t` / `ph_pmc_event_list_t` to
      `include/c_interface/profiler_hub_types.h`
- [ ] Add `ph_get_pmc_events_for_track()` declaration to
      `include/c_interface/profiler_hub.h`
- [ ] Add `ph_ctx::get_pmc_events_for_track(track_id)` to
      `source/profiler_hub_ctx.hpp`/`.cpp` (looks up track by id in
      `m_tracks`, builds `ph_pmc_event_t` vector owned by ctx)
- [ ] Add glue function to `source/profiler_hub_c_api.cpp`
- [ ] Demo the new call in `examples/clients/ph-c-client/main.c`
- [ ] Build project, confirm no new warnings/errors
- [ ] Update CHANGELOG.md

## Test Cases
No tests requested yet.

## Notes
- Track ownership/lifetime: `ph_pmc_event_list_t` follows the same rule as
  `ph_track_list_t` — valid until ctx freed or re-queried.

## Amendment: Track Event Count + Filtering (superseding the "deferred" note above)
User decided to do the empty-track filtering now, not defer it. New plan:

### Approach
Add a single correlated-subquery read statement returning
`(track_id, event_count)` per track directly, joining `rocpd_track` against
all 5 event-bearing tables:
```sql
SELECT T.id AS track_id,
  (SELECT COUNT(*) FROM rocpd_region_<uuid> R WHERE R.nid IS T.nid AND R.pid IS T.pid AND R.tid IS T.tid)
+ (SELECT COUNT(*) FROM rocpd_kernel_dispatch_<uuid> K WHERE K.nid IS T.nid AND K.pid IS T.pid AND K.tid IS T.tid)
+ (SELECT COUNT(*) FROM rocpd_memory_allocate_<uuid> MA WHERE MA.nid IS T.nid AND MA.pid IS T.pid AND MA.tid IS T.tid)
+ (SELECT COUNT(*) FROM rocpd_memory_copy_<uuid> MC WHERE MC.nid IS T.nid AND MC.pid IS T.pid AND MC.tid IS T.tid)
+ (SELECT COUNT(*) FROM rocpd_sample_<uuid> S WHERE S.track_id = T.id)
  AS event_count
FROM rocpd_track_<uuid> T
```
`IS` used instead of `=` for NULL-safe triple matching (many tracks have
NULL pid/tid, e.g. process-level or counter tracks).

### Tasks (new, step by step)
- [x] Add `track_event_count_result` + `track_event_count_statement()` to
      `source/data_storage/read_statements.hpp`
- [x] Add `reader_t::impl::get_track_event_counts()` to
      `source/reader_impl.hpp`/`.cpp` returning `unordered_map<size_t,size_t>`
      (track id -> event count). Also fixed pre-existing bug: `track_info_t.id`
      was never populated in `get_all_tracks()` (always 0)
- [x] Add public `reader_t::get_track_event_counts()` wrapper to
      `include/reader.hpp`
- [x] Add `event_count` field (`uint32_t`) to `ph_track_t` in
      `include/c_interface/profiler_hub_types.h`
- [x] In `ph_ctx::initialize_track_list()`, populate `event_count` per
      track and drop tracks with `event_count == 0` from both `m_tracks` and
      `m_c_tracks`
- [x] Update `examples/clients/ph-c-client/main.c` to print event_count,
      confirm track count drops (403 -> 12 tracks with real data)
- [x] Build + run against test DB to verify
- [ ] Update CHANGELOG.md

### Follow-up refinement
Moved `event_count` off the separate map into `reader_types::track_info_t`
itself (populated once inside `get_all_tracks()`); removed the now-redundant
public `reader_t::get_track_event_counts()` wrapper. `ph_ctx` now just reads
`track->event_count` directly.

### Per-device (agent) track splitting
Found via inspection that a single db `track_id` can be written to by
multiple physical devices (e.g. two GPUs both writing `device_temp`),
silently doubling `event_count` and making per-device values inseparable.
Fixed:
- New `track_agent_count_statement()`: `sample JOIN pmc_event JOIN pmc_info`
  grouped by `(track_id, agent_id)`.
- `reader_types::track_info_t` gets `agent_id` (0 = not device-specific).
- `get_all_tracks()`: for any track with >1 distinct agent, the original
  entry becomes agent #1's data only; additional agents get synthetic
  entries (new incrementing id, same real db track id registered in
  `m_track_ptr_to_db_id` for future per-track queries, same topology).
- `ph_track_t` gets `agent_id`; `ph_ctx`/`main.c` updated. Verified: DB now
  yields 14 tracks (was 12), with `device_memory_usage`/`device_temp` split
  into agent 1 (202) + agent 2 (202) entries; `device_power`/`device_busy_*`
  stay single-agent since gfx1036 never wrote those counters.

### Performance fix
Original `track_event_count_statement()` used 5 correlated subqueries
per-track-row (`EXPLAIN QUERY PLAN` confirmed: full `SCAN` per subquery, once
per track row — O(tracks * event rows)). Replaced with 5 independent
`GROUP BY nid,pid,tid` (or `track_id` for samples) queries, each a single
full-table scan, combined in C++ via `topology_key_t`/`topology_key_hash_t`
(already existed for this exact purpose elsewhere in reader_impl).
Caught and fixed a NULL-collision bug in the same change: `value_or(0)` for
missing pid/tid collapsed "no thread" and "thread id 0" into the same key,
causing every orphan track to inherit unrelated tracks' counts. Fixed with an
out-of-band sentinel (`numeric_limits<size_t>::max()`) instead of 0.

PMC-event-reading tasks from the original plan (above) remain a separate,
not-yet-started follow-up.
