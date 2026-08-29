# Kernel replay in the rocpd schema

Kernel replay executes one dispatch once per counter group. Every one of those executions is a real
dispatch with its own timing and its own counter readings, and every one of them reports the same
`dispatch_id` — that is what makes them passes of one dispatch rather than separate dispatches. The
rocpd export was written before replay existed and treats `dispatch_id` as the identity of an
execution, so it merged things that are not the same execution.

This page records what went wrong, the fix, and the alternatives that were considered and dropped.

## What went wrong

The export kept a single map from `dispatch_id` to the rocpd event row representing that dispatch.
Two separate failures followed from that one assumption.

**Dispatch rows were dropped.** `process_dispatch()` returned early whenever it saw a `dispatch_id`
it had already recorded, so only the first pass produced a `rocpd_kernel_dispatch` row. The
timestamps of every later pass went with it. The drop was silent: the duplicate-dispatch warning sat
after the early return and could never be reached.

**Counter rows were merged and then summed.** The counter path had no duplicate check at all, so
every pass's `rocpd_pmc_event` rows were written — but they all resolved to the first pass's
`event_id`, because that was the only event the map had. Nothing on the row recorded which pass
produced it. `counters_collection` then aggregated with `SUM(PMC_E.value)` grouped by `dispatch_id`,
so a counter collected by more than one pass was summed across all of them.

The second failure is the more serious one. Dropping dispatch rows loses data, which is visible to
anyone who counts them. Summing counters across passes produces a number that is wrong but looks
ordinary. It bites hardest on the counters people are most likely to trust, because the usual way to
write counter groups repeats a sanity counter in each group:

```bash
rocprofv3 --pmc SQ_WAVES,SQ_INSTS_VALU,GRBM_COUNT \
          --pmc SQ_WAVES,SQ_INSTS_VALU,GRBM_GUI_ACTIVE \
          --pmc SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_SALU \
          --kernel-replay-beta-enabled -- ./app
```

`SQ_WAVES` is collected three times, once per pass, each time reporting the same 512 waves. Grouped
by `dispatch_id` that reads as 1536. Counters appearing in only one group were never affected, which
is why the corruption is uneven and hard to spot.

## The fix

A dispatch execution is identified by `dispatch_id` together with the replay pass that produced it.
The event map is keyed on that pair, so each pass keeps its own event and its own
`rocpd_kernel_dispatch` row, and the counters of a pass attach to the execution that produced them
through the `event_id` foreign key that was already there.

Schema 3.0.4 adds one column:

```sql
-- rocpd_kernel_dispatch
"replay_pass" INTEGER,   -- NULL when the run did not use kernel replay
```

and changes `counters_collection` to aggregate per event instead of per `dispatch_id`:

```sql
GROUP BY
    PMC_E.guid,
    E.id,        -- was K.dispatch_id
    PMC_I.name,
    K.agent_id;
```

Outside kernel replay there is exactly one event per `dispatch_id`, so this grouping returns exactly
what grouping on `dispatch_id` returned. That equivalence is the point of the design, and it is
asserted directly in the tests by building the same non-replay data under 3.0.3 and 3.0.4 and
comparing the view output row for row.

Because several rows can now share a `dispatch_id`, `dispatch_id` can no longer double as the
`rocpd_kernel_dispatch` primary key. Replay runs allocate a surrogate row id; runs without replay
keep `id == dispatch_id` exactly as before.

## Why this direction

The obvious alternative, and the one suggested when the problem was first raised, is a new table
keyed on `dispatch_id` holding the replay records, following the `rocpd_sample` table that SPM added
in 3.0.3. Three things argued against it.

**`rocpd_sample` earns its keep and a replay table would not.** An SPM sample carries a payload of
its own — a timestamp and a track — that has nowhere else to live, and the sample genuinely is a
distinct entity from the dispatch. A replay pass carries one integer. A table whose only content is
a pass index adds a join to every counter query and normalizes nothing.

**The 1:N relationship is not counter-to-pass, it is dispatch-to-execution.** `rocpd_kernel_dispatch`
is already the per-execution table. Replay does not need a new place to record executions; it needs
the export to stop collapsing the rows the SDK is already emitting. Once each execution has its own
row, the pass index is an attribute of that row, and it belongs there rather than repeated on every
counter row beneath it.

**Putting the pass on `rocpd_pmc_event` would fix the symptom and leave the cause.** A nullable
`replay_pass` on the counter table would stop the summing, but the dispatch rows would still be
dropped and the per-pass timestamps would still be lost. It also denormalizes: the pass would be
repeated once per counter rather than once per execution, on the largest table in the database.

The view change is where this design pays off most. SPM keeps `counters_collection` backward
compatible with `WHERE PMC_E.sample_id IS NULL`, which works because SPM rows are additional — 
filtering them out leaves the pre-SPM result. That trick does not transfer to replay, where every
counter row belongs to some pass and the equivalent filter would empty the view. Grouping on the
event instead of the `dispatch_id` gives backward compatibility without a filter at all, because the
thing being grouped on is already 1:1 with `dispatch_id` in every database that does not contain
replay data.

## What is still open

**The pass index on the kernel-trace path is derived, not reported.** `tool_counter_record_t` carries
`replay_pass`, so the counter path uses the value the SDK computed. The kernel dispatch buffer
records have no pass field — `rocprofiler_kernel_dispatch_info_t` has `reserved_padding[56]`
available for one, but adding it is an SDK change rather than an output-layer change. Until then,
when kernel tracing is enabled alongside replay, the export takes the n-th record seen for a
`dispatch_id` to be its n-th pass. This holds because replay executes the passes of a dispatch in
sequence and the buffer preserves completion order, but it is an inference rather than a reported
value, and it is the part of this change most worth replacing with an explicit field.

**What the view should report for a replay run is a product decision, not a technical one.** The
grouping above returns one row per pass, which is the only answer that does not discard or invent
data. It does mean a tool that queries `counters_collection` for a replayed run gets more rows than
it would for a non-replayed one, without necessarily knowing why. Whether the default view should
instead collapse to one row per dispatch under some defined rule, and leave the per-pass detail to a
separate view, is a question about what the schema promises its consumers.

**Coverage still runs against the schema rather than against hardware.** The tests here build the
shipped SQL in SQLite and drive it with constructed rows, which is enough to prove the grouping
behavior and the backward compatibility claim without a GPU. What they do not prove is that a real
replayed run populates `replay_pass` correctly end to end. That needs an integration test on the
rocprofv3 side, where replay is actually reachable from the command line.
