# Statistics

Rocjitsu statistics describe observations made by the simulator and measurements
of its host execution. The measurement domain and unit travel with each metric.
Functional scheduler steps are not GPU cycles, and host elapsed time is not
modeled GPU execution time.

## Configuration

Statistics are disabled by default. Add this object to a simulation configuration:

```json
{
  "statistics": {
    "enabled": true,
    "include": ["cache.*"],
    "scopes": ["component"],
    "max_dispatch_scopes": 1024,
    "output": {"format": "jsonl", "directory": "stats"}
  }
}
```

This is a configuration fragment; the usual VM and topology configuration is
still required. `include` accepts exact metric names or prefixes with a trailing
`*`; an omitted or empty list selects every available metric. Select a histogram
or summary by its base name to include all its derived scalar fields. Scopes
default to `component` and `dispatch`. The dispatch scope limit bounds output
cardinality across the session; excess scopes are counted in
`dropped_dispatch_scopes`, and completeness is false when details were dropped.

Configuration loading validates statistics fields without creating files.
Creating the VM activates one session shared by its GPUs. Output files have
unique names and are created exclusively; existing files are never truncated.
The current producer writes one final snapshot after producer teardown. A run
that never starts, is interrupted, or leaves active waves or dispatches is
marked incomplete. The generic reader also accepts multiple completed snapshots.

## Cache observations

Each CU's scalar and vector L1 caches and each XCD's L2 cache have separate
component scopes, including the GPU ordinal. The first cache producers expose:

| Metric | Meaning |
|---|---|
| `cache.read_requests`, `cache.read_bytes` | Attempted incoming read spans and bytes, including retries |
| `cache.read_size` | Histogram of attempted span sizes in bytes |
| `cache.read_hit_chunks` | Demand chunks served from resident bytes |
| `cache.read_miss_chunks` | Demand lookups without usable resident bytes |
| `cache.read_bypass_chunks` | Chunks bypassing lookup or forcing coherent refetch |
| `cache.peer_probe_unavailable` | Peer probes refused before lookup |
| `cache.backing_read_attempts`, `cache.backing_write_attempts` | Existing L2 backing transaction totals, captured after teardown |

L1 vector spans follow coalescing. Scalar dword loads classify dword chunks;
scalar byte loads and L2 classify line chunks. A request crossing a line can
contain several chunks. Admission can fail before classification, and a miss
can subsequently retry, so request totals are not the sum of chunk totals.
Stores, cache maintenance, and the memory-side cache do not currently have a
full hit/miss metric family. Cache observations describe the functional model;
they do not claim measured hardware misses or modeled cache latency.

## Scheduler and memory pipelines

The `scheduler.*` family measures functional CU scheduler steps, after wave
state updates. `scheduler.steps` counts iterations and
`scheduler.active_wave_steps` counts resident active-wave observations. Blocked
wave counters cover wait counters, barriers, GWS, export-buffer allocation,
pixel-parameter readiness, unavailable-memory retries,
ending, memory traps, debugger/runtime pauses, STATUS halt, instruction
readiness, and sleep. Each observed blocked wave receives one reason, with
debugger/runtime/STATUS pauses taking priority over its wait state, then
readiness before sleep. Selection of another runnable slot is not a blocked
reason. These counts cannot be interpreted as cycle-accurate pipeline stalls.

Each CU also exposes `memory.{scalar,global,local,tensor}.*`:

| Suffix | Meaning |
|---|---|
| `accepted` | Instructions whose ownership entered the pipeline |
| `completed`, `cancelled`, `failed` | Mutually exclusive terminal outcomes |
| `pending` | Accepted instructions without a terminal outcome |
| `unavailable_attempts` | Attempts returning transient unavailability |
| `retry_attempts` | Attempts after entering the unavailable queue |
| `retry_depth` | Instructions currently retained in that queue |

`memory.tensor.*` covers tensor DMA instructions handed to the retry queue.
Transfers that complete or fail on their first attempt never enter that pipeline
and do not contribute to these metrics; a kernel whose transfers all complete
on the first attempt can report zero tensor accepted and completed instructions.

An unavailable access discarded for replay is cancelled; a retained retry
remains pending. Deferred architectural writeback also remains pending.
Cancellation and a late callback settle ownership only once, including when a
wave slot has been reused. Final teardown counts residual owned instructions
as cancelled. Within one scope, absent overflow, `accepted` equals `completed +
cancelled + failed + pending`. Queue depth and pending are gauges, not totals.

## Library ownership

`lib/util/include/util/statistics.h` contains both owner-local accumulators and
the reusable registry API in the `stats` namespace: `stats::Session`,
`stats::Scope`, and registered counter, gauge, histogram, and summary handles.
The compiled registry and JSONL writer live in `lib/util/src/statistics.cpp`;
consumers link the `util` target. The namespace identifies the statistics
package independently of its library location. Generic callers provide scope
keys and display names, metric selection, output settings, and optional JSON
metadata. They do not need rocjitsu headers, version information, or a VM.
The offline converter accepts
the default `util.statistics.v1` schema as well as `rocjitsu.statistics.v1`,
retaining application metadata in both SQLite and CSV output.

`rocjitsu/stats/statistics.h` provides the VM adapter. It maps GPU/path identities
to utility scopes, translates rocjitsu configuration, enforces the dispatch
scope limit, and supplies run provenance and completeness. Core event
adapters define modeled events and bind to the simulator's existing event
paths. The rocjitsu adapter
preserves the `rocjitsu.statistics.v1` format consumed by the offline converter.

## Accumulators

`util/statistics.h` also supplies header-only C++20 primitives without a
dependency on the VM, ISA, plugin system, or serialization libraries:

| Type | Meaning |
|---|---|
| `stats::Counter` | Monotonic `uint64_t` total, saturating with an explicit overflow flag |
| `stats::Gauge` | Current `int64_t` value; unavailable until explicitly set |
| `stats::Summary` | Count, sum, minimum, maximum, mean, and population variance |
| `stats::Histogram` | Fixed inclusive upper bounds, disjoint bucket counts, and a summary |

These primitives require a single owner or external synchronization. A snapshot
does not itself synchronize concurrent writers. Histograms own their bounds and
validate them during setup. Recording does not allocate. Histogram merges
require identical bounds; gauges deliberately have no implicit merge operation.

Histogram buckets cover `(-infinity, bound[0]]`, `(bound[0], bound[1]]`, and so on,
with a final unbounded bucket. Bucket counts are disjoint and cumulative over
the accumulator's lifetime. Overflow and rejected nonfinite samples are
explicitly reported. An empty summary has no meaningful minimum, maximum, mean,
or variance; consult its count before using these values.

## Offline analysis

Keep the original JSONL file as the provenance artifact. The standard-library
Python converter validates its records and writes normalized tables:

```bash
python3 scripts/rocjitsu-stats.py run.jsonl --sqlite run.sqlite
python3 scripts/rocjitsu-stats.py run.jsonl --csv run-tables
```

Multiple input files can be supplied before the output option. SQLite imports
are transactional and identical reimports are idempotent. A conflicting record
with the same identity rejects the import. CSV output creates six tables and
refuses to overwrite existing output files.

| Table | Contents |
|---|---|
| `runs` | Run identity, schema, completeness, and provenance |
| `metrics` | Metric identity, name, kind, numeric type, unit, and domain |
| `scopes` | Component or dispatch identity and optional GPU identity |
| `snapshots` | Epoch and snapshot identities and observation completeness |
| `samples` | Scalar values, availability, and overflow status |
| `histogram_buckets` | Bucket indices, inclusive upper bounds, and exact counts |

The `runs.metadata_json` object retains the original records under `run` and,
when present, `session_end`. This preserves footer metadata independently of
initial run metadata, even when both records use the same application field.
Other tables retain their original record directly as `metadata_json`.
This preserves additional provenance without making a consumer guess the
meaning of a label.
Samples and buckets reference a run, epoch, snapshot, scope, and metric. Buckets
also have a bucket index. Identities are local to their run.

### Exact integers

Unsigned 64-bit values are decimal strings in JSONL and in the `exact_value`
column. SQLite's `value_integer` convenience column is populated only when the
value fits a signed 64-bit integer. Otherwise it is null and
`integer_out_of_range` is one. Histogram counts follow the same rule with
`exact_count` and `count_integer`. Both tables retain the producer's `overflow`
flag; a saturated accumulator does not describe an exact total. Floating-point
values use `value_real`.

Do not cast large counter values to floating point to make an SQL query work.
SQLite `SUM()` can overflow even if every input fits signed 64 bits. Python can
sum exact values without that restriction:

```python
import sqlite3

with sqlite3.connect('run.sqlite') as connection:
    run_id = connection.execute('SELECT run_id FROM runs').fetchone()[0]
    rows = connection.execute(
        'SELECT samples.exact_value, samples.availability, samples.overflow '
        'FROM samples JOIN metrics USING (run_id, metric_id) '
        'WHERE samples.run_id = ? AND metrics.name = ? '
        'AND samples.epoch = ? AND samples.snapshot_id = ?',
        (run_id, 'cache.backing_read_attempts', '0', '1'),
    ).fetchall()
    if not rows or any(status != 'available' or overflow for _, status, overflow in rows):
        raise ValueError('the selected observations cannot supply an exact total')
    total = sum(int(value) for value, _, _ in rows)
```

This example selects the first run in the database and its final snapshot;
choose the desired run explicitly when importing multiple runs. The join uses
the metric name because exported metric identities also include their scope.
Scope selection matters: do not
add an aggregate scope to its component scopes, or mix measurements from
different domains.

For pandas, read `exact_value` and `exact_count` as strings before converting
them to nullable `UInt64` columns. For DuckDB, explicitly cast an unsigned
counter column to `UBIGINT`; floating-point inference would lose large integer
precision. Decimal strings sort lexicographically until converted.

### Snapshots and incomplete runs

The reader retains a snapshot only after its matching `snapshot_end` record.
If a file ends with a truncated record or an unfinished snapshot, earlier
finished snapshots remain available and the run remains incomplete. A valid
`session_end` can separately report whether the run completed. An unsupported
or unavailable sample has no numeric value; it is different from zero.

Counter and histogram bucket values are cumulative within an epoch. Calculate
an interval by subtracting compatible snapshots in the same epoch. Never sum
cumulative snapshots, and never subtract across a reset epoch. Histogram
differences require the same metric, scope, and bucket boundaries.

## JSONL contract

The schema identifier is `rocjitsu.statistics.v1`. A stream starts with one
`run` record containing the schema and `run_id`. It defines `metric` and `scope`
records before using them. Each snapshot contains:

1. `snapshot_begin`, carrying `epoch`, `snapshot_id`, and `complete`.
2. `sample` and `histogram_bucket` records referring to defined scopes and metrics.
3. `snapshot_end`, carrying the matching snapshot identity.

A `session_end` record terminates the stream and carries `complete`. Repeated
`schema` and `run_id` fields on records must agree with the run. A sample may
repeat its `epoch` and `snapshot_id`, which must agree with the open snapshot.
Nested snapshots and records following `session_end` are rejected.

Metric kinds are `counter`, `gauge`, `histogram`, and `summary`; scalar numeric
types are `u64`, `i64`, and `f64`. Scalar records have `value`, optional
`availability` (`available`, `unavailable`, or `unsupported`), and optional
`overflow`. Unavailable values are null. Summary observations use separate
scalar metrics for count, sum, minimum, maximum, mean, and variance, preserving
the integer type of the count. A bucket has `bucket_index`, `count`, optional
`overflow`, and `upper_bound`; null denotes its unbounded upper endpoint.
Floating-point NaN and infinity are not valid data.

Run the consumer's independent format and SQL checks with:

```bash
python3 -B tests/stats/test_stats_converter.py -v
```
