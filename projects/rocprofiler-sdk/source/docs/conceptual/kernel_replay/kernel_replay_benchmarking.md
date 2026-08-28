(kernel-replay-benchmarking)=
# Benchmarking Kernel Replay

[Performance assessment](kernel_replay_performance.md) states what kernel replay costs and why the
current regression tests cannot detect a production regression. This page is about the other half of
the problem: how to obtain a number that means something, what the benchmark suite records today,
and what it still cannot express.

## There are three ways to collect several counter groups

Comparing kernel replay against "no replay" is ambiguous, because the profiler will do something
different depending on what it is asked for, and the three possibilities do not collect the same
data. Reading a wall time without knowing which one produced it is how a configuration that
collects a quarter of the data ends up looking four times better.

| | Application runs | What each dispatch ends up with | Cost |
|---|---|---|---|
| Application replay | G | every group | G x application run |
| Multiplexed | 1 | one group, whichever was current | ~1 x application run |
| Kernel replay | 1 | every group | 1 x application run + G passes per dispatch |

**Application replay** runs the whole application once per counter group and merges the results
afterwards. It is the thing kernel replay exists to replace. Its cost is the application's own cost
multiplied by the group count, so it is expensive exactly when the application is expensive to
start, load and warm up, regardless of how much GPU work it does.

**Multiplexed collection** is what `rocprofv3` does when it is given several `--pmc` flags and
`--kernel-replay-beta-enabled` is not set. The tool counts dispatches and rotates the counter group
every `ROCPROF_COUNTER_GROUPS_INTERVAL` of them, so one run yields all the groups but no single
dispatch carries more than one. Where dispatches are homogeneous and numerous, that is a reasonable
statistical trade; where the interesting kernel runs twice, it is not a measurement of that kernel
at all. It belongs in a benchmark as a floor -- the cost of the group machinery when nothing is
replayed -- and not as the comparison.

**Kernel replay** re-executes each dispatch once per group inside one application run, restoring the
agent's tracked device memory between passes, so every dispatch carries every group. It produces the
same data as application replay, which is what makes those two, and only those two, comparable.

So the axis to normalize on is the total cost of collecting every requested group at
full per-dispatch fidelity. On that axis application replay costs G application runs and kernel
replay costs one run plus its replay overhead, and the crossover is a property of the workload:
replay wins on applications that are slow to start and cheap per dispatch, and loses on applications
with a large resident footprint or a very high dispatch count.

## What the benchmark suite records

A benchmark job is a single `rocprofv3` invocation, so application replay cannot be written as a
job: it is G runs whose costs add up. The suite handles this by measuring one of those runs and
projecting the rest.

`benchmark_config` records how a run collected its counters, in `counter_collection_mode`,
`counter_group_count` and `kernel_replay`. These are read from the `rocprofv3` command line rather
than from the configuration JSON the tool emits, because the tool does not report replay there; two
runs that differ only in whether they replayed would otherwise produce identical configuration
records, hash to the same row, and have their measurements averaged together.

The `benchmark_replay_*` views join each run to the single-group run of the same application and
multiply it by the group count, which gives the projected application replay cost alongside the
measured one:

```sql
SELECT benchmark_label, counter_collection_mode, counter_groups,
       measured, application_replay_projected
FROM benchmark_replay_wall_time;
```

The projection assumes the per-group runs are interchangeable in cost, which holds when the groups
are the same size and the application is deterministic. Where that assumption is doubtful,
`benchmark/scripts/replay_perf.py` measures application replay directly by running the application
once per group and adding the runs up. It also covers what replay costs dispatches that opt out,
which is the realistic shape for a large application with one hot kernel: the reader lock is taken
per dispatch whenever a replay service is configured, whether or not anything is replayed.

Either way the measurement is a median over repeated runs taken after untimed warmup runs. A single
timed run on shared CI hardware only detects catastrophic regressions, because the first run of an
application pays for page cache misses and code object loading.

## What it still cannot express

**The snapshot itself is not measured.** `benchmark_metrics` records host process statistics: wall
time, CPU time, RSS, page faults, context switches. The quantity the cost model is written in --
bytes moved across the host link -- is not among them, and neither is peak host staging residency.
A change that halves the bytes copied and a change that speeds up the application for an unrelated
reason are indistinguishable in the current schema. Closing this needs the SDK to report a per-run
replay summary: dispatches seen, dispatches replayed, bytes snapshotted, peak concurrent snapshot
residency, and whether the staging was host or device.

**Wall time and GPU time are not separated.** Replay drains the agent and holds an exclusive
per-agent writer lock across the whole window, which serializes work that would otherwise overlap.
That shows up in wall time identically to an increase in the work itself, so the two cannot be told
apart, and the multi-stream jobs are exactly the ones where the distinction matters most.

**Declines are not recorded.** A run in which replay declined every dispatch looks like a run with
no overhead, because that is what it is. HIP graph launches decline, so a graph-capturing
application gets no replay at all and a benchmark of it reports an encouraging number that means
nothing. `replay-hip-graph-declines` exists to make that visible, but the visibility depends on
someone knowing what the job is for; a recorded decline count and reason would not.

## Range replay

Range replay replays a range of dispatches as a unit, snapshotting once per range rather than once
per dispatch, which changes the cost model from `dispatches x footprint x passes` to
`ranges x footprint x passes`. The saving is proportional to how many dispatches fall inside a
range, and how many do is a property of the application: a range ends where something interferes
with it, and synchronization is the common case.

`kernel-replay.yaml` therefore sweeps how many dispatches the application enqueues between
synchronizations. That axis is worth measuring on its own -- it determines how much a workload loses
to the per-dispatch drain -- and it is the axis range replay changes, so kernel replay numbers along
it exist before there is anything to compare them to. Range replay also declines for a documented
set of reasons, which makes recording decline counts a prerequisite rather than an improvement: a
range replay run that declined everything and a range replay run that worked perfectly are the same
wall time.

## See also

- [Performance assessment](kernel_replay_performance.md) -- the cost model and where it breaks down
- [Test coverage](kernel_replay_testing.md) -- what is tested and what is not
- [Concurrency and isolation](kernel_replay_concurrency_and_isolation.md) -- the drain and the lock
