# A simple execution-thread allocation heuristic

A roughly 1:2:1 split between XCD engines, shared dispatch workers and async
helpers is a useful default for the measured concurrent-dispatch workloads.
Count **dispatch workers**, rather than the inclusive `cpu_dispatch_threads`
setting: E=n, D=2n+1, H=n uses exactly 4n execution threads.

At 32 threads this gives **8/17/8**, which is close to the target-specific
selections. Prefer engine counts that divide the target's XCD count, and retain
the four-thread gfx1250 choice 1/4/0. With these adjustments, the observed median
within-round dispatch penalty is at most about 3% across the tested budgets and
two workloads. These measurements now inform explicit allocation tables in the target
configs. The study below evaluated only these two workloads.

## Direct comparison at 31 and 32 threads

Four fresh processes per target/allocation, interleaved within each round. Times
are medians; percentage changes in this table compare those medians. The selected
controls are 8/13/12 on gfx950 and 8/21/4 on gfx1250. An 8/16/8 allocation uses
31 threads; 8/17/8 uses 32, since B = E + D - 1 + H.

| Target | E/D/H | Threads | Dispatch s | Wall s | Dispatch change vs selected 32-thread allocation | Wall change |
|---|---:|---:|---:|---:|---:|---:|
| gfx1250 | 8/16/8 | 31 | 0.6095 | 2.810 | +2.3% | +0.2% |
| gfx1250 | 8/17/8 | 32 | 0.6127 | 2.820 | +2.9% | +0.5% |
| gfx1250 | 8/21/4 | 32 | 0.5956 | 2.805 | +0.0% | +0.0% |
| gfx950 | 8/13/12 | 32 | 0.3999 | 2.365 | +0.0% | +0.0% |
| gfx950 | 8/16/8 | 31 | 0.4111 | 2.375 | +2.8% | +0.4% |
| gfx950 | 8/17/8 | 32 | 0.3967 | 2.375 | -0.8% | +0.4% |

## Recommended rule for the eight-XCD targets

For a budget B from 1 to 64:

1. Start with E approximately B/4. Round up to one of 1, 2, 4 or 8, capped at
   eight. These counts divide the eight XCDs evenly across engine threads.
2. Split the remaining B-E threads approximately 2:1 between dispatch workers W
   and async helpers H. Round W to the nearest integer.
3. Cap W at 35 for gfx950 or 31 for gfx1250, assigning the remaining budget to H.
   The configuration setting is D=W+1, because D includes the submitting caller.
4. At B=4 on gfx1250, use 1/4/0 to avoid spending a thread on async helpers.

In pseudocode for these targets:

```python
ideal_E = min(8, max(1, B // 4))
E = next(e for e in (1, 2, 4, 8) if e >= ideal_E)
remaining = B - E
W = min(dispatch_cap - 1, (2 * remaining + 1) // 3)
if target == "gfx1250" and B == 4:
    W = remaining
D, H = W + 1, remaining - W
```

Here `dispatch_cap` is 36 on gfx950 and 32 on gfx1250. This produces:

| Budget | gfx950 E/D/H | gfx1250 E/D/H |
|---:|---:|---:|
| 1 | 1/1/0 | 1/1/0 |
| 4 | 1/3/1 | 1/4/0 |
| 8 | 2/5/2 | 2/5/2 |
| 16 | 4/9/4 | 4/9/4 |
| 24 | 8/12/5 | 8/12/5 |
| 32 | 8/17/8 | 8/17/8 |
| 48 | 8/28/13 | 8/28/13 |
| 64 | 8/36/21 | 8/32/25 |

This rule uses the budget and target properties, without taking a workload type
as an input. The runtime selects from explicit config tables using a pure, unit-tested
selector, rather than evaluating this formula. The shipped tables contain
1, 2, 4, 8, 16, 24, 32 and 64-thread entries, with an automatic ceiling of 32.
Thus an explicit ceiling of 48 currently selects the 32-thread granule. See
[configuration](configuration.md#thread-accounting-and-preferred-allocations)
for overrides and the `--thread-budget-table` command.

## Testing the literal ratio before rounding E

The initial rule used E=min(8,max(1,B//4)), followed by the same 2:1 split of the
remaining budget and target dispatch limits. It therefore selected 6/13/6 at
24 threads and 1/3/1 at four threads on both targets.

Four interleaved rounds compare this rule with the previously selected
allocations. Identical configurations share their four measurements. The
32-thread rows reuse the immediately preceding direct comparison. CPU
cycles/task-clock rose from roughly 3.3-3.5 to roughly 4.0 GHz-equivalent during
this experiment. Percentage changes below are therefore **medians of the four
within-round time ratios**, not ratios of the displayed time medians. Wall time
also includes variable framework startup; raw ranges are retained with the data.

| Target | Budget | Heuristic E/D/H | Selected E/D/H | Heuristic dispatch / wall s | Selected dispatch / wall s | Median within-round change (dispatch / wall) |
|---|---:|---:|---:|---:|---:|---:|
| gfx950 | 4 | 1/3/1 | 2/3/0 | 1.2374 / 3.005 | 1.2200 / 3.005 | +1.4% / -0.1% |
| gfx950 | 8 | 2/5/2 | 2/7/0 | 0.7123 / 2.500 | 0.7116 / 2.495 | +1.9% / +0.6% |
| gfx950 | 16 | 4/9/4 | 4/5/8 | 0.4731 / 2.230 | 0.4686 / 2.230 | -1.1% / +0.2% |
| gfx950 | 24 | 6/13/6 | 8/9/8 | 0.4144 / 2.250 | 0.4179 / 2.295 | +6.1% / +0.9% |
| gfx950 | 32 | 8/17/8 | 8/13/12 | 0.3967 / 2.375 | 0.3999 / 2.365 | +1.1% / +0.4% |
| gfx950 | 48 | 8/28/13 | 8/16/25 | 0.3591 / 2.115 | 0.3723 / 2.130 | -3.6% / -0.7% |
| gfx950 | 64 | 8/36/21 | 8/34/23 | 0.3285 / 2.080 | 0.3223 / 2.130 | +0.3% / -0.4% |
| gfx1250 | 4 | 1/3/1 | 1/4/0 | 2.0185 / 3.980 | 1.7721 / 3.715 | +12.9% / +6.4% |
| gfx1250 | 8 | 2/5/2 | 2/5/2 | 1.1873 / 3.160 | 1.1873 / 3.160 | +0.0% / +0.0% |
| gfx1250 | 16 | 4/9/4 | 4/11/2 | 0.7422 / 2.700 | 0.7326 / 2.700 | +1.7% / +0.2% |
| gfx1250 | 24 | 6/13/6 | 8/15/2 | 0.6743 / 2.625 | 0.5892 / 2.545 | +11.3% / +2.7% |
| gfx1250 | 32 | 8/17/8 | 8/21/4 | 0.6127 / 2.820 | 0.5956 / 2.805 | +3.0% / +0.5% |
| gfx1250 | 48 | 8/28/13 | 8/32/9 | 0.5175 / 2.485 | 0.5121 / 2.500 | +2.1% / -0.4% |
| gfx1250 | 64 | 8/32/25 | 8/32/25 | 0.5094 / 2.610 | 0.5094 / 2.610 | +0.0% / +0.0% |

## Rounding E to fit the target's XCD count

The partitioner assigns XCD i to engine i modulo E. With eight XCDs, E=6 gives
loads of 2,2,1,1,1,1 XCDs per engine; E=8 gives one each. This motivates rounding E
up to a divisor of the XCD count. At B=24, that changes the rule from 6/13/6 to
8/12/5 while preserving the total budget.

Four fresh rounds compare both choices with the selected 24-thread allocations.
Percentage changes again use within-round ratios. The rounded rule has a median
dispatch change of -3.4% on gfx950 and +2.1% on gfx1250. This supports the
adjustment, although it does not isolate partition balance as the sole cause.

| Target | E/D/H | Dispatch / wall s | Median within-round change vs selected (dispatch / wall) |
|---|---:|---:|---:|
| gfx1250 | 6/13/6 | 0.6548 / 2.735 | +8.4% / +4.9% |
| gfx1250 | 8/12/5 | 0.5998 / 2.770 | +2.1% / -0.0% |
| gfx1250 | 8/15/2 | 0.5944 / 2.565 | +0.0% / +0.0% |
| gfx950 | 6/13/6 | 0.4103 / 2.210 | +1.5% / +0.9% |
| gfx950 | 8/9/8 | 0.4129 / 2.190 | +0.0% / +0.0% |
| gfx950 | 8/12/5 | 0.3994 / 2.175 | -3.4% / -0.2% |

## Measurement and evidence

These are the same launch-only FP8 GEMMs, frozen concurrent runtime, and reserved
physical cores 32-95 used in [the scaling report](concurrent-dispatch-scaling.md).
No numerical comparisons were performed. All launches succeeded, instruction
signatures matched, and every submitted async operation retired.

The evidence directory is:
`/home/jakub/rocjitsu/misc/async-dispatch-concurrent-8-17-8-20260915`.
It contains 140 unique process results: four warmups, 24 direct-comparison samples,
88 additional literal-heuristic samples and 24 engine-rounding samples. The
heuristic summary also reuses 16 of the direct-comparison samples at 32 threads.

- `run.py`, `heuristic.py`, `alignment.py`: runners and summary generation; repeated
  invocations reuse completed samples rather than launching fresh processes.
- `table.md`, `summary.json`, `summary.csv`: 8/16/8, 8/17/8 and selected controls.
- `heuristic-table.md`, `heuristic-summary.json`: the literal-ratio evaluation,
  including median times, ranges and within-round changes.
- `alignment-table.md`, `alignment-summary.json`: the 24-thread comparison.
- `*-plan.json`, per-phase JSON and `samples/`: complete execution plans,
  configurations, environment, commands, timings, perf counters and output.
- `provenance.json`, validation JSON files and `clock-evidence.csv`: frozen binary
  hashes, successful-result checks and frequency-counter evidence.
