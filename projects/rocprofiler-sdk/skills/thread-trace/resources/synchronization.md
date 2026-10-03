# Barriers and Atomics

Waves of a workgroup wait for each other at `s_barrier`, and wait instructions after
atomic instructions hold the wave until the atomics they wait for complete. ATT shows how long each wave spent on these instructions.

## What the trace shows

- `stats`, `hotspots`, and `lines`: `s_barrier` or the wait instruction right after an
  atomic holds the time (in `wait`), or atomic instructions (`global_atomic_*`, `ds_*`
  atomics) do (in `stall` and issue cycles).
- `barriers`: in each workgroup, one wave spends little of its time at `s_barrier` and the
  others much of theirs; that wave is often, not always, the one the others wait for
  ([reading-the-trace.md](reading-the-trace.md#commands) describes the report). When the same
  wave waits least in every workgroup and the others' barrier shares are much higher, that
  points to work only that wave does (for example code under `if (threadIdx.x == 0)`); the
  lines it ran show what that work is. Which wave arrives last can change from barrier to
  barrier; the per-barrier query in [python-api.md](python-api.md#examples) shows it.
- Per wave, from the decoded records ([python-api.md](python-api.md#examples)): the time
  each wave spent at each barrier, and the instructions a wave ran between two barriers.
- Many barriers per iteration, each costing a little, show in `lines` as several `s_barrier`
  lines with a modest `wait` each; together they show in the `wait_share` of the
  `barrier (s_barrier)` row of `pipes`, and as part of WAIT in `summary`.
- `hotspots` can rank `s_barrier` low even when barriers hold most of the wave time: the
  work that hides one wave's wait can be the work the others wait for. Read barrier time
  with `summary` and `barriers`.

## What to change

Suggestions, each for a pattern the trace shows:

- **One wave in each workgroup spends little time at the barrier while the others wait:**
  spread the work that wave does between barriers across the workgroup's threads, or move it
  out of the loop.
- **Many barriers per iteration, each costing a little:** synchronize less often per unit of
  work: fewer barriers per iteration where correctness allows, or more work per iteration.
- **Atomic instructions, or the waits after them, hold the time:** issue fewer atomic
  instructions, for example by combining updates before issuing them.

## Checking a change

Capture again: `compare` should show the barrier and atomic lines losing cost, and wave
lifetime should fall for the same work.
