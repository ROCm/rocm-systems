# Barriers and Atomics

Waves of a workgroup wait for each other at `s_barrier`, and waits on atomic instructions
last until the atomics complete. ATT shows how long each wave spent on these instructions.

## What the trace shows

- `hotspots` and `stats`: `s_barrier`, atomic instructions (`global_atomic_*`, `ds_*`
  atomics), or the wait instruction right after an atomic hold the time.
- `barriers`: in each workgroup, the wave the others waited for spends little of its time at
  `s_barrier`, and the others spend much of theirs there. It reports which wave that is (rank
  0 is the workgroup's first wave), both barrier shares, the lines that wave ran, and the
  barriers the others waited at. When one rank is the waited-for wave in every workgroup,
  work that only that wave does (for example code under `if (threadIdx.x == 0)`, which runs
  in the first wave) is what the whole workgroup waits for.
- Per wave, from `Capture.waves` ([reading-the-trace.md](reading-the-trace.md#querying-the-trace-directly)):
  the time each wave spent at each barrier, and the instructions a wave ran between two
  barriers.
- Many barriers per loop iteration, each costing a little, add up in `lines`.

## What to change

- **Reach each barrier together.** Give the waves of a workgroup equal work between
  barriers; move work that only some waves do out of the loop, or spread it across all
  threads.
- **Fewer barriers.** Keep only the barriers that separate a write from a read of the same
  shared data.
- **Fewer atomics.** Combine updates within a wave or workgroup first, then issue one atomic
  per group.

## Checking a change

Capture again: `compare` should show the barrier and atomic lines losing cost, and wave
lifetime should fall for the same work.

Waits on another GPU's memory look like slow memory instructions; ATT cannot show the
link's bandwidth or congestion.
