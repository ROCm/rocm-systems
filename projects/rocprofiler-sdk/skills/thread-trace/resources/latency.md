# Waiting on Memory Instructions

When waves spend most of their time in WAIT and wait instructions (`s_waitcnt`, or
`s_wait_*` on gfx12 and later) hold the time, the waves are waiting for memory instructions
they issued earlier.

## What the trace shows

- `summary`: WAIT is the largest wave state. WAIT counts every immediate instruction, so
  check in `hotspots` that wait instructions, not `s_barrier` or `s_nop`, hold it.
- `pipes`: the wait class holds most of `wait_share`, and VALU instructions are issuing
  during a small share of the time.
- `hotspots` and `lines`: wait instructions lead, with most of their cost non-hidden
  (little issue on other pipes overlapped them).
- `lifetime`: `s_wait` latency holds the largest share of wave lifetime; with many traced
  waves, a positive slope and r also show that waves that waited longer tended to live longer.
- The disassembly shows which memory instructions each costly wait follows, and each wave's
  `waitcnt` list names those rocprofv3 works out it required to complete, assuming in-order
  completion ([capture.md](capture.md#the-ui_output-directory)). When a wait
  comes right after the memory instruction it needs, nothing was scheduled in between; when
  a memory instruction's address comes from an earlier load, each step of the chain waits in
  turn.

The trace shows that the waves waited, not why the memory system took that long: it cannot
tell latency from bandwidth.

## Few waves

- `occupancy`: few active waves per SIMD for most of the kernel. `summary` gives the VGPRs
  per wave and the LDS per workgroup that the dispatch allocated.
- With few resident waves, a waiting wave has few other waves to switch to: little on the
  SIMD overlaps its waits, and they appear mostly as non-hidden cost in `hotspots`.

## What to change

Suggestions, each for a pattern the trace shows:

- **A wait comes right after the memory instruction it waits for:** issue that instruction
  earlier, or issue several before the first wait, so other work runs while they are in
  flight.
- **Each load's address comes from the previous load (a chain):** shorten the chain, for
  example by computing addresses instead of loading them, or give each wave several
  independent chains so their waits overlap.
- **Few waves are resident:** allow more waves (more workgroups, fewer registers per wave,
  less LDS per workgroup, or, when LDS per workgroup limits how many workgroups fit, more
  waves per workgroup for the same LDS), so the SIMD has other waves to run during a wait;
  this helps only if the memory system has headroom, which the trace cannot show.
- **The wait and load lines cost as much as the lines that compute with the data:** do more
  work for each value loaded, for example more outputs per thread or a larger block of data
  per loop iteration.

## Checking a change

Capture again and run `att_mine.py <old capture> compare <new capture>`: the wait lines you
targeted should lose cost, per unit of work (`compare` is per wave, so when a wave now does
more work, compare kernel times). WAIT's share of wave time can stay high after a real
improvement, so check the kernel's time.
