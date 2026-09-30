# Waiting on Memory Instructions

When waves spend most of their time in WAIT and wait instructions (`s_waitcnt`, or `s_wait_*`
on gfx12 and later) hold the time, the waves are waiting for memory instructions they
issued earlier, and had nothing else ready to issue meanwhile.

## What the trace shows

- `summary`: WAIT is the largest wave state.
- `pipes`: the wait class holds most of the stall, and vector instructions issue a small
  share of the SIMDs' time, so the SIMDs have spare cycles to fill while waves wait.
- `hotspots` and `lines`: wait instructions lead, with most of their cost non-hidden (no
  other pipe was busy while they waited).
- `lifetime`: `s_wait` latency grows with wave lifetime; the longest waves are the ones
  that waited most.
- The disassembly shows which memory instructions each costly wait follows. When a wait
  comes right after the memory instruction it needs, nothing was scheduled in between; when
  a memory instruction's address comes from an earlier load, each step of the chain waits in
  turn.

## Few waves

- `occupancy`: few active waves per SIMD for most of the kernel. `summary` gives the VGPRs,
  SGPRs, and LDS each wave was allocated, and the GPU's maximum waves per SIMD and LDS
  size.
- With few resident waves, a waiting wave has no other wave to switch to, so its waits
  show up as WAIT for the whole SIMD, and its waits for its own matrix or transcendental
  results show up as STALL ([stalls.md](stalls.md#stall-a-busy-pipe-or-a-result-not-ready)).

## What to change

- **More work between each memory instruction and its wait.** Issue the loads of several
  iterations or elements before the first wait (unroll, or load the next iteration's data
  before using the current one).
- **More waves.** Launch more workgroups, or allocate fewer VGPRs, SGPRs, or less LDS per
  wave.
- **Shorter chains of dependent memory instructions.** Compute addresses instead of loading
  them where possible, or give each wave several independent chains so their waits overlap.

## Checking a change

Capture again and run `compare`: the wait lines you targeted should lose cost, WAIT's
share of wave time should fall, and wave lifetime should fall for the same work.
