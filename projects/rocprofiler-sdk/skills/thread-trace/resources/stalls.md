# Instruction and Wave Timing

Every instruction in the trace has a latency (stall plus issue or execute cycles) and the
idle time before it. Which instructions hold the cost, and which state the waves were in, says what
kind of time it is. The definitions follow the rocprofv3 thread trace documentation.

## Per instruction

| Measure | Meaning |
| --- | --- |
| Hitcount | How many times the traced waves executed it |
| Latency | Stall plus issue cycles (stall plus execute on gfx10 and later) |
| Stall | Cycles the pipe could not issue it, usually because the unit was busy (for example vector memory or LDS backpressure) |
| Idle | Cycles between the previous instruction's completion and this one's start: arbiter loss, a register dependency, or an instruction cache miss |
| Hidden | The decoder's estimate of the instruction's stall, issue, and idle cycles during which another instruction pipe on the same SIMD was busy |
| Non-hidden | Latency plus idle, minus hidden: the cycles no other work overlapped |

## Wave states

The decoder splits each wave's lifetime into states: EXEC (issuing), WAIT (waiting on a
wait instruction), STALL (an instruction could not issue), and IDLE (nothing to issue).
`att_mine.py capture summary` sums them over the traced waves, and `pipes` splits
issue and stall cycles by instruction class.

## Waits: `s_waitcnt` and `s_wait_*`

A wait instruction (`s_waitcnt`; on gfx12 and later `s_wait_loadcnt`, `s_wait_dscnt`, and
the other `s_wait_*` instructions) holds the wave until memory instructions it issued
earlier have completed. A costly wait is time the wave spent waiting for their results.
Which of the earlier instructions it waited for is not in the trace; the disassembly shows
the memory instructions issued before it. See [latency.md](latency.md).

## Barriers: `s_barrier`

A costly `s_barrier` is time a wave spent waiting for the other waves of its workgroup to
arrive. See [synchronization.md](synchronization.md).

## STALL: a busy pipe or a result not ready

A stalled instruction could not issue for one of two reasons, and the trace does not say
which:

- **Its pipe was busy** with earlier instructions, from this wave or others. That is a
  throughput limit: only less work on that pipe helps.
- **It needs the result of an earlier long-latency instruction** on the same wave, such as a
  matrix (`v_mfma*`, `v_wmma*`) or transcendental instruction, that has not completed. That
  is latency, like a wait on memory: more resident waves, or several independent chains per
  wave (for example more accumulators), give the pipe other work meanwhile.

Two checks tell them apart. With one or two waves per SIMD (`occupancy`, and the VGPRs per
wave in `summary`), a wave that stalls usually leaves its pipe idle, so the stall is its own
dependency chain. And a pipe that limits throughput runs near the GPU's peak for its
instructions: divide the kernel's work (FLOPs, or instructions issued) by its time without the
profiler and compare with the peak. A kernel far below that peak is not at the pipe's ceiling,
whatever share of the stall its instructions hold. `pipes` shows how much of the SIMDs' time
each class spent issuing; see [compute.md](compute.md#ceiling-check).

## Idle cycles

Idle before an instruction comes from arbiter loss (another wave issued), a register the
instruction needs from an earlier one, or an instruction cache miss. Idle that concentrates
on the instructions of a dependency chain points to the chain; idle spread over a large
loop body can be instruction fetch.

- Break long chains of dependent instructions: interleave independent work, or process
  several elements per thread so their chains overlap.
- Keep hot loops small enough that their instructions stay cached; heavy unrolling makes
  the loop body larger.

## Hidden cost

`hotspots` subtracts the cycles during which another pipe on the same SIMD was busy. An
instruction whose cost is mostly hidden overlapped useful work; removing it would not
shorten the wave. Compare `cost` with `non_hidden` before attributing time to it.
