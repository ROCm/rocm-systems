# Instruction and Wave Timing

Every instruction in the trace has a latency (stall plus issue or execute cycles) and the
idle time before it. Which instructions hold the cost, and which state the waves were in,
says what kind of time it is.

## Per instruction

| Measure | Meaning |
| --- | --- |
| Hitcount | How many times the traced waves executed it |
| Latency | Stall plus issue cycles (stall plus execute on gfx10 and later) |
| Stall | Cycles the hardware pipe could not issue it, usually because the unit was busy or its queue full (backpressure, for example from the vector L1 cache (TCP) or LDS). Wait and other immediate instructions spend their latency waiting, and on some architectures that is not in their stall ([reading-the-trace.md](reading-the-trace.md#commands) says where `att_mine.py` reports it) |
| Idle | Cycles between the previous instruction's completion and this one's start, which can be caused by arbiter loss, a source or destination register dependency, or an instruction cache miss |
| Hidden | An estimate of the instruction's stall, issue, and idle cycles that overlapped issue on related pipes on the same SIMD ([Hidden cost](#hidden-cost)) |
| Non-hidden | Latency plus idle, minus hidden |

## Wave states

Each wave's lifetime is split into states:

- **STALL:** the wave's next instruction was not accepted by its pipe
  ([below](#stall-the-pipe-did-not-accept-the-instruction)).
- **WAIT:** the wave was in an immediate instruction: `s_waitcnt` (or `s_wait_*` on gfx12
  and later), but also `s_barrier` ([synchronization.md](synchronization.md)), `s_nop`,
  `s_sleep`, and others. `hotspots` shows which. A barrier can be recorded as two
  instructions (on gfx9, a short message and then the wait), which doubles its hits.
- **EXEC:** the wave issued an instruction, and the time until its next one, so EXEC
  includes the idle gaps between instructions.
- **IDLE:** the time before the wave's first instruction.

## Waits: `s_waitcnt` and `s_wait_*`

A wait instruction (`s_waitcnt`; on gfx12 and later `s_wait_loadcnt`, `s_wait_dscnt`, and
the other `s_wait_*` instructions) holds the wave until enough of the memory instructions
it issued earlier have completed for their count to fall to the value in its operand
(`vmcnt(0)` waits for all of them). A costly wait is time the wave spent waiting for their
results. The trace does not record which of the earlier instructions it waited for; rocprofv3
works them out per wave from the instruction sequence, assuming in-order completion, in the
`waitcnt` list of each wave's file, which leaves some waits out
([capture.md](capture.md#the-ui_output-directory)). See [latency.md](latency.md).

## STALL: the pipe did not accept the instruction

The unit can be busy with earlier instructions from this wave or others. Stall happens to
every kind of instruction (VALU, matrix, LDS, vector and scalar memory). The trace does not
record the reason for a stall. See [compute.md](compute.md).

Stall on a memory instruction (a load, store, or LDS access) is its pipe not accepting it,
not the wave waiting for its data: that wait shows on the `s_waitcnt` or `s_wait_*`
instruction before the data is used.

## Idle cycles

Arbiter loss means another wave issued; a register dependency means the instruction needed
the result of an earlier one. Waits the program makes explicit are not idle: `s_waitcnt` and
`s_wait_*` for memory results, and `s_nop` that the compiler inserts between some dependent
instructions, show as WAIT. `summary` reports idle as the idle time between instructions,
and `hotspots` per instruction.

## Hidden cost

`hotspots` subtracts the cycles during which instructions on related pipes were issuing on
the same SIMD, from this wave or others; which pipes count depends on the instruction's
class. For a stall, that can include the same pipe: a VALU stall
counts as hidden only while VALU or matrix instructions issue. So:

- For waits and idle time, a mostly hidden cost overlapped other work, and removing it may
  not shorten the kernel. This does not hold for `s_barrier`: the work that hides one wave's
  barrier wait can be the work the other waves are waiting for. Read barrier time with
  `barriers`, not `non_hidden`.
- A hidden stall can be the stalled pipe doing other work. The pipe may then still be the
  limit; `pipes` shows how much of the time it was issuing.

Compare `cost` with `non_hidden` before attributing time to an instruction.

## What to change

Suggestions, each for a pattern the trace shows:

- **Idle concentrates on instructions that use the previous instruction's result:** put
  independent work between them, for example by processing several elements per thread so
  their chains overlap.
- **Few waves are resident (`summary`'s `waves_per_simd`):** allow more waves, so other
  waves issue while one waits for its result ([latency.md](latency.md#few-waves)).

## Checking a change

Capture again and run `att_mine.py <old capture> compare <new capture>`: the lines you
targeted should lose cost. For idle, `hotspots` should show less in the `idle` column on
those instructions, and `summary`'s idle between instructions should fall.
