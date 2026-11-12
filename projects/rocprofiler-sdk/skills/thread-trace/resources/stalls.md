# Instruction and Wave Timing

Every instruction in the trace has a latency (stall plus issue or execute cycles) and the
idle time before it. Which instructions hold the cost, and whether it is waiting, stall,
issue, or idle, says what kind of time it is.

## Per instruction

| Measure | Meaning |
| --- | --- |
| Hitcount | How many times the traced waves executed it |
| Latency | Stall plus issue cycles (stall plus execute on gfx10 and later) |
| Stall | Cycles the hardware pipe could not issue it, usually because the unit was busy (for example, backpressure from the vector L1 cache (TCP) or LDS). Wait and other immediate instructions spend their latency waiting, and on some architectures that is not in their stall ([reading-the-trace.md](reading-the-trace.md#commands) says where `att_mine.py` reports it) |
| Idle | Cycles between the previous instruction's completion and this one's start, which can be caused by arbiter loss, a source or destination register dependency, or an instruction cache miss |
| Hidden | An estimate of the instruction's stall, issue, and idle cycles that overlapped issue on related pipes on the same SIMD (and, on gfx11 and later, memory issue on the other SIMD; [Hidden cost](#hidden-cost)) |
| Non-hidden | Latency plus idle, minus hidden |

## Wave time

`summary`'s `wave_time` splits the traced waves' lifetime by what their instructions did,
summed from the instruction records:

- **wait (`s_waitcnt`, `s_wait_*`):** the latency of wait instructions, the time the wave
  waited for earlier memory instructions ([below](#waits-s_waitcnt-and-s_wait_)).
- **barrier (`s_barrier`):** the latency of barriers ([synchronization.md](synchronization.md)).
  On gfx9 the decoder records a barrier as two instructions (a short message when the wave
  reaches it, then the wait); `att_mine.py` counts the pair as one. On gfx12,
  `s_barrier_signal` and `s_barrier_wait` are two instructions.
- **other immediate:** the latency of other immediate instructions, such as `s_nop` and
  `s_sleep`.
- **stall:** cycles the other instructions waited for their pipe to accept them
  ([below](#stall-the-pipe-did-not-accept-the-instruction)), also given by class.
- **issue:** the rest of those instructions' latency: issue cycles on gfx9, execution
  cycles on gfx10 and later.
- **idle between instructions:** the idle before every instruction but a wave's first
  ([Idle cycles](#idle-cycles)).
- **before the first instruction** and **after the last instruction:** the time from the
  wave's start to its first instruction ([below](#before-the-first-instruction)), and from
  its last instruction's end to the wave's end.
- **waves without instructions:** the whole lifetime of waves the trace lists with no
  instructions, such as the other SIMDs' waves on gfx9 with `--att-simd-select`.

Instruction latencies can overlap, so the shares can add up to a little more than 100%.
`instructions_cover` is the share the instructions account for (all but the last three
kinds). Below 100%, the rest of the waves' time is not in any instruction's figures.

The decoder also records wave states (STALL, WAIT, EXEC, IDLE). They are sums of the same
records with less detail (WAIT mixes waits and barriers, and EXEC mixes issue with idle
between instructions), so `att_mine.py` does not use them.

## Waits: `s_waitcnt` and `s_wait_*`

A wait instruction (`s_waitcnt`; on gfx12 and later `s_wait_loadcnt`, `s_wait_dscnt`, and
the other `s_wait_*` counter waits) holds the wave until enough of the memory instructions
it issued earlier have completed for their count to fall to the value in its operand. Each
counter covers only some instructions: on gfx9, `vmcnt` counts vector memory instructions
and `lgkmcnt` LDS, scalar memory, and message instructions; on gfx10 and gfx11, vector
stores have their own counter, `vscnt`. So `vmcnt(0)` does not wait for LDS or scalar
memory instructions. A costly wait is time the wave spent waiting for their
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

Idle time can come from arbiter loss, a source or destination register dependency, or an
instruction cache miss; the trace does not say which. Waits the program makes explicit are
not idle: `s_waitcnt`, the `s_wait_*` counter waits, and `s_nop` are instructions with
their own latency. The ALU-dependency waits `s_delay_alu` (gfx11 and later) and
`s_wait_alu` (gfx12 and later) are not recorded, so their time appears as idle on the next
instruction. `summary`
reports the idle between instructions in `wave_time`, and `hotspots` per instruction.

## Before the first instruction

The time from a wave's start to its first instruction. The trace records when the wave
started and when its first instruction issued, not what happened in between, so no
instruction and no page of this skill accounts for it. It weighs most when waves are short
(`summary`'s `wave_lifetime_median`) or few. When it holds most of the wave time, the
instructions' figures describe only the rest (`instructions_cover`): read them with that
share in mind, and check whether a larger problem, or more work per wave, makes the waves'
instruction time dominate.

## Hidden cost

`hotspots` subtracts the cycles during which instructions on related pipes were issuing on
the same SIMD, from this wave or others; which pipes count depends on the instruction's
class. On gfx11 and later, the memory instructions the decoder records for the other SIMD
(`other_simd`) also count as vector-memory issue, so they can hide memory, scalar, and
other instructions' cycles. For a stall, that can include the same pipe: a VALU stall
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
  their chains overlap. This helps only if the idle is a dependency, which the trace cannot
  tell from arbiter loss or an instruction cache miss.
- **Few waves are resident (`summary`'s `waves_per_simd`):** allow more waves, so other
  waves issue while one waits for its result ([latency.md](latency.md#few-waves)); this
  helps only if the memory system has headroom, which the trace cannot show.

## Checking a change

Capture again and run `att_mine.py <old capture> compare <new capture>`: the lines you
targeted should lose cost. `compare` is per wave, so when a change alters the work each wave
does, compare kernel times, or the cost per unit of work. For idle, `hotspots` may show
less in the `idle` column on those instructions, and the idle between instructions in
`summary`'s `wave_time` may fall.
