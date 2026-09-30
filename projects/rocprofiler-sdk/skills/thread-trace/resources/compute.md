# Busy Instruction Pipes

When STALL is the largest wave state, instructions could not issue: their pipe was busy, or
they needed the result of an earlier matrix or transcendental instruction that had not
completed. The instructions that hold the stall cycles say which pipe is involved; the ceiling
check below says whether it is busy or waiting
([stalls.md](stalls.md#stall-a-busy-pipe-or-a-result-not-ready)).

## What the trace shows

- `summary`: STALL is the largest wave state.
- `pipes`: one instruction class holds most of the stall (`stall_share`): VALU, LDS,
  matrix, or transcendental. `issue_share` is the part of the SIMDs' time each class spent
  issuing, and `cycles_each` how many issue cycles one instruction of the class takes.
- `hotspots`: the top instructions are VALU, LDS (`ds_*`), matrix (`v_mfma*`, `v_wmma*`), or
  transcendental (`v_exp`, `v_log`, `v_rcp`, `v_sqrt`, `v_sin`) instructions, with most of
  their cost in the stall column. When every instruction holds a small share, the cost is
  spread over a class of instructions, and `pipes` names the class.
- `lifetime`: VALU latency grows with wave lifetime.
- `hotspots` separates cost other pipes overlapped (hidden) from cost nothing overlapped
  (non-hidden); a pipe that stalls while the others sit idle limits the wave.

## Ceiling check

`pipes` ends with the share of the SIMDs' time that vector instructions (all VALU classes)
spent issuing.

- **Near or above 100%:** the SIMDs issue a vector instruction almost every cycle. The
  kernel is at the ceiling its vector instructions set, and only the first two changes
  below shorten it. Instruction scheduling and extra independent work do not.
- **Well below 100%, with STALL still the largest state:** another pipe is the busy one
  (LDS, or the matrix core), or the stalled instructions wait for earlier results with too
  few waves resident to fill the gap. Look at the class that holds the stall, and at
  `occupancy` ([latency.md](latency.md#few-waves)).
- **Matrix instructions** issue in a few cycles and keep the matrix core busy for longer,
  so their `issue_share` understates the matrix core's load. Stall on a matrix instruction
  does not by itself show a busy matrix core: it is also what a wave waiting for its
  previous matrix instruction's result looks like.

Before concluding that a pipe is at its ceiling, compare with the whole GPU: the trace covers
the waves of one compute unit. Divide the kernel's work (for matrix work, its FLOPs) by its
time without the profiler, and compare with the GPU's peak for those instructions. At a
fraction of the peak the pipe is not saturated: more resident waves, or more independent
chains per wave, can still raise the rate.

For the headroom, divide 100 by the share: at 60%, removing every other cost could make
the kernel at most about 1.7 times faster without cutting vector work.

## What to change

Roughly in order of what they usually gain:

- **Fewer instructions of the kind that stalls.** Compute several outputs per thread so
  work they share (loads, address and index arithmetic, subexpressions that do not depend
  on the output) runs once; hoist work out of loops; reuse values instead of recomputing
  or reloading them.
- **Cheaper instructions** where the result's accuracy allows. `cycles_each` shows which
  instructions cost the most per issue; transcendental instructions take several times as
  many cycles as ordinary VALU instructions. Use fast intrinsics, remove them with
  algebra, or, when an input's range is known and small, replace a chain of them with a
  short polynomial fitted over that range. Check the result against the program's own
  correctness test.
- **Matrix instructions** for matrix products that run on VALU.
- **More independent work while a result is pending**, when few waves are resident and the
  stalled instructions consume an earlier result: more waves (fewer VGPRs per wave), or
  several independent accumulator chains per wave.
- **Overlap the busy pipe with other work.** Interleave independent instructions of other
  kinds, such as the next iteration's loads, with the stalled ones. This helps only while
  the pipe has spare issue cycles (the ceiling check above).

## Checking a change

Capture again: `compare` should show the lines you targeted losing cost, `pipes` should
show the class that held the stall issuing fewer cycles, and STALL's share of wave time
should fall.
