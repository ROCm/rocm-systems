# Busy Instruction Pipes

When STALL is the largest wave state, instructions were not accepted by their pipe, usually
because the unit was busy or its queue full
([stalls.md](stalls.md#stall-the-pipe-did-not-accept-the-instruction)). The class that holds
the stall names the kind of instruction that stalled (on gfx9, FLAT covers `flat_*`,
`global_*`, and `scratch_*` instructions, and `flat_*` can reach LDS or memory); `pipes` also
shows how much of the time each class was issuing.

## What the trace shows

- `summary`: STALL is the largest wave state, or EXEC once `summary`'s idle between
  instructions is set apart (the waves spend more time issuing than in any other state).
- `pipes`: one instruction class holds most of the stall (`stall_share`), and `top_stalled`
  names the opcodes that stalled most ([reading-the-trace.md](reading-the-trace.md#commands)
  defines the columns).
- `hotspots`: the top instructions belong to that class, with most of their cost in the
  stall column. When every instruction holds a small share, the cost is spread over the
  class, and `pipes` names it.
- Compare `hotspots`' `cost` with `non_hidden` (the hidden part can include stall that issue
  on related pipes overlapped) and read it with `pipes` ([stalls.md](stalls.md#hidden-cost)).

## Ceiling check

`pipes` ends with the share of the SIMDs' resident time during which vector instructions
(all VALU classes) were issuing. Instructions that issued in the same cycles count once, so
it is at most 100%. With `--att-simd-select` on gfx9, the other SIMDs' waves still count as
resident and lower it.

- **Near 100%:** vector issue is the likely limit on the traced SIMDs. Fewer or cheaper
  vector instructions are the change most likely to shorten the kernel (time it);
  scheduling or extra independent work cannot add issue cycles. A high share also results
  when one wave issues while the others wait, for example at a barrier (`barriers`); then
  that wave's serial work, not issue, limits the kernel.
- **Well below 100%, with STALL still the largest state:** go by the class that holds the
  stall (above).
- **Well below 100% in any case:** this measures the instructions the kernel uses now. It
  does not show that doing the work with other instructions (for example matrix instead of
  VALU) or another algorithm would not be faster.
- The trace records when an instruction issues, not how long its unit works on it
  afterwards, and not why an instruction stalled. So a low `busy_share` does not show that
  the unit was idle, and a high stall share alone does not show that the unit's throughput
  limits the kernel.
- On gfx10 and later, the cycles after the stall are execution cycles, mostly from a fixed
  table rather than measured, so the share is an estimate there
  ([reading-the-trace.md](reading-the-trace.md#commands)).

## What to change

Suggestions, each for a pattern the trace shows:

- **One class holds the stall and is issuing most of the time:** issue fewer instructions of
  that class. `hotspots` and `lines` name the lines; look there for work repeated per output
  or per iteration that could be done once, or values recomputed that could be reused.
- **Some instructions of the stalled class take many more cycles each to issue** (per
  instruction, `duration - stall` in the decoded records, which is issue cycles on gfx9 and
  execution cycles on gfx10 and later; [python-api.md](python-api.md)):
  where the result allows, use fewer of them on the hot lines.
- **None of these fits, for example a fixed number of matrix instructions:** the trace does
  not settle what would help. Try changes to how the work is divided (work per wave, waves
  per workgroup, the loop structure) and time each one.

## Checking a change

Capture again: `compare` should show the lines you targeted losing cost, per unit of work
(it is per wave), and `pipes` should show fewer instructions of that class per wave
(`per_wave`). The class's `busy_share` falls only if the class was not the limit, and stays
near 100% if it was, so time the kernel.
