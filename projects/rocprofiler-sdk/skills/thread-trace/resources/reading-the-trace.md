# Reading the Trace

## The stats CSV

rocprofv3 writes one `stats_*_dispatch_<n>.csv` per traced dispatch (consecutive kernels
share one): one row per instruction, summed over the traced waves.

| Column | Meaning |
| --- | --- |
| `CodeObj`, `Vaddr` | The code object load id and the instruction's ELF address |
| `Instruction`, `Source` | The instruction and its source line (needs line tables) |
| `Hitcount`, `Latency`, `Stall`, `Idle` | Per-instruction totals over the traced waves; [stalls.md](stalls.md#per-instruction) defines them |

The CSV loses each wave's order; the decoder's records keep it.

## Commands

`python3 <skill dir>/scripts/att_mine.py <capture> <command>`; add `--json` for
machine-readable output (with `--json`, `stats`, `hotspots`, and `lines` omit the `next:` line, and `hotspots` and
`lines` their `note:` line). All but `stats` need the decoder's Python package
([capture.md](capture.md#setup)).

| Command | Answers |
| --- | --- |
| `stats` | The CSV ranked by instruction and by source line by `Latency + Idle`, with the `Latency` of wait and barrier instructions (by opcode: `s_waitcnt`, `s_wait_*`, `s_barrier`) in a `wait` column and their `stall` column set to 0, and per line the number of instructions that ran (`insts`) |
| `summary` | Decoder warnings, waves, unresolved instructions, the wave-state split and the idle time between instructions, the register allocation per wave and LDS per workgroup (VGPRs and LDS rounded up to the allocation granularity, so they can exceed the compiler's counts; on gfx10 and later the SGPR figure can be a fixed 128), and workgroup size of the dispatch the traced waves belong to (and how many dispatches they span), peak active waves, active waves divided by the SIMDs that held a wave, as a time average and at the busiest moment, against the most a SIMD can hold (`waves_per_simd`), `scratch_*` instructions if any ran, and the GPU's properties from rocprofv3's output |
| `pipes` | Per instruction class (VALU, matrix, LDS, memory, waits, barriers, and the decoder's other categories): instructions per wave (a barrier recorded as two instructions, as on gfx9, counts twice), mean issue cycles per instruction (duration minus stall), the share of the SIMDs' resident time during which the class was issuing (instructions issued in the same cycles counted once), the share of stall cycles (instructions their pipe did not accept), the share of waiting (wait, barrier, and other immediate instructions), and the opcodes that stalled most. Ends with the share of the time VALU instructions (matrix included) were issuing, the ceiling check in [compute.md](compute.md#ceiling-check), and, if any ran, the `scratch_*` instructions (scratch memory: register spills or private arrays) per wave and their share of the cost. Resident time is when at least one traced wave was on the SIMD. On gfx10 and later, the cycles after the stall are execution cycles, mostly from a fixed table by instruction type rather than measured (on gfx10 and gfx11, `s_barrier` is measured) |
| `hotspots`, `lines` | Cost per instruction or source line, and the part that no issue on related pipes overlapped ([stalls.md](stalls.md#hidden-cost)). `lines` adds `insts`, the number of distinct instructions each line ran. `wait` holds the latency of wait, barrier, and other immediate (IMMED) instructions, and their `stall` column is set to 0 (that time is in `wait` instead). `hotspots` adds per-hit p50 and p90, which separate steady cost from occasional long hits (stall, wait, or idle) (an instruction recorded twice per execution, such as a barrier on gfx9, mixes two kinds of record) |
| `barriers` | How each workgroup's waves split their time around `s_barrier`: the wave that waited least at `s_barrier` (by launch order; often, not always, the one the others waited for), its barrier share (the latency and idle of its `s_barrier` records over its lifetime, median over workgroups) against the others', the lines it ran, and the barriers the others waited at. Workgroups are estimated as the waves launched together on a compute unit, up to the dispatch's waves per workgroup: an estimate, because the trace's workgroup ids are not reliable (on gfx9 the field does not hold the workgroup index) |
| `lifetime` | Wave lifetime against `s_wait` latency, VALU latency, other latency (barriers included), and idle time: each one's share of lifetime, and the slope and correlation of lifetime with it |
| `occupancy` | Active waves, VGPRs, and SGPRs over time from the occupancy records, and the mean active waves per SIMD that held a wave. The records can span more compute units than the one whose instructions were traced; `occupancy_simds` says how many SIMDs they cover |
| `compare <other capture>` | Cost per source line per wave in two captures, for example before and after a change |

`stats`, `summary`, `pipes`, `hotspots`, and `lines` end with a `next:` line naming the
pages that cover what the capture shows, and why (`stats` chooses from the CSV's costliest
instruction, and for a flat profile points to `summary` and `pipes` instead; a capture with
no waves gets none). The routing is a heuristic with fixed thresholds.

## Your own queries

For a question the reports do not answer, query the decoded records:
[python-api.md](python-api.md).
