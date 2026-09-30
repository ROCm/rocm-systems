# Reading the Trace

## The stats CSV

rocprofv3 writes one `stats_*_dispatch_<n>.csv` per traced dispatch: one row per
instruction, summed over the traced waves.

| Column | Meaning |
| --- | --- |
| `Codeobj`, `Vaddr` | The code object load id and the instruction's ELF address |
| `Instruction`, `Source` | The instruction and its source line (needs line tables) |
| `Hitcount` | How many times the traced waves executed it |
| `Latency` | Stall plus issue cycles (stall plus execute on gfx10 and later) |
| `Stall` | Cycles the pipe could not issue it, usually because the unit was busy |
| `Idle` | Cycles since the previous instruction completed: arbiter loss, a register dependency, or an instruction cache miss |

`python3 <skill dir>/scripts/att_mine.py capture stats` ranks instructions and source
lines by `Latency + Idle`. The CSV sums over waves and loses order; the decoder's records
keep each wave's instructions in order.

## Commands

`python3 <skill dir>/scripts/att_mine.py <capture> <command>`; add `--json` for
machine-readable output. All but `stats` need the decoder's Python package
([capture.md](capture.md#setup)).

| Command | Answers |
| --- | --- |
| `stats` | The CSV ranked by instruction and by source line |
| `summary` | Decoder warnings, waves, unresolved instructions, the wave-state split, the dispatch's VGPRs, SGPRs, LDS, and workgroup size, peak active waves, and the GPU's properties from rocprofv3's output |
| `pipes` | Per instruction class (VALU, transcendental, matrix, LDS, memory, waits, barriers, and the decoder's other categories): instructions per wave, issue cycles per instruction, the share of the SIMDs' time spent issuing, and the share of stall cycles. Issue cycles are an instruction's duration minus its stall. Ends with the share of the SIMDs' time that vector instructions issue, the ceiling check in [compute.md](compute.md#ceiling-check) |
| `hotspots`, `lines` | Cost per instruction or source line, and the part no other pipe overlapped (the decoder's hidden-latency analysis); per-hit p50 against p90 separates steady cost from occasional stalls |
| `barriers` | How each workgroup's waves split their time around `s_barrier`: the wave the others waited for (by rank; rank 0 is the workgroup's first wave), its barrier share against the others', the lines it ran, and the barriers the others waited at. Workgroups are the waves launched together on a compute unit, up to the dispatch's waves per workgroup |
| `lifetime` | Wave lifetime against `s_wait` latency, VALU latency, other latency, and idle time: each one's share of lifetime, and the slope and correlation of lifetime with it |
| `occupancy` | Active waves, VGPRs, and SGPRs over time from the occupancy records, and the mean active waves per SIMD. The records span more compute units than the one whose instructions were traced; `occupancy_simds` says how many SIMDs they cover |
| `compare <other capture>` | Cost per source line per wave in two captures, for example before and after a change |

`summary`, `pipes`, `hotspots`, and `lines` end with a `next:` line that names the page of
this skill that covers what the capture shows, chosen by the largest wave state and the
instruction class that holds the stall (`stats`, which reads only the CSV, chooses by its
costliest instruction), and says what the page adds: the changes that
address it, in order, and how to confirm each in the next capture. `hotspots` and `lines`
add a note when no row holds more than 5% of the cost: the cost is then spread over a class
of instructions, which `pipes` names.

How to read `pipes` depends on the architecture. On CDNA GPUs (gfx9, such as MI300), an
instruction's duration minus its stall is the cycles it held its pipe's issue port, so
`issue_share` is how busy each pipe's issue port was, and `cycles_each` gives the cost of one
instruction of the class (for example, a transcendental instruction takes several times the
cycles of an ordinary VALU one). On RDNA GPUs (gfx10 and later) the trace's duration covers
execution, so there `issue_share` is how long each class kept its pipe busy, and it can exceed
the issue time.

`hotspots`, `lifetime`, and `occupancy` compute what the decoder's samples
`hidden_latency_hotspots.py`, `plot_wave_lifetime.py`, and `plot_occupancy_resources.py`
compute. [stalls.md](stalls.md) defines the measures.

## The decoder's Python API

```python
from rocprof_trace_decoder import CodeObject, Decoder, analyze_hidden_latency, generate_code_artifacts

code = generate_code_artifacts([CodeObject("capture/.../x_code_object_id_2.out", 2)]).code_index
with Decoder() as decoder:
    records = decoder.parse_file("capture/.../x_shader_engine_0_1.att", isa=code)
records.waves       # cu, simd, wave_id, workgroup_id, begin_time, end_time, timeline, instructions
records.occupancy   # time, cu, simd, wave_id, workgroup_id, start (1 when a wave starts, 0 when it ends)
records.dispatches  # vgprs, sgprs, lds_size, thread_dim_x/y/z
hidden = analyze_hidden_latency({0: records}, code_index=code).by_pc
```

Each instruction record has `time`, `duration`, `stall`, `category`, and `pc`;
`code.entries[pc]` gives its text and source line, and each `timeline` entry a wave state
and its duration.

## Querying the trace directly

```python
import sys; sys.path.insert(0, "<skill dir>/scripts")
from att_mine import Capture

cap = Capture("capture")
for wave in cap.waves:            # se, cu, simd, wave_id, workgroup, begin, end, states, insts
    for inst in wave.insts:       # time, duration, stall, latency, idle, cost, category, text, source, pc
        ...
```

For example: the time each wave of a workgroup spent at the same `s_barrier`, or the
instructions a wave issued between two costly waits.

Source constructs do not map one-to-one onto instructions; read the disassembly of the
captured code object before concluding what an instruction belongs to.
