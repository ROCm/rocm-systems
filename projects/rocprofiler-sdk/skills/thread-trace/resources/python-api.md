# Using the rocprof-trace-decoder Python API

The `rocprof_trace_decoder` package decodes a `.att` file into Python records: every traced
wave with its instructions and states, wave starts and ends, and dispatches. Use it
directly when a question is narrower than `att_mine.py`'s reports: one instruction across waves, one loop iteration, one barrier, one wave.

- [Setup](#setup)
- [Decode a capture](#decode-a-capture)
- [Functions](#functions)
- [Records](#records)
- [Examples](#examples)
- [Counters, wall-clock time, and markers](#counters-wall-clock-time-and-markers)
- [Sample scripts](#sample-scripts)

## Setup

Set up the decoder's Python package as for `att_mine.py` ([capture.md](capture.md#setup)).

## Decode a capture

```python
import re
from pathlib import Path
from rocprof_trace_decoder import (CodeObject, Decoder, HiddenLatency, InstCategory,
                                   WaveStateType, analyze_hidden_latency,
                                   generate_code_artifacts)

cap = Path("capture")
objects = [CodeObject(p, int(re.search(r"_code_object_id_(\d+)", p.name).group(1)))
           for p in cap.rglob("*_code_object_id_*.out")]
code = generate_code_artifacts(objects).code_index      # instruction text and source by pc

records_by_file = []                                     # (shader engine, TraceRecords)
with Decoder() as decoder:
    for att in sorted(cap.rglob("*.att")):
        se = int(re.search(r"_shader_engine_(\d+)_", att.name).group(1))
        records = decoder.parse_file(att, isa=code)
        for info in records.info:
            print(att.name, decoder.info_string(info))   # decoder warnings, if any
        records_by_file.append((se, records))
waves = [w for _, r in records_by_file for w in r.waves]

text = lambda pc: code.entries[pc].inst if pc in code.entries else "?"
source = lambda pc: code.entries[pc].source if pc in code.entries else "?"
```

- A directory with several kernels or GPUs mixes them ([capture.md](capture.md#output)
  says what each `.att` file holds), so use a new directory for each capture, with one
  kernel in it, or pick the files you decode.
- Decoder warnings are explained in [capture.md](capture.md#troubleshooting).

## Functions

| Call | Returns |
| --- | --- |
| `generate_code_artifacts([CodeObject(path, id), ...]).code_index` | A `CodeIndex`: `code.entries[pc]` gives an instruction's text (`.inst`) and source line (`.source`; for inlined code, `inner -> ... -> call site`) |
| `Decoder().parse_file(path, isa=code)` | The records of one `.att` file (`TraceRecords`) |
| `decoder.info_string(info)` | The text of a decoder warning from `records.info` |
| `code.accumulate_wave(wave)` | Adds a wave to `code.entries[pc]`'s `hitcount`, `latency`, `stall`, and `idle`, the same counts as rocprofv3's stats CSV |
| `analyze_hidden_latency({se: records}, code_index=code).by_pc[pc]` | A `HiddenLatency`: the instruction's `idle`, `stall`, and `issue` cycles that issue on related pipes overlapped ([stalls.md](stalls.md#hidden-cost)) |

Pass one `.att` file per `analyze_hidden_latency` call and add up the results, as the first
example does: its argument is keyed by shader engine, so two files from the same engine
cannot share a call.

## Records

`TraceRecords` holds, among other fields, `waves`, `occupancy`, and `dispatches`, plus `info` (decoder
warnings); `perf_events`, `realtime`, and `shaderdata` are in
[Counters, wall-clock time, and markers](#counters-wall-clock-time-and-markers). Times are in
shader clock cycles.

| Record | Field | Meaning |
| --- | --- | --- |
| `Wave` | `cu`, `simd`, `wave_id` | Compute unit (gfx9) or WGP (gfx10 and later), SIMD, and wave slot |
| | `begin_time`, `end_time` | When the wave started and ended |
| | `instructions` | Its `Instruction`s, in order |
| | `timeline` | Its `WaveState`s, in order |
| | `workgroup_id` | Not a reliable workgroup index (on gfx9 many waves of different workgroups carry the same value); do not group waves by it |
| `Instruction` | `time` | When the wave first tried to issue it |
| | `stall` | Cycles until it issued |
| | `duration` | Stall plus issue cycles (gfx9), or stall plus execution (gfx10 and later) |
| | `category` | An `InstCategory`: `VALU` (matrix instructions included), `SALU`, `VMEM`, `SMEM`, `FLAT`, `LDS`, `IMMED` (`s_waitcnt`, `s_nop`, ...), `MESSAGE` (on gfx9, `s_endpgm` and similar), `JUMP` and `NEXT` (branch taken and not), and others |
| | `pc` | The key into `code.entries` |
| `WaveState` | `type`, `duration` | A `WaveStateType` (`IDLE`, `EXEC`, `WAIT`, `STALL`) and its length; [stalls.md](stalls.md#wave-states) says what each counts |
| `Occupancy` | `time`, `start`, `cu`, `simd`, `wave_id` | A wave starting (`start` 1) or ending (0); these can cover more compute units than the traced one |
| `Dispatch` | `vgprs`, `sgprs`, `lds_size`, `thread_dim_x`/`_y`/`_z` | Registers allocated per wave (`vgprs` includes accumulation registers; on gfx10 and later `sgprs` can be a fixed 128 rather than the allocation) and LDS bytes per workgroup, rounded up to the allocation granularity, so they can exceed the compiler's counts; and the workgroup size |

The idle time before an instruction is not a field; compute it as the second example does,
which matches the stats CSV. Like the CSV, leave out instructions whose `pc` has address 0
and code object 0: instructions the decoder could not resolve, and trap or context records
(category `CONTEXT`). `InstCategory(c).name` and
`WaveStateType(t).name` give names. A barrier can be recorded as two instructions (on gfx9,
a `MESSAGE` and then an `IMMED`); the barrier examples below merge them.

## Examples

Reproduce the stats CSV from the records, and rank instructions by the part of their cost
that no issue on related pipes overlapped ([stalls.md](stalls.md#hidden-cost)):

```python
for w in waves:
    code.accumulate_wave(w)
hidden = {}
for se, records in records_by_file:
    for pc, h in analyze_hidden_latency({se: records}, code_index=code).by_pc.items():
        hidden.setdefault(pc, HiddenLatency())
        hidden[pc] += h
rows = []
for pc, e in code.entries.items():
    if e.hitcount:
        h = hidden.get(pc, HiddenLatency())
        cost = e.latency + e.idle
        rows.append((cost - (h.stall + h.issue + min(h.idle, e.idle)), cost, e.inst, e.source))
for non_hidden, cost, inst, src in sorted(rows, reverse=True)[:5]:
    print(non_hidden, cost, inst, src)
```

One instruction's latency and the idle before it, across waves (here the costliest one):

```python
from collections import defaultdict
per_pc = defaultdict(list)          # pc -> [(duration, stall, idle), ...]
for w in waves:
    prev_end = w.begin_time
    for i in w.instructions:
        if i.pc.address == 0 and i.pc.code_object_id == 0:
            continue                # unresolved
        idle = max(i.time - prev_end, 0)
        per_pc[i.pc].append((i.duration, i.stall, idle))
        prev_end = max(prev_end, i.time + i.duration)
pc = max(per_pc, key=lambda p: sum(d + idle for d, _, idle in per_pc[p]))
durations = sorted(d for d, _, _ in per_pc[pc])
idles = sorted(i for _, _, i in per_pc[pc])
print(text(pc), source(pc), "latency p50", durations[len(durations) // 2],
      "max", durations[-1], "idle p50", idles[len(idles) // 2])
```

Cycles each wave spent in wait instructions on given source lines, and their spread across
waves:

```python
lines = ("kernel.hip:34", "kernel.hip:35")    # matched at the end of `source` (call site)
per_wave = []
for w in waves:
    per_wave.append(sum(i.duration for i in w.instructions
                        if text(i.pc).startswith(("s_waitcnt", "s_wait_"))
                        and source(i.pc).endswith(lines)))
per_wave.sort()
if per_wave:
    print("median", per_wave[len(per_wave) // 2], "p90", per_wave[int(0.9 * len(per_wave))])
```

Each barrier a wave reached, as the index of its last record and the cycles the wave spent
in it, and what one wave issued between two consecutive barriers:

```python
def barriers(w):
    out = []                        # (index of the barrier's last record, cycles in it)
    for k, i in enumerate(w.instructions):
        if text(i.pc).startswith("s_barrier"):
            if out and out[-1][0] == k - 1:            # second record of the same barrier
                out[-1] = (k, out[-1][1] + i.duration)
            else:
                out.append((k, i.duration))
    return out

w = next((w for w in waves if len(barriers(w)) >= 2), None)
if w:
    b = barriers(w)
    between = [i for i in w.instructions[b[0][0] + 1 : b[1][0]]
               if not text(i.pc).startswith("s_barrier")]
    print(len(between), "instructions:",
          {InstCategory(c).name: sum(i.category == c for i in between)
           for c in {i.category for i in between}})
```

How long each wave spent at its k-th barrier, counting from 0 (a wide spread means some
waves arrive much later than others):

```python
k = 1
waits = sorted(barriers(w)[k][1] for w in waves if len(barriers(w)) > k)
if waits:
    print(waits[0], waits[len(waits) // 2], waits[-1])
```

A wave's time in each state:

```python
totals = defaultdict(int)
for s in waves[0].timeline:
    totals[WaveStateType(s.type).name] += s.duration
```

Gaps when no traced wave was resident on any traced compute unit, for example between
kernels in a capture with `--att-consecutive-kernels`:

```python
spans = sorted((w.begin_time, w.end_time) for w in waves)
end, idle_gaps = spans[0][1], []
for begin, finish in spans[1:]:
    if begin > end:
        idle_gaps.append((end, begin - end))     # (when, how many cycles)
    end = max(end, finish)
```

## Counters, wall-clock time, and markers

### SQ counters (gfx9)

With `--att-activity N`, or `--att-perfcounters` and `--att-perfcounter-ctrl`, the trace also
holds SQ counter samples, in `records.perf_events`. Each `PerfEvent` gives four counters read
at `time` on compute unit `cu`: `events0` to `events3` are counters 0 to 3 when `bank` is 0,
and 4 to 7 when it is 1. rocprofv3 configures the counters in the order they are listed, but
drops a name the GPU does not have, with the warning `counter not found`, which moves the later
ones up. Each sample counts one sampling period. The GPU sends no sample for a period in which
all four counts are zero, though the decoder may add a zero-count record to mark the gap, so
add the samples up for a total. The samples cover every compute unit of the shader engine, not only the traced one, unless
`--att-perfcounter-target-only` is set.

```python
# the eight counters --att-activity lists, in its order
names = ["SQ_BUSY_CU_CYCLES", "SQ_VALU_MFMA_BUSY_CYCLES", "SQ_ACTIVE_INST_VALU",
         "SQ_ACTIVE_INST_LDS", "SQ_ACTIVE_INST_VMEM", "SQ_ACTIVE_INST_FLAT",
         "SQ_ACTIVE_INST_SCA", "SQ_ACTIVE_INST_MISC"]
counts = defaultdict(int)            # (shader engine, cu, counter) -> total
for se, records in records_by_file:
    for p in records.perf_events:
        for k, v in enumerate((p.events0, p.events1, p.events2, p.events3)):
            counts[(se, p.cu, names[4 * p.bank + k])] += v
```

### Wall-clock time

`records.realtime` pairs shader clock readings with a reference clock, whose frequency in Hz
is `records.realtime_frequency` (`None` when the capture did not record it). From them, the
shader clock's average frequency over the capture, to turn cycle counts into time:

```python
for se, records in records_by_file:
    rt, hz = records.realtime, records.realtime_frequency
    if len(rt) >= 2 and hz:
        seconds = (rt[-1].realtime_clock - rt[0].realtime_clock) / hz
        mhz = (rt[-1].shader_clock - rt[0].shader_clock) / seconds / 1e6
        span = max(w.end_time for w in records.waves) - min(w.begin_time for w in records.waves)
        print(se, f"{mhz:.0f} MHz; traced waves span {span / mhz:.1f} us")
```

### Markers from the kernel

A kernel can write values into the trace: `__builtin_amdgcn_s_ttracedata(value)` writes a
32-bit value (`s_ttracedata`, through M0), and on gfx10 and later
`__builtin_amdgcn_s_ttracedata_imm(value)` writes an 8-bit immediate. Each becomes a
`ShaderData` record in `records.shaderdata` with its `time`, `value`, and the wave slot
(`cu`, `simd`, `wave_id`) that wrote it, so markers can label loop iterations or phases in
each wave. On gfx10 and later, `flags` is a bitmask: 1 marks a value from `s_ttracedata_imm`,
and 2 a record the trap handler wrote rather than the kernel; on gfx9 it is always 0. Records can also come from compute units that were not traced, and a record's time can
fall after its wave's `end_time`. Each marker adds instructions, so time the kernel without
them.

To give each traced wave its markers, take the last wave on the same slot, in the same `.att`
file, that began before the record:

```python
marks = defaultdict(list)            # id(wave) -> [(time, value), ...]
for _, records in records_by_file:
    by_slot = defaultdict(list)
    for w in sorted(records.waves, key=lambda w: w.begin_time):
        by_slot[(w.cu, w.simd, w.wave_id)].append(w)
    for s in records.shaderdata:
        if s.flags & 2:              # gfx10 and later: written by the trap handler
            continue
        began = [w for w in by_slot.get((s.cu, s.simd, s.wave_id), []) if w.begin_time <= s.time]
        if began:
            marks[id(began[-1])].append((s.time, s.value))
```

## Sample scripts

The decoder's source tree has command-line samples in
`projects/rocprof-trace-decoder/samples`; two of them plot the trace:

| Script | Output |
| --- | --- |
| `plot_occupancy_resources.py` | Active waves and SGPR and VGPR allocation over time (PNG) |
| `plot_wave_lifetime.py` | Each wave's lifetime against its wait, VALU, other, and idle cycles (PNG) |

Run them from `projects/rocprof-trace-decoder` with the `.att` files and code objects; they need
`matplotlib` and `numpy`, and write the PNG to the directory given with `-d` (default: the
current one):

```bash
PYTHONPATH=python python3 samples/plot_wave_lifetime.py -d <outdir> \
    <capture>/*.att <capture>/*_code_object_id_*.out
```
