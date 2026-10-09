---
name: thread-trace
description: Diagnoses why a compute kernel on an AMD GPU is slow, instruction by instruction, with Advanced Thread Trace (rocprofv3 --att) and the rocprof-trace-decoder Python API. Use when a kernel runs below its expected throughput and you need the instructions and source lines that hold the time, such as waves stalled on busy vector or matrix (MFMA) instructions, waiting on memory (s_waitcnt) or at barriers (__syncthreads), or idle; or when capturing or analyzing .att traces. Not for hardware counters, roofline, PC sampling, host-side timelines, or NVIDIA GPUs.
---

# Advanced Thread Trace (ATT)

ATT records the instructions that the traced waves of a kernel executed, in order (fewer
when trace data is lost or the buffer fills): when the wave first tried to issue it, how
many cycles it stalled before it issued, and its duration, from which the idle time before
the next instruction follows. By default
it covers the waves of one compute unit per traced shader engine (on RDNA GPUs, one SIMD of
a WGP), for the dispatches you select. Timing says how slow a kernel is; the trace says
which instructions the time went to.

## Prerequisites

- rocprofv3 from ROCm 10.0 or later, with thread trace support for your GPU.
- The rocprof-trace-decoder library and, for `scripts/att_mine.py`, its Python package.
- The kernel built as you run it (keep its optimization flags, such as `-O3`) plus
  `-gline-tables-only`, so instructions map to source lines; a debug build (`-O0`) runs
  different code and timing.

[capture.md](resources/capture.md#setup) covers each one, including building the decoder.

## How to use ATT

```bash
rocprofv3 --att --kernel-include-regex '<kernel name regex>' -d capture -- ./app <args>
```

- Only the first dispatch of each matching kernel is traced. To skip warm-up, run the kernel
  in a loop and add `--kernel-iteration-range N-N` to trace the Nth dispatch.
- Use a new directory for each capture, with one kernel in it: `att_mine.py` decodes every
  `.att` file under a directory together, so several kernels from one run are mixed, and a
  second run in the same directory stops it, because each run numbers its code objects
  from 1. Unknown kernel names and the other options (which compute unit, SIMDs, and shader
  engines to trace, the buffer size, consecutive kernels) are in
  [capture.md](resources/capture.md#capture).

## What the trace shows, and where to read

Each row is something `att_mine.py` reports, its possible causes, and the page that
covers them. A capture can match several rows; read each page that applies.

| What you see | Possible causes | Read |
| --- | --- | --- |
| Wait instructions (`s_waitcnt`, or `s_wait_*` on gfx12 and later) hold the time (the `wait` column of `hotspots` and `lines`); WAIT is a large wave state | Latency of memory or LDS instructions not covered by other work, or bandwidth (the trace cannot tell which) | [latency.md](resources/latency.md) |
| `s_barrier`, `global_atomic_*`, or the waits after atomics that return a value hold the time | Waves reaching a barrier at different times, many barriers, or atomics that take long to complete | [synchronization.md](resources/synchronization.md) |
| STALL is the largest wave state; VALU, matrix, LDS, or memory instructions hold the stall, or VALU instructions (matrix included) are issuing most of the time | Instructions their pipe did not accept, usually because the unit was busy with earlier instructions or its queue was full | [compute.md](resources/compute.md) |
| The idle column dominates the top instructions, or `summary`'s idle between instructions is large | A source or destination register dependency (waiting for an earlier result), arbiter loss, or instruction cache misses | [stalls.md](resources/stalls.md#idle-cycles) |
| Few waves are resident for most of the kernel (`summary`'s `waves_per_simd` against its `max`) | Registers per wave, LDS per workgroup (which limits how many workgroups fit, so few waves in each leaves few resident), or too few workgroups | [latency.md](resources/latency.md#few-waves) |
| EXEC is the largest wave state, after `summary`'s idle between instructions is set apart | The waves spend more time issuing than in any other state: more instructions than the work needs, or vector issue near its limit (`pipes`) | [compute.md](resources/compute.md#ceiling-check) |

The trace shows where the time went in the code that ran, not whether a change will help:
a large stall or wait share does not show that the time cannot be reduced or that a unit is
at its limit, and a low `busy_share` (the share of time a class was issuing) says nothing
about moving the work to other instructions or another algorithm. When the trace leaves a change open, build it and time
it rather than ruling it out.

## What ATT shows

- For each instruction, summed over the traced waves: how often it ran, its latency, stall,
  and idle cycles ([stalls.md](resources/stalls.md#per-instruction)), and its source line.
- For each wave: when it started and ended, its time in each state (STALL, WAIT, EXEC, and
  IDLE; [stalls.md](resources/stalls.md#wave-states)), and its instructions in order with
  their category (VALU, SALU, SMEM, VMEM, FLAT, LDS, and others).
- How many waves were resident over time, and the dispatch's VGPR allocation per wave and
  LDS per workgroup.
- An estimate of how many of an instruction's cycles overlapped issue on related
  pipes (hidden latency; [stalls.md](resources/stalls.md#hidden-cost)).

ATT does not show which threads of a wave were active, the addresses instructions accessed,
cache or HBM traffic, host-side work such as API calls and launch overhead, or interconnect
traffic, and its per-instruction and per-wave detail covers only the traced compute units
(the occupancy records can span more). With `--att-consecutive-kernels` it
records several dispatches, including the gaps between them on the GPU; on gfx9,
`--att-activity`, or `--att-perfcounters` with `--att-perfcounter-ctrl`, adds SQ counters to the
trace ([python-api.md](resources/python-api.md#sq-counters-gfx9)). Say so
when a question needs what the trace lacks.

## Reading a capture

A loop that tends to work goes from broad to narrow: the kind of time from the trace, then
the lines that hold it, read against the source. Adapt it to the task.

1. **Read the kernel.** Find the loop that does the work and what each iteration loads,
   computes, and synchronizes on, so that the trace's lines and instructions can be read
   against it. Leave what limits the kernel to the trace.
2. **Capture and check the capture.** `python3 <skill dir>/scripts/att_mine.py capture
   summary` (rather than reading the stats CSV, which sums over waves). Decoder warnings (data lost, incomplete waves, incomplete stitching) and
   unresolved instructions mark parts of the trace whose totals are incomplete; zero waves
   means nothing ran on the traced compute unit (on RDNA, on the traced SIMD; on gfx9 with
   `--att-simd-select`, waves of the other SIMDs are still counted, without instructions)
   ([capture.md](resources/capture.md#troubleshooting)).
3. **Broad: what kind of time it is.** Each command decodes the capture again, which takes
   longer as the trace grows (time the first one to see what each will cost):
   - `summary`: the share of wave time in EXEC, WAIT, STALL, and IDLE, and the idle time
     between instructions (may be counted inside EXEC, depending on the architecture);
   - `pipes`: issue and stall cycles by instruction class (VALU, matrix, LDS, memory,
     waits, barriers), the opcodes that stalled most, and the ceiling check
     ([compute.md](resources/compute.md#ceiling-check));
   - `lifetime` and `occupancy`: what grows with wave lifetime, and how many waves were
     resident.
4. **Narrow: where in the source.** `lines` and `hotspots` rank source lines and
   instructions by the cost no issue on related pipes on the SIMD overlapped
   ([stalls.md](resources/stalls.md#hidden-cost)); `stats` ranks the stats CSV; `barriers`
   shows which wave, by launch order, waited least at `s_barrier` (often, not always, the
   one the others wait for), and in what share of the workgroups (estimated). When no line stands out, the cost is spread over many
   instructions: go back to `pipes`, which shows which classes hold it. When a question is
   narrower than these reports (one instruction across waves, one loop iteration, the time
   between a load and its wait, one barrier), query the decoded records with the Python
   API ([python-api.md](resources/python-api.md)).
5. **Read the page for what you found before changing the kernel**
   ([What the trace shows, and where to read](#what-the-trace-shows-and-where-to-read)).
   Each page says what each cause looks like in the trace, changes that address it, and
   how to confirm them. `stats`, `summary`, `pipes`, `hotspots`, and `lines` end with a
   `next:` line (a capture with no waves gets none); read each page it names.

6. **Change, verify, and repeat.** Build the change, check its output, and time it without
   the profiler. Take figures from the tools rather than from memory: the trace has measured
   where the time goes, `summary` prints the GPU's properties as rocprofv3 recorded them,
   and timing a change settles what estimates cannot. Capture the new version and run
   `att_mine.py <old capture> compare <new capture>` (`delta` is new minus old); it compares
   cost per wave, so when a change alters the work each wave does, compare the timings. The
   kind of time often
   changes after a fix, so start the next round at step 3, and time again the changes the
   earlier trace set aside.
7. **Report.** Give the user, in this order:
   1. the bottleneck: the kind of time (wave states and the instruction class) and what it
      means for this kernel;
   2. where: the instructions and source lines that hold it, with their share;
   3. the evidence: the commands you ran and the numbers they printed;
   4. the change, if you made one, and the kernel's time before and after, measured
      without the profiler;
   5. what the trace could not show, and how you checked it otherwise, if you did.

## Tips

- A high cost on an instruction does not say why: `hotspots` shows how much of it is
  `stall`, `wait` (wait, barrier, and other immediate instructions), and `idle` cycles, the
  rest being issue cycles (execution on gfx10 and later), and `summary` shows which state
  the waves were in.
- One source line can compile to many instructions, especially a call inlined from a header,
  such as a math function: `insts` in `lines` counts the distinct instructions the line ran,
  and the disassembly shows them. Check it before treating a line as one instruction.
- Read the disassembly of the captured code object before concluding what an instruction
  belongs to; `hipcc --offload-device-only -S` also writes it to a `.s` file.
- Inlined code carries two lines: `att_mine.py` prints `kernel.hip:23 (inlined
  kernel.hip:13)`, where the first is the call site in the kernel and the second the line
  inside the inlined function. Name both: the cycles are spent on the inner line, and the
  change may belong at either. `stats` may name only the inner line; `lines` shows both.

## Reference

| File | Contents |
| --- | --- |
| [capture.md](resources/capture.md) | Decoder setup, capture options, output files, troubleshooting |
| [reading-the-trace.md](resources/reading-the-trace.md) | The stats CSV and every `att_mine.py` command |
| [python-api.md](resources/python-api.md) | Using the decoder's Python API: decoding a capture, the calls and what they return, the records' fields, worked queries, SQ counters, wall-clock time, and markers written by the kernel |
| [stalls.md](resources/stalls.md) | What latency, stall, idle, hidden cycles, and wave states mean |
| [latency.md](resources/latency.md) | Waves waiting on earlier memory instructions, and too few waves |
| [compute.md](resources/compute.md) | Waves stalled: instructions their pipe did not accept |
| [synchronization.md](resources/synchronization.md) | Waves waiting at barriers and on atomics |
