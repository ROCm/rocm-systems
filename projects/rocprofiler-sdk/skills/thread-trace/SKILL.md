---
name: thread-trace
description: Diagnoses why a HIP kernel on an AMD GPU is slow, instruction by instruction, by capturing Advanced Thread Trace (ATT) with rocprofv3 --att and reading it with the rocprof-trace-decoder Python API. Use when a kernel runs below its expected throughput and you need to know which instructions and source lines hold the time, such as waves stalled on busy vector, transcendental, or matrix (MFMA) pipes, matrix cores sitting idle, waves waiting on memory (s_waitcnt) or at barriers (__syncthreads), or too few resident waves; and when capturing .att traces, analyzing an existing capture, or working with rocprof-trace-decoder. Not for hardware counters, roofline analysis, PC sampling, or host-side timelines of API calls, copies, and kernel launches (rocprofv3 --pmc, rocprof-compute, PC sampling, and rocprofv3's API and kernel traces cover those), or for NVIDIA GPUs and CUDA tools.
---

# AMD Thread Trace (ATT)

ATT records every instruction that the traced waves of a kernel executed, in order: when
each one issued, how long it stalled, and how long the wave sat idle before it. It covers
the waves of one compute unit (a WGP on RDNA GPUs) per traced shader engine, for the
dispatches you select. Timing says how slow a kernel is; the trace says which instructions
the time went to.

## Prerequisites

- rocprofv3 from ROCm 7.0 or later (the release that added `--att`), with thread trace
  support for your GPU. The
  [supported devices](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html#supported-devices)
  table lists the validated architectures and what each supports; treat others as
  unverified until a capture works, and check `rocprofv3 --help` for the options your
  version has.
- The rocprof-trace-decoder library. If `rocprofv3 --att` reports that it cannot find the
  decoder, build it and set `ROCPROF_ATT_LIBRARY_PATH` as in
  [capture.md](resources/capture.md#setup).
- For `scripts/att_mine.py`: Python 3.10 or later with `pyelftools` 0.31 or later, the
  decoder's Python package on `PYTHONPATH`, and `ROCPROF_TRACE_DECODER_LIB` pointing to
  the library. If `python3 -c "import rocprof_trace_decoder"` fails, use the package in
  rocm-systems (same section).
- Kernels built with `-gline-tables-only` (or `-g`), so instructions map to source lines.

## How to use ATT

```bash
rocprofv3 --att --kernel-include-regex '<kernel name regex>' -d capture -- ./app <args>
```

- Only the first matching dispatch is traced. To skip warm-up, run the kernel in a loop
  and add `--kernel-iteration-range N-N` to trace the Nth dispatch.
- If you do not know the kernel's name, capture without the regex: each traced
  dispatch's CSV starts with its kernel's name.
- Run `rocprofv3 --help` for the other options: which compute unit, SIMDs, and shader
  engines to trace, the buffer size, and consecutive kernels
  ([capture.md](resources/capture.md#capture) lists them).

## ATT outputs

| File | What it is |
| --- | --- |
| `stats_*_dispatch_<n>.csv` | Per-instruction summary of the traced waves: hit count, latency, stall, idle, source line. It sums over waves, so it cannot show which wave waited for which; read it through `att_mine.py` (steps 2 to 4), which adds the per-wave view and names the page to read next. |
| `ui_output_*_dispatch_<n>/` | Per-wave JSON for the ROCprof Compute Viewer |
| `*/*_shader_engine_<se>_<id>.att` | The raw trace, one file per shader engine; the decoder's Python API and `att_mine.py` read it |
| `*/*_code_object_id_<n>.out` | The code objects the trace refers to; `llvm-objdump -d` the one holding your kernel |
| `*_results.db` | rocprofv3's database for the run, including the GPU's properties: compute units, SIMDs per CU, wave size, maximum waves per SIMD, LDS size (`*_agent_info.csv` with `--output-format csv`) |

## What ATT shows

- For each instruction, summed over the traced waves: how often it ran, its latency (stall
  plus issue cycles; stall plus execute on gfx10 and later), its stall cycles (the pipe
  could not issue it), and the idle cycles
  before it (arbiter loss, a register dependency, or an instruction cache miss), with its
  source line.
- For each wave: when it started and ended, its time in each state (EXEC issuing, WAIT,
  STALL, IDLE), and its instructions in order with their category (VALU, SALU, SMEM, VMEM,
  FLAT, LDS, and others).
- How many waves were resident over time, and the VGPRs, SGPRs, and LDS each dispatch
  allocated per wave.
- The decoder's estimate of how many of an instruction's cycles overlapped other
  instruction-pipe work (hidden latency).

ATT does not show which threads of a wave were active, the addresses instructions
accessed, cache or HBM traffic, hardware counters, host overhead or gaps between kernels,
or interconnect traffic. Say so when a question needs them. Traced runs are slower than
normal ones; time the application without the profiler.

To test a limit the trace cannot see, time an experiment: for example, point the loads at
one tile so they hit in cache, or skip part of the computation, and compare the time with
the original. The difference is what that part costs, whether or not the trace shows it.

## Reading a capture

A loop that tends to work goes from broad to narrow: a hypothesis from the source, the kind
of time from the trace, then the lines that hold it. Adapt it to the task.

1. **Start from the source.** Read the kernel and guess what limits it (memory traffic,
   vector or matrix arithmetic, synchronization), which loop does the work, and roughly
   how fast it could run given the bytes it moves and the instructions it issues. The
   trace then confirms the guess or replaces it.
2. **Capture and check the capture.** `python3 <skill dir>/scripts/att_mine.py capture
   summary`. Decoder warnings (data lost, incomplete waves), zero waves, or unresolved
   instructions make it unusable ([capture.md](resources/capture.md#troubleshooting)).
3. **Broad: what kind of time it is.** Each command takes seconds:
   - `summary`: the share of wave time in EXEC, WAIT, STALL, and IDLE;
   - `pipes`: issue and stall cycles by instruction class (VALU, transcendental, matrix,
     LDS, memory, waits, barriers). Its last line is a ceiling check: how much of the
     SIMDs' time vector instructions issue. Near 100%, the kernel runs as fast as its
     vector instructions allow, and only fewer or cheaper instructions will help;
   - `lifetime` and `occupancy`: what grows with wave lifetime, and how many waves were
     resident.
4. **Narrow: where in the source.** `lines` and `hotspots` rank source lines and
   instructions by the cost no other pipe overlapped; `stats` ranks the stats CSV;
   `barriers` shows which wave of each workgroup the others wait for at `s_barrier`. When no
   line stands out, the cost is spread over a class of instructions: go back to `pipes`.
5. **Read the page for what you found before changing the kernel.** The table only names
   a direction; each page lists the changes to try in order, what each looks like in the
   trace, and how to confirm it. `summary`, `pipes`, `hotspots`, and `lines` end with a
   `next:` line that names the page for the capture they read.

   | What you see | Direction | Read |
   | --- | --- | --- |
   | Wait instructions (`s_waitcnt`, or `s_wait_*` on gfx12 and later) hold the time; WAIT is the largest wave state | Hide memory latency | [latency.md](resources/latency.md) |
   | `s_barrier` or atomic instructions hold the time | Balance or remove synchronization | [synchronization.md](resources/synchronization.md) |
   | STALL is the largest wave state; VALU, LDS, matrix, or transcendental instructions hold the time, or vector instructions issue most of the SIMDs' time | Issue less work on the busy pipe | [compute.md](resources/compute.md) |
   | The idle column dominates the top instructions | Shorten dependency chains | [stalls.md](resources/stalls.md#idle-cycles) |
   | Few waves are resident for most of the kernel | Keep more waves resident | [latency.md](resources/latency.md#few-waves) |

6. **Change, verify, and repeat.** Build the change, check its output, and time it
   without the profiler. Take figures from the tools rather than from memory: the trace
   has measured where the time goes, `summary` prints the GPU's properties as rocprofv3
   recorded them, and timing a change settles what estimates cannot. Capture the new
   version and run `att_mine.py <new capture> compare <old capture>`. The kind of time
   often changes after a fix, so start the next round at step 3; when the trace
   contradicts your guess from step 1, revise the guess. When the kernel looks to be at a
   ceiling, check that against the GPU's peak rate and with a timed experiment before
   concluding that nothing more can be gained.
7. **State the finding:** the instruction and source line that hold the time, the command
   and numbers that show it, what you changed and the timings, and what ATT could not show.

[reading-the-trace.md](resources/reading-the-trace.md) describes the stats CSV, every
command, the decoder's Python API, and how to query the trace directly.

## Tips

- A high cost on an instruction does not say why: `hotspots` splits it into stall and idle
  cycles, and `summary` shows which state the waves were in.
- STALL means an instruction could not issue: its pipe was busy, or it needed an earlier
  matrix or transcendental result. With one or two waves per SIMD it is usually the
  latter. Before concluding that a pipe is saturated, compare the kernel's achieved rate
  with the GPU's peak ([stalls.md](resources/stalls.md#stall-a-busy-pipe-or-a-result-not-ready)).
- A large stall on an instruction that runs once per wave (its hits equal the number of
  traced waves), such as one in the kernel's prologue, is the wave waiting for a busy
  pipe, not the cost of that instruction. Reduce the work on that pipe instead of
  changing that line ([compute.md](resources/compute.md)).
- Read the disassembly of the captured code object before concluding what an instruction
  belongs to; `--offload-device-only -S` with hipcc also prints it.
- Inlined code carries two lines: `att_mine.py` prints `kernel.hip:23 (inlined
  kernel.hip:13)`, and the CSV's source column the chain `...:13 -> ...:23`. The first
  `att_mine.py` line (the last in the chain) is the call site in the kernel; the other is the
  line inside the inlined function. Name the call site as the place to change, and the inner
  line as where the cycles go.
- To compare two versions of a kernel, capture both and run `att_mine.py <capture> compare
  <other capture>`.

## Reference

| File | Contents |
| --- | --- |
| [capture.md](resources/capture.md) | Decoder setup, capture options, output files, troubleshooting |
| [reading-the-trace.md](resources/reading-the-trace.md) | The stats CSV, every `att_mine.py` command, the decoder's Python API, custom queries |
| [stalls.md](resources/stalls.md) | What latency, stall, idle, hidden cycles, and wave states mean |
| [latency.md](resources/latency.md) | Waves waiting on earlier memory instructions, and too few waves |
| [compute.md](resources/compute.md) | Waves stalled on a busy instruction pipe |
| [synchronization.md](resources/synchronization.md) | Waves waiting at barriers and on atomics |
