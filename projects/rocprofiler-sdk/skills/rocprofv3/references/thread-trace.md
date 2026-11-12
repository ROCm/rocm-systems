# Advanced Thread Trace (ATT) with rocprofv3

Contents:

1. [Reading the report](#reading-the-report)
2. [Tuning the capture](#tuning-the-capture)
3. [Viewing in ROCprof Compute Viewer](#viewing-in-rocprof-compute-viewer)
4. [Supported GPUs](#supported-gpus)
5. [Parameters](#parameters)
6. [Environment variables and input files](#environment-variables-and-input-files)
7. [Choosing what gets traced](#choosing-what-gets-traced)
8. [Output files](#output-files)
9. [Stats CSV columns](#stats-csv-columns)
10. [ISA cheat sheet for reading hotspots](#isa-cheat-sheet-for-reading-hotspots)
11. [ATT compared with other services](#att-compared-with-other-services)
12. [Troubleshooting](#troubleshooting)

When the dedicated rocprofiler-sdk thread-trace skill is installed, prefer it for capture and
analysis; this page covers the same workflow with this skill's `analyze_att_stats.py`.

Sources: `rocprofv3 --help` (rocprofiler-sdk 1.4.1), the rocprofiler-sdk thread trace guide, and
runs on an AMD Instinct MI325X (gfx942).

## Reading the report

`scripts/analyze_att_stats.py` takes the `--att` output directory or `stats_*.csv` files and
prints, per dispatch: traced waves, total latency and idle, latency by instruction class, top
instructions by latency and by idle, top source lines, and findings. Options: `--kernel NAME`,
`--top N`, `-o report.md`.

- **Latency** is the cycles an instruction took from issue to completion, summed over traced waves
  (stall plus issue on gfx9, stall plus execute on gfx10+). **Stall** is the part where the pipe
  did not accept the instruction (unit busy or its queue full). On a load or store that is the
  memory pipeline backing up, not the wait for data, which shows on the later `s_waitcnt`.
  **Idle** is the cycles between the previous instruction completing and this one starting:
  losing arbitration to another wave, a register dependency on an earlier result, or an
  instruction cache miss. Explicit waits (`s_waitcnt`, `s_nop`) count as latency, not idle.
  **Hitcount** is executions over traced waves; hitcount / traced waves is executions per wave
  (loop trip count).
- **Global memory dominant** (VMEM loads/stores and `s_waitcnt vmcnt` holding most latency): the
  kernel waits on memory. Check coalescing and bytes moved (counter collection), then increase
  memory-level parallelism: more waves, more independent loads in flight before the first wait,
  prefetch into registers or LDS.
- **`s_waitcnt vmcnt` in a hot loop with low latency per hit** (well under 100 cycles): the wait is
  almost always already satisfied; its stall is issue contention with other waves. The loop body
  (VALU, SALU, branch) is the limiter. The script detects this case.
- **`s_waitcnt lgkmcnt`**: waits on scalar loads (kernel arguments, constants) or LDS. Hoist scalar
  loads, interleave LDS reads with math.
- **LDS (`ds_*`) or barrier (`s_barrier`) heavy**: bank conflicts or load imbalance between waves
  of a workgroup.
- **VALU/MFMA heavy with little stall**: compute-bound. Reduce instructions per element, use MFMA
  or lower precision, unroll to cut SALU loop overhead.
- **Idle on `s_endpgm`**: waves waiting for outstanding stores to drain before exiting; count it as
  memory cost.
- Source lines are the outermost frame of the inlining chain (the user's call site); the full
  chain is in the stats CSV.

Explain the hot source lines in terms of the user's code and propose concrete changes. When the
same hotspot appears in a PC sampling or counter profile, cite both.

## Tuning the capture

| Need | Option |
| --- | --- |
| Small grid, empty stats (no wave on the traced CU) | `--att-target-cu <other CU>`, `--att-shader-engine-mask 0xF` (more data), or launch more waves |
| Hardware activity counters in the viewer (gfx9 only) | `--att-activity 8` (not with `--pmc`) |
| Several consecutive dispatches in one trace, including the GPU gaps between them | `--att-consecutive-kernels N --att-gpu-index 0` |
| Trace exactly a code region | `--att --selected-regions` with `roctxProfilerResume(0)` / `roctxProfilerPause(0)` in the app |
| Radeon, waves not captured | `--att-simd-select 0x0` |
| Buffer-full warning printed during the capture | larger `--att-buffer-size` (bytes, for example `0x20000000`), narrower shader-engine mask, or a smaller kernel instance |
| "Data Lost" from the decoder | Part of the trace was dropped for bandwidth; the rest is still decoded but totals miss the dropped part. With streamed counters, raise the `--att-activity` or `--att-perfcounter-ctrl` period or capture without them; otherwise narrow the shader-engine mask |
| Occupancy timeline without instruction detail | `--att-no-detail` |

## Viewing in ROCprof Compute Viewer

Open the `ui_output_agent_<agent>_dispatch_<id>` directory in
[ROCprof Compute Viewer](https://rocm.docs.amd.com/projects/rocprof-compute-viewer/en/amd-mainline/)
for per-wave timelines, ISA with source, wave states, and occupancy. Copy the whole directory
if the viewer runs on another machine. Its files are described under [Output files](#output-files).

## Supported GPUs

| Architecture | GPUs | Support |
| --- | --- | --- |
| CDNA4 (gfx950) | MI350 series | Full: instruction trace and perf counter streaming |
| CDNA3 (gfx942) | MI300 series, MI325X | Full |
| CDNA2 (gfx90a) | MI200 series | Full |
| RDNA4 (gfx1200, gfx1201) | Radeon | Trace only (no `--att-perfcounters`/`--att-activity`) |
| RDNA3.5 (gfx1150 to gfx1153) | Ryzen AI APUs | Trace only; best validated on gfx1151 and gfx1153 |
| RDNA3 (gfx1100 to gfx1102) | Radeon | Trace only |
| RDNA2 (gfx1030) | Radeon | Trace only |

MI100 (gfx908) is expected to work but is not validated.

## Parameters

| Option | Default | Meaning |
| --- | --- | --- |
| `--att`, `--advanced-thread-trace` | off | Enable thread trace |
| `--att-target-cu N` | 1 | CU (WGP on RDNA) that records detailed instruction tokens |
| `--att-shader-engine-mask MASK` | `0x1` | Shader engines to trace; more bits mean more data and more risk of loss |
| `--att-simd-select V` | `0xF` | gfx9: SIMD bitmask; gfx10+: one SIMD ID (try `0x0` on Radeon) |
| `--att-buffer-size BYTES` | 384 MB | Trace buffer shared by all traced shader engines (1 MB to 2 GB) |
| `--att-consecutive-kernels N` | 0 | After a targeted kernel, also trace the next N dispatches into the same files |
| `--att-gpu-index LIST` | all | Comma-separated GPU indexes to trace; recommended with consecutive kernels |
| `--att-activity N` | off | gfx9: stream SQ activity counters every N (1 to 16, 8 recommended); not with `--pmc` |
| `--att-perfcounters "C1 C2"` | off | gfx9: SQ counters to stream (with `--att-perfcounter-ctrl 1-32`); not with `--pmc` or `--att-activity` |
| `--att-perfcounter-target-only` | false | Stream counters only for the target CU to save bandwidth |
| `--att-serialize-all` | false | Serialize untraced kernels too |
| `--att-no-detail` | false | Occupancy only, no instruction detail |
| `--att-no-intercept` | false | Quick-scan device-wide mode without dispatch interception |
| `--att-resource-mode default\|hsa\|code-object` | code-object | When trace queues and buffers are allocated per GPU |
| `--att-library-path DIR ...` | ROCm lib dirs | Where to find `librocprof-trace-decoder.so` |
| `--kernel-include-regex`, `--kernel-exclude-regex` | all kernels | Which kernels to trace (searched in the demangled name) |
| `--kernel-iteration-range` | first instance | Instances of each matching kernel, counted from 1: `2`, `2-4`, `"[1, [5-8]]"` |

## Environment variables and input files

- `ROCPROF_ATT_LIBRARY_PATH`: decoder search path (same as `--att-library-path`).
- `ROCPROF_ATT_PARAM_RESOURCE_MODE`: same as `--att-resource-mode`; the CLI value wins.
- `ROCPROFILER_SQTT_FORCE_HSA=1`: use an HSA queue and HSA memory instead of KFD resources. The SDK
  falls back to HSA automatically, with a warning, when KFD resources are unavailable.
- `HSA_CU_MASK`: restrict the queue to a few CUs so most waves land on the traced CU; this slows
  heavy kernels.

JSON input (`-i att.json`):

```json
{
  "jobs": [
    {
      "advanced_thread_trace": true,
      "kernel_include_regex": "gemm",
      "kernel_iteration_range": "2",
      "att_target_cu": 1,
      "att_shader_engine_mask": "0x1",
      "att_simd_select": "0xF",
      "att_buffer_size": "0x6000000",
      "output_directory": "rocprof_att",
      "output_file": "run"
    }
  ]
}
```

## Choosing what gets traced

- Default: the first instance of every kernel matching the include regex.
- `--kernel-iteration-range` picks other instances; each traced dispatch gets its own files.
- `--att-consecutive-kernels N` starts at a targeted dispatch and continues for N dispatches
  (targeted or not) into one ATT file; a new targeted dispatch inside the window restarts the count.
  It traces on all GPUs unless `--att-gpu-index` is set, and cannot be combined with
  `--selected-regions`.
- `--att --selected-regions`: tracing starts disabled; every dispatch between
  `roctxProfilerResume(0)` and `roctxProfilerPause(0)` is traced. Each resume/pause cycle writes a
  separate set of outputs.
- Thread trace cannot run with `--replay-mode kernel`. It can share a run with tracing options:
  `--att --kernel-trace` also records every dispatch's timing in the database.

## Output files

In the output directory (`-d`), per traced dispatch or batch:

| File | Content |
| --- | --- |
| `stats_ui_output_agent_<agent>_dispatch_<dispatch>.csv` | Per-instruction totals; the input to the analysis script |
| `ui_output_agent_<agent>_dispatch_<dispatch>/` | The decoded trace as JSON, which ROCprof Compute Viewer opens and which scripts can read directly (see the next table) |
| `<name>_<agent>_shader_engine_<se>_<dispatch>.att` | Raw SQTT stream for re-decoding with ROCprof Trace Decoder |
| `<name>_<gfx>_code_object_id_<id>.out` | Code objects (ELF) for ISA tools such as `llvm-objdump -d` |
| `<name>_results.db` | rocpd database with kernel symbols and code objects; it holds dispatch timings only if a tracing option such as `--kernel-trace` was added |

Inside `ui_output_agent_<agent>_dispatch_<dispatch>/`:

| File | What it holds |
| --- | --- |
| `se<se>_sm<simd>_sl<slot>_wv<wave>.json` | One traced wave. `wave.instructions` rows are `[time, category, stall, duration, line]`, where `line` is the row in `code.json`; `wave.timeline` is `[state, duration]`; `wave.waitcnt` lists, for each executed wait, the earlier memory instructions it waited for (`[wait line, [[line, 0], ...]]`). rocprofv3 infers that list assuming in-order completion, so treat it as approximate. |
| `code.json` | The disassembly, one row per instruction with its source line and totals over traced waves; `header` names the columns (ISA, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle) |
| `wstates<k>.json` | How many traced waves were in wave state `k` over time (`time`, `state`); `k` is 1 IDLE, 2 EXEC, 3 WAIT, 4 STALL |
| `occupancy.json` | Wave start and end events per shader engine (`occupancy_fields` names the columns: time, cu, simd, wave_id, start, kernel_id, ...), plus dispatches and other trace events |
| `filenames.json` | GPU generation (`gfxip`), requested counter names, and the index of the wave and other files |
| `realtime.json` | Pairs of GPU clock and real-time clock per shader engine, for converting trace time to wall time |
| `source_*`, `snapshots.json` | Copies of the source files the code objects refer to, when they exist on this machine |
| `se*_perfcounter.json`, `shaderdata_*.json`, `other_simd_se*.json` | Only when the capture has them: streamed SQ counters (`--att-activity`), markers, and (gfx11 and later) memory instructions issued on other SIMDs |

## Stats CSV columns

| Column | Meaning |
| --- | --- |
| `CodeObj` | Code object load ID |
| `Vaddr` | Instruction address within the code object (ELF vaddr) |
| `Instruction` | ISA text. Rows starting with `;` are kernel labels; their `Source` holds the kernel name. |
| `Hitcount` | Executions summed over traced waves |
| `Latency` | Cycles from issue to completion, summed (stall + issue on gfx9, stall + execute on gfx10+) |
| `Stall` | Cycles the pipe could not issue the instruction (unit busy, back pressure, waiting on counters) |
| `Idle` | Gap between the previous instruction completing and this one starting (dependency, arbitration loss, instruction cache miss) |
| `Source` | `file:line`, with inlining chains joined by ` -> `; the last element is the outermost call site |

Rows with `Hitcount` 0 were not executed by any traced wave (other branch paths, or no wave on the
traced CU). `Hitcount` of the first executed instruction approximates the number of traced waves.

## ISA cheat sheet for reading hotspots

| Mnemonic pattern | What it is | Typical cause when hot |
| --- | --- | --- |
| `global_load_*`, `buffer_load_*`, `flat_load_*` | Vector memory loads | Memory latency or bandwidth; issue stalls mean the memory pipeline is backed up |
| `global_store_*`, `buffer_store_*` | Vector memory stores | Write bandwidth; stores also hold up `s_endpgm` |
| `global_atomic_*` | Vector atomics | Contention on the same addresses |
| `s_waitcnt vmcnt(N)` | Wait until at most N vector memory operations are outstanding (gfx9 counts loads and stores) | Data not back yet: too little memory-level parallelism or high miss rate |
| `s_waitcnt vscnt(N)` | Wait on outstanding vector stores (gfx10+) | Store-heavy code |
| `s_waitcnt lgkmcnt(N)` | Wait on LDS, GDS, scalar memory (`s_load`), or messages | Kernel-argument and constant loads not hidden, or LDS latency |
| `s_load_*`, `s_buffer_load_*` | Scalar memory loads | Uniform data (arguments, constants) |
| `ds_read_*`, `ds_write_*` | LDS access | Bank conflicts, narrow accesses |
| `s_barrier` | Workgroup barrier | Waves wait on the slowest wave of the workgroup |
| `v_mfma_*`, `v_smfmac_*` (CDNA), `v_wmma_*` (RDNA) | Matrix instructions | Matrix-core bound, or operands arriving late |
| `v_*` | Vector ALU | Instruction count, dependency chains |
| `s_*` arithmetic, `s_cmp_*`, `s_cbranch_*` | Scalar ALU and branches | Loop control and index math; unroll or simplify |
| `s_nop N` | Inserted wait states | Hardware hazards after certain instructions |
| `s_endpgm` | End of program | Idle here is waiting for outstanding stores to drain |

## ATT compared with other services

| Service | Granularity | Scope | Use for |
| --- | --- | --- | --- |
| Counter collection (`--pmc`) | Per dispatch aggregates | Every dispatch | Memory- vs compute-bound, bandwidth, cache, occupancy |
| PC sampling | Statistical samples of wave PCs, with stall reasons on MI300+ | Many dispatches, long runs | Hotspots across the whole run |
| Thread trace (`--att`) | Every instruction of traced waves, near cycle accurate | One or a few dispatches, one CU per shader engine | Exact per-instruction latency, stalls, and wave scheduling |

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `rocprof-trace-decoder library path not found` | Install the ROCprof Trace Decoder (ROCm 10.0 or later is required for thread trace), or point to it with `--att-library-path <dir>` or `ROCPROF_ATT_LIBRARY_PATH`. |
| No `stats_*.csv` | The regex matched no kernel, the instance in `--kernel-iteration-range` never ran, or the app crashed. Check the demangled name with a kernel trace. |
| Stats CSV has rows but every Hitcount is 0 | No wave ran on the traced CU: change `--att-target-cu`, widen `--att-shader-engine-mask`, launch more workgroups, or restrict CUs with `HSA_CU_MASK`. |
| Empty Source column | Rebuild with `-g`. |
| `--att` rejected with `--replay-mode kernel` | Collect ATT in its own run. |
| `ATT perfcounters cannot be enabled with PMC` | Drop `--pmc` or `--att-activity`/`--att-perfcounters`. |
| `--att-consecutive-kernels` with `--selected-regions` rejected | Use one or the other. |
| Run is very slow | Tracing serializes traced kernels; narrow the filter and iteration range. |
