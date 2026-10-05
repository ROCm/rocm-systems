---
name: rocprofv3
description: >-
  Profiles applications on AMD GPUs with rocprofv3 and explains the results: application
  tracing (hot kernels, memory copies, idle GPU, Perfetto timelines), hardware counters
  (occupancy, cache hit rate, HBM bandwidth, memory- or compute-bound), Advanced Thread Trace,
  PC sampling, process attach, MPI, and rocpd databases. Use when the user asks to profile or
  trace HIP, Triton, OpenMP, or PyTorch GPU code with rocprofv3 or rocprof, collect performance
  counters, or query a rocprofv3 results.db. Prefer the dedicated PC sampling or thread-trace
  skill when installed. Not for PyTorch profiler JSON traces, rocprof-compute, or NVIDIA tools.
---

# rocprofv3

`rocprofv3` (ROCprofiler-SDK) runs an application and records what the GPU did. Pick the
collection mode from the question, run one focused collection, summarize it with the matching
script in this skill, and explain the result in terms of the user's code. Read the reference
file for a mode before interpreting its output.

## Pick the mode

| The user wants to know | Mode | Summarize with | Details |
| --- | --- | --- | --- |
| Where the application spends time: kernels, copies, API calls, idle GPU, a timeline | Tracing | `scripts/analyze_trace.py` | [references/tracing.md](references/tracing.md) |
| Why a kernel is slow: memory-, latency-, or compute-bound; occupancy, cache, bandwidth | Counter collection | `scripts/analyze_counters.py` | [references/counter-collection.md](references/counter-collection.md) |
| Exact per-instruction latency of one kernel dispatch; data for ROCprof Compute Viewer | Advanced Thread Trace (ATT) | thread-trace skill, or `scripts/analyze_att_stats.py` | [references/thread-trace.md](references/thread-trace.md) |
| Which instructions are hot across a long run, with stall reasons | PC sampling (beta) | PC sampling skill, or the recipe in the reference | [references/pc-sampling.md](references/pc-sampling.md) |
| An existing `*_results.db`: convert, summarize, query, merge | rocpd tools | `scripts/analyze_trace.py` or `scripts/analyze_counters.py` | [references/rocpd.md](references/rocpd.md) |

For an open-ended "make it faster" request, go in order: trace the whole run, collect counters
for the hot kernels, then PC sampling or thread trace on the worst kernel. Stop as soon as the
question is answered; each step costs runs and time.

## Prerequisites

- ROCm with `rocprofv3`, `rocprofv3-avail`, and `rocpd` in `/opt/rocm/bin`. Check with
  `rocprofv3 --version`.
- An AMD GPU visible on native Linux: `amd-smi list` or `rocminfo | grep gfx`. Containers need
  `/dev/kfd` and `/dev/dri`. WSL2 support is experimental.
- The application must run correctly without the profiler first. For tracing and counters any
  build works; for thread trace and PC sampling keep the usual optimization flags (`-O3`) and add
  `-g` (or `-gline-tables-only`) so ISA maps to source lines. An `-O0` debug build runs different
  code and is not worth profiling.
- Python 3 for the scripts (standard library only). `pandas` is needed only by
  `rocpd convert -f csv`, `rocpd summary`, and `rocpd query`.
- Run scripts by absolute path: `python3 <this-skill-dir>/scripts/<script>.py ...`.

## Command basics (every mode)

```bash
rocprofv3 [options] -d <out-dir> -o <name> -- <application> [application args]
```

- Options go before `--`; everything after it is the application command line.
- Output is a rocpd SQLite database, `<out-dir>/<name>_results.db` (default
  `./<hostname>/<pid>_results.db`). Keep this default. Direct `--output-format csv|pftrace|otf2`
  is deprecated and can omit data; convert the database afterwards. Use `-f json` only where a
  mode requires it (PC sampling, per-pass kernel replay data).
- Data is written when the process exits normally or on SIGINT/SIGTERM. `kill -9`, the OOM
  killer, or `timeout -s KILL` lose everything.
- Use a short, representative input: profiling adds overhead, counters re-run the application
  once per group, and thread trace serializes traced kernels.
- `rocprofv3 --help` lists every option for the installed version; `rocprofv3-avail` reports
  what the GPU supports (counters, PC sampling methods, SPM). `-i run.json` takes the same
  options from a JSON file (see [references/tracing.md](references/tracing.md#input-files)).

## Tracing

```bash
rocprofv3 --runtime-trace -d rocprof_trace -o run -- ./app <args>
python3 <this-skill-dir>/scripts/analyze_trace.py rocprof_trace/run_results.db
```

`--runtime-trace` covers HIP runtime API, ROCTx, RCCL, OpenMP (OMPT), kernels, memory copies,
allocations, scratch, and KFD page events, and is the right default. `--kernel-trace` alone has
the lowest overhead; `--sys-trace` adds HSA and HIP compiler APIs. `--hip-trace` traces the API
only and records no kernels. Add `--summary --summary-units usec` for a table at exit.

The script reports the GPU busy/idle breakdown, top kernels with resource usage, copies with
effective bandwidth, host API time, ROCTx ranges, the largest idle gaps, and findings:

- **Hot kernels** cover 80% of kernel time (plus any kernel at 3% or more); they are where
  optimization pays. Follow up with counter collection on them.
- **GPU idle at 10% or more** of the activity window: the GPU waits on the host. Check what
  precedes the big gaps (synchronization, host work, copies).
- **Launch-bound kernels** (100+ calls averaging under 10 us): fuse, batch, or use HIP graphs.
- **Slow H2D/D2H copies** (well under 10 GB/s on multi-MiB transfers): pageable host memory.
  Use `hipHostMalloc` and async copies on streams; on MI325X this took a 64 MiB copy from 1.6 to
  21 GB/s.
- **Scratch above 0**: register spills or stack arrays.
- Time inside `hipMemcpy` or `hipDeviceSynchronize` is the host waiting for the GPU, not API
  overhead. The first call of each kind (first `hipMalloc`, first `hipMemcpy`, first launch of
  a kernel) includes lazy initialization; judge steady state from repeated calls.

Timeline: `rocpd convert -i rocprof_trace/run_results.db -f pftrace -d rocprof_trace -o run`, then
open `run_results.pftrace` at <https://ui.perfetto.dev> (`-f otf2` for traces over about 10 GB).

## Counter collection

Check support first; `-d` comes before the subcommand:

```bash
rocprofv3-avail -d 0 list --pmc
rocprofv3-avail -d 0 pmc-check SQ_WAVES GRBM_COUNT TCC_HIT_sum TCC_MISS_sum
```

Default recipe for AMD Instinct (gfx90a, gfx942, gfx950), every group checked on gfx942. Each
`--pmc` group is one full run of the application:

```bash
rocprofv3 --kernel-include-regex '<kernel regex>' \
  --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES SQ_BUSY_CYCLES SQ_WAVE_CYCLES TCC_HIT_sum TCC_MISS_sum \
  --pmc FETCH_SIZE \
  --pmc WRITE_SIZE \
  --pmc SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS SQ_INSTS_VMEM_RD SQ_INSTS_VMEM_WR \
  --pmc VALUBusy \
  --pmc MeanOccupancyPerActiveCU OccupancyPercent \
  --pmc MemUnitStalled LDSBankConflict \
  -d rocprof_pmc -o run -- ./app <args>
python3 <this-skill-dir>/scripts/analyze_counters.py rocprof_pmc
```

- Add `--pmc MfmaUtil` for matrix kernels. On Radeon (RDNA3/RDNA4) swap `TCC_HIT_sum TCC_MISS_sum`
  for `GL2C_HIT_sum GL2C_MISS_sum`, drop `SQ_INSTS_VMEM_*`, `MemUnitStalled`, and `MfmaUtil`, and
  set the performance level to `STABLE_STD` first (`sudo amd-smi set --perf-level STABLE_STD`).
- `FETCH_SIZE` and `WRITE_SIZE` never fit in one group. A group that does not fit fails with
  "Request exceeds the capabilities of the hardware"; split it.
- The workload must be deterministic across runs. Otherwise use
  `--replay-mode kernel --kernel-replay-beta-enabled` (beta) to replay each dispatch in one run.
- The script merges the `pass_N/` databases and reports L2 hit rate, DRAM bandwidth (and % of
  peak on known Instinct GPUs, or `--peak-hbm-tbps`), instructions per wave, DRAM bytes per memory
  instruction, and flags.

Reading it:

- DRAM bandwidth at 70% of peak or more: memory-bandwidth-bound; move fewer bytes.
- DRAM bytes per VMEM wave-instruction above wave size x 16 B (1024 B on wave64): strided or
  scattered access over-fetches. The coalesced ideal for 4-byte elements on wave64 is 256 B.
- L2 hit rate under 50%: working set exceeds L2 or poor locality (streaming kernels miss by design).
- `OccupancyPercent` under 25% with low bandwidth: latency-bound; cut VGPR/LDS use or launch more work.
- `VALUBusy` or `MfmaUtil` high with low bandwidth: compute-bound.
- `GRBM_GUI_ACTIVE` equals `GRBM_COUNT` per dispatch in counter mode, so it is not GPU
  utilization. Some SDK percentages exceed 100% on multi-XCD GPUs (for example `VALUBusy` on
  MI300); compare them between kernels.

## Advanced Thread Trace (ATT)

If the dedicated thread-trace skill from rocprofiler-sdk is installed, use it for ATT capture and
analysis; it goes deeper (wave states, hidden cost, barriers) through the rocprof-trace-decoder
Python API. Otherwise use the steps below.

```bash
rocprofv3 --att --kernel-include-regex '<kernel regex>' --kernel-iteration-range 2 \
  -d rocprof_att -o run -- ./app <args>
python3 <this-skill-dir>/scripts/analyze_att_stats.py rocprof_att
```

- Needs ROCm 10.0 or later and the ROCprof Trace Decoder
  (`/opt/rocm/lib/librocprof-trace-decoder.so`). Full support on gfx90a, gfx942, gfx950; trace
  only on RDNA2 to RDNA4.
- Always filter to one kernel. `--kernel-iteration-range 2` skips the warm-up instance (counting
  from 1); by default only the first instance of each matching kernel is traced.
- Only waves on one CU per traced shader engine are timed (`--att-target-cu 1`,
  `--att-shader-engine-mask 0x1`), so results are a sample. If every hitcount is 0, pick another
  CU, widen the mask, or launch more workgroups.
- Output per dispatch: `stats_*.csv` (per-instruction totals; what the script reads), a
  `ui_output_*` directory with the decoded trace as JSON (per-wave instruction timelines and
  wave states, ISA with per-line totals, occupancy over time) that ROCprof Compute Viewer opens,
  the raw `.att` stream, and the code objects.
- Decoder warnings such as "Data Lost" do not make a trace unusable: the rest is decoded, but the
  totals miss the dropped part. Report the warning as a caveat.

The script ranks instructions and source lines by latency and breaks latency down by class.
Global memory (VMEM issue stalls plus `s_waitcnt vmcnt` waits) dominating means the kernel waits
on memory. A `vmcnt` wait in a hot loop averaging well under 100 cycles per execution is already
satisfied, so the loop body is the limiter (the script flags this). Idle on `s_endpgm` is
waiting for stores to drain.

## PC sampling (beta)

Statistical sampling of wave program counters across the whole run, with stall reasons on
MI300 and later. Check the supported configuration first and match it exactly:

```bash
rocprofv3-avail info --pc-sampling
rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method stochastic --pc-sampling-unit cycles \
  --pc-sampling-interval 1048576 --kernel-trace --output-format json -d rocprof_pcs -o run -- ./app
```

Use `stochastic` (power-of-two cycle intervals) on gfx942/gfx950 and `host_trap` with
`--pc-sampling-unit time --pc-sampling-interval 1000` (microseconds) on MI200. PC sampling needs
`--output-format json` (or `csv`). Analyze the JSON with the PC sampling skill's
`analyze_pc_sampling.py` when it is installed; otherwise use the recipe and stall-reason guide in
[references/pc-sampling.md](references/pc-sampling.md).

## Other ways to run

| Need | How | Details |
| --- | --- | --- |
| Python or PyTorch | `rocprofv3 --runtime-trace -d out -o run -- python3 train.py` | [tracing](references/tracing.md#common-variants) |
| MPI or Slurm | `mpirun -n 4 rocprofv3 ... -o run_%rank% -- ./app`; `--profile-mpi-ranks 0-3` | [tracing](references/tracing.md#mpi-and-multi-process-runs) |
| Running process | start it with `ROCP_TOOL_ATTACH=1`, then `rocprofv3 --attach <PID> --attach-duration-msec 5000 --attach-sync-output ...` | [tracing](references/tracing.md#attaching-to-a-running-process) |
| Only part of a run | `--selected-regions` with `roctxProfilerResume(0)`/`roctxProfilerPause(0)`, or `--collection-period delay:duration:repeat` | [tracing](references/tracing.md#profiling-control) |
| OpenMP offload | `--runtime-trace` (OMPT needs LLVM `libomp`) | [tracing](references/tracing.md#openmp-hip-graphs-and-other-runtimes) |
| Counters over time inside a kernel | `--spm-beta-enabled --spm <counters>` (beta) | [counters](references/counter-collection.md#spm-sampling-beta) |
| Name kernels by ROCTx range | `--marker-trace --kernel-rename` | [tracing](references/tracing.md#kernel-naming) |
| Migrating from rocprof/rocprofv2 | flag mapping | [tracing](references/tracing.md#legacy-rocprof-to-rocprofv3) |

## Reporting results

Use the same structure for every mode so results are comparable between runs:

1. **What was measured**: the exact `rocprofv3` command, GPU (product and gfx target), input
   size, and output path.
2. **Headline**: one sentence naming the main limiter, for example "strided_copy is
   memory-bound: it reads 2 GiB from DRAM to copy 64 MiB".
3. **Findings**, most important first. Each one gets: the evidence (two or three numbers with
   units, copied from the script output), the cause in the user's code (kernel, file:line, or
   API call), the change to make, and the expected effect.
4. **Next step**: the measurement that would confirm a finding or go one level deeper (counters
   after a trace, thread trace or PC sampling after counters, or a before/after rerun).
5. **Caveats**: decoder or profiler warnings, small sample counts, one-time setup costs, and
   anything that could not be measured on this machine.

Never report numbers that did not come from a run. If a mode cannot run here, say so and
explain what it would show. If nothing stands out, say that instead of inventing a problem.

## Procedure checklist

1. Confirm `rocprofv3 --version` and a visible GPU; for counters, ATT, or PC sampling also check
   support with `rocprofv3-avail`.
2. Pick the mode from the table and read its reference file.
3. Build the application (with `-g` for ATT and PC sampling) and choose a short input.
4. Run the collection with `-d <dir> -o <name>`; confirm the output files exist.
5. Summarize with the matching script.
6. Report with the structure in [Reporting results](#reporting-results), and offer the next
   step (deeper mode, Perfetto timeline, or a before/after comparison).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No tracing options were enabled` | Add a tracing flag such as `--runtime-trace` before `--`. |
| No output files | The app crashed or was killed with SIGKILL; check its exit code and stderr. |
| `Request exceeds the capabilities of the hardware to collect` | Split the `--pmc` group; verify with `rocprofv3-avail -d 0 pmc-check`. |
| `Invalid counter name` | Not defined on this GPU; list with `rocprofv3-avail -d 0 list --pmc`. |
| Kernel filter matched nothing | The regex is searched in the demangled name; check names in a kernel trace. |
| `rocprof-trace-decoder library path not found` | Install the decoder or pass `--att-library-path <dir>`. |
| PC sampling options rejected | All of `--pc-sampling-beta-enabled`, method, unit, and interval are required and must match `rocprofv3-avail info --pc-sampling`. |
| `ImportError: pandas` from `rocpd` | `python3 -m pip install pandas`, or use the scripts and `-f pftrace`, which do not need it. |
| `rocprofv3 --attach` returns status 2 | Use the application's own PID and start it with `ROCP_TOOL_ATTACH=1`. |
| Application's own signal handlers never run | Add `--disable-signal-handlers`. |
| Wrong ROCm used | `--rocm-root /opt/rocm-X.Y.Z`. |
