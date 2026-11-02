# Application tracing with rocprofv3

Contents:

1. [Choosing what to trace](#choosing-what-to-trace)
2. [Tracing options](#tracing-options)
3. [Reading the trace report](#reading-the-trace-report)
4. [Common variants](#common-variants)
5. [Output location and naming](#output-location-and-naming)
6. [Input files](#input-files)
7. [Profiling control](#profiling-control)
8. [MPI and multi-process runs](#mpi-and-multi-process-runs)
9. [Attaching to a running process](#attaching-to-a-running-process)
10. [OpenMP, HIP graphs, and other runtimes](#openmp-hip-graphs-and-other-runtimes)
11. [Kernel naming](#kernel-naming)
12. [Legacy rocprof to rocprofv3](#legacy-rocprof-to-rocprofv3)
13. [Troubleshooting](#troubleshooting)

Sources: `rocprofv3 --help` (rocprofiler-sdk 1.4.1, ROCm 10.2), the rocprofiler-sdk how-to
guides, and runs on an AMD Instinct MI325X (gfx942). Database views, SQL recipes, and the `rocpd`
commands are in [rocpd.md](rocpd.md).

## Choosing what to trace

| Goal | Flags |
| --- | --- |
| Overview (default) | `--runtime-trace` |
| Lowest overhead, kernels only | `--kernel-trace` |
| Kernels plus transfers | `--kernel-trace --memory-copy-trace` |
| Also HSA runtime and HIP compiler calls | `--sys-trace` (more overhead, rarely needed) |
| Drop one part of an aggregate | `--runtime-trace --scratch-memory-trace=False` |
| ROCTx ranges only | `--marker-trace` |
| Rename kernels by enclosing ROCTx range | `--marker-trace --kernel-rename` (Kokkos: `--kokkos-trace`) |
| Multi-GPU collectives | `--runtime-trace` already includes `--rccl-trace` |
| OpenMP host regions and target offload | `--runtime-trace` includes `--ompt-trace` (rocpd output only) |
| On-screen summary at exit | add `--summary --summary-units usec` |

## Tracing options

Boolean options accept an optional value, so `--kernel-trace=False` disables part of an
aggregate option.

| Option | Collects |
| --- | --- |
| `-r`, `--runtime-trace` | HIP runtime API, ROCTx, RCCL, rocDecode, rocJPEG, rocSHMEM, hipFILE, OMPT, HIP events and graphs, kernel dispatches, memory copies, allocations, scratch, KFD events |
| `-s`, `--sys-trace` | Everything in `--runtime-trace` plus HIP compiler API and HSA API |
| `--kernel-trace` | Kernel dispatches (name, timestamps, grid, workgroup, VGPR/AGPR/SGPR, LDS, scratch) |
| `--memory-copy-trace` | Async copies (direction, bytes, agents) |
| `--memory-allocation-trace` | HSA allocations and frees (hipMalloc, hipFree) |
| `--scratch-memory-trace` | Scratch allocate, free, and reclaim |
| `--hip-trace` | HIP runtime and compiler API only (no kernels or copies) |
| `--hip-runtime-trace` / `--hip-compiler-trace` | One half of `--hip-trace` |
| `--hsa-trace` (`--hsa-core-trace`, `--hsa-amd-trace`, `--hsa-image-trace`, `--hsa-finalizer-trace`) | HSA API |
| `--marker-trace` | ROCTx marks and ranges (preloads the new ROCTx library, so apps linked to the old roctracer ROCTx are captured too) |
| `--rccl-trace` | RCCL collectives |
| `--kfd-trace` | KFD page migration, page mapping, queue eviction, and dropped events |
| `--hip-event-trace` | GPU-side barriers from `hipEventRecord` / `hipStreamWaitEvent` (JSON and rocpd only) |
| `--hip-graph-trace` | One record per `hipGraphLaunch` (JSON and rocpd only) |
| `--ompt-trace [CATEGORY ...]` | OpenMP host and target events; categories `thread parallel task sync mutex target device error all` (rocpd only) |
| `--kokkos-trace` | Built-in Kokkos Tools; implies `--marker-trace --kernel-rename` |
| `--rocdecode-trace`, `--rocjpeg-trace`, `--rocshmem-trace`, `--hipfile-trace` | Those libraries' APIs |

Post-processing options (need at least one tracing option): `--stats` (per-domain stats files),
`-S`/`--summary` (table on stderr at exit), `-D`/`--summary-per-domain`,
`--summary-groups 'KERNEL_DISPATCH|MEMORY_COPY'`, `--summary-output-file FILE|stdout|stderr`,
`-u`/`--summary-units sec|msec|usec|nsec`.

Other options: `--output-config` (writes the resolved configuration and environment next to the
results), `--log-level warning|info|trace`, `--minimum-output-data KB` (skip writing tiny
outputs), `--disable-signal-handlers`, `--preload LIB ...` (sanitizers), `--rocm-root PATH`,
`-A`/`--agent-index absolute|relative|type-relative`, `--group-by-queue` (Perfetto shows HSA
queues instead of HIP streams), `--process-sync`.

## Reading the trace report

`scripts/analyze_trace.py` accepts one or more `.db` files or directories (searched
recursively), so it also handles MPI runs, `.rpdb` packages, and before/after comparisons. Use
`--top N` for longer tables and `-o report.md` to save the report.

- **GPU busy/idle**: busy is the union of kernel and copy intervals between the first and last
  GPU operation, so overlap is not double counted. Idle of 10% or more means the GPU waits on the
  host; the idle-gap table shows what precedes each gap.
- **Hot kernels**: the kernels covering 80% of kernel time, plus any kernel at 3% or more. Next
  step for each: counter collection (memory- or compute-bound?), then PC sampling or thread trace
  (which instructions?).
- **Launch-bound kernels** (100+ calls averaging under 10 us): launch overhead dominates. Fuse
  kernels, batch work per launch, or capture the sequence in a HIP graph.
- **Memory copies**: a high copy share means transfers dominate. H2D/D2H well below 10 GB/s on
  multi-MiB copies points to pageable host memory: use `hipHostMalloc` and async copies on
  separate streams. Device-to-device and peer copies run at GPU memory or link bandwidth.
- **Scratch (B) above 0**: register spills or stack arrays; check VGPR pressure.
- **Host API**: time inside `hipMemcpy`, `hipDeviceSynchronize`, or `hipStreamSynchronize` is
  the host blocked on the GPU, not API overhead. Many `hipLaunchKernel` calls with a high average
  confirm launch overhead.
- **KFD events** (page faults, migrations, prefetches): the driver moved pages (SVM/HMM, managed
  memory, XNACK). The report names the host call they overlap. Inside the first `hipMemcpy` of a
  run they can be one-time setup; re-measure with more iterations before acting on them.
- **First calls**: the first `hipMalloc`/`hipHostMalloc`, first `hipMemcpy`, and first launch of
  each kernel absorb lazy initialization. Judge steady-state cost from repeated calls.

If nothing stands out, say so and suggest the next mode rather than inventing problems.

Timeline: `rocpd convert -i run_results.db -f pftrace -d out -o run`, then open
`out/run_results.pftrace` at <https://ui.perfetto.dev>. Add `--group-by-queue` for HSA queues,
`--start 25% --end 75%` or `--start-marker`/`--end-marker` to trim. Use `-f otf2` (Vampir) for
converted traces above about 10 GB.

## Common variants

- **Python and PyTorch**: `rocprofv3 --runtime-trace -d out -o run -- python3 train.py`.
  Annotate phases with ROCTx (`import roctx`) or `torch.cuda.nvtx` ranges, which ROCm builds of
  PyTorch implement with ROCTx.
- **Only part of a run**: `--collection-period 5:2:1` or `--selected-regions` (see
  [Profiling control](#profiling-control)).
- **Comparing two runs**: run the script on each database and compare the kernel and copy
  tables, or `rocpd summary -i base.db new.db --summary-by-rank` (needs pandas).
- **Containers**: pass `--device /dev/kfd --device /dev/dri` and run `rocprofv3` inside the
  same container as the application.

## Output location and naming

- `-d DIR` / `--output-directory DIR` and `-o NAME` / `--output-file NAME`. With both, the
  database is `DIR/NAME_results.db`. The default is `./%hostname%/%pid%_results.db`.
- Both accept placeholders: `%hostname%`, `%pid%`, `%ppid%`, `%rank%` (`SLURM_PROCID` or MPI
  rank), `%size%`, `%nid%` (rank if known, else pid), `%job%` (`SLURM_JOB_ID`), `%tag%` (basename
  of the application), `%argt%`, `%launch_time%`, `%env{NAME}%`. Shorthands: `%p`, `%r`, `%j`, `%s`.
- `-f`/`--output-format` takes one or more of `rocpd` (default), `json`, `csv`, `pftrace`, `otf2`.
  Direct `csv`, `pftrace`, and `otf2` are deprecated and can omit data (hipFILE, rocSHMEM, OMPT,
  graph attribution). `json` is a custom schema for programs and does not load in Perfetto.
- Every run also records agent information (`rocpd_info_agent` in the database, or
  `*_agent_info.csv` for CSV output).

## Input files

`-i FILE` accepts JSON or YAML (YAML needs `pyyaml`; prefer JSON). Keys are the long option names
with underscores; each entry in `jobs` is one application run:

```json
{
  "jobs": [
    {
      "runtime_trace": true,
      "scratch_memory_trace": false,
      "summary": true,
      "summary_groups": ["KERNEL_DISPATCH|MEMORY_COPY"],
      "output_directory": "rocprof_trace",
      "output_file": "run"
    }
  ]
}
```

## Profiling control

- `-P`/`--collection-period START:DURATION:REPEAT` (repeat 0 means forever; several triplets
  allowed) with `--collection-period-unit hour|min|sec|msec|usec|nsec`. Not available with attach
  or multi-pass counters.
- `--selected-regions`: collection starts disabled; `roctxProfilerResume(0)` enables it and
  `roctxProfilerPause(0)` disables it again, for every requested service (traces, counters,
  thread trace, PC sampling). Use `--selected-regions-ref-count` when pause/resume pairs nest.
  Without `--selected-regions`, pause/resume hide regions from an otherwise full trace. The two
  mechanisms (`--selected-regions` and `--collection-period`) are mutually exclusive.
- ROCTx in C/C++: include `rocprofiler-sdk-roctx/roctx.h`, link `-lrocprofiler-sdk-roctx`, then
  `roctxRangePush("name")` / `roctxRangePop()` (nested, same thread),
  `roctxRangeStart`/`roctxRangeStop` (any thread), `roctxMark("name")`, and naming helpers such
  as `roctxNameOsThread` and `roctxNameHipStream`.
- ROCTx in Python: `import roctx` (installed under `/opt/rocm/lib/pythonX.Y/site-packages`; add it
  to `PYTHONPATH`), then `roctx.rangePush`, `roctx.mark`, or the
  `roctx.context_decorators.RoctxRange` context manager and decorator.

## MPI and multi-process runs

- Run `rocprofv3` inside the launcher (`mpirun -n N rocprofv3 ... -- ./app`, `srun ... rocprofv3 ...`)
  so the tool loads in each rank. Running it outside the launcher also profiles `mpirun` itself.
- Name outputs per rank: `-o run_%rank%` or `-d %h.%p.%env{OMPI_COMM_WORLD_RANK}%`.
- `--profile-mpi-ranks 0-3,8` limits output to some ranks while the tool still runs everywhere.
  Rank detection order: PBS, Slurm, PMI, MVAPICH2, Open MPI, generic `MPI_RANK`-style variables;
  override with `--mpi-world-rank-variable` and `--mpi-world-size-variable`.
- `--process-sync` makes ranks wait for each other while writing output, for workloads that tear
  down the process group early.
- Pass every rank's database (or the directory) to `scripts/analyze_trace.py`; it prints the
  spread of GPU busy share across process/GPU pairs, which exposes load imbalance.
- Child processes inherit the tool, so a launcher that spawns workers produces one database per
  process.

## Attaching to a running process

1. Start the target with `ROCP_TOOL_ATTACH=1` in its environment.
2. Make sure you may ptrace it: same user or root. `{ : < /proc/$PID/mem; } && echo ok` checks
   the permission. On Yama systems `sudo sysctl kernel.yama.ptrace_scope=0` (system-wide); in
   Docker, `--cap-add SYS_PTRACE`. Run `rocprofv3` in the same container as the target.
3. `rocprofv3 --attach <PID> --attach-duration-msec 5000 --attach-sync-output --runtime-trace -d out -o run`.
   Without `--attach-duration-msec`, press Enter or Ctrl+C to detach. `--attach-sync-output`
   waits for output before returning; without it, output is written asynchronously inside the
   target, so killing the target right away truncates it.
4. Children of the PID are attached too (`--attach-children=false` to skip them). Reattaching
   later requires the same data-collection options.

Use the application's own PID. A wrapper shell's PID fails with
`rocattach ... returned non-zero status 2` and a message that the tool library is not visible
from the target's mount namespace. Multi-pass counters and `--collection-period` do not work in
attach mode.

## OpenMP, HIP graphs, and other runtimes

- OpenMP offload: build with `amdclang++ -fopenmp --offload-arch=gfxNNN`; `--runtime-trace` (or
  `--ompt-trace parallel task target --kernel-trace --memory-copy-trace`) records OpenMP regions
  next to kernels. OMPT data exists only in the rocpd database. It needs an OMPT-capable runtime
  (LLVM `libomp` from ROCm or AOMP; GCC `libgomp` produces nothing). `OMP_NUM_THREADS=1`
  suppresses most parallel-region events.
- HIP graphs: kernels launched from a graph carry `graph_exec_id` and `graph_node_id` in the
  `kernels` view; `graph_launches` has one row per `hipGraphLaunch`. Graph memcpy nodes usually
  appear as `__amd_rocclr_copyBuffer` kernels, not copies.
- Kokkos: `--kokkos-trace` replaces templated kernel names with Kokkos labels.

## Kernel naming

- Names are demangled by default. `-M`/`--mangled-kernels` keeps mangled names; `-T`/`--truncate-kernels`
  drops template and argument lists.
- `--kernel-rename` (with `--marker-trace`) names each kernel after its enclosing
  `roctxRangePush` range, which helps with generic library kernels.
- `--kernel-include-regex` / `--kernel-exclude-regex` / `--kernel-iteration-range` filter
  counter collection and thread trace only; they do not shrink a trace.

## Legacy rocprof to rocprofv3

| rocprof / rocprofv2 | rocprofv3 |
| --- | --- |
| `--hip-trace` (included kernels and copies) | `--runtime-trace`, or `--hip-trace --kernel-trace --memory-copy-trace` |
| `--roctx-trace` | `--marker-trace` |
| `--roctx-rename` | `--kernel-rename` |
| `--basenames on` | `--truncate-kernels` |
| `--stats` (kernel stats by default) | `--stats` with a tracing option, or `--summary` |
| `--trace-period` | `--collection-period` |
| `-i input.txt` with `pmc:` lines | same, or `--pmc` on the command line |
| CSV by default | rocpd database by default; `rocpd convert -f csv` |
| `--list-basic`, `--list-derived` | `rocprofv3 -L` or `rocprofv3-avail list --pmc` |

Timing from rocprofv3 has lower overhead than rocprof and rocprofv2; single-kernel times can differ
by up to about 20%, so do not compare numbers across the tools.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No tracing options were enabled` warning | Add `--runtime-trace` or another tracing flag before `--`. |
| No `.db` file | The app crashed or was killed with SIGKILL; check its exit code and stderr. Under WSL2, see the rocprofiler-sdk WSL guide. |
| Kernels missing from the report | `--hip-trace` alone has no kernels; use `--runtime-trace` or add `--kernel-trace`. |
| Trace is huge or slow | Use `--kernel-trace`, a shorter input, `--collection-period`, or `--selected-regions`. |
| Application's own signal handler never runs | Add `--disable-signal-handlers`. |
| Wrong ROCm picked up | `--rocm-root /opt/rocm-X.Y.Z`. |
| `rocprofv3 --attach` returns status 2 | Pass the application's own PID and start the target with `ROCP_TOOL_ATTACH=1`. |
