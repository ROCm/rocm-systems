# RCCL kernel-timing performance tooling

Kernel timings come from RCCL itself: every dispatch carries a completion signal
whose hardware start/end ticks RCCL converts to CLOCK_BOOTTIME nanoseconds and
buffers per communicator. rccl-tests drains that buffer and writes a CSV. No
profiler is involved, so the numbers are the kernel's own, and they merge with
an external rocprof trace without calibration because the clock domain is the
same.

rocprofv3 is still supported as an alternative source (`--source profiled`), but
it is no longer the default anywhere -- and on the toolchain this was developed
against it does not work at all. srock (ROCm 10.0.0) ships no rocprofv3, and the
system one (1.1.0, ROCm 7.2.1) is too old to trace dispatches against that
runtime. It still intercepts rocTX, so a profiled run "succeeds" and produces
marker traces with no kernels whatsoever. Both the runner and `analyze.py
overhead` now call that out instead of letting it read as a zero-cost profile;
do not trust a profiled column without checking for that note.

## Instrumentation

### RCCL (`RCCL_KERNEL_TIMING=1`)

- Off by default; when off, RCCL launches exactly as it always did.
- Covers every kernel RCCL dispatches: the planned collectives and the
  symmetric kernels through `ncclLaunchKernelTimed`, and the standalone
  launches -- all eight DDA IPC/fabric backends, the hierarchical all-gather
  shuffle, and the one-rank reduction -- through `ncclKernelTimingLaunch`.
  This matters on an 8-GPU node, where the DDA fast path, not the planned path,
  is what actually runs.
- `RCCL_KERNEL_TIMING_LAYOUT`: `none` disables, `1`/`2`/`3` pins a known HIP
  event layout. Unset (the default) probes for the layout at the first dispatch.
- Records carry rank, per-communicator dispatch sequence number, comm hash,
  func, datatype, count, nChannels, nThreads and nColls. The sequence number is
  incremented for every dispatch whether or not it ends up timed, so a gap in
  the sequence means a dispatch went unmeasured rather than unrecorded.

### rccl-tests (`RCCL_TESTS_KERNEL_TIMING=<prefix>`)

- Drains the records at each configuration change and tags them with the same
  message the rocTX marker carries (size, count, type, op, in_place, proc,
  thread, ngpus, graph), plus `phase=setup` for warmup and verification.
- Writes `<prefix>_pid<pid>.csv` at exit, one file per process. Reports the
  dropped-record count, and says so explicitly when it drained nothing.
- rocTX markers (`RCCL_TESTS_ROCTX=1`) are unchanged and still drive the
  rocprofv3 path.

## Tools

Everything goes through `analyze.py`; the `roctx_*.py` modules are the
implementation. Each subcommand takes `--source
{auto,dispatch,baseline,profiled,log}` and `auto` prefers a dispatch trace.

| subcommand | what it answers |
| --- | --- |
| `list` | which runs exist |
| `report` | per-size kernel duration, algbw/busbw, outliers |
| `variance` | per-launch jitter, inter-launch gaps, rank skew |
| `overhead` | what the instrumentation costs against an uninstrumented baseline |
| `compare` | two runs side by side, with bandwidth deltas |
| `plot` | overlay plot (matplotlib or interactive plotly) |
| `export` | tidy-records JSON for external analysis |

`variance` is what the dispatch trace makes possible and rocprof did not: it
reports per-launch on-GPU duration spread, the gap between one dispatch ending
and the next starting on the same rank, the share of wall time spent outside a
kernel, and how far apart the ranks started the same dispatch.

### `roctx_perf_run.py` -- the runner

```
python3 tools/roctx_perf_run.py --test all_reduce --mode dispatch --baseline
python3 tools/roctx_perf_run.py --test all_reduce --mode dispatch,profiled --baseline --repeats 3
```

- `--mode dispatch` (default) sets `RCCL_KERNEL_TIMING=1` and no profiler;
  `--mode profiled` runs rocprofv3 as before; both may be given. A profiled rep
  that ends with no kernel trace is reported as such at the end of the rep.
- `--baseline` adds an uninstrumented run of the same sweep, which is what
  `analyze.py overhead` prices the instrumented runs against.
- `--kernel-timing-layout N` pins the event layout for dispatch runs.
- Auto-detects GPU count, resolves `mpirun` from the build, records
  `metadata.json` (command, environment, rocm-smi, git status, ldd, and
  librccl provenance: path, md5, version, ROCm/HIP build IDs, git hash).
- Stages into `.tmp-*` and renames on clean exit.

## What the instrumentation costs

Measured on 8x gfx950, `-b 8 -e 1G -f 2 -w 20 -n 100`, three repeats, against
the uninstrumented baseline in the same run:

- Free where the timed kernel is dispatched back to back: all-gather is +1.5%
  median (about 0.2us), and all-reduce is +2% at small sizes and 0.0% from 16M
  up, where the kernel dominates.
- About +3.5us per iteration wherever a device-to-device staging copy
  immediately precedes the timed kernel, because the runtime cannot batch a
  signalled dispatch with the copy ahead of it. This is a fixed cost, so it
  reads as +55% on small reduce-scatter and decays to +0.7% by 64M. All-reduce
  shows it only above 256KB, exactly where the DDA path switches from the flat
  kernel to the tree kernel with its staging copy.

So the timing is close to free on the paths without a staging copy, and costs a
fixed few microseconds per iteration on the ones with it.

## Notes

- Timestamps are comparable across the GPUs of a node, which is what makes rank
  skew meaningful; they are CLOCK_BOOTTIME, so they do not survive a reboot.
- `LD_LIBRARY_PATH` must point at the `librccl.so` under test.
- The test binary must be built with MPI (`make MPI=1 MPI_HOME=...`), and the
  `mpirun` used must match the `libmpi` the binary linked against, or every
  rank silently runs as a singleton.

## TODO

1. **Multi-platform** -- run the same battery on at least two platforms.
2. **Algorithm/protocol diagnostics** -- capture algo/proto/channel decisions
   and correlate them to bandwidth inflection points.
3. **Per-test `-b` selection** -- some tests need a minimum message size > 8.
4. **Staging-copy overhead** -- decide whether the +3.5us on staged DDA paths is
   worth avoiding, e.g. by timing the copy and the kernel as one region.
