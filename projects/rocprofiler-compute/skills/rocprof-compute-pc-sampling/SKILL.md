---
name: rocprof-compute-pc-sampling
description: Collects and analyzes rocprof-compute PC sampling to show which individual instructions in an AMD GPU kernel are hot and why wavefronts stall there, with ISA and source line attribution. Use when the user asks which instruction or source line is slow, asks about stall reasons, wants instruction-level or ISA-level detail, or wants hotspots inside a single kernel. Not for CUDA tools, Windows, or whole-application timing.
---

# PC sampling

PC sampling answers a question counters cannot: which instruction inside the
kernel is hot, and what the wavefront was waiting on. Use it after counters
have already identified the kernel worth this level of detail.

PC sampling is experimental. Every command below needs `--experimental`, and
option names can change between releases, so run
`rocprof-compute profile --experimental --help` before choosing flags.

## 1. Check what the hardware supports

```bash
rocprofv3-avail info --pc-sampling
```

This reports the sampling methods the installed device accepts and the valid
interval range. Two methods exist:

- `stochastic` samples in hardware and is the only method that records a stall
  reason. It requires MI300 or newer.
- `host_trap` works more widely but records no stall reason, and its samples
  can land a little past the instruction responsible.

`--pc-sampling-method` defaults to `stochastic`. Fall back to `host_trap` only
when the device rejects stochastic.

Never hardcode an interval. Stochastic intervals are counted in cycles and must
be a power of two; host-trap intervals are microseconds. The accepted range
comes from the device, so take it from the command above.

## 2. Build for source attribution

Samples always map to the kernel's assembly. Mapping assembly back to a source
line requires debug info in the application, for example `hipcc -g`. Without
it, `source_line` reads `N/A` and the `Source` column in CSV output is empty.
Keep optimization flags on; profiling a `-O0` build measures the wrong program.

Ask about the build before collecting, because rebuilding afterwards means
re-profiling.

## 3. Collect

```bash
rocprof-compute profile \
    --experimental \
    --pc-sampling \
    --pc-sampling-method stochastic \
    --pc-sampling-interval <interval-from-the-query> \
    --name <workload_name> \
    -- <application> <args>
```

PC sampling does not collect counters. A PC-sampling-only profile cannot be
combined with the framework tracing options, and it will not populate the
counter panels. When the user needs both, collect two workloads.

## 4. Analyze

```bash
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> --list-stats

rocprof-compute analyze \
    --path ./workloads/<name>/<gpu_model> \
    -k <kernel_id> \
    -b 21 \
    --pc-sampling-sorting-type count \
    --pc-sampling-rows 20
```

`-b 21` prints the PC sampling table on the terminal after Speed-of-Light.
Without it the table is only in `report.txt` in the run folder.

Sorting defaults to `count`, which puts the hottest instruction first. Use
`offset` to read the kernel in program order instead. `--pc-sampling-rows`
defaults to 10; pass `0` for every row.

Read the table as:

| Column | Meaning |
|---|---|
| PC offset | where the instruction sits in the kernel's ISA |
| Sample count | how often execution was there |
| Stall reason | what the wavefront was waiting on, stochastic sampling only |

A high sample count marks a hot instruction, not a defective one. A loop body
is supposed to be hot. What matters is a hot instruction whose stall reason
says it is waiting: a memory stall on a load means the kernel is latency-bound
there, and the fix is usually coalescing, prefetching, or more independent work
between the load and its use.

Stall reasons and what each one implies are listed in
[pc_sampling.rst](../../docs/how-to/pc_sampling.rst). Do not interpret a stall
reason from host-trap data; it does not record one.

## 5. Export annotated disassembly

```bash
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -o pc_report
```

Every analyze run writes its output into a new run folder, here `./pc_report`.
The workload folder `<n>-<workload>/` holds `pc-sampling-summary.csv`, one row
per sampled kernel. Each sampled kernel has its own folder
`kernel-NNN-<name>/` with `kernel-NNN-<name>-pc-sampling-isa.csv`, the
disassembly with sample counts on each instruction line, and
`kernel-NNN-<name>-pc-sampling-src.csv`, the sample counts of each source line.
`kernels.csv` maps each folder to its kernel. `analysis.db` in the same folder
stores the same content. The terminal shows the top stats, system info, and
Speed-of-Light; the full report, PC sampling included, is `report.txt` in the
same folder.

Start with the `-pc-sampling-src.csv` of the hottest kernel to find the hottest
source line.

## 6. When there are too few samples

A short kernel may not accumulate enough samples to be meaningful. Increase the
work the kernel does, or shorten the interval within the range the device
accepts, rather than drawing conclusions from a handful of samples.

For multi-process workloads, keep each process or rank separate; see the
multi-process sections of
[pc_sampling.rst](../../docs/how-to/pc_sampling.rst).
