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
it, the `Source` column in CSV output is empty.
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
rocprof-compute analyze --path ./workloads/<name>/<gpu_model>
```

The PC sampling table shows one row per kernel, hottest first, so it picks the
kernel to look at. `--pc-sampling-rows` defaults to 10 kernels; pass `0` for
every kernel.

Read the table as:

| Column | Meaning |
|---|---|
| Sample count | how often execution was in the kernel |
| Stall reason | what the kernel's wavefronts were waiting on, stochastic sampling only |

For the hot instructions of that kernel, export the annotated disassembly
(step 5) and open the kernel's ISA CSV.

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
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> \
    --output-format csv --output-name pc_report
```

CSV output writes per-kernel annotated disassembly into a
`per_kernel_pc_sampling/` directory, with sample counts and source lines beside
each instruction. `--output-format db` stores the same content in the analysis
database. Both disable terminal output.

CSV output also writes `<file>_pc_sampling.csv` beside each captured source
file, with the sample counts of every source line and the ISA rows each line
compiled to. Start there to find the hottest source line.

## 6. When there are too few samples

A short kernel may not accumulate enough samples to be meaningful. Increase the
work the kernel does, or shorten the interval within the range the device
accepts, rather than drawing conclusions from a handful of samples.

For multi-process workloads, keep each process or rank separate; see the
multi-process sections of
[pc_sampling.rst](../../docs/how-to/pc_sampling.rst).
