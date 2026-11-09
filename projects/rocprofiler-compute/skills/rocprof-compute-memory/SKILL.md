---
name: rocprof-compute-memory
description: Analyzes the AMD GPU memory hierarchy with rocprof-compute, covering the Memory Chart, vL1D and L2 caches, LDS bank conflicts, Infinity Fabric traffic, and the experimental gfx950 memory bandwidth analysis. Use when the user asks about memory bandwidth, cache hit rates, coalescing, LDS conflicts, data locality, HBM traffic, or says a kernel is memory-bound. Not for CUDA tools, Nsight, Windows, host-memory leaks, GPU out-of-memory allocation failures, or system-wide tracing.
---

# Analyze the AMD GPU memory hierarchy

Use this after Speed-of-Light points at memory, or when the user asks about
bandwidth, caches, or data movement directly.

Run `rocprof-compute analyze --help` before choosing flags.

## 1. Start with the Memory Chart

```bash
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> --list-stats
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k <kernel_id> -b memchart
```

The Memory Chart shows the hierarchy at once: requests issued and what each
instrumented cache level served. On CDNA GPUs it also reports HBM traffic where
the required metrics are available. Read it before opening any single cache
level, because it tells you which level to open.

Always analyze one kernel at a time with `-k`.

If there is no workload yet, collect only the memory blocks:

```bash
rocprof-compute profile --name <workload_name> -b memchart vl1d l2 lds -- <application> <args>
```

The cache aliases differ by architecture; for example, some GPUs name their
caches `gl0` and `gl2`.

### Navi48 (gfx1201)

Navi48 GPU metrics and the Memory Chart are available on Linux starting with
ROCm Compute Profiler 3.11 (ROCm 10.3). ROCm 10.2 and 10.1 shipped without
gfx1201 support. Apply the `STABLE_STD` prerequisite in the
[Navi48 guide](../../docs/conceptual/rdna/navi48.rst) before profiling and compare
baselines in the same performance state.

Collect `memchart` first. Add `gl0`, `gl1`, `gl2`, or `wgp` only when the chart
points at that level; LDS metrics belong to `wgp` on Navi48. Discover aliases with
`rocprof-compute --list-metrics-gfx-id gfx1201`.

GL1 is a write-combining stage. Read its utilization and stalls; do not invent a
hit rate when the report prints `N/A`. EA to DRAM reports
`DRAM Read Bandwidth` and `DRAM Write Bandwidth`, including
traffic served or absorbed by Infinity Cache. Those values do not measure bytes
reaching physical GDDR6 and must not be compared against a GDDR6 peak.
`Infinity Cache Hit Rate`, `GDDR6 Read Bandwidth`, and `GDDR6 Write Bandwidth`
remain `N/A`; treat unavailable values as unknown, not zero.

The `GL2 Cache Bandwidth` table reports size-weighted requests at GL2 output
FIFOs. Reads cover uncompressed categories and writes include recompressed
requests. Keep these rates distinct from GL2/EA interface traffic and physical
GDDR6 bandwidth. LDS byte-access rate uses load, store, and atomic dword counters;
its hardware normalization remains provisional.

Treat the WGP occupancy field as provisional. Navi48 living-wave counter units,
instance reduction, and WGP normalization still require validation, so it does
not establish an exact occupancy limit. The WGP Instruction Issue Stalls table
is omitted from the CLI while its values are unavailable. See the Navi48 guide
for the metrics whose restoration is tracked under AIPROFCOMP-907.

### Collect only what the question needs

Never run a full profile. It replays the application once per counter pass and
also runs roofline benchmarks on architectures that support them.

- List block aliases with `rocprof-compute --list-metrics-gfx-id <gfx_arch>`
  and metrics with
  `rocprof-compute --list-metrics-gfx-id <gfx_arch> --list-metrics all`.
  Pass `-b` only the block aliases, block ids, or metric ids (for example
  `2.1.0`) the question needs. Aliases differ by architecture.
- `--set <set_name>` collects a predefined topic in one pass.
  `rocprof-compute profile --list-sets` lists the sets.
- Add `-k <kernel_name>` when the user names the kernels that matter.
- Add `--no-roof` unless the question is about roofline. A `-b` list without
  block 4 (`roof`) already skips it.
- `--iteration-multiplexing kernel` spreads counters over repeated dispatches of
  each kernel, so it only helps when every profiled kernel is dispatched many
  times. Pair it with `-k`. See Iteration multiplexing in
  [profile mode](../../docs/how-to/profile/mode.rst).
- Tell the user which analysis panels the narrower profile leaves empty.

In ROCm 10.2 and 10.1, use `--list-blocks <gfx_arch>` to list blocks and
`--list-metrics <gfx_arch>` to list metrics. The architecture selector changed
in ROCm 10.3.

## 2. Open the level the chart points at

```bash
# Vector L1 data cache
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k <kernel_id> -b vl1d

# L2 cache
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k <kernel_id> -b l2

# L2 per channel, including Infinity Fabric read and write stalls
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k <kernel_id> -b l2_per_channel

# Local Data Share, including bank conflicts and unaligned stalls
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k <kernel_id> -b lds

# Address processing and data return, for coalescing behaviour
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k <kernel_id> -b tatd

# Scalar L1 data cache
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k <kernel_id> -b sl1d
```

Aliases are stable; numeric block ids are architecture-specific. List what this
workload actually has with `--list-metrics`, and the aliases for an
architecture with `rocprof-compute --list-metrics-gfx-id <arch>`.

## 3. Interpret

Each level has its own reference, with the counters, the formulas, and what
they mean:

- [Vector L1 cache](../../docs/conceptual/cdna/vector-l1-cache.rst)
- [L2 cache](../../docs/conceptual/cdna/l2-cache.rst)
- [Local Data Share](../../docs/conceptual/cdna/local-data-share.rst)
- [Infinity Fabric transactions](../../docs/tutorial/includes/infinity-fabric-transactions.rst)
- [Vector memory operation counting](../../docs/tutorial/includes/vector-memory-operation-counting.rst)
- [LDS examples](../../docs/tutorial/includes/lds-examples.rst)

Client APUs use a different hierarchy; see the
[RDNA performance model](../../docs/conceptual/rdna/rdna-performance-model.rst)
for GL0, GL1, and GL2. For Navi48's RDNA4 hierarchy and traffic boundaries, use
the [Navi48 guide](../../docs/conceptual/rdna/navi48.rst).

Read hit rates and utilizations against the `Percent of Peak` and `Peak`
columns the report prints. Do not apply a remembered cutoff such as "L2 hit
rate below 50% is bad"; a streaming kernel with no reuse is supposed to miss.
Judge a rate against what the algorithm should do, the neighbouring kernels,
and a known-good baseline.

What usually follows from what:

| Reading | Usual meaning |
|---|---|
| High request count, low cache hits | poor locality or no reuse to exploit |
| Many requests per instruction in TA/TD | uncoalesced access |
| Sustained LDS bank conflicts | padding or access reordering needed |
| High fabric read or write stalls | traffic exceeds what the fabric can absorb |
| Near-peak HBM with low cache hits | reduce traffic or add reuse; more bandwidth is not available |

## 4. Memory bandwidth analysis (experimental, gfx950)

A guided breakdown that walks the bandwidth tree and names the limiting level.
It is experimental and currently shipped for gfx950 only; see
[memory bandwidth analysis](../../docs/how-to/membw_analysis.rst). It needs
only the memory chart (block 3) and block 30, so profile only those two blocks:

```bash
rocprof-compute profile --experimental --membw-analysis -b 3 30 \
    --name <name> -- <application>

rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k <kernel_id>
```

Report the guidance the tool produced rather than substituting your own numbers.

On any other architecture, use the Memory Chart path above.

## 5. Normalization

For a bandwidth question, `-n per_second` is meaningful and `per_cycle` is not.
`-n` applies to the entire report, so switch it only when the user is asking
about bandwidth alone. Leave it at `per_kernel` for a mixed report.
