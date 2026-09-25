---
name: memory
description: Analyzes the AMD GPU memory hierarchy with rocprof-compute, covering the Memory Chart, vL1D and L2 caches, LDS bank conflicts, Infinity Fabric traffic, and the experimental gfx950 memory bandwidth analysis. Use when the user asks about memory bandwidth, cache hit rates, coalescing, LDS conflicts, data locality, HBM traffic, or says a kernel is memory-bound. Not for CUDA tools, Windows, host memory profiling, or system-wide tracing.
---

# Analyze the AMD GPU memory hierarchy

Use this after Speed-of-Light points at memory, or when the user asks about
bandwidth, caches, or data movement directly.

Run `rocprof-compute analyze --help` before choosing flags.
Never use the GUI or TUI.

## 1. Start with the Memory Chart

```bash
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> --list-stats
rocprof-compute analyze --path ./workloads/<name>/<gpu_model> -k <kernel_id> -b memchart
```

The Memory Chart shows the whole hierarchy at once: requests issued, what each
cache level served, and what reached HBM. Read it before opening any single
cache level, because it tells you which level to open.

Always analyze one kernel at a time with `-k`.

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
workload actually has with `--list-available-metrics`, and the aliases for an
architecture with `rocprof-compute --list-blocks <arch>`.

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
for GL0, GL1, and GL2.

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
It is experimental and currently shipped for gfx950 only. It needs block 30 at
both profile and analyze time:

```bash
rocprof-compute profile --experimental --membw-analysis --name <name> -- <application>

rocprof-compute analyze --path ./workloads/<name>/<gpu_model> \
    --experimental --membw-analysis -k <kernel_id>
```

Its cutoffs and guidance text are data, not something to restate from memory.
They live in `src/membw_analysis/tree_spec/gfx950_membw_tree_spec.yaml` and
`src/membw_analysis/tree_spec/gfx950_membw_guidance.yaml`. Read those files
when you need to explain a verdict, and report the guidance the tool produced
rather than substituting your own numbers.

On any other architecture, use the Memory Chart path above.

## 5. Normalization

For a bandwidth question, `-n per_second` is meaningful and `per_cycle` is not.
`-n` applies to the entire report, so switch it only when the user is asking
about bandwidth alone. Leave it at `per_kernel` for a mixed report.
