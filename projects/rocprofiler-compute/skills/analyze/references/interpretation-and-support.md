# Analysis interpretation and support

Load this reference after the recommended top-down workflow identifies the
likely bottleneck, or when selecting hardware-specific metrics.

## Support matrix and dependencies

| Area | Supported scope | Validation source |
|---|---|---|
| OS | Linux distributions supported by the matching ROCm release | ROCm system requirements |
| GPUs | `gfx908`, `gfx90a`, `gfx940`, `gfx941`, `gfx942`, `gfx950`, `gfx1150`–`gfx1153`, `gfx1250` | Shipped analysis configs |
| Inputs | Workloads from the matching `rocprof-compute profile` version | Analyze integration tests |
| Reports | Terminal, text, per-view CSV, SQLite analysis database | Analyze unit/integration tests |
| APIs | HIP kernels, ROCTX ranges, rocpd data, MPI/multi-process identity | Parser/profile tests |
| Frameworks | Kernel analysis for ROCm frameworks; PyTorch 2.13/2.14 and Triton attribution are experimental | Framework-trace tests |
| PC sampling | Hardware/configuration dependent and experimental | PC-sampling tests |

Use the rocprofiler-compute version paired with the installed ROCm release.
The ROCm 10.2 target is rocprofiler-compute 3.10. Profile mode supports Python
3.8 or newer; analyze mode requires Python 3.9 or newer and the dependencies
in `requirements.txt`. CSV and database reports require a workload collected
through rocpd, which is the current profile backend.

## Metric interpretation

Thresholds below are triage heuristics, not universal pass/fail criteria.
Compare against the device peak, workload intent, neighboring kernels, and a
known-good baseline.

### Roofline

- Left of the ridge: likely memory-bound. Investigate traffic, locality, reuse,
  tiling, and LDS.
- Right of the ridge: likely compute-bound. Investigate instruction mix, MFMA,
  divergence, and instruction-level parallelism.
- Far below both ceilings: inspect occupancy, launch size, and scheduler stalls.

### Memory

| Signal | Triage heuristic | Investigation |
|---|---|---|
| HBM bandwidth utilization | Above 70% can indicate bandwidth saturation | Reduce traffic or increase reuse |
| L2 hit rate | Below 50% can indicate poor reuse | Check layout and working set |
| vL1D hit rate | Below 70% can indicate weak locality | Check access order or LDS tiling |
| LDS bank conflicts | Any sustained conflict rate is suspicious | Pad or reorganize LDS accesses |

### Occupancy

Compare active waves/CU with theoretical occupancy. If both are low, inspect
grid size. If theoretical occupancy is low, inspect VGPR, SGPR, and LDS
allocation. Scratch traffic can indicate register spilling. Low occupancy is
not automatically a defect when the kernel saturates its limiting resource.

### Scheduler and instruction stalls

| Stall | Typical cause | Candidate investigation |
|---|---|---|
| VMEM | Global/scratch latency | Coalescing, prefetch, reuse, independent loads |
| LDS | Bank conflicts or dependency | Access layout and pipelining |
| Barrier | Workgroup imbalance/synchronization | Divergence and barrier frequency |
| SALU | Scalar or branch pressure | Uniformity and control flow |
| VALU idle | Insufficient vector work | Launch size, dependencies, instruction mix |

For GEMM/convolution workloads, low MFMA utilization plus high VMEM stalls
usually means the matrix pipeline is data-starved. Do not apply an arbitrary
MFMA target to kernels that are not intended to use matrix instructions.

## PC sampling

Start with hottest samples:

```bash
rocprof-compute analyze \
    --path <workload> \
    --pc-sampling-sorting-type count \
    --pc-sampling-rows 10 \
    -k <kernel_id>
```

A high sample count identifies a hot instruction, not necessarily a defect.
Interpret its stall reason and surrounding ISA/source context. Host-trap
sampling can skid to a nearby instruction. Very short workloads may not yield
enough samples.

CSV output writes per-kernel annotated disassembly below
`per_kernel_pc_sampling/`; database output stores equivalent views.

## ROCTX and multi-process analysis

`--list-stats` shows markers and dispatches. Select the 1-based dispatch IDs
inside the desired range and analyze with `--dispatch`. For multi-process or
MPI workloads, retain process/rank identity and compare like-for-like ranges;
do not aggregate away load imbalance before inspection.

## Known issues and safeguards

- Block IDs are architecture-specific. Resolve them with
  `--list-available-metrics` instead of assuming a numeric ID.
- A missing panel usually means its counters were not collected. Re-profile
  with the needed block or with the recommended full collection.
- Zero metrics may be caused by a tiny kernel, unsupported counter, or missing
  dispatch—not necessarily zero hardware activity.
- Do not compare runs from different GPU/partition/clock configurations
  without calling out those differences.
- Do not normalize a metric to a physically meaningless unit.
- Experimental PC-sampling and framework-trace behavior must be checked
  against the installed release.

## Excluded workflows

This skill does not cover Windows, offline/air-gapped setup, GUIs/IDEs, CUDA
tools, or generating/replacing kernel source code.
