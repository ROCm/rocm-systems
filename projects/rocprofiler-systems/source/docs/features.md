# Features

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

## Overview

[Omnitrace](https://github.com/AMDResearch/omnitrace) is designed to be highly extensible. Internally, it leverages the
[timemory performance analysis toolkit](https://github.com/NERSC/timemory) to
manage extensions, resources, data, etc.

### Data Collection Modes

- Dynamic instrumentation
  - Runtime instrumentation
    - Instrument executable and shared libraries at runtime
  - Binary rewriting
    - Generate a new executable and/or library with instrumentation built-in
- Statistical sampling
  - Periodic software interrupts per-thread
- Background thread sampling
  - Record process and system-level values while an application executes
- Critical trace generation

### Data Analysis

- Critical trace generation (beta)
- Support for

### Parallelism API Support

- Built-in MPI support
- Kokkos-Tools support

### GPU Metrics

- HIP API tracing
- ROCM HSA API tracing
- Kernel runtime tracing
- System-level sampling (via rocm-smi)
  - Memory usage
  - Power usage
  - Temperature
  - Utilization

### CPU Metrics

- CPU hardware counters sampling and profiles
- CPU frequency sampling
- Various timing metrics
  - Wall time
  - CPU time (process and/or thread)
  - CPU utilization (process and/or thread)
  - User CPU time
  - Kernel CPU time
- Various memory metrics
  - High-water mark (sampling and profiles)
  - Memory page allocation
  - Virtual memory usage
- Network statistics
- I/O metrics
- ... many more

### Third-party API support

- OpenMP-Tools (OMPT)
- TAU
- LIKWID
- Caliper
- CrayPAT
- VTune
- NVTX
- ROCTX
