# Introduction

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

[Browse Omniperf source code on Github](https://github.com/ROCm/omniperf)

## Scope

MI Performance Profiler ([Omniperf](https://github.com/ROCm/omniperf)) is a system performance profiling tool for Machine Learning/HPC workloads running on AMD Instinct (tm) Accelerators. It is currently built on top of the [rocProfiler](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/rocprof.html) to monitor hardware performance counters. The Omniperf tool primarily targets accelerators in the MI100 and MI200 families. Development is in progress to support MI300 and Radeon (tm) RDNA (tm) GPUs.

## Features

The Omniperf tool performs system profiling based on all available hardware counters for the target accelerator. It provides high level performance analysis features including System Speed-of-Light, IP block Speed-of-Light, Memory Chart Analysis, Roofline Analysis, Baseline Comparisons, and more...

Both command line analysis and GUI analysis are supported.

Detailed Feature List:
- MI100 support
- MI200 support
- Standalone GUI Analyzer
- Grafana/MongoDB GUI Analyzer
- Dispatch Filtering
- Kernel Filtering
- GPU ID Filtering
- Baseline Comparison
- Multi-Normalizations
- System Info Panel
- System Speed-of-Light Panel
- Kernel Statistic Panel
- Memory Chart Analysis Panel
- Roofline Analysis Panel (*Supported on MI200 only, SLES 15 SP3 or RHEL8*)
- Command Processor (CP) Panel
- Shader Processing Input (SPI) Panel
- Wavefront Launch Panel
- Compute Unit - Instruction Mix Panel
- Compute Unit - Pipeline Panel
- Local Data Share (LDS) Panel
- Instruction Cache Panel
- Scalar L1D Cache Panel
- Texture Addresser and Data Panel
- Vector L1D Cache Panel
- L2 Cache Panel
- L2 Cache (per-Channel) Panel

## Compatible SOCs

| Platform | Status         |
| :------- | :------------- |
| Vega 20 (MI-50/60)  | No  |
| MI100    | Supported      |
| MI200    | Supported      |
| MI300    | In development |
