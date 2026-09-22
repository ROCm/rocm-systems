# Introduction

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

This documentation was created to provide a detailed breakdown of all facets of Omniperf. In addition to a full deployment guide with installation instructions, we also explain the design of the tool and each of its components. If you are new to Omniperf, these chapters can be followed in order to gradually acquaint you with the tool and progressively introduce its more advanced features.

This project is proudly open source, and we welcome all feedback! For more details on how to contribute, please see our Contribution Guide.

[Browse Omniperf source code on Github](https://github.com/ROCm/omniperf)

## What is Omniperf

Omniperf is a kernel level profiling tool for Machine Learning/HPC workloads running on AMD Instinct (tm) MI accelerators. AMD's Instinct (tm) MI accelerators are Data Center GPUs designed for compute and with some graphics functions disabled or removed. Omniperf is currently built on top of [rocProf](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/rocprof.html) to monitor hardware performance counters. The Omniperf tool primarily targets accelerators in the MI100, MI200, and MI300 families. Development is in progress to support Radeon (tm) RDNA (tm) GPUs.

## Features

The Omniperf tool performs profiling based on all available hardware counters for the target accelerator. It provides high level performance analysis features including System Speed-of-Light, Hardware block level Speed-of-Light, Memory Chart Analysis, Roofline Analysis, Baseline Comparisons, and more...

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
- Roofline Analysis Panel (_Supported on MI200 only, Ubuntu 20.04, SLES 15 SP3 or RHEL8_)
- Command Processor (CP) Panel
- Workgroup Manager (SPI) Panel
- Wavefront Launch Panel
- Compute Unit - Instruction Mix Panel
- Compute Unit - Pipeline Panel
- Local Data Share (LDS) Panel
- Instruction Cache Panel
- Scalar L1D Cache Panel
- L1 Address Processing Unit, a.k.a. Texture Addresser (TA) / L1 Backend Data Processing Unit, a.k.a. Texture Data (TD) panel(s)
- Vector L1D Cache Panel
- L2 Cache Panel
- L2 Cache (per-Channel) Panel

## Compatible SoCs

| Platform          | Status     |
| :---------------- | :--------- |
| Vega 20 (MI50/60) | No support |
| MI100             | Supported  |
| MI200             | Supported  |
| MI300             | Supported  |
