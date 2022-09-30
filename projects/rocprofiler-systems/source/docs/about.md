# About

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

## Overview

> ***[Omnitrace](https://github.com/AMDResearch/omnitrace) is an AMD open source research project and is not supported as part of the ROCm software stack.***

[Browse Omnitrace source code on Github](https://github.com/AMDResearch/omnitrace)

[Omnitrace](https://github.com/AMDResearch/omnitrace) is designed for both high-level profiling and
comprehensive tracing of applications running on the CPU or the CPU+GPU via dynamic binary instrumentation,
call-stack sampling, and various other means for determining currently executing function and line information.

Visualization of the comprehensive omnitrace results can be viewed in any modern web browser by visiting
[ui.perfetto.dev](https://ui.perfetto.dev/) and loading the perfetto output (`.proto` files) produced by omnitrace.

Aggregated high-level results are available in text files for human consumption and JSON files for programmatic analysis.
The JSON output files are compatible with the python package [hatchet](https://github.com/hatchet/hatchet) which converts
the performance data into pandas dataframes and facilitate multi-run comparisons, filtering, visualization in Jupyter notebooks,
and much more.

[Omnitrace](https://github.com/AMDResearch/omnitrace) has two distinct configuration steps when instrumenting:

1. Configuring which functions and modules are instrumented in the target binaries (i.e. executable and/or libraries)
   - [Instrumenting with Omnitrace](instrumenting.md)
2. Configuring what the instrumentation does happens when the instrumented binaries are executed
   - [Customizing Omnitrace Runtime](runtime.md)

## Omnitrace Use Cases

When analyzing the performance of an application, ***it is always best to NOT assume you know where the performance bottlenecks are***
***and why they are happening.*** Omnitrace is a ***tool for the entire execution of application***. It is the sort of tool which is
ideal for *characterizing* where optimization would have the greatest impact on the end-to-end execution of the application and/or
viewing what else is happening on the system during a performance bottleneck.

Especially when GPUs are involved, there is a tendency to assume that the quickest path to performance improvement is minimizing
the runtime of the GPU kernels. This is a highly flawed assumption: if you optimize the runtime of a kernel from 1 millisecond
to 1 microsecond (1000x speed-up) but the original application *never spent time waiting* for kernel(s) to complete,
you will see zero statistically significant speed-up in end-to-end runtime of your application. In other words, it does not matter
how fast or slow the code on GPU is if the application is not bottlenecked waiting on the GPU.

Use Omnitrace to obtain a high-level view of the entire application. Use it to determine where the performance bottlenecks are and
obtain clues to why these bottlenecks are happening. If you want ***extensive*** insight into the execution of individual kernels
on the GPU, AMD Research is working on another tool for this but you should start with the tool which characterizes the
broad picture: Omnitrace.

With regard to the CPU, Omnitrace does not target any specific vendor, it works just as well with non-AMD CPUs as with AMD CPUs.
With regard to the GPU, Omnitrace is currently restricted to the HIP and HSA APIs and kernels executing on AMD GPUs.
