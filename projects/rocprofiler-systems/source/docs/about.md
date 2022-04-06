# About

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

> ***[Omnitrace](https://github.com/AMDResearch/omnitrace) is an AMD research project and should***
> ***not be treated as an offical part of the ROCm software stack.***

[Browse Omnitrace source code on Github](https://github.com/AMDResearch/omnitrace)

[Omnitrace](https://github.com/AMDResearch/omnitrace) is designed for both high-level and
comprehensive application tracing and profiling on both the CPU and GPU.
[Omnitrace](https://github.com/AMDResearch/omnitrace) supports both binary instrumentation
and sampling as a means of collecting various metrics.

Visualization of the comprehensive omnitrace results can be viewed in any modern web browser by visiting [ui.perfetto.dev](https://ui.perfetto.dev/)
and loading the perfetto output (`.proto` files) produced by omnitrace.

Aggregated high-level results are available in text files for human consumption and JSON files for programmatic analysis.
The JSON output files are compatible with the python package [hatchet](https://github.com/hatchet/hatchet) which converts
the performance data into pandas dataframes and facilitate multi-run comparisons, filtering, visualization in Jupyter notebooks, and much more.

[Omnitrace](https://github.com/AMDResearch/omnitrace) has two distinct configuration steps:

1. Configuring which functions and modules are instrumented in the target binaries (i.e. executable and/or libraries)
   - [Instrumenting with Omnitrace](instrumenting.md)
2. Configuring what the instrumentation does happens when the instrumented binaries are executed
   - [Customizing Omnitrace Runtime](runtime.md)
