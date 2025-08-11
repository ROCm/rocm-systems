.. meta::
   :description: ROCm Systems Profiler introduction, explanation, and reference
   :keywords: rocprof-sys, rocprofiler-systems, Omnitrace, ROCm, profiler, explanation, introduction, what is, tracking, visualization, tool, Instinct, accelerator, AMD

******************************
What is ROCm Systems Profiler?
******************************

ROCm Systems Profiler is designed for the high-level profiling and comprehensive tracing
of applications running on the CPU or the CPU and GPU. It supports dynamic binary
instrumentation, call-stack sampling, and various other features for determining
which function and line number are currently executing.

A visualization of the comprehensive ROCm Systems Profiler results can be observed in any modern
web browser. Upload the Perfetto (``.proto``) output files produced by ROCm Systems Profiler at
`ui.perfetto.dev <https://ui.perfetto.dev/>`_ to see the details.

.. important::
   If you are using a version of ROCm prior to ROCm 6.3.1 and are experiencing problems viewing your
   trace in the latest version of [Perfetto](http://ui.perfetto.dev), then try using
   [Perfetto UI v46.0](https://ui.perfetto.dev/v46.0-35b3d9845/#!/).

Aggregated high-level results are available as human-readable text files and
JSON files for programmatic analysis. The JSON output files are compatible with the
`hatchet <https://github.com/hatchet/hatchet>`_ Python package. Hatchet converts
the performance data into pandas data frames and facilitates multi-run comparisons, filtering,
and visualization in Jupyter notebooks.

To use ROCm Systems Profiler for instrumentation, follow these two configuration steps:

#. Indicate the functions and modules to :doc:`instrument <./how-to/instrumenting-rewriting-binary-application>` in the target binaries, including the executable and any libraries
#. Specify the :doc:`instrumentation parameters <./how-to/configuring-runtime-options>` to use when the instrumented binaries are launched

