.. meta::
   :description: Omnitrace documentation and reference
   :keywords: Omnitrace, ROCm, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

******************
What is Omnitrace?
******************

Omnitrace is designed for the high-level profiling and comprehensive tracing
of applications running on the CPU or the CPU and GPU. It supports dynamic binary
instrumentation, call-stack sampling, and various other features for determining
which function and line number are currently executing.

A visualization of the comprehensive Omnitrace results can be observed in any modern
web browser. Upload the Perfetto (``.proto``) output files produced by Omnitrace at
`ui.perfetto.dev <https://ui.perfetto.dev/>`_ to see the details.

.. important::
   Perfetto validation is done with trace_processor v46.0 as there is a known issue with v47.0.
   If you are experiencing problems viewing your trace in the latest version of `Perfetto <http://ui.perfetto.dev>`_,
   then try using `Perfetto UI v46.0 <https://ui.perfetto.dev/v46.0-35b3d9845/#!/>`_.

Aggregated high-level results are available as human-readable text files and
JSON files for programmatic analysis. The JSON output files are compatible with the
`hatchet <https://github.com/hatchet/hatchet>`_ Python package. Hatchet converts
the performance data into pandas data frames and facilitates multi-run comparisons, filtering,
and visualization in Jupyter notebooks.

To use Omnitrace for instrumentation, follow these two configuration steps:

#. Indicate the functions and modules to :doc:`instrument <./how-to/instrumenting-rewriting-binary-application>` in the target binaries, including the executable and any libraries
#. Specify the :doc:`instrumentation parameters <./how-to/configuring-runtime-options>` to use when the instrumented binaries are launched

