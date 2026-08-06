.. meta::
   :description: ROCm Systems Profiler introduction, explanation, and reference
   :keywords: rocprof-sys, rocprofiler-systems, Omnitrace, ROCm, profiler, explanation, introduction, what is, tracking, visualization, tool, Instinct, accelerator, AMD, rocpd, perfetto, optiq, json, text, hatchet, output format

******************************
What is ROCm Systems Profiler?
******************************

ROCm Systems Profiler is designed for the high-level profiling and comprehensive tracing
of applications running on the CPU or the CPU and GPU. It supports dynamic binary
instrumentation, call-stack sampling, and various other features for determining
which function and line number are currently executing.

ROCm Systems Profiler can emit results in several output formats, selected with the
``--output-format`` command-line argument. The detailed trace formats capture the full
timeline of events, while the aggregated formats summarize high-level results:

* **ROCm Profiling Data (rocpd)**: A detailed trace stored as a SQLite3 database
  (``.db``), selected with ``--output-format rocpd``. This will soon be the default
  output format. You can view and analyze ``rocpd`` files with the
  `ROCm Optiq <https://rocm.docs.amd.com/projects/roc-optiq/en/latest/>`_ tool.
* **Perfetto**: A detailed trace stored as a protocol buffer (``.proto``), selected with
  ``--output-format proto``. Upload the Perfetto output files at
  `ui.perfetto.dev <https://ui.perfetto.dev/>`_ to visualize the results in any modern
  web browser.
* **Text**: Aggregated high-level results as human-readable text files, selected with
  ``--output-format text`` (``txt`` is an alias).
* **JSON**: Aggregated high-level results as JSON files for programmatic analysis,
  selected with ``--output-format json``. The JSON output files are compatible with the
  `hatchet <https://github.com/hatchet/hatchet>`_ Python package, which converts the
  performance data into pandas data frames and facilitates multi-run comparisons,
  filtering, and visualization in Jupyter notebooks.

Tokens are space- or comma-separated, so multiple formats can be requested at once, for
example ``--output-format proto rocpd``. For details on all formats, see
:doc:`./how-to/understanding-rocprof-sys-output`.

To use ROCm Systems Profiler for instrumentation, follow these two configuration steps:

#. Indicate the functions and modules to :doc:`instrument <./how-to/instrumenting-rewriting-binary-application>` in the target binaries, including the executable and any libraries
#. Specify the :doc:`instrumentation parameters <./how-to/configuring-runtime-options>` to use when the instrumented binaries are launched
