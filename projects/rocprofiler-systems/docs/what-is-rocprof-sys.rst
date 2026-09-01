.. meta::
   :description: Introduction to ROCm Systems Profiler, plus a high-level overview of its architecture, instrumentation modes, and command-line tools
   :keywords: ROCm Systems Profiler overview, rocprofiler-systems, rocprof-sys, Omnitrace, what is, introduction, instrumentation, sampling, presets, rocpd, perfetto, optiq, json, text, hatchet, output format, MPI, OpenMP, GPU metrics, tracking, visualization, tool, Instinct, accelerator, AMD

.. _rocprofiler-systems-at-a-glance:

******************************
What is ROCm Systems Profiler?
******************************

ROCm Systems Profiler (rocprofiler-systems) is a system-level profiler for applications running on the CPU or the CPU and GPU. Using dynamic binary instrumentation, call-stack sampling, and other techniques, it captures HIP/HSA APIs, kernel dispatches, memory copies, RCCL, and GPU telemetry (temperature, power, utilization, interconnect) on one timeline, down to the function and line number currently executing. It also correlates this GPU-side data with host call stacks, MPI, OpenMP, and Python activity, so you can see why the device is busy, stalled, or waiting.

For example, in a distributed training job, the GPU can appear idle not because its kernels are slow, but because a data loader is starved, an MPI collective is blocking, or Python's GIL is stalling the dispatch queue.

This topic orients you to how ROCm Systems Profiler is put together and how to invoke it. For the full, categorized feature catalog and use cases, see :doc:`conceptual/rocprof-sys-feature-set`.

.. _glance-capabilities:

Capabilities
==============

The feature set is spread across several how-to and conceptual pages; the following tables are a quick index into them:

Tracing
-------

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 25 45 30

   * - Feature
     - Description
     - See also
   * - ROCm API tracing
     - Traces ROCm API and runtime domains via ROCprofiler-SDK, including HIP, HSA, ROCTx, RCCL, kernel dispatches, memory events, rocDecode, and rocJPEG. List available domains with ``rocprof-sys-avail --list-domains``.
     - :ref:`rocprof-sys-feature-gpu-metrics`
   * - MPI and communication tracing
     - Records collective and point-to-point operation timing with per-rank attribution across MPI (via standard PMPI wrappers), RCCL, UCX, and OpenSHMEM/rocSHMEM.
     - :doc:`how-to/communication-runtime-profiling`
   * - OpenMP tracing
     - Captures thread team creation, parallel regions, task execution, and synchronization via the OMPT callback interface.
     - :doc:`how-to/openmp-profiling`
   * - Unified memory
     - Tracks page faults and host/device migrations for HIP managed memory (KFD).
     - :doc:`how-to/unified-memory-profiling`

Metrics
-------

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 25 45 30

   * - Feature
     - Description
     - See also
   * - GPU telemetry
     - Samples GPU temperature, power, utilization, clocks, memory, XGMI/PCIe bandwidth, and VCN/JPEG engine activity via ``amd-smi``.
     - :ref:`rocprof-sys-feature-gpu-metrics`, :doc:`how-to/xgmi-pcie-sdma-sampling`, :doc:`how-to/vcn-jpeg-sampling`
   * - GPU hardware counters
     - Collects GPU hardware counters (occupancy, VALU utilization, wavefront counts, etc.) via ``--gpu-events``. See ``rocprof-sys-avail -H -c GPU`` for the device-specific list.
     - :ref:`rocprof-sys-feature-gpu-metrics`
   * - CPU hardware counters
     - Reads IPC, cache misses, branch mispredictions, and other CPU PMU events via the Linux ``perf`` subsystem. See ``rocprof-sys-avail -H -c CPU`` for the device-specific list.
     - :ref:`rocprof-sys-feature-cpu-metrics`
   * - Network (NIC) metrics
     - Samples NIC throughput and errors (PAPI); AINIC where supported.
     - :doc:`how-to/nic-profiling`

Collection mechanism (host-side correlation)
---------------------------------------------

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 25 45 30

   * - Feature
     - Description
     - See also
   * - Binary instrumentation
     - Inserts probes at selected function entry/exit with no source changes (runtime instrumentation or binary rewrite). Complete, deterministic host call-stack data.
     - :ref:`glance-modes-compared`
   * - Call-stack sampling
     - Periodically interrupts the application and records the host call stack (and related metrics). Lower-overhead statistical data to correlate with the GPU timeline.
     - :ref:`glance-modes-compared`
   * - Python hooks
     - Instruments Python interpreter call frames for mixed Python/C++/HIP workload analysis.
     - :doc:`how-to/profiling-python-scripts`
   * - Process attachment
     - Attaches to an already-running process without restarting it.
     - :doc:`how-to/attaching-to-running-process`

Analysis
--------

.. list-table::
   :header-rows: 1
   :width: 100%
   :widths: 25 45 30

   * - Feature
     - Description
     - See also
   * - Trace
     - Event timeline of GPU APIs, kernels, copies, and correlated host activity. Default analysis mode.
     - :doc:`conceptual/data-collection-modes`, :doc:`how-to/understanding-rocprof-sys-output`
   * - Profile
     - High-level summary profiles with statistical aggregations (mean, min, max, stddev) per function. Lower overhead than a full timeline; JSON and text outputs are generated.
     - :doc:`conceptual/data-collection-modes`
   * - Causal profiling
     - Estimates the end-to-end speedup from optimizing a given function or line by selectively slowing other code regions.
     - :doc:`how-to/performing-causal-profiling`

.. _glance-architecture:

How it works
=============

ROCm Systems Profiler couples GPU kernel dispatches, memory copies, and device telemetry with the host-side activity around them. GPU data is captured via ROCprofiler-SDK callbacks and ``amd-smi`` polling; host call stacks, MPI, OpenMP, and Python frames are captured via binary instrumentation, statistical sampling, callback APIs, or symbol interception. Everything is merged into a single trace/profile output, correlated on the GPU timeline.

.. image:: data/how_systems_profiler_works.png
   :width: 60%
   :align: center

For the full explanation of each collection mode, including overhead trade-offs and a worked instrumentation-vs-sampling example, see :doc:`conceptual/data-collection-modes`.

.. _glance-modes-compared:

Binary instrumentation vs. call-stack sampling
================================================

Binary instrumentation and call-stack sampling can be used independently or together, trading overhead against completeness:

.. list-table::
   :header-rows: 1

   * - Mode
     - Overhead
     - Best for
   * - Binary instrumentation
     - Higher
     - Function-level profiling of selected functions when complete, deterministic call data is needed and the added overhead is acceptable.
   * - Call-stack sampling
     - Low
     - Whole application context at low overhead. For example, long-running jobs, MPI workloads, etc.
   * - Combined (both enabled)
     - Higher than sampling alone
     - Enable statistical sampling after binary instrumentation to help "fill in the gaps" between instrumented regions.

To use ROCm Systems Profiler for instrumentation, follow these two configuration steps:

#. Indicate the functions and modules to :doc:`instrument <how-to/instrumenting-rewriting-binary-application>` in the target binaries, including the executable and any libraries.
#. Specify the :doc:`instrumentation parameters <how-to/configuring-runtime-options>` to use when the instrumented binaries are launched.

.. _glance-cli-tools:

Command-line tools
====================

ROCm Systems Profiler ships as a set of standalone executables, each oriented toward a different profiling workflow:

.. list-table::
   :header-rows: 1
   :widths: 15 35 25 25

   * - Tool
     - Purpose
     - Example
     - See also
   * - ``rocprof-sys-avail``
     - Lists what is available to configure and collect on your system (settings, domains, counters, components).
     - ``rocprof-sys-avail -d`` or ``rocprof-sys-avail -H -c GPU``
     - :doc:`how-to/general-tips-using-rocprof-sys`
   * - ``rocprof-sys-instrument``
     - Performs dynamic binary instrumentation, including binary rewriting.
     - ``rocprof-sys-instrument -o ./app.inst -- ./app``
     - :doc:`how-to/instrumenting-rewriting-binary-application`
   * - ``rocprof-sys-run``
     - Launches a binary-rewritten executable, or profiles an application directly using preset/domain flags.
     - ``rocprof-sys-run -- ./app.inst``
     - :doc:`how-to/instrumenting-rewriting-binary-application`, :ref:`using-preset-profiles-quick-start`
   * - ``rocprof-sys-sample``
     - Performs call-stack sampling without instrumentation.
     - ``rocprof-sys-sample -f 1000 -- ./app``
     - :doc:`how-to/sampling-call-stack`
   * - ``rocprof-sys-attach``
     - Attaches to an already-running process.
     - ``rocprof-sys-attach -p $(pidof my_app)``
     - :doc:`how-to/attaching-to-running-process`
   * - ``rocprof-sys-causal``
     - Performs causal profiling to estimate optimization impact.
     - ``rocprof-sys-causal -l foo -- ./app``
     - :doc:`how-to/performing-causal-profiling`

.. _glance-presets:

Preset profiles
=================

Presets replace manually setting numerous environment variables with a single ``--preset`` flag available on ``rocprof-sys-run`` and ``rocprof-sys-sample``. Presets were introduced in ROCm 7.12 and expanded into an extensible JSON-based system in ROCm 7.13.

Discover, inspect, and apply a preset as follows:

#. List available presets:

   .. code-block:: shell

      rocprof-sys-run --list-presets

#. See what a preset configures:

   .. code-block:: shell

      rocprof-sys-run --explain=balanced

#. Run with the preset (``balanced`` is a good starting point):

   .. code-block:: shell

      rocprof-sys-run --preset=balanced -- ./my_app

#. Optionally, combine the preset with domain flags for finer control:

   .. code-block:: shell

      rocprof-sys-run --preset=balanced --gpu=temp,power -- ./my_app

Each built-in preset enables a different combination of tracing, profiling, sampling, and domains, tuned for a specific workload scenario:

.. list-table::
   :header-rows: 1
   :widths: 18 22 40 20

   * - Category
     - Presets
     - Best for
     - See also
   * - General
     - ``balanced``, ``profile-only``, ``detailed``
     - Most profiling scenarios; minimal-overhead production profiling; in-depth full-system analysis
     - :ref:`using-preset-profiles-general`
   * - GPU and workload
     - ``trace-gpu``, ``workload-trace``, ``trace-hw-counters``
     - GPU device activity; AI/ML, HPC, and GPU-accelerated training; hardware counter collection
     - :ref:`using-preset-profiles-gpu-workload`
   * - HPC
     - ``trace-hpc``, ``trace-openmp``, ``profile-mpi``
     - MPI/OpenMP/Kokkos applications; OpenMP target offload; MPI communication latency analysis
     - :ref:`using-preset-profiles-hpc`
   * - API tracing
     - ``sys-trace``, ``runtime-trace``
     - Full system API visibility; runtime-only API tracing
     - :ref:`using-preset-profiles-api-tracing`

For the exact configuration behind a preset, run ``rocprof-sys-run --explain=<name>``. For domain flags, configuration export, and custom presets, see :doc:`how-to/using-preset-profiles`.

.. _glance-output-formats:

Output formats
================

ROCm Systems Profiler supports several output formats, each suited to a different analysis or visualization workflow. ``rocpd`` is expected to become the default output format in an upcoming release.

.. list-table::
   :header-rows: 1

   * - Format
     - File extension
     - Description
     - Viewer
   * - Perfetto (proto)
     - ``.proto``
     - Detailed trace stored as a protocol buffer for interactive timeline visualization
     - `ui.perfetto.dev <https://ui.perfetto.dev>`_
   * - ROCm Profiling Data (rocpd)
     - ``.db``
     - Detailed trace and counter data stored as a SQLite3 database; queryable with SQL or convertible to other formats via ``rocpd convert``
     - `ROCm Optiq <https://rocm.docs.amd.com/projects/roc-optiq/en/latest/>`_
   * - Text
     - ``.txt``
     - Aggregated results (mean, min, max, stddev per function) as human-readable text
     - Text editor
   * - JSON
     - ``.json``
     - Aggregated high-level results for programmatic analysis; hatchet-compatible
     - Any JSON viewer, `hatchet <https://github.com/hatchet/hatchet>`_, custom scripts

Output-format selection differs by tool:

* When no ``--output-format`` is specified, ``rocprof-sys-run`` and ``rocprof-sys-sample`` produce a rocpd (SQLite3 database) trace:

  .. code-block:: shell

     rocprof-sys-run -- ./my_app

* ``--output-format`` (introduced in ROCm 7.14) selects one or more formats explicitly:

  .. code-block:: shell

     rocprof-sys-run --output-format proto rocpd json text -- ./my_app

* ``rocprof-sys-attach`` uses its own ``-F`` flag with different token names for the same formats (``perfetto`` instead of ``proto``):

  .. code-block:: shell

     rocprof-sys-attach -p 12345 -F perfetto,rocpd

For the legacy flags and the environment variables each ``--output-format`` token maps to, see :ref:`data-collection-modes-output-formats`. For output path conventions, metadata, and per-format details, see :doc:`how-to/understanding-rocprof-sys-output`.
