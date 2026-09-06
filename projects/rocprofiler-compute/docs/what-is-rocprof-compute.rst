.. meta::
   :description: What is ROCm Compute Profiler?
   :keywords: Omniperf, ROCm, profiler, tool, Instinct, accelerator, AMD, rocprofiler-compute,
              rocprof-compute, kernel profiling, roofline, speed-of-light, wavefront,
              instruction mix, baseline comparison

******************************
What is ROCm Compute Profiler?
******************************

ROCm Compute Profiler is a kernel-level profiling tool for machine learning and high
performance computing (HPC) workloads running on AMD Instinct™ GPUs. It measures how
efficiently a single GPU kernel dispatch uses the underlying hardware: compute unit
occupancy, cache and memory bandwidth utilization, instruction mix, and achieved versus
peak arithmetic throughput.

This kernel-level view matters once you already know *which* kernel is worth optimizing.
Where ROCm Systems Profiler characterizes an entire application to find where time goes,
ROCm Compute Profiler drills into a specific dispatch to explain why it runs at the speed
it does, down to the hardware block level.

AMD Instinct MI-Series GPUs are data center-class GPUs designed for
compute and have some graphics capabilities disabled or removed.
ROCm Compute Profiler primarily targets use with
:doc:`GPUs in the AMD Instinct MI300, MI200, and MI100 Series <rocm:conceptual/gpu-arch>`.
Development is in progress to support Radeon™ (RDNA) GPUs.

ROCm Compute Profiler is built on top of :doc:`ROCprofiler-SDK <rocprofiler-sdk:index>` to
monitor hardware performance counters. For the full performance-model reference, see
:doc:`conceptual/performance-model`.

.. _glance-architecture:

How it works
=============

ROCm Compute Profiler is a Python-based tool that profiles an application in up to two
stages: it first replays the application as many times as needed to collect the requested
hardware counters per kernel dispatch, then, unless disabled with ``--no-roof``, runs a set
of accelerator-specific micro-benchmarks to establish the empirical roofline. The roofline
model is not available on accelerators pre-MI200. See :ref:`profiling-routine` for the exact
stage breakdown.

.. image:: /data/how_compute_profiler_works.png
   :width: 60%
   :align: center
   :alt: Diagram of ROCm Compute Profiler replaying an application across multiple counter-collection passes, running roofline micro-benchmarks, deriving metrics, and producing analysis views available as CLI output by default, or as CSV, an analysis database, or an HTML roofline plot

Counter collection is performed by injecting two libraries into the target process alongside :doc:`ROCprofiler-SDK <rocprofiler-sdk:index>`: a native tool (``librocprofiler-compute-tool.so``) that collects hardware performance counters per dispatch, and an SDK tool (``librocprofiler-sdk-tool.so``) that handles kernel tracing and output database generation. See :ref:`core-install-rocprof-var` for the full backend and library breakdown.

.. warning::

   Because associating counters with a specific kernel dispatch requires serializing dispatches, kernels launched on separate HIP streams on the same GPU don't execute concurrently while profiling. Profiled kernel duration and utilization metrics reflect this serialized execution, not the concurrent behavior of a normal run. See the full warning and its consequences in :doc:`how-to/profile/mode`.

Once counters are collected, ROCm Compute Profiler derives higher-level metrics, such as utilizations and ratios, from the raw counter values to populate the analysis views described next.

ROCm Compute Profiler standalone GUI analyzer (experimental)
-------------------------------------------------------------

ROCm Compute Profiler provides a :doc:`standalone GUI <how-to/analyze/standalone-gui>` to
enable basic performance analysis.

.. _glance-key-features:

Key features
==============

The feature set is documented across several how-to and conceptual pages; this table is a quick index into them.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Feature
     - Description
     - See also
   * - Kernel-level counter collection
     - Collects hardware performance counters per kernel dispatch, replaying the application as many times as needed to gather every requested counter.
     - :ref:`glance-architecture`
   * - System and hardware block Speed-of-Light
     - Compares achieved utilization against theoretical peak, both for overall GPU throughput and for individual hardware blocks such as the command processor, workgroup manager, and caches.
     - :ref:`glance-analysis-views`
   * - Memory chart analysis
     - Visualizes transaction counts and bandwidth at each level of the memory hierarchy.
     - :ref:`glance-analysis-views`
   * - Empirical roofline analysis
     - Benchmarks achievable peak compute throughput and memory bandwidth, then plots kernel arithmetic intensity against that roofline.
     - :ref:`standalone-roofline`
   * - Wavefront and instruction mix analysis
     - Reports launch and runtime statistics, and the breakdown of VALU, VMEM, scalar, and matrix instructions issued by a kernel.
     - :doc:`conceptual/cdna/pipeline-metrics`
   * - Baseline comparison
     - Compares two or more profiled runs of the same SoC side by side to check the effect of a code change.
     - :ref:`analysis-baseline-comparison`
   * - Multiple analysis interfaces
     - The command line and ROCm Optiq both read the same profiling output.
     - :doc:`install/quickstart`
   * - Iteration multiplexing
     - Collects a large number of performance counters with minimal profiling overhead by splitting counter collection across kernel iterations instead of full application replays.
     - :ref:`iteration-multiplexing`

.. _glance-analysis-views:

Analysis views
================

ROCm Compute Profiler's analysis report is organized into the following views:

.. list-table::
   :header-rows: 1
   :widths: 20 55 25

   * - View
     - What it tells you
     - See also
   * - System Speed-of-Light
     - What percentage of the GPU's theoretical peak performance the kernel achieves, across compute and memory subsystems.
     - :ref:`CDNA Speed-of-Light <sys-sol>`, :ref:`RDNA Speed-of-Light <sys-sol-rdna>`
   * - Hardware block Speed-of-Light
     - Which specific hardware block — the compute unit, a cache level, the command processor, or the workgroup manager — is the bottleneck for a kernel, based on utilization and stall metrics for that block.
     - :ref:`profiling-hw-component-filtering`, :doc:`conceptual/performance-model`
   * - Memory chart
     - Read, write, and atomic transaction counts, hit rates, and latencies at each level of the memory hierarchy: LDS, the L1 caches, the L2 cache, and HBM.
     - :doc:`conceptual/cdna/vector-l1-cache`, :doc:`conceptual/cdna/l2-cache`
   * - Roofline
     - Classifies whether a kernel is compute-bound or memory-bound, and plots its exact position against attainable peak compute throughput and memory bandwidth; combine with kernel filtering for a per-kernel breakdown.
     - :ref:`standalone-roofline`, :ref:`per-kernel-roofline`
   * - Wavefront analysis
     - Launch statistics (grid size, workgroup size, VGPR/AGPR/SGPR usage) and runtime statistics (wavefront occupancy, active cycles) for profiled kernels.
     - :doc:`conceptual/cdna/pipeline-metrics`
   * - Instruction mix
     - Breakdown of VALU, VMEM, scalar, LDS, and matrix (MFMA/WMMA) instructions issued by a kernel.
     - :doc:`conceptual/cdna/pipeline-metrics`
   * - Baseline comparison
     - Side-by-side comparison of any of the preceding views across two or more profiled runs of the same SoC — for example, before and after an optimization.
     - :ref:`analysis-baseline-comparison`

.. _glance-cli-tools:

Key commands
==============

ROCm Compute Profiler exposes two primary modes on the ``rocprof-compute`` executable: ``profile`` to collect data, and ``analyze`` to read it back. See :ref:`modes` for the full list of modes and :ref:`basic-operations` for the required arguments per operation.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Command
     - Purpose
   * - ``rocprof-compute profile -n my_run -- ./app``
     - Profiles ``./app``, collecting all available counters for all kernels, and writes results to ``workloads/my_run/<SoC>/``.
   * - ``rocprof-compute profile -n my_run -k vecCopy -- ./app``
     - Profiles only kernels whose name matches ``vecCopy``.
   * - ``rocprof-compute profile -n my_run -d 1 2 3 -- ./app``
     - Profiles only the 1st, 2nd, and 3rd dispatch of each kernel.
   * - ``rocprof-compute profile -n my_run -b 2 5 -- ./app``
     - Profiles only the counters needed for the specified analysis report blocks, which speeds up the profiling run.
   * - ``rocprof-compute profile -n my_run --roof-only -- ./app``
     - Runs only the roofline micro-benchmarks, skipping standard counter collection.
   * - ``rocprof-compute analyze -p workloads/my_run/MI300X/``
     - Analyzes a profiled run in the terminal.
   * - ``rocprof-compute analyze -p workloads/my_run/MI300X/ --output-format db``
     - Writes the analysis to a SQLite analysis database instead of printing to the terminal.
   * - ``rocprof-compute analyze -p run_a/MI300X/ run_b/MI300X/``
     - Compares two profiled runs of the same SoC side by side.

For full walkthroughs, see :doc:`how-to/profile/mode` and :doc:`how-to/analyze/cli`.

.. _glance-flags:

Flags reference
==================

ROCm Compute Profiler supports many flags for filtering kernels, dispatches, and
hardware blocks during profiling and analysis, and for selecting the analysis output
format. See :doc:`how-to/profile/mode` for the full list of profile mode flags and
:doc:`how-to/analyze/mode` for analyze mode flags, or run ``rocprof-compute profile -h``
or ``rocprof-compute analyze -h`` for the complete set.

.. _glance-output-formats:

Output formats
================

``analyze`` accepts an ``--output-format`` option for the analysis report. The following table lists the available values, plus the HTML roofline plots that ``analyze`` writes automatically, and the ``rocpd`` database that ``profile`` always writes:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Format
     - Produced by
     - Description
   * - ``rocpd``
     - profile
     - SQLite database of raw counters per dispatch, written automatically. Required for ``analyze --output-format csv`` or ``db``.
   * - ``stdout`` (default)
     - analyze
     - Analysis results printed to the terminal only.
   * - ``txt``
     - analyze
     - Analysis results written to a ``rocprof_compute_<uuid>.txt`` file.
   * - ``csv``
     - analyze
     - Analysis results written to a folder of CSV files. Requires the workload's ``rocpd`` database from profiling.
   * - ``db``
     - analyze
     - Analysis results written to a ``rocprof_compute_<uuid>.db`` SQLite analysis database. Same requirement as ``csv``.
   * - ``html``
     - analyze
     - Standalone Plotly roofline plot(s) (``empirRoof_gpu-<id>...html``), written automatically whenever the workload includes roofline data, independent of ``--output-format``. See :ref:`standalone-roofline`.

Select the analyze output format with ``--output-format`` and override the file name with ``--output-name``. See :ref:`analysis-output-format` for details.

.. _glance-supported-hardware:

Supported hardware
=====================

Roofline micro-benchmark support varies by :ref:`SoC family <def-soc>` independently of general counter and Speed-of-Light support. See :doc:`reference/compatible-accelerators` for the full, tool-wide support table.

.. list-table::
   :header-rows: 1
   :widths: 35 15 15 35

   * - SoC family
     - Counters / SOL support
     - Roofline support
     - Notes
   * - AMD Instinct™ MI350 Series (CDNA4, gfx950)
     - ✅
     - ✅
     - Full feature support; MI350P support introduced in ROCm Compute Profiler 3.6.0.
   * - AMD Instinct MI300 Series (CDNA3)
     - ✅
     - ✅
     - Full feature support.
   * - AMD Instinct MI200 Series (CDNA2)
     - ✅
     - ✅
     - Full feature support.
   * - AMD Instinct MI100 Series (CDNA1)
     - ✅
     - ❌
     - Roofline micro-benchmarks require MI200 or later. See :ref:`standalone-roofline`.
   * - AMD Ryzen™ AI Max / Ryzen AI Max+ 300 and 400 Series and Ryzen AI 5/7 PRO 4xx Series (Strix/Halo, Gorgon/Halo; gfx1150/gfx1151/gfx1152/gfx1153)
     - ✅
     - ✅
     - Full feature support; roofline benchmarking for gfx1150 and gfx1152 was added in ROCm Compute Profiler 3.7.0, with the gfx1151 (Strix Halo) gap closed in 3.8.0. Uses Wave Matrix Multiply Accumulate (WMMA) in place of MFMA.
   * - AMD Instinct MI50, MI60 (Vega 20)
     - ❌
     - ❌
     - —
