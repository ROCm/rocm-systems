.. meta::
   :description: What is ROCm Compute Profiler?
   :keywords: Omniperf, ROCm, profiler, tool, Instinct, accelerator, AMD

******************************
What is ROCm Compute Profiler?
******************************

ROCm Compute Profiler is a kernel-level profiling tool for machine learning and high
performance computing (HPC) workloads running on AMD Instinct™ accelerators.

AMD Instinct MI-series accelerators are data center-class GPUs designed for
compute and have some graphics capabilities disabled or removed.
ROCm Compute Profiler primarily targets use with
:doc:`accelerators in the MI300, MI200, and MI100 families <rocm:conceptual/gpu-arch>`.
Development is in progress to support Radeon™ (RDNA) GPUs.

ROCm Compute Profiler is built on top of :doc:`ROCProfiler <rocprofiler:index>` to
monitor hardware performance counters.

.. _high-level-design:

High-level design
=================

The architecture of ROCm Compute Profiler consists of three major components shown in the
following diagram.

Core ROCm Compute Profiler
--------------------------

Acquires raw performance counters via application replay using ``rocprof``.
Counters are stored in a comma-separated-values format for further
:doc:`analysis <how-to/analyze/mode>`. It runs a set of accelerator-specific
micro-benchmarks to acquire hierarchical roofline data. The roofline model is
not available on accelerators pre-MI200.

ROCm Compute Profiler standalone GUI analyzer
---------------------------------------------

ROCm Compute Profiler provides a :doc:`standalone GUI <how-to/analyze/standalone-gui>` to
enable basic performance analysis.

Features
========

ROCm Compute Profiler offers comprehensive profiling based on all available hardware counters
for the target accelerator. It delivers advanced performance analysis features,
such as system Speed-of-Light (SOL) and hardware block-level SOL evaluations.
Additionally, ROCm Compute Profiler provides in-depth memory chart analysis, roofline
analysis, baseline comparisons, and more, ensuring a thorough understanding of
system performance.

ROCm Compute Profiler supports analysis through both the :doc:`command line </how-to/analyze/cli>`.
The following list describes ROCm Compute Profiler's features at a high level.

* :doc:`Support for AMD Instinct MI300, MI200, and MI100 accelerators <reference/compatible-accelerators>`

* :doc:`Standalone GUI analyzer </how-to/analyze/standalone-gui>`

* :ref:`Filtering <filtering>` to reduce profiling time

  * Filtering by dispatch

  * Filter by kernel

  * Filtering by GPU ID

* :ref:`Baseline comparisons <analysis-baseline-comparison>`

* :ref:`Multiple normalizations <normalization-units>`
