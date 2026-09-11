.. meta::
  :description: A high-level overview of ROCprofiler-SDK architecture, profiling services, and the rocprofv3 command-line tool
  :keywords: ROCprofiler-SDK overview, rocprofv3, profiling services, ATT, PC sampling, counter collection, SPM, API tracing, GPU profiling

.. _rocprofiler-sdk-at-a-glance:

************************************************
ROCprofiler-SDK at a glance
************************************************

ROCprofiler-SDK is a profiling infrastructure for GPU compute applications on ROCm. It provides hardware performance counters, API tracing, PC sampling, thread trace, and streaming performance monitoring through a unified, context-based API. The ``rocprofv3`` command-line tool exposes all of these capabilities without requiring source code changes or tool library development.

This topic orients you to the SDK's design and services. For step-by-step usage, follow the links to the relevant how-to guides.

.. _glance-context-model:

Context model
==============

The context is the central design concept in ROCprofiler-SDK — a bundle of profiling services declared upfront during tool initialization. Profiling overhead is scoped to exactly the services requested; services that aren't configured impose no interception cost on the application.

.. image:: /data/rocprofiler_sdk_context_model.png
   :width: 60%
   :align: center

Multiple tools can run simultaneously, each with its own context. ROCprofiler-SDK assigns a priority to each tool at registration, so a lower-priority tool can inspect what higher-priority tools have already configured.

.. _glance-advanced-features:

Advanced profiling features
============================

ATT, PC sampling, and SPM provide hardware-level observability beyond standard counter collection and API tracing.

.. _glance-att:

Advanced Thread Trace (ATT)
-----------------------------

ATT records the complete instruction-level execution history of GPU wavefronts on targeted compute units. It is exposed via ``rocprofiler-sdk/experimental/thread_trace.h`` and delivers raw SQTT (Shader Queue Thread Trace) data. The ``rocprof-trace-decoder`` library decodes this data for visualization in :doc:`ROCprof Compute Viewer <rocprof-compute-viewer:index>` (RCV).

.. _glance-att-comparison:

How ATT differs from counter-based services
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unlike PMC collection, ATT doesn't accumulate counter values — it records every instruction issued or stalled on the selected compute unit for the duration of the traced dispatch.

The following table highlights the key differences in granularity, data type, and output size across Dispatch PMC, PC sampling, and ATT.

.. list-table::
   :header-rows: 1

   * - Property
     - Dispatch PMC
     - PC sampling
     - ATT
   * - Granularity
     - Accumulated per kernel
     - Statistical sample per wavefront
     - Every instruction on traced CU
   * - Data type
     - Hardware counter values
     - PC + execution state snapshot
     - Full wavefront execution history
   * - Kernel serialization required
     - Yes
     - No
     - No (only traced kernels serialized)
   * - Output size
     - Small
     - Medium
     - Large (raw trace per dispatch)
   * - Use case
     - What fraction of peak?
     - Which code is hot?
     - Which instruction is stalling, and why?

Raw ``.att`` files are decoded by the ``rocprof-trace-decoder`` library into ``ui_output_agent_{agent_id}_dispatch_{dispatch_id}/`` directories containing JSON and CSV files, which RCV reads directly. This output is separate from the ``--output-format`` pipeline (which defaults to rocpd for tracing data).

.. _glance-att-cli:

CLI options
^^^^^^^^^^^^

Use ``--att`` to enable Advanced Thread Trace. Additional options control kernel filtering, dispatch count, and SQ-block counter streaming (``--att-perfcounters``, gfx9 only). For the full parameter reference and runnable examples, see :ref:`thread-trace-parameters`.

.. _glance-att-hw:

Hardware support
^^^^^^^^^^^^^^^^^

ATT support varies by GPU architecture: AMD Instinct MI200/MI300/MI350 series (CDNA2/CDNA3/CDNA4) have full support for both instruction trace and perfmon streaming, while RDNA2-4 (gfx10-12) are trace-only and don't support ``--att-perfcounters``. For the full per-architecture matrix, see :ref:`Supported devices <thread-trace-supported-devices>`.

ATT is limited to one compute unit per shader engine. For production workloads, use ``--kernel-include-regex`` to limit tracing to the target kernel and collect from the minimum number of shader engines needed.

For full usage details, see :ref:`using-thread-trace` and :ref:`thread-trace`.

.. _glance-pc-sampling:

PC sampling
------------

PC sampling periodically captures the program counter, execution state, and hardware context of active GPU wavefronts. It is exposed in ROCprofiler-SDK via ``rocprofiler-sdk/pc_sampling.h``. Aggregating these program counter snapshots builds a statistical histogram of which code is executing, answering "what code is running?" SPM instead samples hardware counter values at fixed intervals, answering "how are hardware resources utilized over time?"

.. note::

  PC sampling is a beta feature. It requires the environment variable ``ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1`` and can't be used simultaneously with counter collection services in the same context.

.. warning::

  PC sampling carries a risk of hardware freeze that requires a cold restart.

.. _glance-pc-sampling-methods:

Sampling methods
^^^^^^^^^^^^^^^^^

ROCprofiler-SDK supports two PC sampling methods: HOST_TRAP, which uses a software interrupt, and STOCHASTIC, which uses a hardware PMU. The following table compares their properties.

.. list-table::
   :header-rows: 1

   * - Property
     - HOST_TRAP
     - STOCHASTIC
   * - Mechanism
     - Software interrupt (trap handler)
     - Hardware PMU
   * - Supported hardware
     - MI200+ (gfx90a+)
     - MI300+ (gfx942+)
   * - Interval basis
     - Time (microseconds)
     - Cycles / instructions
   * - Additional data captured
     - PC, exec mask, hardware IDs
     - PC, instruction type, stall reason, wave_issued flag
   * - Recommendation
     - MI200 series
     - Preferred on gfx942 and later

STOCHASTIC sampling probes waves actively running on the GPU and records whether each sampled wave issued an instruction at the captured PC, providing stall attribution that HOST_TRAP can't deliver.

.. _glance-pc-sampling-cli:

CLI options
^^^^^^^^^^^^

Use ``--pc-sampling-beta-enabled``, ``--pc-sampling-method``, ``--pc-sampling-unit``, and ``--pc-sampling-interval`` to enable and configure PC sampling with ``rocprofv3``. For the full option reference, see `PC sampling <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/quick-reference/rocprofv3-cli-options.html#pc-sampling>`_ in the CLI options reference.

Use ``rocprofv3-avail list --pc-sampling`` or ``info --pc-sampling`` to query which PC sampling configurations (methods, units, and interval ranges) each agent supports, without collecting a trace. See :ref:`using-rocprofv3-avail` for full usage details.

.. _glance-pc-sampling-hw:

Hardware support
^^^^^^^^^^^^^^^^^

Stochastic PC sampling requires AMD Instinct MI300-series (gfx942) or later; host-trap is supported on MI200-series (gfx90a) and later. For the full per-GPU support matrix, see :ref:`pc-sampling-supported-gpus`.

.. note::

   PC sampling is disabled by default and requires ``ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1``. This beta feature carries a risk of hardware freeze requiring a cold restart. Stochastic PC sampling is recommended over host-trap starting with gfx942.

For full usage details, see :ref:`using-pc-sampling` and :ref:`pc-sampling`.

.. _glance-spm:

Streaming Performance Monitoring (SPM)
----------------------------------------

SPM is a hardware capability on AMD Radeon™ and Instinct™ GPUs that streams counter values continuously into a memory ring buffer at a configurable hardware interval, independent of kernel dispatch boundaries. It is an experimental API in ROCprofiler-SDK, exposed via ``rocprofiler-sdk/experimental/spm.h``. Structs and enums are tagged ``ROCPROFILER_SDK_EXPERIMENTAL``. The top-level ``rocprofiler-sdk/spm.h`` reincludes the experimental header and emits a deprecation warning directing users to the experimental path.

.. _glance-spm-comparison:

How SPM differs from other counter services
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ROCprofiler-SDK provides three hardware counter collection mechanisms. The right choice depends on the granularity and timing requirements of your use case:

.. list-table::
   :header-rows: 1

   * - Property
     - Dispatch PMC
     - Device counter collection
     - SPM
   * - Granularity
     - One accumulated value per kernel
     - Accumulated over a configurable epoch
     - One value per hardware sampling interval
   * - Timing
     - Bounded to dispatch start/end
     - User-controlled flush
     - Continuous, independent of dispatches
   * - Kernel serialization required
     - Yes
     - No
     - No
   * - Concurrent workload support
     - No
     - Partial
     - Yes
   * - Use case
     - Identify expensive kernels; measure peak utilization
     - GPU-wide utilization over a time window
     - Time-series monitoring, DCGM-style telemetry, overlapping kernels

SPM captures how hardware resources are utilized over time. This is distinct from PC sampling, which reconstructs an execution histogram from program counter snapshots, and from ATT counter streaming (``--att-perfcounters``), which embeds SQ-block counter values into the thread trace ring buffer and requires ATT to be active. SPM operates as an independent hardware path.

.. _glance-spm-cli:

CLI options
^^^^^^^^^^^^

Use ``--spm-beta-enabled`` and ``--spm <COUNTERS>`` to enable and configure SPM collection with ``rocprofv3``, or ``rocprofv3-avail list --spm`` / ``list --spm-config`` to query SPM support per agent. For the full option reference, see :ref:`spm-cli-options`.

Counter listing output includes an SPM column (Supported / Not Supported) alongside each counter's name, description, and dimensions.

SPM records are written to all output backends: JSON, CSV, rocpd (SQLite3), and stats. Each ``rocprofiler_spm_counter_record_t`` includes a ``ROCPROFILER_SPM_RECORD_FLAG_DISPATCH_END`` flag that marks dispatch-boundary sentinel records, which carry no counter data.

For full usage details, see :ref:`using-spm`.

.. _glance-hardware:

Supported hardware
===================

Hardware support varies by GPU architecture, firmware version, ROCm release, and feature-gate state. To verify capability on your system, run ``rocprofv3 -L`` or ``rocprofv3-avail``. For per-feature hardware support tables, see :ref:`glance-att-hw` and :ref:`glance-pc-sampling-hw`.

.. note::

   **Counter collection**: On gfx11 and gfx12 architectures (AMD Radeon RX 7000 series and later), counter collection requires a stable power state. Use ``amd-smi`` to set the power state before profiling.

.. _glance-dependencies:

Dependencies
=============

The following table lists the components that ROCprofiler-SDK depends on and their roles.

.. list-table::
   :header-rows: 1

   * - Component
     - Role
   * - ``aqlprofile``
     - Generates PM4/AQL packets for counter collection and thread trace; bundled as a static library inside the SDK
   * - ``rocprofiler-register``
     - Coordinates intercept table modification across multiple tool libraries
   * - ROCm runtime (HIP + HSA)
     - Provides API interception points
   * - KFD kernel driver
     - Provides hardware access for PC sampling and counter collection

.. _glance-rocprofv3:

rocprofv3 — command-line profiling interface
=============================================

``rocprofv3`` is the official CLI for ROCprofiler-SDK. Run your application under ``rocprofv3`` and it produces structured output files with trace data, counter values, or PC samples, without requiring source code changes or recompilation.

Internally, ``rocprofv3`` sets ``ROCP_TOOL_LIBRARIES`` to load ``librocprofiler-sdk-tool.so``, a tool library built on ROCprofiler-SDK. Note that ``rocprofiler-compute`` uses ROCprofiler-SDK directly through its own tool library (``librocprofiler-compute-tool.so``) and doesn't route through ``rocprofv3``. Similarly, ``rocprofiler-systems`` uses ROCprofiler-SDK directly.

A companion tool, ``rocprofv3-avail``, lists all available hardware performance counters, derived metrics, and supported architectures for the installed GPUs.

How it works
-------------

The following figure shows how ``rocprofv3`` loads the tool library, creates profiling contexts, and writes output files after the application exits.

.. image:: /data/how_rocprofv3_works.png
   :width: 100%
   :align: center

Key CLI options
----------------

The following tables summarize the key ``rocprofv3`` options by category. For the complete CLI reference, see :ref:`cli-options`.

.. _glance-tracing:

Tracing
^^^^^^^^

Use these options to collect API and activity traces from the GPU runtime: ``--sys-trace`` (full system trace), ``--hip-trace``, ``--hsa-trace``, ``--kernel-trace``, and ``--rccl-trace``. For full descriptions and usage, see `Aggregate tracing <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/quick-reference/rocprofv3-cli-options.html#aggregate-tracing>`_ and `Basic tracing <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/quick-reference/rocprofv3-cli-options.html#basic-tracing>`_ in the CLI options reference.

.. _glance-counter-collection:

Counter collection
^^^^^^^^^^^^^^^^^^^

Use ``--pmc`` to collect hardware performance counters. Specify multiple ``--pmc`` flags to collect more counters across multiple passes.

For the full option description and usage, see `Counter collection <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/quick-reference/rocprofv3-cli-options.html#counter-collection>`_ in the CLI options reference.

PC sampling (beta)
^^^^^^^^^^^^^^^^^^^

For PC sampling CLI options, see :ref:`PC sampling CLI options <glance-pc-sampling-cli>`.

Thread Trace (ATT)
^^^^^^^^^^^^^^^^^^^

For ATT CLI options, see :ref:`ATT CLI options <glance-att-cli>`.

SPM (beta)
^^^^^^^^^^^

For SPM CLI options, see :ref:`SPM CLI options <glance-spm-cli>`.

.. _glance-process-attachment-cli:

Process attachment
^^^^^^^^^^^^^^^^^^^

Process attachment lets you begin profiling an already-running process, instead of launching it under ``rocprofv3`` from the start, using ``-p``/``--pid``/``--attach``. For the full option description, see `Dynamic process attachment <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/quick-reference/rocprofv3-cli-options.html#dynamic-process-attachment>`_ in the CLI options reference; for prerequisites and a full walkthrough, see :ref:`rocprofv3-process-attachment`.

.. _glance-output-format-cli:

Output format options
^^^^^^^^^^^^^^^^^^^^^^

The ``--output-format`` option lets you specify one or more output formats in a single run. The default output format is rocpd (SQLite3).

For the full description of ``--output-format``, ``--output-file``, and related I/O options, see `I/O options <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/quick-reference/rocprofv3-cli-options.html#io-options>`_ in the CLI options reference.

Supported output formats
-------------------------

The following table describes the supported output formats, their file extensions, and the tools that can open them.

.. list-table::
   :header-rows: 1

   * - Format
     - File extension
     - Description
     - Viewer
   * - rocpd (default)
     - ``.db``
     - SQLite3 database containing all trace and counter data; queryable with SQL or convertible using ``rocpd convert``
     - SQLite browser, custom scripts
   * - JSON
     - ``.json``
     - Structured JSON records
     - Any JSON viewer, custom scripts
   * - CSV
     - ``.csv``
     - Flat tabular data for scripted analysis
     - Spreadsheet, pandas, R
   * - PFTrace (Perfetto)
     - ``.pftrace``
     - Perfetto protobuf format for interactive timeline visualization
     - `ui.perfetto.dev <https://ui.perfetto.dev>`_
   * - OTF2
     - ``.otf2``
     - Open Trace Format 2 for HPC trace analysis
     - Vampir, Score-P, Cube

For comprehensive documentation, see :ref:`using-rocpd-output-format`.
