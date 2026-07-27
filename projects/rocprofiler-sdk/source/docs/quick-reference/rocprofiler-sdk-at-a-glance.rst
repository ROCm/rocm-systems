.. meta::
  :description: A high-level overview of ROCprofiler-SDK architecture, profiling services, and the rocprofv3 command-line tool
  :keywords: ROCprofiler-SDK overview, rocprofv3, profiling services, SPM, ATT, PC sampling, counter collection, API tracing, GPU profiling

.. _rocprofiler-sdk-at-a-glance:

==============================
ROCprofiler-SDK at a glance
==============================

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

SPM, ATT, and PC sampling provide hardware-level observability beyond standard counter collection and API tracing.

.. _glance-spm:

Streaming Performance Monitoring (SPM)
----------------------------------------

SPM is a hardware capability on AMD Radeon™ and Instinct™ GPUs that streams counter values continuously into a memory ring buffer at a configurable hardware interval, independent of kernel dispatch boundaries. It is an experimental API in ROCprofiler-SDK, exposed via ``rocprofiler-sdk/experimental/spm.h``. Structs and enums are tagged ``ROCPROFILER_SDK_EXPERIMENTAL``. The top-level ``rocprofiler-sdk/spm.h`` reincludes the experimental header and emits a deprecation warning directing users to the experimental path.

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

CLI options
^^^^^^^^^^^^

Use the following options to enable and configure SPM collection with ``rocprofv3``.

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``--spm-beta-enabled``
     - Required to enable SPM collection
   * - ``--spm <COUNTERS>``
     - Counters to collect, comma- or space-separated. All counters must fit in a single hardware pass; if they don't, the job fails.
   * - ``--spm-sample-interval <N>``
     - Sampling interval, interpreted in the unit set by ``--spm-sample-interval-unit``
   * - ``--spm-sample-interval-unit <UNIT>``
     - Interval unit; currently accepts ``sclk_cycles`` only

``rocprofv3-avail`` provides two sub-commands for SPM:

- ``list --spm`` — lists SPM-capable counters per agent
- ``list --spm-config`` — lists agents with SPM configuration support, including minimum and maximum sampling intervals

Counter listing output includes an SPM column (Supported / Not Supported) alongside each counter's name, description, and dimensions.

SPM records are written to all output backends: JSON, CSV, rocpd (SQLite3), and stats. Each ``rocprofiler_spm_counter_record_t`` includes a ``ROCPROFILER_SPM_RECORD_FLAG_DISPATCH_END`` flag that marks dispatch-boundary sentinel records, which carry no counter data.

For full usage details, see :ref:`using-spm`.

.. _glance-att:

Advanced Thread Trace (ATT)
-----------------------------

ATT records the complete instruction-level execution history of GPU wavefronts on targeted compute units. It is exposed via ``rocprofiler-sdk/experimental/thread_trace.h`` and delivers raw SQTT (Shader Queue Thread Trace) data. The ``rocprof-trace-decoder`` library decodes this data for visualization in ROCprof Compute Viewer (RCV).

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

CLI options
^^^^^^^^^^^^

The following options control ATT collection. Use them to filter kernels, embed SQ-block counter values in the trace, and set the number of dispatches to trace.

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``--att``
     - Enable Advanced Thread Trace
   * - ``--att-perfcounter-ctrl <N>``
     - PMU counter polling during trace (gfx9 only)
   * - ``--att-perfcounters <COUNTERS>``
     - Embed SQ-block counter values in the trace ring buffer
   * - ``--kernel-include-regex <RE>``
     - Filter kernels to trace by name regex
   * - ``--kernel-iteration-range <R>``
     - Trace specific dispatch iterations
   * - ``--att-consecutive-kernels <N>``
     - Number of consecutive kernels to trace (default 0)

The output is written to ``ui_output_agent_*/dispatch_*/`` directories of JSON and CSV files. Open the decoded directory in ROCprof Compute Viewer (RCV) for interactive visualization.

Hardware support
^^^^^^^^^^^^^^^^^

ATT support varies by GPU architecture. Full support includes both instruction trace and perfmon streaming; trace-only architectures don't support ``--att-perfcounters``.

.. list-table::
   :header-rows: 1

   * - Architecture
     - Support
     - Notes
   * - CDNA4 (MI350 series)
     - Full
     - gfx950
   * - CDNA3 (MI300 series)
     - Full
     - gfx942
   * - CDNA2 (MI200 series)
     - Full
     - gfx90a
   * - RDNA2 / gfx10
     - Trace-only
     - No perfmon streaming
   * - RDNA3 / gfx11
     - Trace-only
     - No perfmon streaming
   * - RDNA4 / gfx12
     - Trace-only
     - No perfmon streaming

ATT is limited to one compute unit per shader engine. For production workloads, use ``--kernel-include-regex`` to limit tracing to the target kernel and collect from the minimum number of shader engines needed.

For full usage details, see :ref:`using-thread-trace` and :ref:`thread-trace`.

.. _glance-pc-sampling:

PC sampling
------------

PC sampling periodically captures the program counter, execution state, and hardware context of active GPU wavefronts. It is exposed in ROCprofiler-SDK via ``rocprofiler-sdk/pc_sampling.h``. PC sampling answers "what code is running?" by building a statistical execution histogram — as distinct from SPM, which answers "how are hardware resources utilized over time?"

PC sampling is a **beta** feature. It requires the environment variable ``ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1`` and can't be used simultaneously with counter collection services in the same context.

.. warning::

   PC sampling carries a risk of hardware freeze that requires a cold restart.

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

CLI options
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Option
     - Description
     - Example
   * - ``--pc-sampling-beta-enabled``
     - Required to enable PC sampling; also sets ``ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1``
     - ``--pc-sampling-beta-enabled``
   * - ``--pc-sampling-method <METHOD>``
     - Sampling method: ``stochastic`` (default on gfx942+) or ``host_trap`` (MI200+)
     - ``--pc-sampling-method stochastic``
   * - ``--pc-sampling-unit <UNIT>``
     - Interval unit: ``time`` (microseconds, for host_trap) or ``cycles`` (for stochastic on gfx942)
     - ``--pc-sampling-unit time``
   * - ``--pc-sampling-interval <N>``
     - Numeric sampling interval in ``time`` or ``cycles``
     - ``--pc-sampling-interval 1``

To list available PC sampling configurations per agent:

.. code-block:: bash

   rocprofv3-avail list --pc-sampling

To get detailed information about the PC sampling configurations per agent:

.. code-block:: bash

   rocprofv3-avail info --pc-sampling

Hardware support
^^^^^^^^^^^^^^^^^

The following table lists AMD Instinct GPUs that support PC sampling and shows which sampling methods are available on each.

.. list-table::
   :header-rows: 1

   * - AMD Instinct GPU
     - Architecture
     - Stochastic
     - Host-trap
   * - MI355X
     - CDNA4
     - ✅
     - ✅
   * - MI350X
     - CDNA4
     - ✅
     - ✅
   * - MI325X
     - CDNA3
     - ✅
     - ✅
   * - MI300X / MI300A
     - CDNA3
     - ✅
     - ✅
   * - MI250X / MI250 / MI210
     - CDNA2
     - ❌
     - ✅

.. note::

   PC sampling is disabled by default and requires ``ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1``. This beta feature carries a risk of hardware freeze requiring a cold restart. Stochastic PC sampling is recommended over host-trap starting with gfx942.

For full usage details, see :ref:`using-pc-sampling` and :ref:`pc-sampling`.

.. _glance-hardware:

Supported hardware
===================

Hardware support varies by GPU architecture, firmware version, ROCm release, and feature-gate state. To verify capability on your system, run ``rocprofv3 -L`` or ``rocprofv3-avail``. For per-feature hardware tables, see :ref:`glance-att`, :ref:`glance-pc-sampling`, and :ref:`glance-spm`.

**Counter collection**: On gfx11 and gfx12 architectures (AMD Radeon RX 7000 series and newer), counter collection requires a stable power state. Use ``amd-smi`` to set the power state before profiling.

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
^^^^^^^^^^^^^

The following diagram shows how ``rocprofv3`` loads the tool library, creates profiling contexts, and writes output files after the application exits.

.. image:: /data/how_rocprofv3_works.png
   :width: 100%
   :align: center

Key CLI options
^^^^^^^^^^^^^^^^

The following tables summarize the key ``rocprofv3`` options by category. For the complete CLI reference, see :ref:`cli-options`.

.. _glance-tracing:

Tracing
""""""""

Use these options to collect API and activity traces from the GPU runtime.

.. list-table::
   :header-rows: 1

   * - Option
     - Description
     - Example
   * - ``--sys-trace``
     - Full system trace: HIP, HSA, ROCTx, memory, scratch, kernel dispatch, RCCL
     - ``rocprofv3 --sys-trace -- ./my_app``
   * - ``--hip-trace``
     - HIP runtime and compiler API tracing
     - ``rocprofv3 --hip-trace -- ./my_app``
   * - ``--hsa-trace``
     - HSA API tracing (core, AMD ext, image, finalizer)
     - ``rocprofv3 --hsa-trace -- ./my_app``
   * - ``--kernel-trace``
     - Kernel dispatch records only
     - ``rocprofv3 --kernel-trace -- ./my_app``
   * - ``--rccl-trace``
     - RCCL collective communication tracing
     - ``rocprofv3 --rccl-trace -- ./my_app``

.. _glance-counter-collection:

Counter collection
"""""""""""""""""""

Use ``--pmc`` to collect hardware performance counters. Specify multiple ``--pmc`` flags to collect more counters across multiple passes.

.. list-table::
   :header-rows: 1

   * - Option
     - Description
     - Example
   * - ``--pmc``
     - Hardware counter collection; multiple flags trigger multiple passes
     - ``rocprofv3 --pmc SQ_WAVES,SQ_INSTS_VALU -- ./my_app``

.. _glance-spm-cli:

SPM (beta)
"""""""""""

Use these options to enable and configure SPM collection. ``--spm-beta-enabled`` is required and must be combined with ``--spm`` to specify counters.

+--------------------+----------------------------------+----------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| Tool               | Option                           | Description                                              | Example                                                                                       |
+====================+==================================+==========================================================+===============================================================================================+
| ``rocprofv3``      | ``--spm-beta-enabled``           | Required to enable SPM collection                        | ``rocprofv3 --spm-beta-enabled --spm SQ_WAVES --spm-sample-interval 1000``                   |
|                    |                                  |                                                          | ``--spm-sample-interval-unit sclk_cycles -- ./my_app``                                       |
+                    +----------------------------------+----------------------------------------------------------+-----------------------------------------------------------------------------------------------+
|                    | ``--spm <COUNTERS>``             | Counters to collect; all must fit in a single            | ``rocprofv3 --spm-beta-enabled --spm SQ_WAVES,SQ_BUSY_CYCLES --spm-sample-interval 1000``    |
|                    |                                  | hardware pass                                            | ``--spm-sample-interval-unit sclk_cycles -- ./my_app``                                       |
+                    +----------------------------------+----------------------------------------------------------+-----------------------------------------------------------------------------------------------+
|                    | ``--spm-sample-interval <N>``    | Sampling interval in ``sclk_cycles``                     | ``rocprofv3 --spm-beta-enabled --spm SQ_WAVES --spm-sample-interval 1000``                   |
|                    |                                  |                                                          | ``--spm-sample-interval-unit sclk_cycles -- ./my_app``                                       |
+                    +----------------------------------+----------------------------------------------------------+-----------------------------------------------------------------------------------------------+
|                    | ``--spm-sample-interval-unit``   | Interval unit; currently accepts ``sclk_cycles`` only    | ``rocprofv3 --spm-beta-enabled --spm SQ_WAVES --spm-sample-interval 1000``                   |
|                    | ``<UNIT>``                       |                                                          | ``--spm-sample-interval-unit sclk_cycles -- ./my_app``                                       |
+--------------------+----------------------------------+----------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| ``rocprofv3-avail``| ``list --spm``                   | Lists SPM-capable counters per agent                     | ``rocprofv3-avail list --spm``                                                                |
+                    +----------------------------------+----------------------------------------------------------+-----------------------------------------------------------------------------------------------+
|                    | ``list --spm-config``            | Lists agents with SPM configuration support,             | ``rocprofv3-avail list --spm-config``                                                         |
|                    |                                  | including minimum and maximum sampling intervals         |                                                                                               |
+--------------------+----------------------------------+----------------------------------------------------------+-----------------------------------------------------------------------------------------------+

.. _glance-pc-sampling-cli:

PC sampling (beta)
"""""""""""""""""""

Use these options to enable and configure PC sampling. ``--pc-sampling-beta-enabled`` is required and also sets ``ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1`` in the environment.

.. list-table::
   :header-rows: 1

   * - Option
     - Description
     - Example
   * - ``--pc-sampling-beta-enabled``
     - Required to enable PC sampling; also sets ``ROCPROFILER_PC_SAMPLING_BETA_ENABLED=1``
     - ``rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method stochastic --pc-sampling-unit time --pc-sampling-interval 1000 -- ./my_app``
   * - ``--pc-sampling-method <METHOD>``
     - Sampling method: ``stochastic`` (default on gfx942+) or ``host_trap`` (MI200+)
     - ``rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method host_trap --pc-sampling-unit time --pc-sampling-interval 1 -- ./my_app``
   * - ``--pc-sampling-unit <UNIT>``
     - Interval unit: ``time`` (microseconds) or ``cycles``
     - ``rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method stochastic --pc-sampling-unit cycles --pc-sampling-interval 1048576 -- ./my_app``
   * - ``--pc-sampling-interval <N>``
     - Numeric sampling interval in ``time`` or ``cycles``
     - ``rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method stochastic --pc-sampling-unit time --pc-sampling-interval 1000 -- ./my_app``

.. code-block:: bash

   # Stochastic sampling — recommended on gfx942+
   rocprofv3 --pc-sampling-beta-enabled \
     --pc-sampling-method stochastic \
     --pc-sampling-unit time \
     --pc-sampling-interval 1000 \
     -- ./my_app

   # Host-trap sampling — MI200+
   rocprofv3 --pc-sampling-beta-enabled \
     --pc-sampling-method host_trap \
     --pc-sampling-unit time \
     --pc-sampling-interval 1000 \
     -- ./my_app

.. _glance-att-cli:

Thread Trace (ATT)
"""""""""""""""""""

Use ``--att`` to enable instruction-level wavefront tracing. Combine with the options below to filter kernels, embed counter values, or control the number of traced dispatches.

.. list-table::
   :header-rows: 1

   * - Option
     - Description
     - Example
   * - ``--att``
     - Enable Advanced Thread Trace
     - ``rocprofv3 --att -- ./my_app``
   * - ``--att-perfcounter-ctrl <N>``
     - PMU counter polling during trace (gfx9 only)
     - ``rocprofv3 --att --att-perfcounter-ctrl 3 -- ./my_app``
   * - ``--att-perfcounters <COUNTERS>``
     - Embed SQ-block counter values in the trace ring buffer
     - ``rocprofv3 --att --att-perfcounters SQ_INST_LEVEL_LDS -- ./my_app``
   * - ``--kernel-include-regex <RE>``
     - Filter kernels to trace by name regex
     - ``rocprofv3 --att --kernel-include-regex my_kernel -- ./my_app``
   * - ``--kernel-iteration-range <R>``
     - Trace specific dispatch iterations
     - ``rocprofv3 --att --kernel-iteration-range 1:5 -- ./my_app``
   * - ``--att-consecutive-kernels <N>``
     - Number of consecutive kernels to trace
     - ``rocprofv3 --att --att-consecutive-kernels 2 -- ./my_app``

.. code-block:: bash

   # Basic ATT collection
   rocprofv3 --att -- ./my_app

   # ATT with PMU counter streaming (gfx9 only) and kernel filter
   rocprofv3 --att \
     --att-perfcounter-ctrl 3 \
     --att-perfcounters SQ_INST_LEVEL_LDS \
     --kernel-include-regex my_kernel \
     -- ./my_app

.. _glance-process-attachment-cli:

Process attachment
"""""""""""""""""""

Use ``-p`` to attach ``rocprofv3`` to a running process. The target process must be started with ``ROCP_TOOL_ATTACH=1`` to enable attachment.

.. list-table::
   :header-rows: 1

   * - Option
     - Description
     - Example
   * - ``-p <PID>``
     - Attach to a running process; requires ``ROCP_TOOL_ATTACH=1`` at process start
     - ``rocprofv3 -p $(pidof my_training_job) --hip-trace``

.. code-block:: bash

   # Step 1: start the target with attach support enabled
   ROCP_TOOL_ATTACH=1 ./my_training_job

   # Step 2: attach from another terminal
   rocprofv3 -p $(pidof my_training_job) --hip-trace
   rocprofv3 -p $(pidof my_training_job) --pmc SQ_WAVES,SQ_INSTS_VALU
   rocprofv3 -p $(pidof my_training_job) --att
   rocprofv3 -p $(pidof my_training_job) --spm-beta-enabled --spm SQ_WAVES

.. note::

   When reattaching to a previously profiled process, data-collection options (tracing, PC sampling, ATT, counter collection) generally can't change. ``rocprofv3`` raises a ``RuntimeError`` if it detects an unsupported configuration change. Attaching requires ptrace permissions (same user or ``CAP_SYS_PTRACE``).

.. _glance-output-format-cli:

Output format
""""""""""""""

Use ``--output-format`` to specify one or more output formats in a single run. The default is rocpd (SQLite3).

.. code-block:: bash

   # Specify one or more output formats simultaneously
   rocprofv3 --output-format csv json pftrace otf2 --output-file my_run -- ./my_app

   # Default output is rocpd (SQLite3 .db)
   rocprofv3 --hip-trace -- ./my_app

Output formats
^^^^^^^^^^^^^^^

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

For comprehensive documentation, see :ref:`using-rocprofv3`.
