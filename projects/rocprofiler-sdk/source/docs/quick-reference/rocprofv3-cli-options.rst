.. meta::
  :description: ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software
  :keywords: ROCprofiler-SDK tool usage, rocprofv3 user manual, rocprofv3 usage, rocprofv3 user guide, using rocprofv3, ROCprofiler-SDK tool user guide, ROCprofiler-SDK tool user manual, using ROCprofiler-SDK tool, ROCprofiler-SDK command-line tool, ROCprofiler-SDK CLI, ROCprofiler-SDK command line tool

.. _cli-options:

==============================
rocprofv3 command-line options
==============================

The following tables list the ``rocprofv3`` command-line options categorized according to their purpose.

I/O options
=============

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``-i INPUT`` | ``--input INPUT``
     - Specifies the path to the input file. JSON and YAML formats support configuration of all command-line options for tracing and profiling whereas the text format supports only the specification of HW counters. See `collecting traces using input file <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#collecting-traces-using-input-file>`_ and `counter collection using input file <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#counter-collection-using-input-file>`_.
   * - ``-o OUTPUT_FILE`` | ``--output-file OUTPUT_FILE``
     - Specifies the output file name. If nothing is specified, the default path is ``%hostname%/%pid%``. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/rocprofv3-io-options.html#output-file>`_
   * - ``-d OUTPUT_DIRECTORY`` | ``--output-directory OUTPUT_DIRECTORY``
     - Specifies the output path for saving the output files. If nothing is specified, the default path is ``%hostname%/%pid%``. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/rocprofv3-io-options.html#output-directory>`_
   * - ``-f {csv,json,pftrace,otf2,rocpd} [...]`` | ``--output-format {csv,json,pftrace,otf2,rocpd} [...]``
     - Specifies output format. Supported formats: CSV, JSON, PFTrace, OTF2, and rocpd. Direct CSV, PFTrace, and OTF2 output is deprecated; collect in the default rocpd format and use ``rocpd convert`` instead. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#output-formats>`_
   * - ``--output-config [BOOL]``
     - Generates a configuration output file containing the resolved rocprofv3 settings and options used for the profiling session. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#configuration-output>`_
   * - ``--log-level {fatal,error,warning,info,trace,env,config}``
     - Sets the desired log level.
   * - ``-E EXTRA_COUNTERS`` | ``--extra-counters EXTRA_COUNTERS``
     - Specifies the path to a YAML file consisting of extra counter definitions. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#extra-counters>`_

Dynamic process attachment
=============================

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``-p PID`` | ``--pid PID`` | ``--attach PID``
     - Attaches to a running process by process ID and profiles it dynamically. This enables profiling of applications that are already running without needing to restart them from the profiler. The profiler instruments the target process and collects the specified tracing or counter data for the configured duration. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3-process-attachment.html>`_
   * - ``--attach-duration-msec ATTACH_DURATION_MSEC``
     - When ``--pid`` is used, sets the amount of time in milliseconds the profiler stays attached before detaching. When unset, the profiler waits until Enter is pressed to detach. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3-process-attachment.html>`_
   * - ``--attach-children [BOOL]``
     - When ``--pid`` is used, attaches to the target process and all of its descendant processes. Enabled by default; use ``--attach-children=false`` to attach only to the specified PID. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3-process-attachment.html>`_
   * - ``--attach-sync-output [BOOL]``
     - Attach mode only. Generates output files synchronously during detachment instead of the default asynchronous behavior. Use this option when scripts need to access output files immediately after ``rocprofv3`` exits. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3-process-attachment.html>`_

Aggregate tracing
====================

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``-r [BOOL]`` | ``--runtime-trace [BOOL]``
     - Collects tracing data for the HIP runtime API, marker (ROCTx) API, RCCL API, rocDecode API, rocJPEG API, rocSHMEM API, hipFILE API, HIP event barriers, memory operations (copies, scratch, and allocation), and kernel dispatches. Similar to ``--sys-trace`` without HIP compiler API and the underlying HSA API tracing. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#runtime-trace>`_
   * - ``-s [BOOL]`` | ``--sys-trace [BOOL]``
     - Collects tracing data for the HIP API, HSA API, marker (ROCTx) API, RCCL API, rocDecode API, rocJPEG API, rocSHMEM API, hipFILE API, HIP event barriers, memory operations (copies, scratch, and allocations), and kernel dispatches. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#system-trace>`_

PC sampling
==============

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``--pc-sampling-beta-enabled [BOOL]``
     - Enables PC sampling and sets the ``ROCPROFILER_PC_SAMPLING_BETA_ENABLED`` environment variable. Note that PC sampling support is in beta version. For more details, see `PC sampling <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html>`_.
   * - ``--pc-sampling-unit {instructions,cycles,time}``
     - Specifies the unit for the PC sampling interval: time (microseconds, used with host_trap) or cycles (used with stochastic). For more details, see `PC sampling <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html>`_.
   * - ``--pc-sampling-method {stochastic,host_trap}``
     - Specifies the PC sampling method: stochastic (AMD Instinct™ MI300 series and later, recommended starting gfx942) or host_trap (AMD Instinct MI200 series and later). For more details, see `PC sampling <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html>`_.
   * - ``--pc-sampling-interval PC_SAMPLING_INTERVAL``
     - Specifies the PC sample generation frequency. For more details, see `PC sampling <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html>`_.

Basic tracing
================

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``--hip-trace [BOOL]``
     - Combination of ``--hip-runtime-trace`` and ``--hip-compiler-trace`` (which in turn enables ``--hip-event-trace`` and ``--hip-graph-trace``). This option enables only the HIP API tracing. Unlike previous iterations of rocprofv3, this option doesn't enable kernel tracing, memory copy tracing, and so on. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#hip-trace>`_
   * - ``--marker-trace [BOOL]``
     - Collects marker (ROCTx) traces. Similar to the ``--roctx-trace`` option in earlier rocprof versions, but with improved ROCTx library with more features. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofiler-sdk-roctx.html#using-roctx-in-the-application>`_
   * - ``--kernel-trace [BOOL]``
     - Collects kernel dispatch traces. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#kernel-trace>`_
   * - ``--hip-graph-trace [BOOL]``
     - Collects one record per ``hipGraphLaunch`` invocation, including the graph execution ID and kernel dispatch count for each launch. Independent of ``--kernel-trace``. Automatically enabled by ``--hip-trace`` and ``--hip-runtime-trace``.
   * - ``--hip-event-trace [BOOL]``
     - Traces GPU-side HIP event barriers produced by ``hipEventRecord`` and ``hipStreamWaitEvent``. Independent of ``--kernel-trace``. Automatically enabled by ``--hip-trace`` and ``--hip-runtime-trace``. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#hip-event-trace>`_
   * - ``--memory-copy-trace [BOOL]``
     - Collects memory copy traces. This was a part of the HIP and HSA traces in previous rocprof versions. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#memory-copy-trace>`_
   * - ``--memory-allocation-trace [BOOL]``
     - Collects memory allocation traces. Displays starting address, allocation size, and the agent where allocation occurs. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#memory-allocation-trace>`_
   * - ``--kfd-trace``
     - Collects ``--kfd-page-migration-trace``, ``--kfd-page-mapping-trace``, ``--kfd-queue-trace``, and ``--kfd-dropped-events-trace``. KFD (Kernel Fusion Driver) traces capture low-level driver routines including mapping, unmapping, and migration of data between GPU and system memories, as well as eviction or restoration of GPU queues to facilitate such routines.
   * - ``--scratch-memory-trace [BOOL]``
     - Collects scratch memory operations traces. Helps in determining scratch allocations and manage them efficiently. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#scratch-memory-trace>`_
   * - ``--hsa-trace [BOOL]``
     - Collects ``--hsa-core-trace``, ``--hsa-amd-trace``, ``--hsa-image-trace``, and ``--hsa-finalizer-trace``. This option only enables the HSA API tracing. Unlike previous iterations of rocprof, this doesn't enable kernel tracing, memory copy tracing, and so on. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#hsa-trace>`_
   * - ``--rccl-trace [BOOL]``
     - Collects traces for RCCL (ROCm Communication Collectives Library). `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#rccl-trace>`_
   * - ``--kokkos-trace [BOOL]``
     - Enables builtin Kokkos tools support, which implies enabling ``--marker-trace`` collection and ``--kernel-rename``. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#kokkos-trace>`_
   * - ``--rocdecode-trace [BOOL]``
     - Collects traces for rocDecode APIs. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#rocdecode-trace>`_
   * - ``--rocjpeg-trace [BOOL]``
     - Collects traces for rocJPEG APIs.
   * - ``--rocshmem-trace [BOOL]``
     - Collects traces for rocSHMEM (ROCm SHared MEMory) host-stream APIs.
   * - ``--hipfile-trace [BOOL]``
     - Collects traces for hipFILE APIs. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#hipfile-trace>`_
   * - ``--ompt-trace [CATEGORY ...]``
     - Collects OMPT (OpenMP Tools) traces from applications linked against an OMPT-capable OpenMP runtime (for example, LLVM-based ``libomp`` from AOMP / ROCm). Captures host parallel regions, tasks, sync, mutex, and target-offload events. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#ompt-trace>`_

Granular tracing
===================

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``--hip-runtime-trace [BOOL]``
     - Collects HIP Runtime API traces. For example, public HIP API functions starting with ``hip`` such as ``hipSetDevice``.
   * - ``--hip-compiler-trace [BOOL]``
     - Collects HIP compiler generated code traces. For example, HIP API functions starting with ``__hip`` such as ``__hipRegisterFatBinary``.
   * - ``--hsa-core-trace [BOOL]``
     - Collects HSA API traces (core API). For example, HSA functions prefixed with only ``hsa_`` such as ``hsa_init``. For more details, see `HSA trace <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#hsa-trace>`_.
   * - ``--hsa-amd-trace [BOOL]``
     - Collects HSA API traces (AMD-extension API). For example, HSA functions prefixed with ``hsa_amd_`` such as ``hsa_amd_coherency_get_type``. For more details, see `HSA trace <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#hsa-trace>`_.
   * - ``--hsa-image-trace [BOOL]``
     - Collects HSA API traces (image-extension API). For example, HSA functions prefixed with only ``hsa_ext_image_`` such as ``hsa_ext_image_get_capability``. For more details, see `HSA trace <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#hsa-trace>`_.
   * - ``--hsa-finalizer-trace [BOOL]``
     - Collects HSA API traces (Finalizer-extension API). For example, HSA functions prefixed with only ``hsa_ext_program_`` such as ``hsa_ext_program_create``. For more details, see `HSA trace <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#hsa-trace>`_.
   * - ``--kfd-page-migration-trace``
     - Collects traces of KFD events including migration of pages across device memories.
   * - ``--kfd-page-mapping-trace``
     - Collects traces of KFD events including faulting, mapping, and page validation.
   * - ``--kfd-queue-trace``
     - Collects traces of KFD events including GPU queue eviction and restoration operations.
   * - ``--kfd-dropped-events-trace``
     - Collects traces of KFD events dropped by the KFD device driver.

Counter collection
======================

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``--pmc [PMC ...]``
     - Specifies performance monitoring counters to be collected. Use comma or space to specify more than one counter. For multi-pass collection, use multiple ``--pmc`` flags where each flag defines a separate counter group. The job fails if a counter group can't be collected in a single pass. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#counter-collection-using-command-line>`_

Kernel replay (beta)
=======================

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``--replay-mode {kernel,application}``
     - Selects the counter-collection replay strategy. 'kernel' collects all ``--pmc`` groups in a single application run by replaying each dispatch once per group with device-memory snapshot/restore between passes, instead of re-running the whole application per group. Requires ``--pmc`` and ``--kernel-replay-beta-enabled``. 'application' (default) re-runs the whole application once per counter group. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-kernel-replay-rocprofv3.html>`_
   * - ``--kernel-replay-beta-enabled [BOOL]``
     - Acknowledges that ``--replay-mode kernel`` is a beta feature. Required when ``--replay-mode kernel`` is specified. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-kernel-replay-rocprofv3.html>`_

Post-processing tracing
===========================

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``--stats [BOOL]``
     - Collects statistics of enabled tracing types. Must be combined with one or more tracing options. Doesn't include default kernel stats unlike previous rocprof versions. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#stats>`_
   * - ``-S [BOOL]`` | ``--summary [BOOL]``
     - Displays a single summary of tracing data for the enabled tracing type, after conclusion of the profiling session. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#summary>`_
   * - ``-D [BOOL]`` | ``--summary-per-domain [BOOL]``
     - Displays a summary of each tracing domain for the enabled tracing type, after conclusion of the profiling session. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#summary-per-domain>`_
   * - ``--summary-groups REGULAR_EXPRESSION [...]``
     - Displays a summary for each set of domains matching the specified regular expression. For example, ``KERNEL_DISPATCH|MEMORY_COPY`` generates a summary of all the tracing data in the KERNEL_DISPATCH and MEMORY_COPY domains. Similarly, ``*._API`` generates a summary of all the tracing data in the HIP_API, HSA_API, and MARKER_API domains. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#summary-groups>`_

Summary
==========

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``--summary-output-file SUMMARY_OUTPUT_FILE``
     - Outputs summary to a file, stdout, or stderr. By default, outputs to stderr. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#summary-output-file>`_
   * - ``-u {sec,msec,usec,nsec}`` | ``--summary-units {sec,msec,usec,nsec}``
     - Specifies timing unit for output summary.

Kernel naming
================

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``-M [BOOL]`` | ``--mangled-kernels [BOOL]``
     - Overrides the default demangling of kernel names. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/kernel-naming-filtering.html#kernel-name-mangling>`_
   * - ``-T [BOOL]`` | ``--truncate-kernels [BOOL]``
     - Truncates the demangled kernel names for improved readability. In earlier rocprof versions, this was known as ``--basenames [on/off]``. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/kernel-naming-filtering.html#kernel-name-truncation>`_
   * - ``--kernel-rename [BOOL]``
     - Uses region names defined using ``roctxRangePush`` or ``roctxRangePop`` to rename the kernels. Was known as ``--roctx-rename`` in earlier rocprof versions. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/kernel-naming-filtering.html#kernel-rename>`_

Filtering
============

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``--profile-mpi-ranks RANK_SPECIFICATION``
     - Specifies which MPI ranks should provide profile/trace output using comma-separated ranges and individual ranks (for example, ``0-3,8,10-15``). If not specified, all ranks provide output. The tool runs on all ranks but only selected ranks generate output files. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3-with-mpi.html#selective-rank-profiling>`_
   * - ``--mpi-world-rank-variable ENVIRONMENT_VARIABLE``
     - Specifies the environment variable to use for determining the MPI rank. If not specified, the tool automatically detects the rank from common MPI environment variables. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3-with-mpi.html#selective-rank-profiling>`_
   * - ``--mpi-world-size-variable ENVIRONMENT_VARIABLE``
     - Specifies the environment variable to use for determining the MPI world size. If not specified, the tool automatically detects the world size from common MPI environment variables. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3-with-mpi.html#selective-rank-profiling>`_
   * - ``--kernel-include-regex REGULAR_EXPRESSION``
     - Filters counter-collection and thread-trace data to include the kernels matching the specified regular expression. Non-matching kernels are excluded. For more details, see `kernel filtering <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/kernel-naming-filtering.html#kernel-filtering>`_.
   * - ``--kernel-exclude-regex REGULAR_EXPRESSION``
     - Filters counter-collection and thread-trace data to exclude the kernels matching the specified regular expression. Applied after the ``--kernel-include-regex`` option. For more details, see `kernel filtering <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/kernel-naming-filtering.html#kernel-filtering>`_.
   * - ``--kernel-iteration-range KERNEL_ITERATION_RANGE [...]``
     - Specifies the iteration range for each kernel matching the filter ``[start-stop]``. For more details, see `kernel filtering <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/kernel-naming-filtering.html#kernel-filtering>`_.
   * - ``-P (START_DELAY_TIME):(COLLECTION_TIME):(REPEAT) [...]`` | ``--collection-period (START_DELAY_TIME):(COLLECTION_TIME):(REPEAT) [...]``
     - START_DELAY_TIME is the time in seconds before data collection begins. COLLECTION_TIME is the duration of data collection in seconds. REPEAT is the number of times the data collection cycle is repeated (0 repeats indefinitely). The default unit for time is seconds, which can be changed using the ``--collection-period-unit`` option. You can specify multiple configurations, each defined by a triplet in the format ``start_delay_time:collection_time:repeat``. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/rocprofv3-io-options.html#collection-period>`_
   * - ``--collection-period-unit {hour,min,sec,msec,usec,nsec}``
     - Changes the unit of time used in ``--collection-period`` or ``-P``. The available units are ``hour``, ``min``, ``sec``, ``msec``, ``usec``, and ``nsec``. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/rocprofv3-io-options.html#collection-period>`_
   * - ``--selected-regions [BOOL]``
     - If set, rocprofv3 profiles only regions of code surrounded by ``roctxProfilerResume(0)`` and ``roctxProfilerPause(0)``. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofiler-sdk-roctx.html#using-selected-regions-option>`_
   * - ``--selected-regions-ref-count [BOOL]``
     - If set, rocprofv3 reference counts ``roctxProfilerResume`` and ``roctxProfilerPause`` calls so that only the outermost pair of nested pause/resume calls toggles profiling. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofiler-sdk-roctx.html#using-selected-regions-option>`_

Perfetto-specific
=====================

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``--perfetto-backend {inprocess,system}``
     - Specifies the backend for Perfetto data collection. When selecting 'system' mode, ensure to run the Perfetto traced daemon and then start a Perfetto session. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/rocprofv3-io-options.html#perfetto-specific-options>`_
   * - ``--perfetto-buffer-size KB``
     - Specifies the buffer size for Perfetto output in KB. Default: 1 GB. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/rocprofv3-io-options.html#perfetto-specific-options>`_
   * - ``--perfetto-buffer-fill-policy {discard,ring_buffer}``
     - Specifies the policy for handling new records when Perfetto reaches the buffer limit. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/rocprofv3-io-options.html#perfetto-specific-options>`_
   * - ``--perfetto-shmem-size-hint KB``
     - Specifies the Perfetto shared memory size hint in KB. Default: 64 KB. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/rocprofv3-io-options.html#perfetto-specific-options>`_

Display
==========

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``-L [BOOL]`` | ``--list-avail [BOOL]``
     - Lists the PC sampling configurations, SPM configurations, and metrics available in the ``counter_defs.yaml`` file for counter collection. In earlier rocprof versions, this was known as ``--list-basic``, ``--list-derived``, and ``--list-counters``. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#kernel-counter-collection>`_
   * - ``--group-by-queue [BOOL]``
     - By default, kernel and memory copy operations are grouped by the HIP stream they were submitted to when displaying a Perfetto trace. Enabling ``--group-by-queue`` instead groups them by the HSA queue they were submitted to. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/advanced-rocprofv3-options.html#group-by-queue>`_

Advanced Thread Trace (ATT)
===============================

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``--advanced-thread-trace [BOOL]`` | ``--att [BOOL]``
     - Enables thread trace. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html>`_
   * - ``--att-no-intercept [BOOL]``
     - Enables ATT quick-scan mode without kernel-dispatch interception. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html>`_
   * - ``--att-library-path ATT_LIBRARY_PATH [...]``
     - Specifies the search path to the decoder library. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html>`_
   * - ``--att-target-cu ATT_TARGET_CU``
     - Defines the compute unit (or WGP) used to gather detail tokens. Default: 1. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html#thread-trace-parameters>`_
   * - ``--att-simd-select ATT_SIMD_SELECT``
     - Defines one or more SIMDs to be traced: bitmask on gfx9 and SIMD ID on Navi. Default 0xF. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html#thread-trace-parameters>`_
   * - ``--att-consecutive-kernels ATT_CONSECUTIVE_KERNELS``
     - Starting at the targeted kernel, enables thread trace for the next N kernel dispatches, sharing a single ATT file, ``stats.csv``, and UI directory. Default: 0. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html#thread-trace-parameters>`_
   * - ``--att-buffer-size ATT_BUFFER_SIZE``
     - Specifies the thread trace buffer size, shared across shader engines. Default: 384 MB. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html#thread-trace-parameters>`_
   * - ``--att-resource-mode {default,hsa,code-object}``
     - Selects when thread trace resources (queues, signals, and memory) are allocated for each GPU. Default: code-object. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html#thread-trace-parameters>`_
   * - ``--att-shader-engine-mask ATT_SHADER_ENGINE_MASK``
     - Specifies the bitmask of shader engines to enable. Default: 0x1. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html#thread-trace-parameters>`_
   * - ``--att-gpu-index ATT_GPU_INDEX``
     - Specifies a comma-separated list of GPU index(es) to enable thread trace. Default: all. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html#thread-trace-parameters>`_
   * - ``--att-perfcounters ATT_PERFCOUNTERS``
     - (gfx9 only) Specifies the list of performance counters, and optionally their SIMD mask. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html#thread-trace-parameters>`_
   * - ``--att-perfcounter-ctrl ATT_PERFCOUNTER_CTRL``
     - (gfx9 only) Integer in the [1,32] range specifying the collection period; 0 disables collection. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html#thread-trace-parameters>`_
   * - ``--att-perfcounter-target-only [BOOL]``
     - (gfx9 only) Enables performance counters only for the target compute unit, which heavily reduces bandwidth use. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html#thread-trace-parameters>`_
   * - ``--att-activity ATT_ACTIVITY``
     - (gfx9 only) Collects HW activity counters. Integer in the [1,16] range specifying the collection period. Recommended: 8. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html#thread-trace-parameters>`_
   * - ``--att-serialize-all [BOOL]``
     - Serializes all kernels, not just the traced ones. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html#thread-trace-parameters>`_
   * - ``--att-no-detail [BOOL]``
     - Collects occupancy data without instruction-level detail. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html#thread-trace-parameters>`_

Others
=========

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``--preload PRELOAD``
     - Specifies libraries to prepend to ``LD_PRELOAD``. Useful for sanitizer libraries and custom instrumentation tools. Multiple libraries can be specified. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/advanced-rocprofv3-options.html#library-preloading>`_
   * - ``--minimum-output-data KB``
     - Specifies the minimum output data size threshold in KB. Output files are generated only if the collected profiling data exceeds this threshold. This prevents creation of empty or very small output files. Default: 0 (no threshold). `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/advanced-rocprofv3-options.html#minimum-output-data-threshold>`_
   * - ``--disable-signal-handlers [BOOL]``
     - Controls signal handler prioritization. When set to true, disables rocprofv3 signal handler prioritization, allowing application signal handlers to take precedence. Useful for applications with custom crash handling or when integrating with testing frameworks. Default: false (rocprofv3 handlers have priority). `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/advanced-rocprofv3-options.html#signal-handler-control>`_
   * - ``--process-sync [BOOL]``
     - Enables process synchronization in the rocprofv3 tool finalization stage. When set to true, the rocprofv3 tool forces the process to wait for its peer processes to finish writing trace data before proceeding. Note that some workloads terminate the process group when one of the processes finishes.
   * - ``--rocm-root PATH``
     - Specifies custom ROCm installation directory instead of automatic detection. Useful for multiple ROCm installations, custom builds, or non-standard locations. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/advanced-rocprofv3-options.html#rocm-root-path-configuration>`_
   * - ``--sdk-soversion SDK_SOVERSION``
     - Specifies the shared object version number for ROCProfiler SDK library resolution. Controls which major version of ``librocprofiler-sdk.so.X`` to use. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/advanced-rocprofv3-options.html#sdk-shared-object-version-control>`_
   * - ``--sdk-version SDK_VERSION``
     - Specifies the exact version number for ROCProfiler SDK library resolution. Controls library selection with full semantic versioning (X.Y.Z format). `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/advanced-rocprofv3-options.html#sdk-version-specification>`_
   * - ``-A {absolute,relative,type-relative}`` | ``--agent-index {absolute,relative,type-relative}``
     - Specifies how agent indices are represented in the output files. Default: relative. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/advanced-rocprofv3-options.html#agent-index>`_

SPM
======

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``--spm-beta-enabled [BOOL]``
     - Required to enable SPM collection. Note that SPM support is in beta version. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-spm.html#spm-cli-options>`_
   * - ``--spm [COUNTERS ...]``
     - Specifies SPM events to collect (comma or space separated for more than one counter). All counters must fit in a single hardware pass. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-spm.html#spm-cli-options>`_
   * - ``--spm-sample-interval SPM_SAMPLE_INTERVAL``
     - Specifies the sampling interval for SPM counter collection. Used with ``--spm-sample-interval-unit`` to define how frequently counters are sampled. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-spm.html#spm-cli-options>`_
   * - ``--spm-sample-interval-unit {sclk_cycles,refclk_cycles}``
     - Specifies the unit for the SPM sample interval. Used with ``--spm-sample-interval`` to define the sampling interval. `Read more... <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-spm.html#spm-cli-options>`_

To see the exhaustive list of ``rocprofv3`` options, use:

.. code-block:: bash

    rocprofv3 -h
    rocprofv3 --help

To display the version information for ``rocprofv3``, use:

.. code-block:: bash

    rocprofv3 -v
    rocprofv3 --version

The version command provides comprehensive build and system information including the following:

.. code-block:: shell

    $ rocprofv3 -v
                 version: 1.0.0
            git_revision: a1b2c3d4e5f6789012345678901234567890abcd
            library_arch: x86_64-linux-gnu
             system_name: Linux
        system_processor: x86_64
          system_version: 6.8.0-57-generic
             compiler_id: GNU
        compiler_version: 11.4.0
            rocm_version: 6.2.0
