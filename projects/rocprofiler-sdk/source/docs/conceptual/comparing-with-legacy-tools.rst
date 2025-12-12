.. meta::
  :description: ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software
  :keywords: ROCprofiler-SDK tool, ROCprofiler-SDK library, rocprofv3, ROCm, API, reference

.. _comparing-with-legacy-tools:

=========================================================
Comparing ROCprofiler-SDK to legacy ROCm profiling tools
=========================================================

This topic highlights the differences between the ROCprofiler-SDK and the legacy ROCm profiling tools: `ROCProfiler <https://rocm.docs.amd.com/projects/rocprofiler/en/latest/index.html>`_ and `ROCTracer <https://rocm.docs.amd.com/projects/roctracer/en/latest/index.html>`_. The comparison also includes differences between the ROCprofiler-SDK command-line (CLI) tool: ``rocprofv3`` and legacy ROCProfiler CLI tools: ``rocprof`` and ``rocprofv2``.

ROCprofiler-SDK is an improved version of the ROCm profiling tools that enables more efficient implementations and better thread safety while avoiding the problems that hindered the legacy ROCm profiling tools.
Here are the distinct ROCprofiler-SDK features, which also highlight the improvements over ROCProfiler and ROCTracer:

- Improved tool initialization
- Support for simultaneous use of the same services by multiple tools
- Simplified control of one or more data collection services
- Improved error checking and logging
- Backward ABI compatibility
- PC sampling (beta implementation)

The legacy ROCm profiling tools allowed a tool to access any of the services provided by ROCProfiler or ROCTracer, such as API tracing and kernel tracing, by calling ``roctracer_init()`` when an ROCm runtime is initially loaded.
Since the services to be used by the calling tool are not required to be specified during initialization, the libraries must be effectively prepared for any service to be available at any time.
This behavior introduced unnecessary overhead and made thread-safe data management difficult, as tools generally don't use all the available services.
For example, ROCTracer always installed wrappers around every runtime API and added indirection overhead through the ROCTracer library to check for the current service configuration in a thread-safe manner.

ROCprofiler-SDK introduces ``context`` to solve the preceding issues. Contexts are effectively bundles of service configurations. ROCprofiler-SDK provides a single opportunity for a tool to create as many contexts as required.
A tool can group all services into a single context, create a separate context for each service, or use a mix.
This change in the design allows ROCprofiler-SDK to be aware of the services that might be requested by a tool at any given time.
The design change empowers ROCprofiler-SDK to:

- Avoid unnecessary preparation for services that are never used. If no registered contexts request HSA API tracing, no wrappers need to be generated.
- Perform more extensive checks during service specification and inform the tool about potential issues early.
- Allow multiple tools to use certain services simultaneously.
- Improve thread safety without introducing parallel bottlenecks.
- Manage internal data and allocations more efficiently.

Command-line tool options: rocprofv3 versus rocprof and rocprofv2
==================================================================

The following table provides a comparison between the CLI tool options of ``rocprofv3``, ``rocprof``, and ``rocprofv2``. The comparison indicates that ``rocprofv3`` is more efficient and flexible than ``rocprof`` and ``rocprofv2``.

.. list-table:: Comparing CLI tool options
   :header-rows: 1

   * - Category
     - Feature
     - rocprof
     - rocprofv2
     - rocprofv3
     - Improvements
     - Notes
   * - Basic tracing options
     - HIP trace
     - ``--hip-trace``
     - ``--hip-api``, ``--hip-trace``
     - ``--hip-trace``
     - No change
     - | rocprof and rocprofv2 ``--hip-trace`` options include kernel dispatches and memory copy activities,
       | unlike ``rocprofv3``.
   * - Basic tracing options
     - HSA trace
     - ``--hsa-trace``
     - ``--hsa-trace``
     - ``--hsa-trace``
     - No change
     - | rocprof and rocprofv2 ``--hsa-trace`` options include kernel dispatches and memory copy activities,
       | unlike ``rocprofv3``.
   * - Basic tracing options
     - Scratch memory trace
     - *Not available*
     - *Not available*
     - ``--scratch-memory-trace``
     - New option to trace scratch memory operations.
     -
   * - Basic tracing options
     - Marker trace (ROCTx)
     - ``--roctx-trace``
     - ``--roctx-trace``
     - ``--marker-trace``
     - Improved ROCTx library with more features.
     -
   * - Basic tracing options
     - Memory copy trace
     - Part of HIP and HSA traces
     - Part of HIP and HSA traces
     - ``--memory-copy-trace``
     - Provides granularity for memory move operations.
     -
   * - Basic tracing options
     - Memory allocation trace
     - *Not available*
     - *Not available*
     - ``--memory-allocation-trace``
     - New option for collecting memory allocation traces. Displays starting address, allocation size, and agent where allocation occurred.
     -
   * - Basic tracing options
     - Kernel trace
     - ``--kernel-trace``
     - ``--kernel-trace``
     - ``--kernel-trace``
     - Performance improvement.
     -
   * - Granular tracing options
     - HIP runtime trace
     - Part of ``--hip-trace`` option
     - Part of ``--hip-trace`` option
     - ``--hip-runtime-trace``
     - For collecting HIP runtime API traces. For example, public HIP API functions starting with "hip" such as ``hipSetDevice``.
     -
   * - Granular tracing options
     - HIP compiler trace
     - *Not available*
     - *Not available*
     - ``--hip-compiler-trace``
     - For collecting HIP compiler-generated code traces. For example, HIP API functions starting with "__hip" such as ``__hipRegisterFatBinary``.
     -
   * - Granular tracing options
     - HSA core API trace
     - Part of ``--hsa-trace`` option
     - Part of ``--hsa-trace`` option
     - ``--hsa-core-trace``
     - New option for collecting only HSA API traces (core API). For example, HSA functions prefixed only with "hsa_" such as ``hsa_init``.
     -
   * - Granular tracing options
     - HSA AMD trace
     - Part of ``--hsa-trace`` option
     - Part of ``--hsa-trace`` option
     - ``--hsa-amd-trace``
     - For collecting HSA API traces (AMD-extension API). For example, HSA functions prefixed with "hsa_amd_" such as ``hsa_amd_coherency_get_type``.
     -
   * - Granular tracing options
     - HSA image extension trace
     - Part of ``--hsa-trace`` option
     - Part of ``--hsa-trace`` option
     - ``--hsa-image-trace``
     - New option for collecting HSA API traces (Image-extension API). For example, HSA functions prefixed only with ``hsa_ext_image_`` such as ``hsa_ext_image_get_capability``.
     -
   * - Granular tracing options
     - HSA finalizer trace
     - Part of ``--hsa-trace`` option
     - Part of ``--hsa-trace`` option
     - ``--hsa-finalizer-trace``
     - New option for collecting HSA API traces (Finalizer-extension API). For example, HSA functions prefixed only with "hsa_ext_program_" such as ``hsa_ext_program_create``.
     -
   * - Advanced tracing options
     - Kokkos trace
     - *Not available*
     - *Not available*
     - ``--kokkos-trace``
     - New option to enable built-in Kokkos tools support (implies ``--marker-trace`` and ``--kernel-rename``).
     -
   * - Advanced tracing options
     - RCCL trace
     - *Not available*
     - *Not available*
     - ``--rccl-trace``
     - For collecting ROCm Communication Collectives Library (RCCL, also pronounced as "Rickle") traces.
     -
   * - Advanced tracing options
     - Scratch memory trace
     - *Not available*
     - *Not available*
     - ``--scratch-memory-trace``
     - For collecting scratch memory event traces.
     -
   * - Advanced tracing options
     - rocDecode trace
     - *Not available*
     - *Not available*
     - ``--rocdecode-trace``
     - Tracing ``rocDecode`` library.
     -
   * - Advanced tracing options
     - rocJPEG trace
     - *Not available*
     - *Not available*
     - ``--rocjpeg-trace``
     - Tracing rocJPEG library.
     -
   * - Aggregate tracing options
     - Sys trace
     - ``--sys-trace`` [``hip-trace``| ``hsa-trace``| ``roctx-trace``| ``kernel-trace``].
     - ``--sys-trace`` [``hip-trace``| ``hsa-trace``| ``roctx-trace``| ``kernel-trace``].
     - ``-s``, ``--sys-trace`` [``hip-trace``| ``hsa-trace``| ``scratch-trace``| ``memory-copy-trace``| ``roctx-trace``| ``kernel-trace``].
     - Extends the ``sys-trace`` options with more features.
     -
   * - Aggregate tracing options
     - Runtime trace
     - *Not available*
     - *Not available*
     - ``-r``, ``--runtime-trace`` [``hip-runtime-trace``| ``scratch-trace``| ``memory-copy-trace``| ``roctx-trace``| ``kernel-trace``].
     - New option to aggregate trace operations.
     -
   * - Kernel naming options
     - Kernel name mangling
     - *Not available*
     - *Not available*
     - ``-M``, ``--mangled-kernels``
     - New option for mangled kernel names.
     -
   * - Kernel naming options
     - Kernel name truncation
     - ``--basenames <on|off>``
     - ``--basenames``
     - ``-T``, ``--truncate-kernels``
     - New option for truncating the demangled kernel names.
     -
   * - Kernel naming options
     - Kernel rename
     - ``--roctx-rename``
     - *Not available*
     - ``--kernel-rename``
     - New option to use region names defined by ``roctxRangePush``/ ``roctxRangePop`` regions for renaming the kernels.
     -
   * - Post-processing tracing options
     - Statistics
     - ``--stats``
     - *Not available*
     - ``--stats``
     - Statistics for the collected traces.
     -
   * - Post-processing tracing options
     - Summary
     - *Not available*
     - *Not available*
     - ``-S``, ``--summary``
     - New option to output a single summary of tracing data after the profiling session.
     - ``rocprof`` generated the post-processing step's summary, stats, JSON, and database files with lesser information.
   * - Post-processing tracing options
     - Summary per domain
     - *Not available*
     - *Not available*
     - ``-D``, ``--summary-per-domain``
     - New option to output a summary for each tracing domain after the profiling session.
     - Compared to ``rocprofv3``, ``rocprof --stats`` option captured lesser number of domains in the summary reports.
   * - Post-processing tracing options
     - Summary groups
     - *Not available*
     - *Not available*
     - ``--summary-groups REGULAR_EXPRESSION``
     - New option to output a summary for each set of domains matching the regular expression. For example, ``KERNEL_DISPATCH``| ``MEMORY_COPY`` generate a summary from all the tracing data in the ``KERNEL_DISPATCH`` and ``MEMORY_COPY`` domains.
     -
   * - Summary options
     - Summary output file
     - *Not available*
     - *Not available*
     - ``--summary-output-file SUMMARY_OUTPUT_FILE``
     - New option to output summary to a file, stdout, or stderr (default: stderr).
     -
   * - Summary options
     - Summary units
     - *Not available*
     - *Not available*
     - ``-u`` , ``--summary-units``
     - New option to output summary in desired time units {sec,msec,usec,nsec}.
     -
   * - Display options
     - List available basic and derived metrics and PC sampling configurations
     - ``--list-basic``, ``--list-derived``
     - ``--list-counters``
     - ``-L``, ``--list-avail``
     - A valid YAML is supported for this option now.
     -
   * - Perfetto-specific options
     - Perfetto data collection backend
     - *Not available*
     - *Not available*
     - ``--perfetto-backend`` {in-process,system}
     - New option for Perfetto data collection backend. "system" mode requires starting traces and Perfetto daemons.
     - ``rocprofv2`` used only in-process collection for Perfetto plugin while ``rocprofv3`` gives options to the user.
   * - Perfetto-specific options
     - Perfetto buffer size
     - *Not available*
     - Setting environment variable ``rocprofiler_PERFETTO_MAX_BUFFER_SIZE_KIB`` to the desired buffer size.
     - ``--perfetto-buffer-size`` {KB}
     - New option to define the buffer size for Perfetto output in KB. default: 1 GB.
     -
   * - Perfetto-specific options
     - Perfetto buffer fill policy
     - *Not available*
     - *Not available*
     - ``--perfetto-buffer-fill-policy {discard,ring_buffer}``
     - New option for handling new records when Perfetto has reached the buffer limit.
     - ``rocprofv2`` always used ``TraceConfig_BufferConfig_FillPolicy_RING_BUFFER`` fill policy.
   * - Perfetto-specific options
     - Perfetto shared memory size
     - *Not available*
     - *Not available*
     - ``--perfetto-shmem-size-hint`` KB
     - New option to define Perfetto shared memory size hint in KB. default: 64 KB.
     -
   * - Filtering options
     - Kernel filtration options for counter collection
     - Supported in input.xml file (supports range, gpu and kernel filtration).
     - kernel: <kernel_name> (can only be provided in input.txt file).
     - ``--kernel-include-regex``, ``--kernel-exclude-regex``, ``--kernel-iteration-range``
     - Extensive control over output options using regular expressions.
     -
   * - I/O options
     - Output directory
     - ``-d`` <data directory>
     - ``-d`` | ``--output-directory``
     - ``-d OUTPUT_DIRECTORY``, ``--output-directory OUTPUT_DIRECTORY``
     - ``rocprofv3`` supports special keys for runtime values. For example, "%pid%" gets replaced with the process ID.
     -
   * - I/O options
     - Output file
     - ``-o`` <output file>
     - ``-o`` | ``--output-file-name``
     - ``-o OUTPUT_FILE``, ``--output-file OUTPUT_FILE``
     - ``rocprofv3`` supports special keys for runtime values. For example, "%pid%" gets replaced with the process ID.
     -
   * - I/O options
     - Logging
     - Minimal logging using environment variable.
     - Minimal logging using environment variable.
     - ``--log-level {fatal,error,warning,info,trace,env}``
     - Extensive logging options
     -
   * - I/O options
     - Plugins
     - *Not available*
     - plugin support for different output formats.
     - Replaced with ``--output-format`` option.
     - Not needed as ``rocprofv3`` supports multiple output formats.
     -
   * - I/O options
     - Output formats
     - CSV, JSON (Chrome-Tracing format)
     - CSV, JSON (Chrome-Tracing format), Perfetto, CTF
     - CSV, JSON (custom schema), Perfetto, OTF2
     - | Multiple output formats can be supported in a single run.
       | OTF2 can visualize larger trace files compared to Perfetto.
     - The Perfetto UI doesn't accept the JSON output format generated by ``rocprofv3``. With JSON Chrome tracing format being treated as legacy, Perfetto encourages the ``rocprofv3``-supported binary Perfetto protobuf format (``.pftrace`` extension).
   * - I/O options
     - Counter collection
     - Supports input text and XML format.
     - Only supports input text format.
     - Input support for text, YAML and JSON formats.
     - | It's not possible to check for valid text file. Hence ``rocprofv3`` supports strongly typed input formats.
       | YAML and JSON formats are more readable and easier to maintain.
       | Allows flexibility to add more features for the tool input.
     -
   * - I/O options
     - Command-line counter collection
     - *Not available*
     - *Not available*
     - ``--pmc``
     - New option to collect performance counters from command line. When specifying multiple counters, they must be comma or space-separated.
     -
   * - I/O options
     - Providing custom metrics file
     - ``-m <metric file>``
     - ``-m` <metric file>``
     - ``-E <metric file> --pmc <counter>``
     - This option has been modified in ``rocprofv3`` to collect performance counters from the command line by using ``--pmc`` option and specifying a file with custom metrics.
     -
   * - Advanced options
     - Preload
     - *Not available*
     - *Not available*
     - ``--preload``
     - Libraries to prepend to ``LD_PRELOAD`` (usually for sanitizers).
     -
   * - Trace control options
     - Trace period
     - ``--trace-period``
     - ``-tp`` | ``--trace-period``
     - ``-P`` | ``--collection-period``, ``--collection-period-unit``
     - Users can specify multiple configurations, each defined by a triplet in the format "start_delay:collection_time:repeat", with the ability to change the unit of time in the given configurations.
     -
   * - Trace control options
     - Trace start
     - ``--trace-start <on|off>``
     - *Not available*
     - *Not available*
     - Not available in ``rocprofv3``
     -
   * - Trace control options
     - Flush interval
     - ``--flush-rate``
     - ``--flush-interval``
     - *Not available*
     - Not applicable for ``rocprofv3``
     -
   * - Trace control options
     - Merge traces
     - ``--merge-traces``
     - *Not available*
     - *Not available*
     - Not available in ``rocprofv3``
     -
   * - PC sampling options
     - PC sampling
     - *Not available*
     - *Not available*
     - ``--pc-sampling-beta-enabled``
     - Enables PC sampling support; beta version.
     -
   * - Legacy options
     - Timestamp On/Off
     - ``--timestamp <on|off>``
     - *Not available*
     - *Not available*
     - Not applicable for ``rocprofv3``.
     -
   * - Legacy options
     - Context wait
     - ``--ctx-wait``
     - *Not available*
     - *Not available*
     - Not applicable for ``rocprofv3``.
     -
   * - Legacy options
     - Context limit
     - ``--ctx-limit <max number>``
     - *Not available*
     - *Not available*
     - Not applicable for ``rocprofv3``.
     -
   * - Legacy options
     - Code object tracking
     - ``--obj-tracking <on|off>``
     - Always ``ON`` in ``rocprofv2``.
     - Always ``ON`` in ``rocprofv3``.
     -
     -
   * - Legacy options
     - Heartbeat
     - ``--heartbeat <rate sec>``
     - *Not available*
     - *Not available*
     - Not applicable for ``rocprofv3``
     -

Timing information: rocprofv3 versus rocprof and rocprofv2
===========================================================

``rocprofv3`` provides more accurate timing information by reducing the tool overhead required to collect data and the interference with the timing of the kernel being measured. This results in a reduced kernel time variance received for the same kernel execution and more accurate timing overall. These changes are not available in ``rocprof`` and ``rocprofv2``, so there can be substantial differences (up to 20 percent) in execution times reported by ``rocprof`` and ``rocprofv2`` for a single kernel execution in comparison with ``rocprofv3``. Across a large number of samples of the same kernel, the difference in the average execution time is limited to a single digit, with a much tighter variance of results on ``rocprofv3``.

Default behavior: rocprofv3 versus rocprof and rocprofv2
=========================================================

When run without an option, ``rocprofv3`` behaves differently than ``rocprof`` and ``rocprofv2``. The default behavior of ``rocprofv3`` is to collect all available agents on the system and output them in ``CSV`` format, while ``rocprof`` and ``rocprofv2`` output the kernel traces in ``CSV`` format by default. On ``rocprofv3``, kernel traces are generated using ``--kernel-trace`` option.
