.. meta::
  :description: Documentation of the installation, configuration, use of the ROCprofiler-SDK, and rocprofv3 command-line tool 
  :keywords: ROCprofiler-SDK tool, ROCprofiler-SDK library, rocprofv3, ROCm, API, reference

.. _comparing-with-legacy-tools:

========================================================
Comparing ROCprofiler-SDK to other ROCm profiling tools
========================================================

ROCprofiler-SDK is an improved version of ROCm profiling tools that enables more efficient implementations and better thread safety while avoiding problems that plague the former implementations of ROCProfiler and ROCTracer.
Here are the distinct ROCprofiler-SDK features, which also highlight the improvements over ROCProfiler and ROCTracer:

- Improved tool initialization
- Support for simultaneous use of the same services by multiple tools
- Simplified control of one or more data collection services
- Improved error checking and logging
- Backward ABI compatibility
- PC sampling (beta implementation)

The former implementations allow a tool to access any of the services provided by ROCProfiler or ROCTracer, such as API tracing and kernel tracing, by calling ``roctracer_init()`` when an ROCm runtime is initially loaded.
As the calling tool is not required to specify during initialization, the services it needs to use, the libraries must be effectively prepared for any service to be available anytime. 
This behavior introduces unnecessary overhead and makes thread-safe data management difficult, as tools generally don't use all the available services.
For example, ROCTracer always installs wrappers around every runtime API and adds indirection overhead through the ROCTracer library to check for the current service configuration in a thread-safe manner.

ROCprofiler-SDK introduces `context` to solve the preceding issues. Contexts are effectively bundles of service configurations. ROCprofiler-SDK provides a single opportunity for a tool to create as many contexts as required. 
A tool can group all services into one context, create one context per service, or choose a mix.
This change in the design allows ROCprofiler-SDK to be aware of the services that might be requested by a tool at any given time.
The design change empowers ROCprofiler-SDK to:

- Avoid unnecessary preparation for services that are never used. If no registered contexts request HSA API tracing, no wrappers need to be generated.
- Perform more extensive checks during service specification and inform a tool about potential issues early.
- Allow multiple tools to use certain services simultaneously.
- Improve thread safety without introducing parallel bottlenecks.
- Manage internal data and allocations more efficiently.

===================================================================================================
Comparing command-line tool options: ROCprofiler(rocprof, rocprofv2) and ROCprofiler-SDK(rocprofv3)
===================================================================================================

ROCprofiler-SDK introduces a new command-line tool, `rocprofv3`, which is a more efficient and flexible version of the ROCprofiler tool.

.. list-table:: Comparison of ROCprofiler Command-Line Tool's options
   :header-rows: 1

   * - Category
     - Feature
     - rocprof
     - rocprofv2
     - rocprofv3
     - Improvements
     - Notes
   * - Basic tracing options 
     - HIP Trace
     - `--hip-trace`
     - `--hip-api`, `--hip-trace`
     - `--hip-trace` 
     - No change
     - | rocprof and rocprofv2 `--hip-trace` options include kernel dispatches and memory copy activities,
       | which is not the case in rocprofv3
   * - Basic tracing options 
     - HSA Trace
     - `--hsa-trace`
     - `--hsa-trace`
     - `--hsa-trace`
     - No change
     - | rocprof and rocprofv2 `--hsa-trace` options include kernel dispatches and memory copy activities, 
       | which is not the case in rocprofv3
   * - Basic tracing options 
     - Scratch Memory Trace
     - *Not Available*
     - *Not Available*
     - `--scratch-memory-trace`
     - New option to trace scratch memory operations
     - 
   * - Basic tracing options
     - Marker Trace(ROCTx)
     - `--roctx-trace`
     - `--roctx-trace`
     - `--marker-trace`
     - Improved ROCTx library with more features
     - 
   * - Basic tracing options
     - Memory Copy Trace
     - Part of HIP and HSA Traces
     - Part of HIP and HSA Traces
     - `--memory-copy-trace`
     - Provides granularity for memory move operations
     - 
   * - Basic tracing options
     - Kernel Trace
     - `--kernel-trace`
     - `--kernel-trace`
     - `--kernel-trace`
     - Performance improvement.
     - 
   * - Granular tracing options
     - HIP runtime trace
     - Part of `--hip-trace` option
     - Part of `--hip-trace` option
     - `--hip-runtime-trace`
     - For collecting HIP Runtime API Traces, e.g. public HIP API functions starting with 'hip' (i.e. hipSetDevice).
     - 
   * - Granular tracing options
     - HIP compiler trace
     - *Not Available*
     - *Not Available*
     - `--hip-compiler-trace`
     - For collecting HIP Compiler generated code Traces, e.g. HIP API functions starting with '__hip' (i.e. __hipRegisterFatBinary).
     - 
   * - Granular tracing options
     - HSA core API trace
     - Part of `--hsa-trace` option
     - Part of `--hsa-trace` option
     - `--hsa-core-trace`
     - New option for collecting only HSA API Traces (core API), e.g. HSA functions prefixed with only `hsa_` (i.e. hsa_init)
     - 
   * - Granular tracing options
     - HSA AMD trace
     - Part of `--hsa-trace` option
     - Part of `--hsa-trace` option
     - `--hsa-amd-trace`
     - For collecting HSA API Traces (AMD-extension API), e.g. HSA function prefixed with `hsa_amd_` (i.e. hsa_amd_coherency_get_type)
     - 
   * - Granular tracing options
     - HSA Image Extension trace
     - Part of `--hsa-trace` option
     - Part of `--hsa-trace` option
     - `--hsa-image-trace`
     - New option for collecting HSA API Traces (Image-extenson API), e.g. HSA functions prefixed with only `hsa_ext_image_` (i.e. hsa_ext_image_get_capability).
     - 
   * - Granular tracing options
     - HSA Finalizer trace
     - Part of `--hsa-trace` option
     - Part of `--hsa-trace` option
     - `--hsa-finalizer-trace`
     - New option for collecting HSA API Traces (Finalizer-extension API), e.g. HSA functions prefixed with only `hsa_ext_program_` (i.e. hsa_ext_program_create)
     - 
   * - Aggregate tracing options
     - Sys Trace
     - `--sys-trace` [hip-trace|hsa-trace|roctx-trace|kernel-trace]
     - `--sys-trace` [hip-trace|hsa-trace|roctx-trace|kernel-trace]
     - ` -s, --sys-trace` [hip-trace|hsa-trace|scratch-trace|memory-copy-trace|roctx-trace|kernel-trace]
     - Extends the sys trace options with more features
     - 
   * - Aggregate tracing options
     - Runtime Trace
     - *Not available*
     - *Not available*
     - ` -r, --runtime-trace` [hip-runtime-trace|scratch-trace|memory-copy-trace|roctx-trace|kernel-trace]
     - New option to aggregate trace operations
     - 
   * - Kernel naming options
     - Kernel Name Mangling
     - *Not Available*
     - *Not Available*
     - `-M`, `--mangled-kernels`
     - New option for mangled  kernel names
     - 
   * - Kernel naming options
     - Kernel Name Truncation
     - `--basenames  <on|off>`
     - `--basenames`
     - `-T`, `--truncate-kernels`
     - New option for truncating the demangled  kernel names
     - 
   * - Kernel naming options
     - Kernel Rename
     - `--roctx-rename`
     - *Not available*
     - `--kernel-rename`
     - New option to use region names defined by roctxRangePush/roctxRangePop regions to rename the kernels
     - 
   * - Post-processing tracing options
     - Statistics
     - --stats
     - *Not Available*
     - --stats
     - Statistics for the collected traces
     - 
   * - Post-processing tracing options
     - Summary
     - *Not available*
     - *Not available*
     - `-S, --summary`
     - New option to output a single summary of tracing data after the profiling session
     - `rocprof` generated the post-processing step's summary, stats, JSON, and database files with much less information.
   * - Post-processing tracing options
     - Summary Per Domain
     - *Not available*
     - *Not available*
     - `-D, --summary-per-domain`
     - New option to output summary for each tracing domain after the profiling session
     - `rocprof --stats` option had less number of domains in the summary reports than `rocprofv3`
   * - Post-processing tracing options
     - Summary Groups
     - *Not available*
     - *Not available*
     - `--summary-groups REGULAR_EXPRESSION`
     - New option to output a summary for each set of domains matching the regular expression, e.g. 'KERNEL_DISPATCH|MEMORY_COPY' will generate a summary from all the tracing data in the KERNEL_DISPATCH and MEMORY_COPY domains
     - 
   * - Summary options
     - Summary Output File
     - *Not available*
     - *Not available*
     - `--summary-output-file SUMMARY_OUTPUT_FILE`
     - New option to output summary to a file, stdout, or stderr (default: stderr)
     - 
   * - Summary options
     - Summary Units
     - *Not available*
     - *Not available*
     - `-u , --summary-units`
     - New option to output summary in desired time units {sec,msec,usec,nsec}
     - 
   * - Display options
     - List Metrics
     - `--list-basic`, `--list-derived`
     - `--list-counters`
     - `-L`, `--list-metrics`
     - A valid YAML is supported for this option now
     - 
   * - Perfetto-specific options
     - Perfetto data collection backend
     - *Not available*
     - *Not available*
     - `--perfetto-backend` {inprocess,system}
     - New option for perfetto data collection backend. 'system' mode requires starting traced and perfetto daemons
     - `rocprofv2` used only in-process collection for perfetto plugin, However, `rocprofv3` give the option to the user
   * - Perfetto-specific options
     - Perfetto Buffer Size
     - *Not available*
     - Setting env variable `rocprofiler_PERFETTO_MAX_BUFFER_SIZE_KIB` to the desired buffer size
     - `--perfetto-buffer-size` {KB}
     - New option to define size of buffer for perfetto output in KB. default: 1 GB
     - 
   * - Perfetto-specific options
     - Perfetto Buffer fill Policy
     - *Not available*
     - *Not available*
     - `--perfetto-buffer-fill-policy` {discard,ring_buffer}
     - New option or handling new records when perfetto has reached the buffer limit
     - `rocprofv2` always used `TraceConfig_BufferConfig_FillPolicy_RING_BUFFER` fill policy.
   * - Perfetto-specific options
     - Perfetto shared memory size
     - *Not available*
     - *Not available*
     - `--perfetto-shmem-size-hint` KB
     - New option to define perfetto shared memory size hint in KB. default: 64 KB
     - 
   * - Filtering options
     - Kernel Filtration options for Counter Collection
     - Supported in input.xml file (supports range, gpu and kernel filtration)
     - kernel: <kernel_name> (can only be provided in input.txt file)
     - `--kernel-include-regex`, `--kernel-exclude-regex`, `--kernel-iteration-range`
     - Extensive control over output options using regular expressions
     - 
   * - I/O options
     - Output Directory
     - `-d` <data directory>
     - `-d`   | `--output-directory`
     - `-d` OUTPUT_DIRECTORY, `--output-directory` OUTPUT_DIRECTORY
     - rocprofv3 supports special keys for runtime values, e.g. %pid% gets replaced by the process ID
     - 
   * - I/O options
     - Output File
     - `-o` <output file>
     - `-o`   | `--output-file-name`
     - `-o` OUTPUT_FILE, `--output-file` OUTPUT_FILE
     - rocprofv3 supports special keys for runtime values, e.g. %pid% gets replaced by the process ID
     - 
   * - I/O options
     - Logging
     - Minimal logging via environment variable
     - Minimal logging via environment variable
     - --log-level {fatal,error,warning,info,trace,env}
     - Extensive logging options
     - 
   * - I/O options
     - Plugins
     - *Not Available*
     - plugin support for different output formats
     - Replaced by `--output-format` option
     - Not needed as rocprofv3 supports multiple output formats
     - 
   * - I/O options
     - Output Formats
     - CSV, JSON (Chrome-Tracing format)
     - CSV, JSON (Chrome-Tracing format), Perfetto, CTF
     - CSV, JSON (custom schema), Perfetto, OTF2
     - | # Multiple output formats can be supported in single run. 
       | # OTF2 can visualize larger trace files compared to perfetto.
     - The Perfetto UI does not accept the JSON output format produced by rocprofv3. Perfetto is dropping support for the JSON Chrome tracing format in favor of the binary Perfetto protobuf format (``.pftrace`` extension), which is supported by rocprofv3.
   * - I/O options
     - Counter Collection
     - Supports input text and XML format
     - Only supports input text format
     - Input support for text, YAML and JSON formats
     - | # Its not possible to check for valid text file. Hence rocprofv3 supports strongly typed input formats.
       | # YAML and JSON formats are more readable and easy to maintain.
       | # Allows flexibility to add more features for the tool input
     -
   * - I/O options   
     - Providing Custom metrics file
     - `-m`  <metric file>
     - `-m`  <metric file>
     - Not available
     - Not yet in rocprofv3
     - 
   * - Advanced options
     - Preload
     - *Not Available*
     - *Not Available*
     - --preload
     - Libraries to prepend to LD_PRELOAD (usually for sanitizers)
     - 
   * - Trace Control options
     - Trace Period
     - `--trace-period`
     - `-tp | --trace-period`
     - *Not available*
     - Not yet in rocprofv3
     - 
   * - Trace Control options
     - Trace start
     -  `--trace-start <on|off>`
     - *Not available*
     - *Not available*
     - Not yet in rocprofv3
     - 
   * - Trace Control options
     - Flush Interval
     - `--flush-rate`
     - `--flush-interval`
     - *Not available*
     - Not applicable for rocprofv3
     - 
   * - Trace Control options
     - Merge Traces
     - `--merge-traces`
     - *Not available*
     - *Not available*
     - Not yet in rocprofv3
     - 
   * - Legacy options
     - Timestamp On/Off
     - `--timestamp <on|off>`
     - *Not available*
     - *Not available*
     - Not applicable for rocprofv3
     - 
   * - Legacy options
     - Context wait
     - `--ctx-wait`
     - *Not available*
     - *Not available*
     - Not applicable for rocprofv3
     - 
   * - Legacy options
     - Context Limit
     - `--ctx-limit <max number>`
     - *Not available*
     - *Not available*
     - Not applicable for rocprofv3
     - 
   * - Legacy options
     - Code Object Tracking
     - `--obj-tracking <on|off>`
     - Always ``ON`` in rocprofv2
     - Always ``ON`` in rocprofv3
     -
     - 
   * - Legacy options
     - Heartbeat
     - `--heartbeat <rate sec>`
     - *Not available*
     - *Not available*
     - Not applicable for rocprofv3
     - 