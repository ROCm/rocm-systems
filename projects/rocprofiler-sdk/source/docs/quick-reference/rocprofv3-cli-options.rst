.. meta::
  :description: ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software
  :keywords: ROCprofiler-SDK tool usage, rocprofv3 user manual, rocprofv3 usage, rocprofv3 user guide, using rocprofv3, ROCprofiler-SDK tool user guide, ROCprofiler-SDK tool user manual, using ROCprofiler-SDK tool, ROCprofiler-SDK command-line tool, ROCprofiler-SDK CLI, ROCprofiler-SDK command line tool

.. _cli-options:

==============================
roprofv3 command-line options
==============================

The following table lists the commonly used ``rocprofv3`` command-line options categorized according to their purpose.

.. raw:: html

    <div class="pst-scrollable-table-container">
        <table id="rocprov3-cli-options" class="table">
            <thead>
                <tr>
                    <th>Purpose</th>
                    <th>Option</th>
                    <th>Description</th>
                </tr>
            </thead>
            <colgroup>
                <col span="1">
                <col span="1">
            </colgroup>
            <tbody class="cli-options">
                <tr>
                    <th rowspan="7">I/O options</th>
                    <td>-i INPUT | --input INPUT</td>
                    <td>Specifies the path to the input file. JSON and YAML formats support configuration of all command-line options for tracing and profiling whereas the text format supports only the specification of HW counters. See <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#collecting-traces-using-input-file">collecting traces using input file</a> and <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#counter-collection-using-input-file">counter collection using input file.</a></td>
                </tr>
                <tr>
                    <td>-o OUTPUT_FILE | --output-file OUTPUT_FILE</td>
                    <td>Specifies output file name. If nothing is specified, the default path is %hostname%/%pid%. <a href ="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/rocprofv3-io-options.html#output-file">Read more...</a></td>
                </tr>
                <tr>
                    <td>-d OUTPUT_DIRECTORY | --output-directory OUTPUT_DIRECTORY</td>
                    <td>Specifies the output path for saving the output files. If nothing is specified, the default path is %hostname%/%pid%. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/rocprofv3-io-options.html#output-directory">Read more...</a></td>
                </tr>
                <tr>
                    <td>-f {csv,json,pftrace,otf2,rocpd} [{csv,json,pftrace,otf2,rocpd} ...] | --output-format {csv,json,pftrace,otf2,rocpd} [{csv,json,pftrace,otf2,rocpd} ...]</td>
                    <td>Specifies output format. Supported formats: CSV, JSON, PFTrace, OTF2 and rocpd. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#output-formats">Read more...</a></td>
                </tr>
                <tr>
                    <td>--output-config [BOOL]</td>
                    <td>Generates a configuration output file containing the resolved rocprofv3 settings and options used for the profiling session. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#configuration-output">Read more...</a></td>
                </tr>
                <tr>
                    <td>--log-level {fatal,error,warning,info,trace,env}</td>
                    <td>Sets the desired log level.</td>
                </tr>
                <tr>
                    <td>-E EXTRA_COUNTERS | --extra-counters EXTRA_COUNTERS</td>
                    <td>Specifies the path to a YAML file consisting of extra counter definitions. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#extra-counters">Read more...</a></td>
                </tr>
                <tr>
                    <th>Dynamic process attachment</th>
                    <td>-p PID | --pid PID | --attach PID</td>
                    <td>Attaches to a running process by process ID and profiles it dynamically. This enables profiling of applications that are already running without needing to restart them from the profiler. The profiler will instrument the target process and collect the specified tracing or counter data for the configured duration. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3-process-attachment.html">Read more...</a></td>
                </tr>
                <tr>
                    <th rowspan="2">Aggregate tracing</th>
                    <td>-r [BOOL] | --runtime-trace [BOOL]</td>
                    <td>Collects tracing data for HIP runtime API, marker (ROCTx) API, RCCL API, memory operations (copies, scratch, and allocation), and kernel dispatches. Similar to --sys-trace without HIP compiler API and the underlying HSA API tracing. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#runtime-trace">Read more...</a></td>
                </tr>
                <tr>
                    <td>-s [BOOL] | --sys-trace [BOOL]</td>
                    <td>Collects tracing data for HIP API, HSA API, marker (ROCTx) API, RCCL API, memory operations (copies, scratch, and allocations), and kernel dispatches. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#system-trace">Read more...</a></td>
                </tr>
                <tr>
                    <th rowspan="4">PC sampling</th>
                    <td>--pc-sampling-beta-enabled [BOOL]</td>
                    <td>Enables PC sampling and sets the ROCPROFILER_PC_SAMPLING_BETA_ENABLED environment variable. Note that PC sampling support is in beta version. For more details, see <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html">PC sampling.</a></td>
                </tr>
                <tr>
                    <td>--pc-sampling-unit {instructions,cycles,time}</td>
                    <td>Specifies the unit for PC sampling type or method. Note that only units of time are supported. For more details, see <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html">PC sampling.</a></td>
                </tr>
                <tr>
                    <td>--pc-sampling-method {stochastic,host_trap}</td>
                    <td>Specifies the PC sampling type. Note that only host trap method is supported. For more details, see <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html">PC sampling.</a></td>
                </tr>
                <tr>
                    <td>--pc-sampling-interval PC_SAMPLING_INTERVAL</td>
                    <td>Specifies the PC sample generation frequency. For more details, see <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html">PC sampling.</a></td>
                </tr>
                <tr>
                    <th rowspan="11">Basic tracing</th>
                    <td>--hip-trace [BOOL]</td>
                    <td>Combination of --hip-runtime-trace and --hip-compiler-trace. This option enables only the HIP API tracing. Unlike previous iterations of rocprofv3, this option doesn’t enable kernel tracing, memory copy tracing, and so on. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#hip-trace">Read more...</a></td>
                </tr>
                <tr>
                    <td>--marker-trace [BOOL]</td>
                    <td>Collects marker (ROCTx) traces. Similar to --roctx-trace option in earlier rocprof versions, but with improved ROCTx library with more features. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofiler-sdk-roctx.html#using-roctx-in-the-application">Read more...</a></td>
                </tr>
                <tr>
                    <td>--kernel-trace [BOOL]</td>
                    <td>Collects kernel dispatch traces. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#kernel-trace">Read more...</a></td>
                </tr>
                <tr>
                    <td>--memory-copy-trace [BOOL]</td>
                    <td>Collects memory copy traces. This was a part of the HIP and HSA traces in previous rocprof versions. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#memory-copy-trace">Read more...</a></td>
                <tr>
                    <td>--memory-allocation-trace [BOOL]</td>
                    <td>Collects memory allocation traces. Displays starting address, allocation size, and the agent where allocation occurs. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#memory-allocation-trace">Read more...</a></td>
                </tr>
                <tr>
                    <td>--kfd-trace</td>
                    <td>Collects --kfd-page-migration-trace, --kfd-page-mapping-trace, --kfd-queue-trace, and --kfd-dropped-events-trace. KFD (Kernel Fusion Driver) traces capture low-level driver routines including mapping, unmapping, and migration of data between GPU and system memories, as well as eviction or restoration of GPU queues to facilitate such routines.</td>
                </tr>
                <tr>
                    <td>--scratch-memory-trace [BOOL]</td>
                    <td>Collects scratch memory operations traces. Helps in determining scratch allocations and manage them efficiently. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#scratch-memory-trace">Read more...</a></td>
                </tr>
                <tr>
                    <td>--hsa-trace [BOOL]</td>
                    <td>Collects --hsa-core-trace, --hsa-amd-trace, --hsa-image-trace, and --hsa-finalizer-trace. This option only enables the HSA API tracing. Unlike previous iterations of rocprof, this doesn’t enable kernel tracing, memory copy tracing, and so on. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#hsa-trace">Read more...</a></td>
                </tr>
                <tr>
                    <td>--rccl-trace [BOOL]</td>
                    <td>Collects traces for RCCL (ROCm Communication Collectives Library), which is also pronounced as ‘Rickle’. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#rccl-trace">Read more...</a></td>
                </tr>
                <tr>
                    <td>--kokkos-trace [BOOL]</td>
                    <td>Enables builtin Kokkos tools support, which implies enabling --marker-trace collection and --kernel-rename. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#kokkos-trace">Read more...</a></td>
                </tr>
                <tr>
                    <td>--rocdecode-trace [BOOL]</td>
                    <td>Collects traces for rocDecode APIs. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#rocdecode-trace">Read more...</a></td>
                </tr>
                <tr>
                    <th rowspan="10">Granular tracing</th>
                    <td>--hip-runtime-trace [BOOL]</td>
                    <td>Collects HIP Runtime API traces. For example, public HIP API functions starting with hip such as hipSetDevice.</td>
                </tr>
                <tr>
                    <td>--hip-compiler-trace [BOOL]</td>
                    <td>Collects HIP compiler generated code traces. For example, HIP API functions starting with __hip such as __hipRegisterFatBinary.</td>
                </tr>
                <tr>
                    <td>--hsa-core-trace [BOOL]</td>
                    <td>Collects HSA API traces (core API). For example, HSA functions prefixed with only hsa_ such as hsa_init. For more details, see <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#hsa-trace">HSA trace</a></td>
                </tr>
                <tr>
                    <td>--hsa-amd-trace [BOOL]</td>
                    <td>Collects HSA API traces (AMD-extension API). For example, HSA functions prefixed with hsa_amd_ such as hsa_amd_coherency_get_type. For more details, see <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#hsa-trace">HSA trace</a></td>
                </tr>
                <tr>
                    <td>--hsa-image-trace [BOOL]</td>
                    <td>Collects HSA API traces (image-extenson API). For example, HSA functions prefixed with only hsa_ext_image_ such as hsa_ext_image_get_capability. For more details, see <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#hsa-trace">HSA trace</a></td>
                </tr>
                <tr>
                    <td>--hsa-finalizer-trace [BOOL]</td>
                    <td>Collects HSA API traces (Finalizer-extension API). For example, HSA functions prefixed with only hsa_ext_program_ such as hsa_ext_program_create. For more details, see <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#hsa-trace">HSA trace</a></td>
                </tr>
                <tr>
                    <td>--kfd-page-migration-trace</td>
                    <td>Collects traces of KFD events including migration of pages across device memories.</td>
                </tr>
                <tr>
                    <td>--kfd-page-mapping-trace</td>
                    <td>Collects traces of KFD events including faulting, mapping, and page validation.</td>
                </tr>
                <tr>
                    <td>--kfd-queue-trace</td>
                    <td>Collects traces of KFD events including GPU queue eviction and restoration operations.</td>
                </tr>
                <tr>
                    <td>--kfd-dropped-events-trace</td>
                    <td>Collects traces of KFD events dropped by the KFD device driver.</td>
                </tr>
                <tr>
                    <th>Counter collection</th>
                    <td>--pmc [PMC …]</td>
                    <td>Specifies performance monitoring counters to be collected. Use comma or space to specify more than one counter. For multi-pass collection, use multiple --pmc flags where each flag defines a separate counter group. The job fails if a counter group can't be collected in a single pass. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#counter-collection-using-command-line">Read more...</a></td>
                </tr>
                <tr>
                    <th rowspan="4">Post-processing tracing</th>
                    <td>--stats [BOOL]</td>
                    <td>Collects statistics of enabled tracing types. Must be combined with one or more tracing options. Doesn’t include default kernel stats unlike previous rocprof versions. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#stats">Read more...</a></td>
                </tr>
                <tr>
                    <td>-S [BOOL] | --summary [BOOL]</td>
                    <td>Displays single summary of tracing data for the enabled tracing type, after conclusion of the profiling session. Displays a summary of tracing data for the enabled tracing type, after conclusion of the profiling session. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#summary">Read more...</a></td>
                </tr>
                <tr>
                    <td>-D [BOOL] | --summary-per-domain [BOOL]</td>
                    <td>Displays a summary of each tracing domain for the enabled tracing type, after conclusion of the profiling session. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#summary-per-domain">Read more...</a></td>
                </tr>
                <tr>
                    <td>--summary-groups REGULAR_EXPRESSION [REGULAR_EXPRESSION …]</td>
                    <td>Displays a summary for each set of domains matching the specified regular expression. For example, ‘KERNEL_DISPATCH|MEMORY_COPY’ generates a summary of all the tracing data in the KERNEL_DISPATCH and MEMORY_COPY domains. Similarly ‘*._API’ generates a summary of all the tracing data in the HIP_API, HSA_API, and MARKER_API domains. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#summary-groups">Read more...</a></td>
                </tr>
                <tr>
                    <th rowspan="2">Summary</th>
                    <td>--summary-output-file SUMMARY_OUTPUT_FILE</td>
                    <td>Outputs summary to a file, stdout, or stderr. By default, outputs to stderr. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#summary-output-file">Read more...</a></td>
                </tr>
                <tr>
                    <td>-u {sec,msec,usec,nsec} | --summary-units {sec,msec,usec,nsec}</td>
                    <td>Specifies timing unit for output summary.</td>
                </tr>
                <tr>
                    <th rowspan="3">Kernel naming</th>
                    <td>-M [BOOL] | --mangled-kernels [BOOL]</td>
                    <td>Overrides the default demangling of kernel names. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/kernel-naming-filtering.html#kernel-name-mangling">Read more...</a></td>
                </tr>
                <tr>
                    <td>-T [BOOL] | --truncate-kernels [BOOL]</td>
                    <td>Truncates the demangled kernel names for improved readability. In earlier rocprof versions, this was known as --basenames [on/off]. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/kernel-naming-filtering.html#kernel-name-truncation">Read more...</a></td>
                </tr>
                <tr>
                    <td>--kernel-rename [BOOL]</td>
                    <td>Uses region names defined using roctxRangePush or roctxRangePop to rename the kernels. Was known as --roctx-rename in earlier rocprof versions. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/kernel-naming-filtering.html#kernel-rename">Read more...</a></td>
                </tr>
                <tr>
                    <th rowspan="5">Filtering</th>
                    <td>--kernel-include-regex REGULAR_EXPRESSION</td>
                    <td>Filters counter-collection and thread-trace data to include the kernels matching the specified regular expression. Non-matching kernels are excluded. For more details, see <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/kernel-naming-filtering.html#kernel-filtering">kernel filtering</a></td>
                </tr>
                <tr>
                    <td>--kernel-exclude-regex REGULAR_EXPRESSION</td>
                    <td>Filters counter-collection and thread-trace data to exclude the kernels matching the specified regular expression. It is applied after --kernel-include-regex option. For more details, see <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/kernel-naming-filtering.html#kernel-filtering">kernel filtering</a></td>
                </tr>
                <tr>
                    <td>--kernel-iteration-range KERNEL_ITERATION_RANGE [KERNEL_ITERATION_RANGE …]</td>
                    <td>Specifies iteration range for each kernel matching the filter [start-stop]. For more details, see <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/kernel-naming-filtering.html#kernel-filtering">kernel filtering</a></td>
                </tr>
                <tr>
                    <td>-P (START_DELAY_TIME):(COLLECTION_TIME):(REPEAT) [(START_DELAY_TIME):(COLLECTION_TIME):(REPEAT) …] | --collection-period (START_DELAY_TIME):(COLLECTION_TIME):(REPEAT) [(START_DELAY_TIME):(COLLECTION_TIME):(REPEAT) …]</td>
                    <td>START_DELAY_TIME: Time in seconds before the data collection begins.<br>
                    COLLECTION_TIME: Duration of data collection in seconds.<br>
                    REPEAT: Number of times the data collection cycle is repeated.<br>
                    The default unit for time is seconds, which can be changed using the --collection-period-unit option. To repeat the cycle indefinitely, specify repeat as 0. You can specify multiple configurations, each defined by a triplet in the format start_delay_time:collection_time:repeat. For example, the command -P 10:10:1 5:3:0 specifies two configurations, the first one with a start delay time of 10 seconds, a collection time of 10 seconds, and a repeat of 1 (the cycle repeats once), and the second with a start delay time of 5 seconds, a collection time of 3 seconds, and a repeat of 0 (the cycle repeats indefinitely). <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/rocprofv3-io-options.html#collection-period">Read more...</a></td>
                </tr>
                <tr>
                    <td>--collection-period-unit {hour,min,sec,msec,usec,nsec}</td>
                    <td>To change the unit of time used in --collection-period or -P, specify the desired unit using the --collection-period-unit option. The available units are hour for hours, min for minutes, sec for seconds, msec for milliseconds, usec for microseconds, and nsec for nanoseconds. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/rocprofv3-io-options.html#collection-period">Read more...</a></td>
                </tr>
                <tr>
                    <th rowspan="4">Perfetto-specific</th>
                    <td>--perfetto-backend {inprocess,system}</td>
                    <td>Specifies backend for Perfetto data collection. When selecting ‘system’ mode, ensure to run the Perfetto traced daemon and then start a Perfetto session. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/rocprofv3-io-options.html#perfetto-specific-options">Read more...</a></td>
                </tr>
                <tr>
                    <td>--perfetto-buffer-size KB</td>
                    <td>Specifies buffer size for Perfetto output in KB. Default: 1 GB. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/rocprofv3-io-options.html#perfetto-specific-options">Read more...</a></td>
                </tr>
                <tr>
                    <td>--perfetto-buffer-fill-policy {discard,ring_buffer}</td>
                    <td>Specifies policy for handling new records when Perfetto reaches the buffer limit. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/rocprofv3-io-options.html#perfetto-specific-options">Read more...</a></td>
                </tr>
                <tr>
                    <td>--perfetto-shmem-size-hint KB</td>
                    <td>Specifies Perfetto shared memory size hint in KB. Default: 64 KB. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/rocprofv3-io-options.html#perfetto-specific-options">Read more...</a></td>
                </tr>
                <tr>
                    <th rowspan="2">Display</th>
                    <td>-L [BOOL] | --list-avail [BOOL]</td>
                    <td>Lists the PC sampling configurations and metrics available in the counter_defs.yaml file for counter collection. In earlier rocprof versions, this was known as --list-basic, --list-derived, and --list-counters. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html#kernel-counter-collection">Read more...</a></td>
                </tr>
                <tr>
                    <td>--group-by-queue [BOOL]</td>
                    <td>For displaying the HSA Queues that kernels and memory copy operations are submitted to rather than the default grouping of HIP Streams for perfetto. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/advanced-rocprofv3-options.html#group-by-queue">Read more...</a></td>
                </tr>
                <tr>
                    <th rowspan="7">Others</th>
                    <td>--preload PRELOAD</td>
                    <td>Specifies libraries to prepend to LD_PRELOAD. Useful for sanitizer libraries and custom instrumentation tools. Multiple libraries can be specified. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/advanced-rocprofv3-options.html#library-preloading">Read more...</a></td>
                </tr>
                <tr>
                    <td>--minimum-output-data KB</td>
                    <td>Specifies the minimum output data size threshold in KB. Output files are generated only if the collected profiling data exceeds this threshold. This prevents creation of empty or very small output files. Default is 0 (no threshold). <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/advanced-rocprofv3-options.html#minimum-output-data-threshold">Read more...</a></td>
                </tr>
                <tr>
                    <td>--disable-signal-handlers [BOOL]</td>
                    <td>Controls signal handler prioritization. When set to true, disables rocprofv3 signal handler prioritization, allowing application signal handlers to take precedence. Useful for applications with custom crash handling or when integrating with testing frameworks. Default value is false (rocprofv3 handlers have priority). <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/advanced-rocprofv3-options.html#signal-handler-control">Read more...</a></td>
                </tr>
                <tr>
                    <td>--rocm-root PATH</td>
                    <td>Specifies custom ROCm installation directory instead of automatic detection. Useful for multiple ROCm installations, custom builds, or non-standard locations. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/advanced-rocprofv3-options.html#rocm-root-path-configuration">Read more...</a></td>
                </tr>
                <tr>
                    <td>--sdk-soversion SDK_SOVERSION</td>
                    <td>Specifies the shared object version number for ROCProfiler SDK library resolution. Controls which major version of librocprofiler-sdk.so.X to use. <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/advanced-rocprofv3-options.html#sdk-shared-object-version-control">Read more...</a></td>
                </tr>
                <tr>
                    <td>--sdk-version SDK_VERSION</td>
                    <td>Specifies the exact version number for ROCProfiler SDK library resolution. Controls library selection with full semantic versioning (X.Y.Z format). <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/advanced-rocprofv3-options.html#sdk-version-specification">Read more...</a></td>
                </tr>
                <tr>
                    <td>--selected-regions</td>
                    <td>If set, rocprofv3 profiles only regions of code surrounded by roctxMark(name) and roctxMark(0). <a href="https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofiler-sdk-roctx.html#using-selected-regions-option">Read more...</a></td>
                </tr>
            </tbody>
        </table>
    </div>

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
