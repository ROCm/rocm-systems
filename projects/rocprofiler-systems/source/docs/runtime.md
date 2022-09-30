# Customizing Omnitrace Runtime

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

## omnitrace-avail Executable

The `omnitrace-avail` executable provides information about the runtime settings, data collection capabilities, and
available hardware counters (when built with PAPI support). In contrast to this documentation, it is effectively
self-updating: when new capabilities and settings are added to the omnitrace source code, it is effectively
propagated to `omnitrace-avail`, thus it should be viewed as the single source of truth if any conflicting
information or missing feature is found in this documentation.

It is recommended to create a default configuration file in `${HOME}/.omnitrace.cfg`. This can be done via
executing `omnitrace-avail -G ~/.omnitrace.cfg`, or optionally, use `omnitrace-avail -G ~/.omnitrace.cfg --all`
for a verbose configuration file with descriptions, categories, etc.

Modify `${HOME}/.omnitrace.cfg` as desired, e.g. enable [perfetto](https://perfetto.dev/),
[timemory](https://github.com/NERSC/timemory), sampling, and process-level sampling by default
and tweak some sampling default values:

```console
# ...
OMNITRACE_USE_PERFETTO         = true
OMNITRACE_USE_TIMEMORY         = true
OMNITRACE_USE_SAMPLING         = true
OMNITRACE_USE_PROCESS_SAMPLING = true
# ...
OMNITRACE_SAMPLING_FREQ        = 50
OMNITRACE_SAMPLING_CPUS        = all
OMNITRACE_SAMPLING_GPUS        = $env:HIP_VISIBLE_DEVICES
```

### Exploring Runtime Settings

In order to view the list of the available runtime settings, their current value, and descriptions for each setting:

```shell
omnitrace-avail --description
```

> ***Hint: use `--brief` to suppress printing current value and/or `-c 0` to suppress truncation of the descriptions***

Any setting which is boolean (`omnitrace-avail --settings --value --brief --filter bool`) accepts a case insensitive
match to nearly all common expressions for boolean logic: ON, OFF, YES, NO, TRUE, FALSE, 0, 1, etc.

### Exploring Components

[Omnitrace](https://github.com/AMDResearch/omnitrace) uses [timemory](https://github.com/NERSC/timemory) extensively to provide various capabilities and manage
data and resources. By default, when `OMNITRACE_USE_TIMEMORY=ON`, omnitrace will only collect wall-clock
timing values; however, by modifying the `OMNITRACE_TIMEMORY_COMPONENTS` setting, omnitrace can be configured to
collect hardware counters, CPU-clock timers, memory usage, context-switches, page-faults, network statistics,
and many more. In fact, omnitrace can actually be used as a dynamic instrumentation vehicle for other 3rd-party profiling
APIs such as [Caliper](https://github.com/LLNL/Caliper) and [LIKWID](https://github.com/RRZE-HPC/likwid) by building omnitrace
from source with the CMake option(s) `TIMEMORY_USE_CALIPER=ON` and/or `TIMEMORY_USE_LIKWID=ON` and then adding
`caliper_marker` and/or `likwid_marker` to `OMNITRACE_TIMEMORY_COMPONENTS`.

View all possible components and their descriptions:

```shell
omnitrace-avail --components --description
```

Restrict to available components and view the string identifiers for `OMNITRACE_TIMEMORY_COMPONENTS`:

```shell
omnitrace-avail --components --available --string --brief
```

### Exploring Hardware Counters

[Omnitrace](https://github.com/AMDResearch/omnitrace) supports collecting hardware counters via PAPI and ROCm.
Generally, PAPI is used to collect CPU-based hardware counters and ROCm is used to collect GPU-based hardware
counters; although it is possible to install PAPI with ROCm support and collect GPU-based hardware counters
via PAPI but this is not recommended because CPU hardware counters via PAPI cannot be collected simultaneously.

View all possible hardware counters and their descriptions:

```shell
omnitrace-avail --hw-counters --description
```

Additionally, you can pass `-c CPU` to restrict the hardware counters to the counters available via PAPI and
`-c GPU` to restrict the hardware counters displayed to the counters available via ROCm.

### Enabling Hardware Counters

Hardware counters via PAPI are configured with the `OMNITRACE_PAPI_EVENTS` configuration variable.
Hardware counters via ROCm are configured with the `OMNITRACE_ROCM_EVENTS` configuration variable.
It should be noted that ROCm hardware counters also require the `OMNITRACE_USE_ROCPROFILER` configuration
variable to be enabled (i.e., `OMNITRACE_USE_ROCPROFILER=ON`).

Example configuration for hardware counters:

```console

```

#### OMNITRACE_PAPI_EVENTS

In order to collect the majority of hardware counters via PAPI, you need to make sure the `/proc/sys/kernel/perf_event_paranoid`
has a value of less than 2. If you have sudo access, you can use the following command to modify the value:

```shell
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

However this value will not be retained upon reboot. The following command will preserve this setting between reboots:

```shell
echo 'kernel.perf_event_paranoid=0' | sudo tee -a /etc/sysctl.conf
```

PAPI event use something similar to a namespace. All specified hardware counters must be from the same namespace.
For hardware counters starting with the `PAPI_` prefix, these are high-level aggregates of multiple hardware counters.
Otherwise, most events use two or three colons (`::` or `:::`) between the component name and the counter name, e.g.,
`amd64_rapl::RAPL_ENERGY_PKG`, `perf::PERF_COUNT_HW_CPU_CYCLES`, etc.

For example, the following is a valid configuration:

```console
OMNITRACE_PAPI_EVENTS = perf::INSTRUCTIONS  perf::CACHE-REFERENCES  perf::CACHE-MISSES
```

However, the following effectively specifies the same set of hardware counters but is an invalid configuration because it mixes
PAPI components from different namespaces:

```console
OMNITRACE_PAPI_EVENTS = PAPI_TOT_INS        perf::CACHE-REFERENCES  perf::CACHE-MISSES
```

> ***If omnitrace was configured with `OMNITRACE_BUILD_PAPI=ON` (the default), the standard PAPI command line tools such as***
> ***`papi_avail`, `papi_event_chooser`, etc. will not be able to provide information about the PAPI library used by omnitrace***
> ***(omnitrace statically links to libpapi). However, all of these tools are installed with the prefix `omnitrace-` and all***
> ***underscores are replaced with hypens, e.g. `papi_avail` -> `omnitrace-papi-avail`.***

#### OMNITRACE_ROCM_EVENTS

Omnitrace reads the ROCm events from the `${ROCM_PATH}/lib/rocprofiler/metrics.xml` file. Use the `ROCP_METRICS` environment
variable to point omnitrace to a different XML metrics file, e.g., `export ROCP_METRICS=${PWD}/custom_metrics.xml`.
`omnitrace-avail -H -c GPU` will show event names with a suffix of `:device=N` where `N` is the device number.
For example, if you have two devices, you will see:

```console
| Wavefronts:device=0                   | Derived counter: SQ_WAVES             |
...
| Wavefronts:device=1                   | Derived counter: SQ_WAVES             |
```

If you wish to collect the event on all the devices, simply specify the event, e.g. `Wavefronts`, withouth the `:device=` suffix.
If you wish to collect the event only on specific device(s), use the `:device=` suffix.

For example:

```console
OMNITRACE_ROCM_EVENTS = GPUBusy     SQ_WAVES:device=0    SQ_INSTS_VALU:device=1
```

- Records the percentage of time the GPU was busy on all devices
- Counts the number of waves sent to SQs on device 0
- Counts the number of VALU instructions issued on device 1

### omnitrace-avail Examples

#### Generating Default Configuration

```console
$ omnitrace-avail -G ~/.omnitrace.cfg
[omnitrace-avail] Outputting text configuration file '/home/user/.omnitrace.cfg'...
$ cat ~/.omnitrace.cfg
# auto-generated by omnitrace-avail (version 1.2.0) on 2022-06-27 @ 19:15

OMNITRACE_CONFIG_FILE                              =
OMNITRACE_MODE                                     = trace
OMNITRACE_USE_PERFETTO                             = true
OMNITRACE_USE_TIMEMORY                             = false
OMNITRACE_USE_SAMPLING                             = false
OMNITRACE_USE_PROCESS_SAMPLING                     = true
OMNITRACE_USE_ROCTRACER                            = true
OMNITRACE_USE_ROCM_SMI                             = true
OMNITRACE_USE_KOKKOSP                              = false
OMNITRACE_USE_CODE_COVERAGE                        = false
OMNITRACE_USE_PID                                  = true
OMNITRACE_OUTPUT_PATH                              = omnitrace-%tag%-output
OMNITRACE_OUTPUT_PREFIX                            =
OMNITRACE_CI                                       = false
OMNITRACE_CRITICAL_TRACE                           = false
OMNITRACE_CRITICAL_TRACE_BUFFER_COUNT              = 2000
OMNITRACE_CRITICAL_TRACE_COUNT                     = 0
OMNITRACE_CRITICAL_TRACE_DEBUG                     = false
OMNITRACE_THREAD_POOL_SIZE                         = 8
OMNITRACE_CRITICAL_TRACE_PER_ROW                   = 0
OMNITRACE_CRITICAL_TRACE_SERIALIZE_NAMES           = false
OMNITRACE_DEBUG                                    = false
OMNITRACE_DL_VERBOSE                               = 0
OMNITRACE_INSTRUMENTATION_INTERVAL                 = 1
OMNITRACE_KOKKOS_KERNEL_LOGGER                     = false
OMNITRACE_PAPI_EVENTS                              = PAPI_TOT_CYC
OMNITRACE_PERFETTO_BACKEND                         = inprocess
OMNITRACE_PERFETTO_BUFFER_SIZE_KB                  = 1024000
OMNITRACE_PERFETTO_COMBINE_TRACES                  = true
OMNITRACE_PERFETTO_FILE                            = perfetto-trace.proto
OMNITRACE_PERFETTO_FILL_POLICY                     = discard
OMNITRACE_PERFETTO_SHMEM_SIZE_HINT_KB              = 4096
OMNITRACE_ROCTRACER_HSA_ACTIVITY                   = false
OMNITRACE_ROCTRACER_HSA_API                        = false
OMNITRACE_ROCTRACER_HSA_API_TYPES                  =
OMNITRACE_SAMPLING_CPUS                            =
OMNITRACE_SAMPLING_DELAY                           = 0.5
OMNITRACE_SAMPLING_FREQ                            = 10
OMNITRACE_SAMPLING_GPUS                            = all
OMNITRACE_TIME_OUTPUT                              = true
OMNITRACE_TIMEMORY_COMPONENTS                      = wall_clock
OMNITRACE_TRACE_THREAD_LOCKS                       = false
OMNITRACE_VERBOSE                                  = 0
OMNITRACE_COLLAPSE_PROCESSES                       = false
OMNITRACE_COLLAPSE_THREADS                         = false
OMNITRACE_COUT_OUTPUT                              = false
OMNITRACE_CPU_AFFINITY                             = false
OMNITRACE_DIFF_OUTPUT                              = false
OMNITRACE_ENABLE_SIGNAL_HANDLER                    = true
OMNITRACE_ENABLED                                  = true
OMNITRACE_FILE_OUTPUT                              = true
OMNITRACE_FLAT_PROFILE                             = false
OMNITRACE_INPUT_EXTENSIONS                         = json,xml
OMNITRACE_INPUT_PATH                               =
OMNITRACE_INPUT_PREFIX                             =
OMNITRACE_JSON_OUTPUT                              = true
OMNITRACE_MAX_DEPTH                                = 65535
OMNITRACE_MAX_WIDTH                                = 120
OMNITRACE_MEMORY_PRECISION                         = -1
OMNITRACE_MEMORY_SCIENTIFIC                        = false
OMNITRACE_MEMORY_UNITS                             = MB
OMNITRACE_MEMORY_WIDTH                             = -1
OMNITRACE_NETWORK_INTERFACE                        =
OMNITRACE_NODE_COUNT                               = 0
OMNITRACE_PAPI_FAIL_ON_ERROR                       = false
OMNITRACE_PAPI_MULTIPLEXING                        = false
OMNITRACE_PAPI_OVERFLOW                            = 0
OMNITRACE_PAPI_QUIET                               = false
OMNITRACE_PAPI_THREADING                           = true
OMNITRACE_PRECISION                                = -1
OMNITRACE_SCIENTIFIC                               = false
OMNITRACE_STRICT_CONFIG                            = true
OMNITRACE_SUPPRESS_CONFIG                          = true
OMNITRACE_SUPPRESS_PARSING                         = true
OMNITRACE_TEXT_OUTPUT                              = true
OMNITRACE_TIME_FORMAT                              = %F_%H.%M
OMNITRACE_TIMELINE_PROFILE                         = false
OMNITRACE_TIMING_PRECISION                         = 6
OMNITRACE_TIMING_SCIENTIFIC                        = false
OMNITRACE_TIMING_UNITS                             = sec
OMNITRACE_TIMING_WIDTH                             = -1
OMNITRACE_TREE_OUTPUT                              = true
OMNITRACE_WIDTH                                    = -1
```

- Recommendations
    - Use the `--all` option for descriptions, choices, etc. in the configuration file.
    - If you want to create a new configuration without inheriting from an existing `${HOME}/.omnitrace.cfg`,
    set `OMNITRACE_SUPPRESS_CONFIG=ON` in the environment before executing.
    - If you want to create a new configuration with some minor tweaks to an existing config,
    set `OMNITRACE_CONFIG_FILE=/path/to/existing/file` and define the tweaks as environment variables before generating.

#### Viewing Setting Descriptions

```console
$ omnitrace-avail -S -bd
|-----------------------------------------|-----------------------------------------|
|          ENVIRONMENT VARIABLE           |               DESCRIPTION               |
|-----------------------------------------|-----------------------------------------|
| OMNITRACE_CI                            | Enable some runtime validation check... |
| OMNITRACE_ADD_SECONDARY                 | Enable/disable components adding sec... |
| OMNITRACE_COLLAPSE_PROCESSES            | Enable/disable combining process-spe... |
| OMNITRACE_COLLAPSE_THREADS              | Enable/disable combining thread-spec... |
| OMNITRACE_CONFIG_FILE                   | Configuration file for omnitrace        |
| OMNITRACE_COUT_OUTPUT                   | Write output to stdout                  |
| OMNITRACE_CPU_AFFINITY                  | Enable pinning threads to CPUs (Linu... |
| OMNITRACE_CRITICAL_TRACE                | Enable generation of the critical trace |
| OMNITRACE_CRITICAL_TRACE_BUFFER_COUNT   | Number of critical trace records to ... |
| OMNITRACE_CRITICAL_TRACE_COUNT          | Number of critical trace to export (... |
| OMNITRACE_CRITICAL_TRACE_DEBUG          | Enable debugging for critical trace     |
| OMNITRACE_THREAD_POOL_SIZE              | Number of threads to use when genera... |
| OMNITRACE_CRITICAL_TRACE_PER_ROW        | How many critical traces per row in ... |
| OMNITRACE_CRITICAL_TRACE_SERIALIZE_N... | Include names in serialization of cr... |
| OMNITRACE_DEBUG                         | Enable debug output                     |
| OMNITRACE_DIFF_OUTPUT                   | Generate a difference output vs. a p... |
| OMNITRACE_DL_VERBOSE                    | Verbosity within the omnitrace-dl li... |
| OMNITRACE_ENABLED                       | Activation state of timemory            |
| OMNITRACE_ENABLE_SIGNAL_HANDLER         | Enable signals in timemory_init         |
| OMNITRACE_FILE_OUTPUT                   | Write output to files                   |
| OMNITRACE_FLAT_PROFILE                  | Set the label hierarchy mode to defa... |
| OMNITRACE_INPUT_EXTENSIONS              | File extensions used when searching ... |
| OMNITRACE_INPUT_PATH                    | Explicitly specify the input folder ... |
| OMNITRACE_INPUT_PREFIX                  | Explicitly specify the prefix for in... |
| OMNITRACE_INSTRUMENTATION_INTERVAL      | Instrumentation only takes measureme... |
| OMNITRACE_JSON_OUTPUT                   | Write json output files                 |
| OMNITRACE_KOKKOS_KERNEL_LOGGER          | Enables kernel logging                  |
| OMNITRACE_MAX_DEPTH                     | Set the maximum depth of label hiera... |
| OMNITRACE_MAX_THREAD_BOOKMARKS          | Maximum number of times a worker thr... |
| OMNITRACE_MAX_WIDTH                     | Set the maximum width for component ... |
| OMNITRACE_MEMORY_PRECISION              | Set the precision for components wit... |
| OMNITRACE_MEMORY_SCIENTIFIC             | Set the numerical reporting format f... |
| OMNITRACE_MEMORY_UNITS                  | Set the units for components with 'u... |
| OMNITRACE_MEMORY_WIDTH                  | Set the output width for components ... |
| OMNITRACE_NETWORK_INTERFACE             | Default network interface               |
| OMNITRACE_NODE_COUNT                    | Total number of nodes used in applic... |
| OMNITRACE_OUTPUT_FILE                   | Perfetto filename                       |
| OMNITRACE_OUTPUT_PATH                   | Explicitly specify the output folder... |
| OMNITRACE_OUTPUT_PREFIX                 | Explicitly specify a prefix for all ... |
| OMNITRACE_PAPI_EVENTS                   | PAPI presets and events to collect (... |
| OMNITRACE_PAPI_FAIL_ON_ERROR            | Configure PAPI errors to trigger a r... |
| OMNITRACE_PAPI_MULTIPLEXING             | Enable multiplexing when using PAPI     |
| OMNITRACE_PAPI_OVERFLOW                 | Value at which PAPI hw counters trig... |
| OMNITRACE_PAPI_QUIET                    | Configure suppression of reporting P... |
| OMNITRACE_PAPI_THREADING                | Enable multithreading support when u... |
| OMNITRACE_PERFETTO_BACKEND              | Specify the perfetto backend to acti... |
| OMNITRACE_PERFETTO_BUFFER_SIZE_KB       | Size of perfetto buffer (in KB)         |
| OMNITRACE_PERFETTO_COMBINE_TRACES       | Combine Perfetto traces. If not expl... |
| OMNITRACE_PERFETTO_FILL_POLICY          | Behavior when perfetto buffer is ful... |
| OMNITRACE_PERFETTO_SHMEM_SIZE_HINT_KB   | Hint for shared-memory buffer size i... |
| OMNITRACE_PRECISION                     | Set the global output precision for ... |
| OMNITRACE_ROCTRACER_HSA_ACTIVITY        | Enable HSA activity tracing support     |
| OMNITRACE_ROCTRACER_HSA_API             | Enable HSA API tracing support          |
| OMNITRACE_ROCTRACER_HSA_API_TYPES       | HSA API type to collect                 |
| OMNITRACE_SAMPLING_CPUS                 | CPUs to collect frequency informatio... |
| OMNITRACE_SAMPLING_DELAY                | Number of seconds to wait before the... |
| OMNITRACE_SAMPLING_FREQ                 | Number of software interrupts per se... |
| OMNITRACE_SAMPLING_GPUS                 | Devices to query when OMNITRACE_USE_... |
| OMNITRACE_SCIENTIFIC                    | Set the global numerical reporting t... |
| OMNITRACE_STRICT_CONFIG                 | Throw errors for unknown setting nam... |
| OMNITRACE_SUPPRESS_CONFIG               | Disable processing of setting config... |
| OMNITRACE_SUPPRESS_PARSING              | Disable parsing environment             |
| OMNITRACE_TEXT_OUTPUT                   | Write text output files                 |
| OMNITRACE_TIMELINE_PROFILE              | Set the label hierarchy mode to defa... |
| OMNITRACE_TIMEMORY_COMPONENTS           | List of components to collect via ti... |
| OMNITRACE_TIME_FORMAT                   | Customize the folder generation when... |
| OMNITRACE_TIME_OUTPUT                   | Output data to subfolder w/ a timest... |
| OMNITRACE_TIMING_PRECISION              | Set the precision for components wit... |
| OMNITRACE_TIMING_SCIENTIFIC             | Set the numerical reporting format f... |
| OMNITRACE_TIMING_UNITS                  | Set the units for components with 'u... |
| OMNITRACE_TIMING_WIDTH                  | Set the output width for components ... |
| OMNITRACE_TRACE_THREAD_LOCKS            | Enable tracking calls to pthread_mut... |
| OMNITRACE_TREE_OUTPUT                   | Write hierarchical json output files    |
| OMNITRACE_USE_CODE_COVERAGE             | Enable support for code coverage        |
| OMNITRACE_USE_KOKKOSP                   | Enable support for Kokkos Tools         |
| OMNITRACE_USE_OMPT                      | Enable support for OpenMP-Tools         |
| OMNITRACE_USE_PERFETTO                  | Enable perfetto backend                 |
| OMNITRACE_USE_PID                       | Enable tagging filenames with proces... |
| OMNITRACE_USE_ROCM_SMI                  | Enable sampling GPU power, temp, uti... |
| OMNITRACE_USE_ROCTRACER                 | Enable ROCM tracing                     |
| OMNITRACE_USE_SAMPLING                  | Enable statistical sampling of call-... |
| OMNITRACE_USE_PROCESS_SAMPLING          | Enable a background thread which sam... |
| OMNITRACE_USE_TIMEMORY                  | Enable timemory backend                 |
| OMNITRACE_VERBOSE                       | Verbosity level                         |
| OMNITRACE_WIDTH                         | Set the global output width for comp... |
|-----------------------------------------|-----------------------------------------|
```

#### Viewing Components

```console
$ omnitrace-avail -C -bd
|-----------------------------------|----------------------------------------------|
|             COMPONENT             |                 DESCRIPTION                  |
|-----------------------------------|----------------------------------------------|
| allinea_map                       | Controls the AllineaMAP sampler.             |
| caliper_marker                    | Generic forwarding of markers to Caliper ... |
| caliper_config                    | Caliper configuration manager.               |
| caliper_loop_marker               | Variant of caliper_marker with support fo... |
| cpu_clock                         | Total CPU time spent in both user- and ke... |
| cpu_util                          | Percentage of CPU-clock time divided by w... |
| craypat_counters                  | Names and value of any counter events tha... |
| craypat_flush_buffer              | Writes all the recorded contents in the d... |
| craypat_heap_stats                | Undocumented by 'pat_api.h'.                 |
| craypat_record                    | Toggles CrayPAT recording on calling thread. |
| craypat_region                    | Adds region labels to CrayPAT output.        |
| current_peak_rss                  | Absolute value of high-water mark of memo... |
| gperftools_cpu_profiler           | Control switch for gperftools CPU profiler.  |
| gperftools_heap_profiler          | Control switch for the gperftools heap pr... |
| hip_event                         | Records the time interval between two poi... |
| kernel_mode_time                  | CPU time spent executing in kernel mode (... |
| likwid_marker                     | LIKWID perfmon (CPU) marker forwarding.      |
| likwid_nvmarker                   | LIKWID nvmon (GPU) marker forwarding.        |
| malloc_gotcha                     | GOTCHA wrapper for memory allocation func... |
| memory_allocations                | Number of bytes allocated/freed instead o... |
| monotonic_clock                   | Wall-clock timer which will continue to i... |
| monotonic_raw_clock               | Wall-clock timer unaffected by frequency ... |
| network_stats                     | Reports network bytes, packets, errors, d... |
| num_io_in                         | Number of times the filesystem had to per... |
| num_io_out                        | Number of times the filesystem had to per... |
| num_major_page_faults             | Number of page faults serviced that requi... |
| num_minor_page_faults             | Number of page faults serviced without an... |
| page_rss                          | Amount of memory allocated in pages of me... |
| papi_array<8ul>                   | Fixed-size array of PAPI HW counters.        |
| papi_vector                       | Dynamically allocated array of PAPI HW co... |
| peak_rss                          | Measures changes in the high-water mark f... |
| perfetto_trace                    | Provides Perfetto Tracing SDK: system pro... |
| priority_context_switch           | Number of context switch due to higher pr... |
| process_cpu_clock                 | CPU-clock timer for the calling process (... |
| process_cpu_util                  | Percentage of CPU-clock time divided by w... |
| read_bytes                        | Number of bytes which this process really... |
| read_char                         | Number of bytes which this task has cause... |
| roctx_marker                      | Generates high-level region markers for H... |
| system_clock                      | CPU time spent in kernel-mode.               |
| tau_marker                        | Forwards markers to TAU instrumentation (... |
| thread_cpu_clock                  | CPU-clock timer for the calling thread.      |
| thread_cpu_util                   | Percentage of CPU-clock time divided by w... |
| timestamp                         | Provides a timestamp for every sample and... |
| trip_count                        | Counts number of invocations.                |
| user_clock                        | CPU time spent in user-mode.                 |
| user_mode_time                    | CPU time spent executing in user mode (vi... |
| virtual_memory                    | Records the change in virtual memory.        |
| voluntary_context_switch          | Number of context switches due to a proce... |
| vtune_event                       | Creates events for Intel profiler running... |
| vtune_frame                       | Creates frames for Intel profiler running... |
| vtune_profiler                    | Control switch for Intel profiler running... |
| wall_clock                        | Real-clock timer (i.e. wall-clock timer).    |
| written_bytes                     | Number of bytes sent to the storage layer.   |
| written_char                      | Number of bytes which this task has cause... |
| omnitrace                         | Invokes instrumentation functions 'omnitr... |
| roctracer                         | High-precision ROCm API and kernel tracing.  |
| sampling_wall_clock               | Wall-clock timing. Derived from statistic... |
| sampling_cpu_clock                | CPU-clock timing. Derived from statistica... |
| sampling_percent                  | Fraction of wall-clock time spent in func... |
| sampling_gpu_power                | GPU Power Usage via ROCm-SMI. Derived fro... |
| sampling_gpu_temp                 | GPU Temperature via ROCm-SMI. Derived fro... |
| sampling_gpu_busy                 | GPU Utilization (% busy) via ROCm-SMI. De... |
| sampling_gpu_memory_usage         | GPU Memory Usage via ROCm-SMI. Derived fr... |
|-----------------------------------|----------------------------------------------|
```

#### Viewing Hardware Counters

```console
$ omnitrace-avail -H -bd
|---------------------------------------|---------------------------------------|
|           HARDWARE COUNTER            |              DESCRIPTION              |
|---------------------------------------|---------------------------------------|
|                  CPU                  |                                       |
|---------------------------------------|---------------------------------------|
| PAPI_L1_DCM                           | Level 1 data cache misses             |
| PAPI_L1_ICM                           | Level 1 instruction cache misses      |
| PAPI_L2_DCM                           | Level 2 data cache misses             |
| PAPI_L2_ICM                           | Level 2 instruction cache misses      |
| PAPI_L3_DCM                           | Level 3 data cache misses             |
| PAPI_L3_ICM                           | Level 3 instruction cache misses      |
| PAPI_L1_TCM                           | Level 1 cache misses                  |
| PAPI_L2_TCM                           | Level 2 cache misses                  |
| PAPI_L3_TCM                           | Level 3 cache misses                  |
| PAPI_CA_SNP                           | Requests for a snoop                  |
| PAPI_CA_SHR                           | Requests for exclusive access to s... |
| PAPI_CA_CLN                           | Requests for exclusive access to c... |
| PAPI_CA_INV                           | Requests for cache line invalidation  |
| PAPI_CA_ITV                           | Requests for cache line intervention  |
| PAPI_L3_LDM                           | Level 3 load misses                   |
| PAPI_L3_STM                           | Level 3 store misses                  |
| PAPI_BRU_IDL                          | Cycles branch units are idle          |
| PAPI_FXU_IDL                          | Cycles integer units are idle         |
| PAPI_FPU_IDL                          | Cycles floating point units are idle  |
| PAPI_LSU_IDL                          | Cycles load/store units are idle      |
| PAPI_TLB_DM                           | Data translation lookaside buffer ... |
| PAPI_TLB_IM                           | Instruction translation lookaside ... |
| PAPI_TLB_TL                           | Total translation lookaside buffer... |
| PAPI_L1_LDM                           | Level 1 load misses                   |
| PAPI_L1_STM                           | Level 1 store misses                  |
| PAPI_L2_LDM                           | Level 2 load misses                   |
| PAPI_L2_STM                           | Level 2 store misses                  |
| PAPI_BTAC_M                           | Branch target address cache misses    |
| PAPI_PRF_DM                           | Data prefetch cache misses            |
| PAPI_L3_DCH                           | Level 3 data cache hits               |
| PAPI_TLB_SD                           | Translation lookaside buffer shoot... |
| PAPI_CSR_FAL                          | Failed store conditional instructions |
| PAPI_CSR_SUC                          | Successful store conditional instr... |
| PAPI_CSR_TOT                          | Total store conditional instructions  |
| PAPI_MEM_SCY                          | Cycles Stalled Waiting for memory ... |
| PAPI_MEM_RCY                          | Cycles Stalled Waiting for memory ... |
| PAPI_MEM_WCY                          | Cycles Stalled Waiting for memory ... |
| PAPI_STL_ICY                          | Cycles with no instruction issue      |
| PAPI_FUL_ICY                          | Cycles with maximum instruction issue |
| PAPI_STL_CCY                          | Cycles with no instructions completed |
| PAPI_FUL_CCY                          | Cycles with maximum instructions c... |
| PAPI_HW_INT                           | Hardware interrupts                   |
| PAPI_BR_UCN                           | Unconditional branch instructions     |
| PAPI_BR_CN                            | Conditional branch instructions       |
| PAPI_BR_TKN                           | Conditional branch instructions taken |
| PAPI_BR_NTK                           | Conditional branch instructions no... |
| PAPI_BR_MSP                           | Conditional branch instructions mi... |
| PAPI_BR_PRC                           | Conditional branch instructions co... |
| PAPI_FMA_INS                          | FMA instructions completed            |
| PAPI_TOT_IIS                          | Instructions issued                   |
| PAPI_TOT_INS                          | Instructions completed                |
| PAPI_INT_INS                          | Integer instructions                  |
| PAPI_FP_INS                           | Floating point instructions           |
| PAPI_LD_INS                           | Load instructions                     |
| PAPI_SR_INS                           | Store instructions                    |
| PAPI_BR_INS                           | Branch instructions                   |
| PAPI_VEC_INS                          | Vector/SIMD instructions (could in... |
| PAPI_RES_STL                          | Cycles stalled on any resource        |
| PAPI_FP_STAL                          | Cycles the FP unit(s) are stalled     |
| PAPI_TOT_CYC                          | Total cycles                          |
| PAPI_LST_INS                          | Load/store instructions completed     |
| PAPI_SYC_INS                          | Synchronization instructions compl... |
| PAPI_L1_DCH                           | Level 1 data cache hits               |
| PAPI_L2_DCH                           | Level 2 data cache hits               |
| PAPI_L1_DCA                           | Level 1 data cache accesses           |
| PAPI_L2_DCA                           | Level 2 data cache accesses           |
| PAPI_L3_DCA                           | Level 3 data cache accesses           |
| PAPI_L1_DCR                           | Level 1 data cache reads              |
| PAPI_L2_DCR                           | Level 2 data cache reads              |
| PAPI_L3_DCR                           | Level 3 data cache reads              |
| PAPI_L1_DCW                           | Level 1 data cache writes             |
| PAPI_L2_DCW                           | Level 2 data cache writes             |
| PAPI_L3_DCW                           | Level 3 data cache writes             |
| PAPI_L1_ICH                           | Level 1 instruction cache hits        |
| PAPI_L2_ICH                           | Level 2 instruction cache hits        |
| PAPI_L3_ICH                           | Level 3 instruction cache hits        |
| PAPI_L1_ICA                           | Level 1 instruction cache accesses    |
| PAPI_L2_ICA                           | Level 2 instruction cache accesses    |
| PAPI_L3_ICA                           | Level 3 instruction cache accesses    |
| PAPI_L1_ICR                           | Level 1 instruction cache reads       |
| PAPI_L2_ICR                           | Level 2 instruction cache reads       |
| PAPI_L3_ICR                           | Level 3 instruction cache reads       |
| PAPI_L1_ICW                           | Level 1 instruction cache writes      |
| PAPI_L2_ICW                           | Level 2 instruction cache writes      |
| PAPI_L3_ICW                           | Level 3 instruction cache writes      |
| PAPI_L1_TCH                           | Level 1 total cache hits              |
| PAPI_L2_TCH                           | Level 2 total cache hits              |
| PAPI_L3_TCH                           | Level 3 total cache hits              |
| PAPI_L1_TCA                           | Level 1 total cache accesses          |
| PAPI_L2_TCA                           | Level 2 total cache accesses          |
| PAPI_L3_TCA                           | Level 3 total cache accesses          |
| PAPI_L1_TCR                           | Level 1 total cache reads             |
| PAPI_L2_TCR                           | Level 2 total cache reads             |
| PAPI_L3_TCR                           | Level 3 total cache reads             |
| PAPI_L1_TCW                           | Level 1 total cache writes            |
| PAPI_L2_TCW                           | Level 2 total cache writes            |
| PAPI_L3_TCW                           | Level 3 total cache writes            |
| PAPI_FML_INS                          | Floating point multiply instructions  |
| PAPI_FAD_INS                          | Floating point add instructions       |
| PAPI_FDV_INS                          | Floating point divide instructions    |
| PAPI_FSQ_INS                          | Floating point square root instruc... |
| PAPI_FNV_INS                          | Floating point inverse instructions   |
| PAPI_FP_OPS                           | Floating point operations             |
| PAPI_SP_OPS                           | Floating point operations; optimiz... |
| PAPI_DP_OPS                           | Floating point operations; optimiz... |
| PAPI_VEC_SP                           | Single precision vector/SIMD instr... |
| PAPI_VEC_DP                           | Double precision vector/SIMD instr... |
| PAPI_REF_CYC                          | Reference clock cycles                |
| perf::PERF_COUNT_HW_CPU_CYCLES        | PERF_COUNT_HW_CPU_CYCLES              |
| perf::PERF_COUNT_HW_CPU_CYCLES:u=0    | perf::PERF_COUNT_HW_CPU_CYCLES + m... |
| perf::PERF_COUNT_HW_CPU_CYCLES:k=0    | perf::PERF_COUNT_HW_CPU_CYCLES + m... |
| perf::PERF_COUNT_HW_CPU_CYCLES:h=0    | perf::PERF_COUNT_HW_CPU_CYCLES + m... |
| perf::PERF_COUNT_HW_CPU_CYCLES:per... | perf::PERF_COUNT_HW_CPU_CYCLES + s... |
| perf::PERF_COUNT_HW_CPU_CYCLES:freq=0 | perf::PERF_COUNT_HW_CPU_CYCLES + s... |
| perf::PERF_COUNT_HW_CPU_CYCLES:pre... | perf::PERF_COUNT_HW_CPU_CYCLES + p... |
| perf::PERF_COUNT_HW_CPU_CYCLES:excl=0 | perf::PERF_COUNT_HW_CPU_CYCLES + e... |
| perf::PERF_COUNT_HW_CPU_CYCLES:mg=0   | perf::PERF_COUNT_HW_CPU_CYCLES + m... |
| perf::PERF_COUNT_HW_CPU_CYCLES:mh=0   | perf::PERF_COUNT_HW_CPU_CYCLES + m... |
| perf::PERF_COUNT_HW_CPU_CYCLES:cpu=0  | perf::PERF_COUNT_HW_CPU_CYCLES + C... |
| perf::PERF_COUNT_HW_CPU_CYCLES:pin... | perf::PERF_COUNT_HW_CPU_CYCLES + p... |
| perf::CYCLES                          | PERF_COUNT_HW_CPU_CYCLES              |
| perf::CYCLES:u=0                      | perf::CYCLES + monitor at user level  |
| perf::CYCLES:k=0                      | perf::CYCLES + monitor at kernel l... |
| perf::CYCLES:h=0                      | perf::CYCLES + monitor at hypervis... |
| perf::CYCLES:period=0                 | perf::CYCLES + sampling period        |
| perf::CYCLES:freq=0                   | perf::CYCLES + sampling frequency ... |
| perf::CYCLES:precise=0                | perf::CYCLES + precise event sampling |
| perf::CYCLES:excl=0                   | perf::CYCLES + exclusive access       |
| perf::CYCLES:mg=0                     | perf::CYCLES + monitor guest execu... |
| perf::CYCLES:mh=0                     | perf::CYCLES + monitor host execution |
| perf::CYCLES:cpu=0                    | perf::CYCLES + CPU to program         |
| perf::CYCLES:pinned=0                 | perf::CYCLES + pin event to counters  |
| perf::CPU-CYCLES                      | PERF_COUNT_HW_CPU_CYCLES              |
| perf::CPU-CYCLES:u=0                  | perf::CPU-CYCLES + monitor at user... |
| perf::CPU-CYCLES:k=0                  | perf::CPU-CYCLES + monitor at kern... |
| perf::CPU-CYCLES:h=0                  | perf::CPU-CYCLES + monitor at hype... |
| perf::CPU-CYCLES:period=0             | perf::CPU-CYCLES + sampling period    |
| perf::CPU-CYCLES:freq=0               | perf::CPU-CYCLES + sampling freque... |
| perf::CPU-CYCLES:precise=0            | perf::CPU-CYCLES + precise event s... |
| perf::CPU-CYCLES:excl=0               | perf::CPU-CYCLES + exclusive access   |
| perf::CPU-CYCLES:mg=0                 | perf::CPU-CYCLES + monitor guest e... |
| perf::CPU-CYCLES:mh=0                 | perf::CPU-CYCLES + monitor host ex... |
| perf::CPU-CYCLES:cpu=0                | perf::CPU-CYCLES + CPU to program     |
| perf::CPU-CYCLES:pinned=0             | perf::CPU-CYCLES + pin event to co... |
| perf::PERF_COUNT_HW_INSTRUCTIONS      | PERF_COUNT_HW_INSTRUCTIONS            |
| perf::PERF_COUNT_HW_INSTRUCTIONS:u=0  | perf::PERF_COUNT_HW_INSTRUCTIONS +... |
| perf::PERF_COUNT_HW_INSTRUCTIONS:k=0  | perf::PERF_COUNT_HW_INSTRUCTIONS +... |
| perf::PERF_COUNT_HW_INSTRUCTIONS:h=0  | perf::PERF_COUNT_HW_INSTRUCTIONS +... |
| perf::PERF_COUNT_HW_INSTRUCTIONS:p... | perf::PERF_COUNT_HW_INSTRUCTIONS +... |
| perf::PERF_COUNT_HW_INSTRUCTIONS:f... | perf::PERF_COUNT_HW_INSTRUCTIONS +... |
| perf::PERF_COUNT_HW_INSTRUCTIONS:p... | perf::PERF_COUNT_HW_INSTRUCTIONS +... |
| perf::PERF_COUNT_HW_INSTRUCTIONS:e... | perf::PERF_COUNT_HW_INSTRUCTIONS +... |
| perf::PERF_COUNT_HW_INSTRUCTIONS:mg=0 | perf::PERF_COUNT_HW_INSTRUCTIONS +... |
| perf::PERF_COUNT_HW_INSTRUCTIONS:mh=0 | perf::PERF_COUNT_HW_INSTRUCTIONS +... |
| perf::PERF_COUNT_HW_INSTRUCTIONS:c... | perf::PERF_COUNT_HW_INSTRUCTIONS +... |
| perf::PERF_COUNT_HW_INSTRUCTIONS:p... | perf::PERF_COUNT_HW_INSTRUCTIONS +... |
| ... etc. ...                          |                                       |
| perf_raw::r0000                       | perf_events raw event syntax: r[0-... |
| perf_raw::r0000:u=0                   | perf_raw::r0000 + monitor at user ... |
| perf_raw::r0000:k=0                   | perf_raw::r0000 + monitor at kerne... |
| perf_raw::r0000:h=0                   | perf_raw::r0000 + monitor at hyper... |
| perf_raw::r0000:period=0              | perf_raw::r0000 + sampling period     |
| perf_raw::r0000:freq=0                | perf_raw::r0000 + sampling frequen... |
| perf_raw::r0000:precise=0             | perf_raw::r0000 + precise event sa... |
| perf_raw::r0000:excl=0                | perf_raw::r0000 + exclusive access    |
| perf_raw::r0000:mg=0                  | perf_raw::r0000 + monitor guest ex... |
| perf_raw::r0000:mh=0                  | perf_raw::r0000 + monitor host exe... |
| perf_raw::r0000:cpu=0                 | perf_raw::r0000 + CPU to program      |
| perf_raw::r0000:pinned=0              | perf_raw::r0000 + pin event to cou... |
| perf_raw::r0000:hw_smpl=0             | perf_raw::r0000 + enable hardware ... |
| L1_ITLB_MISS_L2_ITLB_HIT              | Number of instruction fetches that... |
| L1_ITLB_MISS_L2_ITLB_HIT:e=0          | L1_ITLB_MISS_L2_ITLB_HIT + edge level |
| L1_ITLB_MISS_L2_ITLB_HIT:i=0          | L1_ITLB_MISS_L2_ITLB_HIT + invert     |
| L1_ITLB_MISS_L2_ITLB_HIT:c=0          | L1_ITLB_MISS_L2_ITLB_HIT + counter... |
| L1_ITLB_MISS_L2_ITLB_HIT:g=0          | L1_ITLB_MISS_L2_ITLB_HIT + measure... |
| L1_ITLB_MISS_L2_ITLB_HIT:u=0          | L1_ITLB_MISS_L2_ITLB_HIT + monitor... |
| L1_ITLB_MISS_L2_ITLB_HIT:k=0          | L1_ITLB_MISS_L2_ITLB_HIT + monitor... |
| L1_ITLB_MISS_L2_ITLB_HIT:period=0     | L1_ITLB_MISS_L2_ITLB_HIT + samplin... |
| L1_ITLB_MISS_L2_ITLB_HIT:freq=0       | L1_ITLB_MISS_L2_ITLB_HIT + samplin... |
| L1_ITLB_MISS_L2_ITLB_HIT:excl=0       | L1_ITLB_MISS_L2_ITLB_HIT + exclusi... |
| L1_ITLB_MISS_L2_ITLB_HIT:mg=0         | L1_ITLB_MISS_L2_ITLB_HIT + monitor... |
| L1_ITLB_MISS_L2_ITLB_HIT:mh=0         | L1_ITLB_MISS_L2_ITLB_HIT + monitor... |
| L1_ITLB_MISS_L2_ITLB_HIT:cpu=0        | L1_ITLB_MISS_L2_ITLB_HIT + CPU to ... |
| L1_ITLB_MISS_L2_ITLB_HIT:pinned=0     | L1_ITLB_MISS_L2_ITLB_HIT + pin eve... |
| L1_ITLB_MISS_L2_ITLB_MISS             | Number of instruction fetches that... |
| L1_ITLB_MISS_L2_ITLB_MISS:IF1G        | L1_ITLB_MISS_L2_ITLB_MISS + Number... |
| L1_ITLB_MISS_L2_ITLB_MISS:IF2M        | L1_ITLB_MISS_L2_ITLB_MISS + Number... |
| L1_ITLB_MISS_L2_ITLB_MISS:IF4K        | L1_ITLB_MISS_L2_ITLB_MISS + Number... |
| L1_ITLB_MISS_L2_ITLB_MISS:e=0         | L1_ITLB_MISS_L2_ITLB_MISS + edge l... |
| L1_ITLB_MISS_L2_ITLB_MISS:i=0         | L1_ITLB_MISS_L2_ITLB_MISS + invert    |
| L1_ITLB_MISS_L2_ITLB_MISS:c=0         | L1_ITLB_MISS_L2_ITLB_MISS + counte... |
| L1_ITLB_MISS_L2_ITLB_MISS:g=0         | L1_ITLB_MISS_L2_ITLB_MISS + measur... |
| L1_ITLB_MISS_L2_ITLB_MISS:u=0         | L1_ITLB_MISS_L2_ITLB_MISS + monito... |
| L1_ITLB_MISS_L2_ITLB_MISS:k=0         | L1_ITLB_MISS_L2_ITLB_MISS + monito... |
| L1_ITLB_MISS_L2_ITLB_MISS:period=0    | L1_ITLB_MISS_L2_ITLB_MISS + sampli... |
| L1_ITLB_MISS_L2_ITLB_MISS:freq=0      | L1_ITLB_MISS_L2_ITLB_MISS + sampli... |
| L1_ITLB_MISS_L2_ITLB_MISS:excl=0      | L1_ITLB_MISS_L2_ITLB_MISS + exclus... |
| L1_ITLB_MISS_L2_ITLB_MISS:mg=0        | L1_ITLB_MISS_L2_ITLB_MISS + monito... |
| L1_ITLB_MISS_L2_ITLB_MISS:mh=0        | L1_ITLB_MISS_L2_ITLB_MISS + monito... |
| L1_ITLB_MISS_L2_ITLB_MISS:cpu=0       | L1_ITLB_MISS_L2_ITLB_MISS + CPU to... |
| L1_ITLB_MISS_L2_ITLB_MISS:pinned=0    | L1_ITLB_MISS_L2_ITLB_MISS + pin ev... |
| RETIRED_SSE_AVX_FLOPS                 | This is a retire-based event. The ... |
| RETIRED_SSE_AVX_FLOPS:ADD_SUB_FLOPS   | RETIRED_SSE_AVX_FLOPS + Addition/s... |
| RETIRED_SSE_AVX_FLOPS:MULT_FLOPS      | RETIRED_SSE_AVX_FLOPS + Multiplica... |
| RETIRED_SSE_AVX_FLOPS:DIV_FLOPS       | RETIRED_SSE_AVX_FLOPS + Division F... |
| RETIRED_SSE_AVX_FLOPS:MAC_FLOPS       | RETIRED_SSE_AVX_FLOPS + Double pre... |
| RETIRED_SSE_AVX_FLOPS:ANY             | RETIRED_SSE_AVX_FLOPS + Double pre... |
| RETIRED_SSE_AVX_FLOPS:e=0             | RETIRED_SSE_AVX_FLOPS + edge level    |
| RETIRED_SSE_AVX_FLOPS:i=0             | RETIRED_SSE_AVX_FLOPS + invert        |
| RETIRED_SSE_AVX_FLOPS:c=0             | RETIRED_SSE_AVX_FLOPS + counter-ma... |
| RETIRED_SSE_AVX_FLOPS:g=0             | RETIRED_SSE_AVX_FLOPS + measure in... |
| RETIRED_SSE_AVX_FLOPS:u=0             | RETIRED_SSE_AVX_FLOPS + monitor at... |
| RETIRED_SSE_AVX_FLOPS:k=0             | RETIRED_SSE_AVX_FLOPS + monitor at... |
| RETIRED_SSE_AVX_FLOPS:period=0        | RETIRED_SSE_AVX_FLOPS + sampling p... |
| RETIRED_SSE_AVX_FLOPS:freq=0          | RETIRED_SSE_AVX_FLOPS + sampling f... |
| RETIRED_SSE_AVX_FLOPS:excl=0          | RETIRED_SSE_AVX_FLOPS + exclusive ... |
| RETIRED_SSE_AVX_FLOPS:mg=0            | RETIRED_SSE_AVX_FLOPS + monitor gu... |
| RETIRED_SSE_AVX_FLOPS:mh=0            | RETIRED_SSE_AVX_FLOPS + monitor ho... |
| RETIRED_SSE_AVX_FLOPS:cpu=0           | RETIRED_SSE_AVX_FLOPS + CPU to pro... |
| RETIRED_SSE_AVX_FLOPS:pinned=0        | RETIRED_SSE_AVX_FLOPS + pin event ... |
| DIV_CYCLES_BUSY_COUNT                 | Number of cycles when the divider ... |
| DIV_CYCLES_BUSY_COUNT:e=0             | DIV_CYCLES_BUSY_COUNT + edge level    |
| DIV_CYCLES_BUSY_COUNT:i=0             | DIV_CYCLES_BUSY_COUNT + invert        |
| DIV_CYCLES_BUSY_COUNT:c=0             | DIV_CYCLES_BUSY_COUNT + counter-ma... |
| DIV_CYCLES_BUSY_COUNT:g=0             | DIV_CYCLES_BUSY_COUNT + measure in... |
| DIV_CYCLES_BUSY_COUNT:u=0             | DIV_CYCLES_BUSY_COUNT + monitor at... |
| DIV_CYCLES_BUSY_COUNT:k=0             | DIV_CYCLES_BUSY_COUNT + monitor at... |
| DIV_CYCLES_BUSY_COUNT:period=0        | DIV_CYCLES_BUSY_COUNT + sampling p... |
| DIV_CYCLES_BUSY_COUNT:freq=0          | DIV_CYCLES_BUSY_COUNT + sampling f... |
| DIV_CYCLES_BUSY_COUNT:excl=0          | DIV_CYCLES_BUSY_COUNT + exclusive ... |
| DIV_CYCLES_BUSY_COUNT:mg=0            | DIV_CYCLES_BUSY_COUNT + monitor gu... |
| DIV_CYCLES_BUSY_COUNT:mh=0            | DIV_CYCLES_BUSY_COUNT + monitor ho... |
| DIV_CYCLES_BUSY_COUNT:cpu=0           | DIV_CYCLES_BUSY_COUNT + CPU to pro... |
| DIV_CYCLES_BUSY_COUNT:pinned=0        | DIV_CYCLES_BUSY_COUNT + pin event ... |
| DIV_OP_COUNT                          | Number of divide uops.                |
| DIV_OP_COUNT:e=0                      | DIV_OP_COUNT + edge level             |
| DIV_OP_COUNT:i=0                      | DIV_OP_COUNT + invert                 |
| DIV_OP_COUNT:c=0                      | DIV_OP_COUNT + counter-mask in ran... |
| DIV_OP_COUNT:g=0                      | DIV_OP_COUNT + measure in guest       |
| DIV_OP_COUNT:u=0                      | DIV_OP_COUNT + monitor at user level  |
| DIV_OP_COUNT:k=0                      | DIV_OP_COUNT + monitor at kernel l... |
| DIV_OP_COUNT:period=0                 | DIV_OP_COUNT + sampling period        |
| DIV_OP_COUNT:freq=0                   | DIV_OP_COUNT + sampling frequency ... |
| DIV_OP_COUNT:excl=0                   | DIV_OP_COUNT + exclusive access       |
| DIV_OP_COUNT:mg=0                     | DIV_OP_COUNT + monitor guest execu... |
| DIV_OP_COUNT:mh=0                     | DIV_OP_COUNT + monitor host execution |
| DIV_OP_COUNT:cpu=0                    | DIV_OP_COUNT + CPU to program         |
| DIV_OP_COUNT:pinned=0                 | DIV_OP_COUNT + pin event to counters  |
| ... etc. ...                          |                                       |
| amd64_rapl::RAPL_ENERGY_PKG           | Number of Joules consumed by all c... |
| amd64_rapl::RAPL_ENERGY_PKG:u=0       | amd64_rapl::RAPL_ENERGY_PKG + moni... |
| amd64_rapl::RAPL_ENERGY_PKG:k=0       | amd64_rapl::RAPL_ENERGY_PKG + moni... |
| amd64_rapl::RAPL_ENERGY_PKG:period=0  | amd64_rapl::RAPL_ENERGY_PKG + samp... |
| amd64_rapl::RAPL_ENERGY_PKG:freq=0    | amd64_rapl::RAPL_ENERGY_PKG + samp... |
| amd64_rapl::RAPL_ENERGY_PKG:excl=0    | amd64_rapl::RAPL_ENERGY_PKG + excl... |
| amd64_rapl::RAPL_ENERGY_PKG:mg=0      | amd64_rapl::RAPL_ENERGY_PKG + moni... |
| amd64_rapl::RAPL_ENERGY_PKG:mh=0      | amd64_rapl::RAPL_ENERGY_PKG + moni... |
| amd64_rapl::RAPL_ENERGY_PKG:cpu=0     | amd64_rapl::RAPL_ENERGY_PKG + CPU ... |
| amd64_rapl::RAPL_ENERGY_PKG:pinned=0  | amd64_rapl::RAPL_ENERGY_PKG + pin ... |
| appio:::READ_BYTES                    | Bytes read                            |
| appio:::READ_CALLS                    | Number of read calls                  |
| appio:::READ_ERR                      | Number of read calls that resulted... |
| appio:::READ_INTERRUPTED              | Number of read calls that timed ou... |
| appio:::READ_WOULD_BLOCK              | Number of read calls that would ha... |
| appio:::READ_SHORT                    | Number of read calls that returned... |
| appio:::READ_EOF                      | Number of read calls that returned... |
| appio:::READ_BLOCK_SIZE               | Average block size of reads           |
| appio:::READ_USEC                     | Real microseconds spent in reads      |
| appio:::WRITE_BYTES                   | Bytes written                         |
| appio:::WRITE_CALLS                   | Number of write calls                 |
| appio:::WRITE_ERR                     | Number of write calls that resulte... |
| appio:::WRITE_SHORT                   | Number of write calls that wrote l... |
| appio:::WRITE_INTERRUPTED             | Number of write calls that timed o... |
| appio:::WRITE_WOULD_BLOCK             | Number of write calls that would h... |
| appio:::WRITE_BLOCK_SIZE              | Mean block size of writes             |
| appio:::WRITE_USEC                    | Real microseconds spent in writes     |
| appio:::OPEN_CALLS                    | Number of open calls                  |
| appio:::OPEN_ERR                      | Number of open calls that resulted... |
| appio:::OPEN_FDS                      | Number of currently open descriptors  |
| appio:::SELECT_USEC                   | Real microseconds spent in select ... |
| appio:::RECV_BYTES                    | Bytes read in recv/recvmsg/recvfrom   |
| appio:::RECV_CALLS                    | Number of recv/recvmsg/recvfrom calls |
| appio:::RECV_ERR                      | Number of recv/recvmsg/recvfrom ca... |
| appio:::RECV_INTERRUPTED              | Number of recv/recvmsg/recvfrom ca... |
| appio:::RECV_WOULD_BLOCK              | Number of recv/recvmsg/recvfrom ca... |
| appio:::RECV_SHORT                    | Number of recv/recvmsg/recvfrom ca... |
| appio:::RECV_EOF                      | Number of recv/recvmsg/recvfrom ca... |
| appio:::RECV_BLOCK_SIZE               | Average block size of recv/recvmsg... |
| appio:::RECV_USEC                     | Real microseconds spent in recv/re... |
| appio:::SOCK_READ_BYTES               | Bytes read from socket                |
| appio:::SOCK_READ_CALLS               | Number of read calls on socket        |
| appio:::SOCK_READ_ERR                 | Number of read calls on socket tha... |
| appio:::SOCK_READ_SHORT               | Number of read calls on socket tha... |
| appio:::SOCK_READ_WOULD_BLOCK         | Number of read calls on socket tha... |
| appio:::SOCK_READ_USEC                | Real microseconds spent in read(s)... |
| appio:::SOCK_WRITE_BYTES              | Bytes written to socket               |
| appio:::SOCK_WRITE_CALLS              | Number of write calls to socket       |
| appio:::SOCK_WRITE_ERR                | Number of write calls to socket th... |
| appio:::SOCK_WRITE_SHORT              | Number of write calls to socket th... |
| appio:::SOCK_WRITE_WOULD_BLOCK        | Number of write calls to socket th... |
| appio:::SOCK_WRITE_USEC               | Real microseconds spent in write(s... |
| appio:::SEEK_CALLS                    | Number of seek calls                  |
| appio:::SEEK_ABS_STRIDE_SIZE          | Average absolute stride size of seeks |
| appio:::SEEK_USEC                     | Real microseconds spent in seek calls |
| coretemp:::hwmon2:in0_input           | V, amdgpu module, label vddgfx        |
| coretemp:::hwmon2:temp1_input         | degrees C, amdgpu module, label edge  |
| coretemp:::hwmon2:temp2_input         | degrees C, amdgpu module, label ju... |
| coretemp:::hwmon2:temp3_input         | degrees C, amdgpu module, label mem   |
| coretemp:::hwmon2:fan1_input          | RPM, amdgpu module, label ?           |
| coretemp:::hwmon0:temp1_input         | degrees C, nvme module, label Comp... |
| coretemp:::hwmon0:temp2_input         | degrees C, nvme module, label Sens... |
| coretemp:::hwmon0:temp3_input         | degrees C, nvme module, label Sens... |
| coretemp:::hwmon3:temp1_input         | degrees C, k10temp module, label Tctl |
| coretemp:::hwmon3:temp2_input         | degrees C, k10temp module, label Tdie |
| coretemp:::hwmon3:temp5_input         | degrees C, k10temp module, label T... |
| coretemp:::hwmon3:temp7_input         | degrees C, k10temp module, label T... |
| coretemp:::hwmon1:temp1_input         | degrees C, enp1s0 module, label PH... |
| coretemp:::hwmon1:temp2_input         | degrees C, enp1s0 module, label MA... |
| io:::rchar                            | Characters read.                      |
| io:::wchar                            | Characters written.                   |
| io:::syscr                            | Characters read by system calls.      |
| io:::syscw                            | Characters written by system calls.   |
| io:::read_bytes                       | Binary bytes read.                    |
| io:::write_bytes                      | Binary bytes written.                 |
| io:::cancelled_write_bytes            | Binary write bytes cancelled.         |
| net:::lo:rx:bytes                     | lo receive bytes                      |
| net:::lo:rx:packets                   | lo receive packets                    |
| net:::lo:rx:errors                    | lo receive errors                     |
| net:::lo:rx:dropped                   | lo receive dropped                    |
| net:::lo:rx:fifo                      | lo receive fifo                       |
| net:::lo:rx:frame                     | lo receive frame                      |
| net:::lo:rx:compressed                | lo receive compressed                 |
| net:::lo:rx:multicast                 | lo receive multicast                  |
| net:::lo:tx:bytes                     | lo transmit bytes                     |
| net:::lo:tx:packets                   | lo transmit packets                   |
| net:::lo:tx:errors                    | lo transmit errors                    |
| net:::lo:tx:dropped                   | lo transmit dropped                   |
| net:::lo:tx:fifo                      | lo transmit fifo                      |
| net:::lo:tx:colls                     | lo transmit colls                     |
| net:::lo:tx:carrier                   | lo transmit carrier                   |
| net:::lo:tx:compressed                | lo transmit compressed                |
| net:::enp1s0:rx:bytes                 | enp1s0 receive bytes                  |
| net:::enp1s0:rx:packets               | enp1s0 receive packets                |
| net:::enp1s0:rx:errors                | enp1s0 receive errors                 |
| net:::enp1s0:rx:dropped               | enp1s0 receive dropped                |
| net:::enp1s0:rx:fifo                  | enp1s0 receive fifo                   |
| net:::enp1s0:rx:frame                 | enp1s0 receive frame                  |
| net:::enp1s0:rx:compressed            | enp1s0 receive compressed             |
| net:::enp1s0:rx:multicast             | enp1s0 receive multicast              |
| net:::enp1s0:tx:bytes                 | enp1s0 transmit bytes                 |
| net:::enp1s0:tx:packets               | enp1s0 transmit packets               |
| net:::enp1s0:tx:errors                | enp1s0 transmit errors                |
| net:::enp1s0:tx:dropped               | enp1s0 transmit dropped               |
| net:::enp1s0:tx:fifo                  | enp1s0 transmit fifo                  |
| net:::enp1s0:tx:colls                 | enp1s0 transmit colls                 |
| net:::enp1s0:tx:carrier               | enp1s0 transmit carrier               |
| net:::enp1s0:tx:compressed            | enp1s0 transmit compressed            |
| net:::vxlan.calico:rx:bytes           | vxlan.calico receive bytes            |
| net:::vxlan.calico:rx:packets         | vxlan.calico receive packets          |
| net:::vxlan.calico:rx:errors          | vxlan.calico receive errors           |
| net:::vxlan.calico:rx:dropped         | vxlan.calico receive dropped          |
| net:::vxlan.calico:rx:fifo            | vxlan.calico receive fifo             |
| net:::vxlan.calico:rx:frame           | vxlan.calico receive frame            |
| net:::vxlan.calico:rx:compressed      | vxlan.calico receive compressed       |
| net:::vxlan.calico:rx:multicast       | vxlan.calico receive multicast        |
| net:::vxlan.calico:tx:bytes           | vxlan.calico transmit bytes           |
| net:::vxlan.calico:tx:packets         | vxlan.calico transmit packets         |
| net:::vxlan.calico:tx:errors          | vxlan.calico transmit errors          |
| net:::vxlan.calico:tx:dropped         | vxlan.calico transmit dropped         |
| net:::vxlan.calico:tx:fifo            | vxlan.calico transmit fifo            |
| net:::vxlan.calico:tx:colls           | vxlan.calico transmit colls           |
| net:::vxlan.calico:tx:carrier         | vxlan.calico transmit carrier         |
| net:::vxlan.calico:tx:compressed      | vxlan.calico transmit compressed      |
| net:::cali59d6fabc2aa:rx:bytes        | cali59d6fabc2aa receive bytes         |
| net:::cali59d6fabc2aa:rx:packets      | cali59d6fabc2aa receive packets       |
| net:::cali59d6fabc2aa:rx:errors       | cali59d6fabc2aa receive errors        |
| net:::cali59d6fabc2aa:rx:dropped      | cali59d6fabc2aa receive dropped       |
| net:::cali59d6fabc2aa:rx:fifo         | cali59d6fabc2aa receive fifo          |
| net:::cali59d6fabc2aa:rx:frame        | cali59d6fabc2aa receive frame         |
| net:::cali59d6fabc2aa:rx:compressed   | cali59d6fabc2aa receive compressed    |
| net:::cali59d6fabc2aa:rx:multicast    | cali59d6fabc2aa receive multicast     |
| net:::cali59d6fabc2aa:tx:bytes        | cali59d6fabc2aa transmit bytes        |
| net:::cali59d6fabc2aa:tx:packets      | cali59d6fabc2aa transmit packets      |
| net:::cali59d6fabc2aa:tx:errors       | cali59d6fabc2aa transmit errors       |
| net:::cali59d6fabc2aa:tx:dropped      | cali59d6fabc2aa transmit dropped      |
| net:::cali59d6fabc2aa:tx:fifo         | cali59d6fabc2aa transmit fifo         |
| net:::cali59d6fabc2aa:tx:colls        | cali59d6fabc2aa transmit colls        |
| net:::cali59d6fabc2aa:tx:carrier      | cali59d6fabc2aa transmit carrier      |
| net:::cali59d6fabc2aa:tx:compressed   | cali59d6fabc2aa transmit compressed   |
|---------------------------------------|---------------------------------------|
|                  GPU                  |                                       |
|---------------------------------------|---------------------------------------|
| TCC_EA1_WRREQ[0]:device=0             | Number of transactions (either 32-... |
| TCC_EA1_WRREQ[1]:device=0             | Number of transactions (either 32-... |
| TCC_EA1_WRREQ[2]:device=0             | Number of transactions (either 32-... |
| TCC_EA1_WRREQ[3]:device=0             | Number of transactions (either 32-... |
| TCC_EA1_WRREQ[4]:device=0             | Number of transactions (either 32-... |
| TCC_EA1_WRREQ[5]:device=0             | Number of transactions (either 32-... |
| TCC_EA1_WRREQ[6]:device=0             | Number of transactions (either 32-... |
| TCC_EA1_WRREQ[7]:device=0             | Number of transactions (either 32-... |
| TCC_EA1_WRREQ[8]:device=0             | Number of transactions (either 32-... |
| TCC_EA1_WRREQ[9]:device=0             | Number of transactions (either 32-... |
| TCC_EA1_WRREQ[10]:device=0            | Number of transactions (either 32-... |
| TCC_EA1_WRREQ[11]:device=0            | Number of transactions (either 32-... |
| TCC_EA1_WRREQ[12]:device=0            | Number of transactions (either 32-... |
| TCC_EA1_WRREQ[13]:device=0            | Number of transactions (either 32-... |
| TCC_EA1_WRREQ[14]:device=0            | Number of transactions (either 32-... |
| TCC_EA1_WRREQ[15]:device=0            | Number of transactions (either 32-... |
| TCC_EA1_WRREQ_64B[0]:device=0         | Number of 64-byte transactions goi... |
| TCC_EA1_WRREQ_64B[1]:device=0         | Number of 64-byte transactions goi... |
| TCC_EA1_WRREQ_64B[2]:device=0         | Number of 64-byte transactions goi... |
| TCC_EA1_WRREQ_64B[3]:device=0         | Number of 64-byte transactions goi... |
| TCC_EA1_WRREQ_64B[4]:device=0         | Number of 64-byte transactions goi... |
| TCC_EA1_WRREQ_64B[5]:device=0         | Number of 64-byte transactions goi... |
| TCC_EA1_WRREQ_64B[6]:device=0         | Number of 64-byte transactions goi... |
| TCC_EA1_WRREQ_64B[7]:device=0         | Number of 64-byte transactions goi... |
| TCC_EA1_WRREQ_64B[8]:device=0         | Number of 64-byte transactions goi... |
| TCC_EA1_WRREQ_64B[9]:device=0         | Number of 64-byte transactions goi... |
| TCC_EA1_WRREQ_64B[10]:device=0        | Number of 64-byte transactions goi... |
| TCC_EA1_WRREQ_64B[11]:device=0        | Number of 64-byte transactions goi... |
| TCC_EA1_WRREQ_64B[12]:device=0        | Number of 64-byte transactions goi... |
| TCC_EA1_WRREQ_64B[13]:device=0        | Number of 64-byte transactions goi... |
| TCC_EA1_WRREQ_64B[14]:device=0        | Number of 64-byte transactions goi... |
| TCC_EA1_WRREQ_64B[15]:device=0        | Number of 64-byte transactions goi... |
| TCC_EA1_WRREQ_STALL[0]:device=0       | Number of cycles a write request w... |
| TCC_EA1_WRREQ_STALL[1]:device=0       | Number of cycles a write request w... |
| TCC_EA1_WRREQ_STALL[2]:device=0       | Number of cycles a write request w... |
| TCC_EA1_WRREQ_STALL[3]:device=0       | Number of cycles a write request w... |
| TCC_EA1_WRREQ_STALL[4]:device=0       | Number of cycles a write request w... |
| TCC_EA1_WRREQ_STALL[5]:device=0       | Number of cycles a write request w... |
| TCC_EA1_WRREQ_STALL[6]:device=0       | Number of cycles a write request w... |
| TCC_EA1_WRREQ_STALL[7]:device=0       | Number of cycles a write request w... |
| TCC_EA1_WRREQ_STALL[8]:device=0       | Number of cycles a write request w... |
| TCC_EA1_WRREQ_STALL[9]:device=0       | Number of cycles a write request w... |
| TCC_EA1_WRREQ_STALL[10]:device=0      | Number of cycles a write request w... |
| TCC_EA1_WRREQ_STALL[11]:device=0      | Number of cycles a write request w... |
| TCC_EA1_WRREQ_STALL[12]:device=0      | Number of cycles a write request w... |
| TCC_EA1_WRREQ_STALL[13]:device=0      | Number of cycles a write request w... |
| TCC_EA1_WRREQ_STALL[14]:device=0      | Number of cycles a write request w... |
| TCC_EA1_WRREQ_STALL[15]:device=0      | Number of cycles a write request w... |
| TCC_EA1_RDREQ[0]:device=0             | Number of TCC/EA read requests (ei... |
| TCC_EA1_RDREQ[1]:device=0             | Number of TCC/EA read requests (ei... |
| TCC_EA1_RDREQ[2]:device=0             | Number of TCC/EA read requests (ei... |
| TCC_EA1_RDREQ[3]:device=0             | Number of TCC/EA read requests (ei... |
| TCC_EA1_RDREQ[4]:device=0             | Number of TCC/EA read requests (ei... |
| TCC_EA1_RDREQ[5]:device=0             | Number of TCC/EA read requests (ei... |
| TCC_EA1_RDREQ[6]:device=0             | Number of TCC/EA read requests (ei... |
| TCC_EA1_RDREQ[7]:device=0             | Number of TCC/EA read requests (ei... |
| TCC_EA1_RDREQ[8]:device=0             | Number of TCC/EA read requests (ei... |
| TCC_EA1_RDREQ[9]:device=0             | Number of TCC/EA read requests (ei... |
| TCC_EA1_RDREQ[10]:device=0            | Number of TCC/EA read requests (ei... |
| TCC_EA1_RDREQ[11]:device=0            | Number of TCC/EA read requests (ei... |
| TCC_EA1_RDREQ[12]:device=0            | Number of TCC/EA read requests (ei... |
| TCC_EA1_RDREQ[13]:device=0            | Number of TCC/EA read requests (ei... |
| TCC_EA1_RDREQ[14]:device=0            | Number of TCC/EA read requests (ei... |
| TCC_EA1_RDREQ[15]:device=0            | Number of TCC/EA read requests (ei... |
| TCC_EA1_RDREQ_32B[0]:device=0         | Number of 32-byte TCC/EA read requ... |
| TCC_EA1_RDREQ_32B[1]:device=0         | Number of 32-byte TCC/EA read requ... |
| TCC_EA1_RDREQ_32B[2]:device=0         | Number of 32-byte TCC/EA read requ... |
| TCC_EA1_RDREQ_32B[3]:device=0         | Number of 32-byte TCC/EA read requ... |
| TCC_EA1_RDREQ_32B[4]:device=0         | Number of 32-byte TCC/EA read requ... |
| TCC_EA1_RDREQ_32B[5]:device=0         | Number of 32-byte TCC/EA read requ... |
| TCC_EA1_RDREQ_32B[6]:device=0         | Number of 32-byte TCC/EA read requ... |
| TCC_EA1_RDREQ_32B[7]:device=0         | Number of 32-byte TCC/EA read requ... |
| TCC_EA1_RDREQ_32B[8]:device=0         | Number of 32-byte TCC/EA read requ... |
| TCC_EA1_RDREQ_32B[9]:device=0         | Number of 32-byte TCC/EA read requ... |
| TCC_EA1_RDREQ_32B[10]:device=0        | Number of 32-byte TCC/EA read requ... |
| TCC_EA1_RDREQ_32B[11]:device=0        | Number of 32-byte TCC/EA read requ... |
| TCC_EA1_RDREQ_32B[12]:device=0        | Number of 32-byte TCC/EA read requ... |
| TCC_EA1_RDREQ_32B[13]:device=0        | Number of 32-byte TCC/EA read requ... |
| TCC_EA1_RDREQ_32B[14]:device=0        | Number of 32-byte TCC/EA read requ... |
| TCC_EA1_RDREQ_32B[15]:device=0        | Number of 32-byte TCC/EA read requ... |
| GRBM_COUNT:device=0                   | Tie High - Count Number of Clocks     |
| GRBM_GUI_ACTIVE:device=0              | The GUI is Active                     |
| SQ_WAVES:device=0                     | Count number of waves sent to SQs.... |
| SQ_INSTS_VALU:device=0                | Number of VALU instructions issued... |
| SQ_INSTS_VMEM_WR:device=0             | Number of VMEM write instructions ... |
| SQ_INSTS_VMEM_RD:device=0             | Number of VMEM read instructions i... |
| SQ_INSTS_SALU:device=0                | Number of SALU instructions issued... |
| SQ_INSTS_SMEM:device=0                | Number of SMEM instructions issued... |
| SQ_INSTS_FLAT:device=0                | Number of FLAT instructions issued... |
| SQ_INSTS_FLAT_LDS_ONLY:device=0       | Number of FLAT instructions issued... |
| SQ_INSTS_LDS:device=0                 | Number of LDS instructions issued ... |
| SQ_INSTS_GDS:device=0                 | Number of GDS instructions issued.... |
| SQ_WAIT_INST_LDS:device=0             | Number of wave-cycles spent waitin... |
| SQ_ACTIVE_INST_VALU:device=0          | regspec 71? Number of cycles the S... |
| SQ_INST_CYCLES_SALU:device=0          | Number of cycles needed to execute... |
| SQ_THREAD_CYCLES_VALU:device=0        | Number of thread-cycles used to ex... |
| SQ_LDS_BANK_CONFLICT:device=0         | Number of cycles LDS is stalled by... |
| TA_TA_BUSY[0]:device=0                | TA block is busy. Perf_Windowing n... |
| TA_TA_BUSY[1]:device=0                | TA block is busy. Perf_Windowing n... |
| TA_TA_BUSY[2]:device=0                | TA block is busy. Perf_Windowing n... |
| TA_TA_BUSY[3]:device=0                | TA block is busy. Perf_Windowing n... |
| TA_TA_BUSY[4]:device=0                | TA block is busy. Perf_Windowing n... |
| TA_TA_BUSY[5]:device=0                | TA block is busy. Perf_Windowing n... |
| TA_TA_BUSY[6]:device=0                | TA block is busy. Perf_Windowing n... |
| TA_TA_BUSY[7]:device=0                | TA block is busy. Perf_Windowing n... |
| TA_TA_BUSY[8]:device=0                | TA block is busy. Perf_Windowing n... |
| TA_TA_BUSY[9]:device=0                | TA block is busy. Perf_Windowing n... |
| TA_TA_BUSY[10]:device=0               | TA block is busy. Perf_Windowing n... |
| TA_TA_BUSY[11]:device=0               | TA block is busy. Perf_Windowing n... |
| TA_TA_BUSY[12]:device=0               | TA block is busy. Perf_Windowing n... |
| TA_TA_BUSY[13]:device=0               | TA block is busy. Perf_Windowing n... |
| TA_TA_BUSY[14]:device=0               | TA block is busy. Perf_Windowing n... |
| TA_TA_BUSY[15]:device=0               | TA block is busy. Perf_Windowing n... |
| TA_FLAT_READ_WAVEFRONTS[0]:device=0   | Number of flat opcode reads proces... |
| TA_FLAT_READ_WAVEFRONTS[1]:device=0   | Number of flat opcode reads proces... |
| TA_FLAT_READ_WAVEFRONTS[2]:device=0   | Number of flat opcode reads proces... |
| TA_FLAT_READ_WAVEFRONTS[3]:device=0   | Number of flat opcode reads proces... |
| TA_FLAT_READ_WAVEFRONTS[4]:device=0   | Number of flat opcode reads proces... |
| TA_FLAT_READ_WAVEFRONTS[5]:device=0   | Number of flat opcode reads proces... |
| TA_FLAT_READ_WAVEFRONTS[6]:device=0   | Number of flat opcode reads proces... |
| TA_FLAT_READ_WAVEFRONTS[7]:device=0   | Number of flat opcode reads proces... |
| TA_FLAT_READ_WAVEFRONTS[8]:device=0   | Number of flat opcode reads proces... |
| TA_FLAT_READ_WAVEFRONTS[9]:device=0   | Number of flat opcode reads proces... |
| TA_FLAT_READ_WAVEFRONTS[10]:device=0  | Number of flat opcode reads proces... |
| TA_FLAT_READ_WAVEFRONTS[11]:device=0  | Number of flat opcode reads proces... |
| TA_FLAT_READ_WAVEFRONTS[12]:device=0  | Number of flat opcode reads proces... |
| TA_FLAT_READ_WAVEFRONTS[13]:device=0  | Number of flat opcode reads proces... |
| TA_FLAT_READ_WAVEFRONTS[14]:device=0  | Number of flat opcode reads proces... |
| TA_FLAT_READ_WAVEFRONTS[15]:device=0  | Number of flat opcode reads proces... |
| TA_FLAT_WRITE_WAVEFRONTS[0]:device=0  | Number of flat opcode writes proce... |
| TA_FLAT_WRITE_WAVEFRONTS[1]:device=0  | Number of flat opcode writes proce... |
| TA_FLAT_WRITE_WAVEFRONTS[2]:device=0  | Number of flat opcode writes proce... |
| TA_FLAT_WRITE_WAVEFRONTS[3]:device=0  | Number of flat opcode writes proce... |
| TA_FLAT_WRITE_WAVEFRONTS[4]:device=0  | Number of flat opcode writes proce... |
| TA_FLAT_WRITE_WAVEFRONTS[5]:device=0  | Number of flat opcode writes proce... |
| TA_FLAT_WRITE_WAVEFRONTS[6]:device=0  | Number of flat opcode writes proce... |
| TA_FLAT_WRITE_WAVEFRONTS[7]:device=0  | Number of flat opcode writes proce... |
| TA_FLAT_WRITE_WAVEFRONTS[8]:device=0  | Number of flat opcode writes proce... |
| TA_FLAT_WRITE_WAVEFRONTS[9]:device=0  | Number of flat opcode writes proce... |
| TA_FLAT_WRITE_WAVEFRONTS[10]:device=0 | Number of flat opcode writes proce... |
| TA_FLAT_WRITE_WAVEFRONTS[11]:device=0 | Number of flat opcode writes proce... |
| TA_FLAT_WRITE_WAVEFRONTS[12]:device=0 | Number of flat opcode writes proce... |
| TA_FLAT_WRITE_WAVEFRONTS[13]:device=0 | Number of flat opcode writes proce... |
| TA_FLAT_WRITE_WAVEFRONTS[14]:device=0 | Number of flat opcode writes proce... |
| TA_FLAT_WRITE_WAVEFRONTS[15]:device=0 | Number of flat opcode writes proce... |
| TCC_HIT[0]:device=0                   | Number of cache hits.                 |
| TCC_HIT[1]:device=0                   | Number of cache hits.                 |
| TCC_HIT[2]:device=0                   | Number of cache hits.                 |
| TCC_HIT[3]:device=0                   | Number of cache hits.                 |
| TCC_HIT[4]:device=0                   | Number of cache hits.                 |
| TCC_HIT[5]:device=0                   | Number of cache hits.                 |
| TCC_HIT[6]:device=0                   | Number of cache hits.                 |
| TCC_HIT[7]:device=0                   | Number of cache hits.                 |
| TCC_HIT[8]:device=0                   | Number of cache hits.                 |
| TCC_HIT[9]:device=0                   | Number of cache hits.                 |
| TCC_HIT[10]:device=0                  | Number of cache hits.                 |
| TCC_HIT[11]:device=0                  | Number of cache hits.                 |
| TCC_HIT[12]:device=0                  | Number of cache hits.                 |
| TCC_HIT[13]:device=0                  | Number of cache hits.                 |
| TCC_HIT[14]:device=0                  | Number of cache hits.                 |
| TCC_HIT[15]:device=0                  | Number of cache hits.                 |
| TCC_MISS[0]:device=0                  | Number of cache misses. UC reads c... |
| TCC_MISS[1]:device=0                  | Number of cache misses. UC reads c... |
| TCC_MISS[2]:device=0                  | Number of cache misses. UC reads c... |
| TCC_MISS[3]:device=0                  | Number of cache misses. UC reads c... |
| TCC_MISS[4]:device=0                  | Number of cache misses. UC reads c... |
| TCC_MISS[5]:device=0                  | Number of cache misses. UC reads c... |
| TCC_MISS[6]:device=0                  | Number of cache misses. UC reads c... |
| TCC_MISS[7]:device=0                  | Number of cache misses. UC reads c... |
| TCC_MISS[8]:device=0                  | Number of cache misses. UC reads c... |
| TCC_MISS[9]:device=0                  | Number of cache misses. UC reads c... |
| TCC_MISS[10]:device=0                 | Number of cache misses. UC reads c... |
| TCC_MISS[11]:device=0                 | Number of cache misses. UC reads c... |
| TCC_MISS[12]:device=0                 | Number of cache misses. UC reads c... |
| TCC_MISS[13]:device=0                 | Number of cache misses. UC reads c... |
| TCC_MISS[14]:device=0                 | Number of cache misses. UC reads c... |
| TCC_MISS[15]:device=0                 | Number of cache misses. UC reads c... |
| TCC_EA_WRREQ[0]:device=0              | Number of transactions (either 32-... |
| TCC_EA_WRREQ[1]:device=0              | Number of transactions (either 32-... |
| TCC_EA_WRREQ[2]:device=0              | Number of transactions (either 32-... |
| TCC_EA_WRREQ[3]:device=0              | Number of transactions (either 32-... |
| TCC_EA_WRREQ[4]:device=0              | Number of transactions (either 32-... |
| TCC_EA_WRREQ[5]:device=0              | Number of transactions (either 32-... |
| TCC_EA_WRREQ[6]:device=0              | Number of transactions (either 32-... |
| TCC_EA_WRREQ[7]:device=0              | Number of transactions (either 32-... |
| TCC_EA_WRREQ[8]:device=0              | Number of transactions (either 32-... |
| TCC_EA_WRREQ[9]:device=0              | Number of transactions (either 32-... |
| TCC_EA_WRREQ[10]:device=0             | Number of transactions (either 32-... |
| TCC_EA_WRREQ[11]:device=0             | Number of transactions (either 32-... |
| TCC_EA_WRREQ[12]:device=0             | Number of transactions (either 32-... |
| TCC_EA_WRREQ[13]:device=0             | Number of transactions (either 32-... |
| TCC_EA_WRREQ[14]:device=0             | Number of transactions (either 32-... |
| TCC_EA_WRREQ[15]:device=0             | Number of transactions (either 32-... |
| TCC_EA_WRREQ_64B[0]:device=0          | Number of 64-byte transactions goi... |
| TCC_EA_WRREQ_64B[1]:device=0          | Number of 64-byte transactions goi... |
| TCC_EA_WRREQ_64B[2]:device=0          | Number of 64-byte transactions goi... |
| TCC_EA_WRREQ_64B[3]:device=0          | Number of 64-byte transactions goi... |
| TCC_EA_WRREQ_64B[4]:device=0          | Number of 64-byte transactions goi... |
| TCC_EA_WRREQ_64B[5]:device=0          | Number of 64-byte transactions goi... |
| TCC_EA_WRREQ_64B[6]:device=0          | Number of 64-byte transactions goi... |
| TCC_EA_WRREQ_64B[7]:device=0          | Number of 64-byte transactions goi... |
| TCC_EA_WRREQ_64B[8]:device=0          | Number of 64-byte transactions goi... |
| TCC_EA_WRREQ_64B[9]:device=0          | Number of 64-byte transactions goi... |
| TCC_EA_WRREQ_64B[10]:device=0         | Number of 64-byte transactions goi... |
| TCC_EA_WRREQ_64B[11]:device=0         | Number of 64-byte transactions goi... |
| TCC_EA_WRREQ_64B[12]:device=0         | Number of 64-byte transactions goi... |
| TCC_EA_WRREQ_64B[13]:device=0         | Number of 64-byte transactions goi... |
| TCC_EA_WRREQ_64B[14]:device=0         | Number of 64-byte transactions goi... |
| TCC_EA_WRREQ_64B[15]:device=0         | Number of 64-byte transactions goi... |
| TCC_EA_WRREQ_STALL[0]:device=0        | Number of cycles a write request w... |
| TCC_EA_WRREQ_STALL[1]:device=0        | Number of cycles a write request w... |
| TCC_EA_WRREQ_STALL[2]:device=0        | Number of cycles a write request w... |
| TCC_EA_WRREQ_STALL[3]:device=0        | Number of cycles a write request w... |
| TCC_EA_WRREQ_STALL[4]:device=0        | Number of cycles a write request w... |
| TCC_EA_WRREQ_STALL[5]:device=0        | Number of cycles a write request w... |
| TCC_EA_WRREQ_STALL[6]:device=0        | Number of cycles a write request w... |
| TCC_EA_WRREQ_STALL[7]:device=0        | Number of cycles a write request w... |
| TCC_EA_WRREQ_STALL[8]:device=0        | Number of cycles a write request w... |
| TCC_EA_WRREQ_STALL[9]:device=0        | Number of cycles a write request w... |
| TCC_EA_WRREQ_STALL[10]:device=0       | Number of cycles a write request w... |
| TCC_EA_WRREQ_STALL[11]:device=0       | Number of cycles a write request w... |
| TCC_EA_WRREQ_STALL[12]:device=0       | Number of cycles a write request w... |
| TCC_EA_WRREQ_STALL[13]:device=0       | Number of cycles a write request w... |
| TCC_EA_WRREQ_STALL[14]:device=0       | Number of cycles a write request w... |
| TCC_EA_WRREQ_STALL[15]:device=0       | Number of cycles a write request w... |
| TCC_EA_RDREQ[0]:device=0              | Number of TCC/EA read requests (ei... |
| TCC_EA_RDREQ[1]:device=0              | Number of TCC/EA read requests (ei... |
| TCC_EA_RDREQ[2]:device=0              | Number of TCC/EA read requests (ei... |
| TCC_EA_RDREQ[3]:device=0              | Number of TCC/EA read requests (ei... |
| TCC_EA_RDREQ[4]:device=0              | Number of TCC/EA read requests (ei... |
| TCC_EA_RDREQ[5]:device=0              | Number of TCC/EA read requests (ei... |
| TCC_EA_RDREQ[6]:device=0              | Number of TCC/EA read requests (ei... |
| TCC_EA_RDREQ[7]:device=0              | Number of TCC/EA read requests (ei... |
| TCC_EA_RDREQ[8]:device=0              | Number of TCC/EA read requests (ei... |
| TCC_EA_RDREQ[9]:device=0              | Number of TCC/EA read requests (ei... |
| TCC_EA_RDREQ[10]:device=0             | Number of TCC/EA read requests (ei... |
| TCC_EA_RDREQ[11]:device=0             | Number of TCC/EA read requests (ei... |
| TCC_EA_RDREQ[12]:device=0             | Number of TCC/EA read requests (ei... |
| TCC_EA_RDREQ[13]:device=0             | Number of TCC/EA read requests (ei... |
| TCC_EA_RDREQ[14]:device=0             | Number of TCC/EA read requests (ei... |
| TCC_EA_RDREQ[15]:device=0             | Number of TCC/EA read requests (ei... |
| TCC_EA_RDREQ_32B[0]:device=0          | Number of 32-byte TCC/EA read requ... |
| TCC_EA_RDREQ_32B[1]:device=0          | Number of 32-byte TCC/EA read requ... |
| TCC_EA_RDREQ_32B[2]:device=0          | Number of 32-byte TCC/EA read requ... |
| TCC_EA_RDREQ_32B[3]:device=0          | Number of 32-byte TCC/EA read requ... |
| TCC_EA_RDREQ_32B[4]:device=0          | Number of 32-byte TCC/EA read requ... |
| TCC_EA_RDREQ_32B[5]:device=0          | Number of 32-byte TCC/EA read requ... |
| TCC_EA_RDREQ_32B[6]:device=0          | Number of 32-byte TCC/EA read requ... |
| TCC_EA_RDREQ_32B[7]:device=0          | Number of 32-byte TCC/EA read requ... |
| TCC_EA_RDREQ_32B[8]:device=0          | Number of 32-byte TCC/EA read requ... |
| TCC_EA_RDREQ_32B[9]:device=0          | Number of 32-byte TCC/EA read requ... |
| TCC_EA_RDREQ_32B[10]:device=0         | Number of 32-byte TCC/EA read requ... |
| TCC_EA_RDREQ_32B[11]:device=0         | Number of 32-byte TCC/EA read requ... |
| TCC_EA_RDREQ_32B[12]:device=0         | Number of 32-byte TCC/EA read requ... |
| TCC_EA_RDREQ_32B[13]:device=0         | Number of 32-byte TCC/EA read requ... |
| TCC_EA_RDREQ_32B[14]:device=0         | Number of 32-byte TCC/EA read requ... |
| TCC_EA_RDREQ_32B[15]:device=0         | Number of 32-byte TCC/EA read requ... |
| TCP_TCP_TA_DATA_STALL_CYCLES[0]:de... | TCP stalls TA data interface. Now ... |
| TCP_TCP_TA_DATA_STALL_CYCLES[1]:de... | TCP stalls TA data interface. Now ... |
| TCP_TCP_TA_DATA_STALL_CYCLES[2]:de... | TCP stalls TA data interface. Now ... |
| TCP_TCP_TA_DATA_STALL_CYCLES[3]:de... | TCP stalls TA data interface. Now ... |
| TCP_TCP_TA_DATA_STALL_CYCLES[4]:de... | TCP stalls TA data interface. Now ... |
| TCP_TCP_TA_DATA_STALL_CYCLES[5]:de... | TCP stalls TA data interface. Now ... |
| TCP_TCP_TA_DATA_STALL_CYCLES[6]:de... | TCP stalls TA data interface. Now ... |
| TCP_TCP_TA_DATA_STALL_CYCLES[7]:de... | TCP stalls TA data interface. Now ... |
| TCP_TCP_TA_DATA_STALL_CYCLES[8]:de... | TCP stalls TA data interface. Now ... |
| TCP_TCP_TA_DATA_STALL_CYCLES[9]:de... | TCP stalls TA data interface. Now ... |
| TCP_TCP_TA_DATA_STALL_CYCLES[10]:d... | TCP stalls TA data interface. Now ... |
| TCP_TCP_TA_DATA_STALL_CYCLES[11]:d... | TCP stalls TA data interface. Now ... |
| TCP_TCP_TA_DATA_STALL_CYCLES[12]:d... | TCP stalls TA data interface. Now ... |
| TCP_TCP_TA_DATA_STALL_CYCLES[13]:d... | TCP stalls TA data interface. Now ... |
| TCP_TCP_TA_DATA_STALL_CYCLES[14]:d... | TCP stalls TA data interface. Now ... |
| TCP_TCP_TA_DATA_STALL_CYCLES[15]:d... | TCP stalls TA data interface. Now ... |
| TCC_EA1_RDREQ_32B_sum:device=0        | Number of 32-byte TCC/EA read requ... |
| TCC_EA1_RDREQ_sum:device=0            | Number of TCC/EA read requests (ei... |
| TCC_EA1_WRREQ_sum:device=0            | Number of transactions (either 32-... |
| TCC_EA1_WRREQ_64B_sum:device=0        | Number of 64-byte transactions goi... |
| TCC_WRREQ1_STALL_max:device=0         | Number of cycles a write request w... |
| RDATA1_SIZE:device=0                  | The total kilobytes fetched from t... |
| WDATA1_SIZE:device=0                  | The total kilobytes written to the... |
| FETCH_SIZE:device=0                   | The total kilobytes fetched from t... |
| WRITE_SIZE:device=0                   | The total kilobytes written to the... |
| WRITE_REQ_32B:device=0                | The total number of 32-byte effect... |
| TA_BUSY_avr:device=0                  | TA block is busy. Average over TA ... |
| TA_BUSY_max:device=0                  | TA block is busy. Max over TA inst... |
| TA_BUSY_min:device=0                  | TA block is busy. Min over TA inst... |
| TA_FLAT_READ_WAVEFRONTS_sum:device=0  | Number of flat opcode reads proces... |
| TA_FLAT_WRITE_WAVEFRONTS_sum:device=0 | Number of flat opcode writes proce... |
| TCC_HIT_sum:device=0                  | Number of cache hits. Sum over TCC... |
| TCC_MISS_sum:device=0                 | Number of cache misses. Sum over T... |
| TCC_EA_RDREQ_32B_sum:device=0         | Number of 32-byte TCC/EA read requ... |
| TCC_EA_RDREQ_sum:device=0             | Number of TCC/EA read requests (ei... |
| TCC_EA_WRREQ_sum:device=0             | Number of transactions (either 32-... |
| TCC_EA_WRREQ_64B_sum:device=0         | Number of 64-byte transactions goi... |
| TCC_WRREQ_STALL_max:device=0          | Number of cycles a write request w... |
| GPUBusy:device=0                      | The percentage of time GPU was busy.  |
| Wavefronts:device=0                   | Total wavefronts.                     |
| VALUInsts:device=0                    | The average number of vector ALU i... |
| SALUInsts:device=0                    | The average number of scalar ALU i... |
| VFetchInsts:device=0                  | The average number of vector fetch... |
| SFetchInsts:device=0                  | The average number of scalar fetch... |
| VWriteInsts:device=0                  | The average number of vector write... |
| FlatVMemInsts:device=0                | The average number of FLAT instruc... |
| LDSInsts:device=0                     | The average number of LDS read or ... |
| FlatLDSInsts:device=0                 | The average number of FLAT instruc... |
| GDSInsts:device=0                     | The average number of GDS read or ... |
| VALUUtilization:device=0              | The percentage of active vector AL... |
| VALUBusy:device=0                     | The percentage of GPUTime vector A... |
| SALUBusy:device=0                     | The percentage of GPUTime scalar A... |
| FetchSize:device=0                    | The total kilobytes fetched from t... |
| WriteSize:device=0                    | The total kilobytes written to the... |
| MemWrites32B:device=0                 | The total number of effective 32B ... |
| L2CacheHit:device=0                   | The percentage of fetch, write, at... |
| MemUnitBusy:device=0                  | The percentage of GPUTime the memo... |
| MemUnitStalled:device=0               | The percentage of GPUTime the memo... |
| WriteUnitStalled:device=0             | The percentage of GPUTime the Writ... |
| ALUStalledByLDS:device=0              | The percentage of GPUTime ALU unit... |
| LDSBankConflict:device=0              | The percentage of GPUTime LDS is s... |
|---------------------------------------|---------------------------------------|
```

## Creating a Configuration File

[Omnitrace](https://github.com/AMDResearch/omnitrace) supports 3 configuration file formats: JSON, XML, and plain text.
Use `omnitrace-avail -G <filename> -F txt json xml` to generate default configuration files of each format and, optionally,
include the `--all` flag for descriptions, etc.
Configuration files are specified via the `OMNITRACE_CONFIG_FILE` environment variable
and by default will look for `${HOME}/.omnitrace.cfg` and `${HOME}/.omnitrace.json`.
Multiple configuration files can be concatenated via `:`, e.g.:

```shell
export OMNITRACE_CONFIG_FILE=~/.config/omnitrace.cfg:~/.config/omnitrace.json
```

If a configuration variable is specified in both a configuration file and in the environment,
the environment variable takes precedence.

### Sample Text Configuration File

Text files support very basic variables and are case-insensitive.
Variables are created when an lvalue starts with a $ and are
dereferenced when they appear as rvalues.

Entries in the text configuration file which do not match to a known setting
in `omnitrace-avail` but are prefixed with `OMNITRACE_` are interpreted as
environment variables and are exported via `setenv`
but do not override an existing value for the environment variable.

```shell
# lvals starting with $ are variables
$ENABLE                         = ON
$SAMPLE                         = OFF

# use fields
OMNITRACE_USE_PERFETTO          = $ENABLE
OMNITRACE_USE_TIMEMORY          = $ENABLE
OMNITRACE_USE_SAMPLING          = $SAMPLE
OMNITRACE_USE_PROCESS_SAMPLING  = $SAMPLE

# debug
OMNITRACE_DEBUG                 = OFF
OMNITRACE_VERBOSE               = 1

# output fields
OMNITRACE_OUTPUT_PATH           = omnitrace-output
OMNITRACE_OUTPUT_PREFIX         = %tag%/
OMNITRACE_TIME_OUTPUT           = OFF
OMNITRACE_USE_PID               = OFF

# timemory fields
OMNITRACE_PAPI_EVENTS           = PAPI_TOT_INS PAPI_FP_INS
OMNITRACE_TIMEMORY_COMPONENTS   = wall_clock peak_rss trip_count
OMNITRACE_MEMORY_UNITS          = MB
OMNITRACE_TIMING_UNITS          = sec

# sampling fields
OMNITRACE_SAMPLING_FREQ         = 50
OMNITRACE_SAMPLING_DELAY        = 0.1
OMNITRACE_SAMPLING_CPUS         = 0-3
OMNITRACE_SAMPLING_GPUS         = $env:HIP_VISIBLE_DEVICES

# misc env variables (see metadata JSON file after run)
$env:OMNITRACE_SAMPLING_KEEP_DYNINST_SUFFIX  = OFF
```

### Sample JSON Configuration File

The full JSON specification for a configuration value contains a lot of information:

```json
{
    "omnitrace": {
        "settings": {
            "OMNITRACE_ADD_SECONDARY": {
                "count": -1,
                "name": "add_secondary",
                "data_type": "bool",
                "initial": true,
                "value": true,
                "max_count": 1,
                "cmdline": [
                    "--omnitrace-add-secondary"
                ],
                "environ": "OMNITRACE_ADD_SECONDARY",
                "cereal_class_version": 1,
                "categories": [
                    "component",
                    "data",
                    "native"
                ],
                "description": "Enable/disable components adding secondary (child) entries when available. E.g. suppress individual CUDA kernels, etc. when using Cupti components"
            }
        }
    }
}
```

However when writing an JSON configuration file, the following is minimally acceptable to set `OMNITRACE_ADD_SECONDARY=false`:

```json
{
    "omnitrace": {
        "settings": {
            "OMNITRACE_ADD_SECONDARY": {
                "value": true
            }
        }
    }
}
```

### Sample XML Configuration File

The full XML specification for a configuration value contains the same information as the JSON specification:

```xml
<?xml version="1.0" encoding="utf-8"?>
<timemory_xml>
    <omnitrace>
        <settings>
            <cereal_class_version>2</cereal_class_version>
            <!-- Full setting specification -->
            <OMNITRACE_ADD_SECONDARY>
                <cereal_class_version>1</cereal_class_version>
                <name>add_secondary</name>
                <environ>OMNITRACE_ADD_SECONDARY</environ>
                <description>...</description>
                <count>-1</count>
                <max_count>1</max_count>
                <cmdline>
                    <value0>--omnitrace-add-secondary</value0>
                </cmdline>
                <categories>
                    <value0>component</value0>
                    <value1>data</value1>
                    <value2>native</value2>
                </categories>
                <data_type>bool</data_type>
                <initial>true</initial>
                <value>true</value>
            </OMNITRACE_ADD_SECONDARY>
            <!-- etc. -->
        </settings>
    </omnitrace>
</timemory_xml>
```

However, when writing an XML configuration file, the following is minimally acceptable to set `OMNITRACE_ADD_SECONDARY=false`:

```xml
<?xml version="1.0" encoding="utf-8"?>
<timemory_xml>
    <omnitrace>
        <settings>
            <OMNITRACE_ADD_SECONDARY>
                <value>false</value>
            </OMNITRACE_ADD_SECONDARY>
        </settings>
    </omnitrace>
</timemory_xml>
```
