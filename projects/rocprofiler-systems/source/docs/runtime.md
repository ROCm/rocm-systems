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

### Exploring Runtime Settings

In order to view the list of the available runtime settings, their current value, and descriptions for each setting:

```shell
omnitrace-avail --description
```

> HINT: use `--brief` to suppress printing current value and/or `-c 0` to suppress truncation of the descriptions

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

[Omnitrace](https://github.com/AMDResearch/omnitrace) supports collecting hardware counters via PAPI.

View all possible hardware counters and their descriptions:

```shell
omnitrace-avail --hw-counters --description
```

### omnitrace-avail Examples

#### Settings

```console
$ omnitrace-avail -S -bd
|-----------------------------------------|-----------------------------------------|
|          ENVIRONMENT VARIABLE           |               DESCRIPTION               |
|-----------------------------------------|-----------------------------------------|
| OMNITRACE_ADD_SECONDARY                 | Enable/disable components adding sec... |
| OMNITRACE_BACKEND                       | Specify the perfetto backend to acti... |
| OMNITRACE_BUFFER_SIZE_KB                | Size of perfetto buffer (in KB)         |
| OMNITRACE_COLLAPSE_PROCESSES            | Enable/disable combining process-spe... |
| OMNITRACE_COLLAPSE_THREADS              | Enable/disable combining thread-spec... |
| OMNITRACE_CONFIG_FILE                   | Configuration file for omnitrace        |
| OMNITRACE_COUT_OUTPUT                   | Write output to stdout                  |
| OMNITRACE_CRITICAL_TRACE                | Enable generation of the critical trace |
| OMNITRACE_CRITICAL_TRACE_BUFFER_COUNT   | Number of critical trace records to ... |
| OMNITRACE_CRITICAL_TRACE_COUNT          | Number of critical trace to export (... |
| OMNITRACE_CRITICAL_TRACE_DEBUG          | Enable debugging for critical trace     |
| OMNITRACE_CRITICAL_TRACE_NUM_THREADS    | Number of threads to use when genera... |
| OMNITRACE_CRITICAL_TRACE_PER_ROW        | How many critical traces per row in ... |
| OMNITRACE_CRITICAL_TRACE_SERIALIZE_N... | Include names in serialization of cr... |
| OMNITRACE_DEBUG                         | Enable debug output                     |
| OMNITRACE_DIFF_OUTPUT                   | Generate a difference output vs. a p... |
| OMNITRACE_ENABLED                       | Activation state of timemory            |
| OMNITRACE_ENABLE_SIGNAL_HANDLER         | Enable signals in timemory_init         |
| OMNITRACE_FILE_OUTPUT                   | Write output to files                   |
| OMNITRACE_FLAT_PROFILE                  | Set the label hierarchy mode to defa... |
| OMNITRACE_FLAT_SAMPLING                 | Ignore hierarchy in all statistical ... |
| OMNITRACE_INPUT_EXTENSIONS              | File extensions used when searching ... |
| OMNITRACE_INPUT_PATH                    | Explicitly specify the input folder ... |
| OMNITRACE_INPUT_PREFIX                  | Explicitly specify the prefix for in... |
| OMNITRACE_INSTRUMENTATION_INTERVAL      | Instrumentation only takes measureme... |
| OMNITRACE_JSON_OUTPUT                   | Write json output files                 |
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
| OMNITRACE_PRECISION                     | Set the global output precision for ... |
| OMNITRACE_ROCM_SMI_DEVICES              | Devices to query when OMNITRACE_USE_... |
| OMNITRACE_ROCTRACER_FLAT_PROFILE        | Ignore hierarchy in all kernels entr... |
| OMNITRACE_ROCTRACER_HSA_ACTIVITY        | Enable HSA activity tracing support     |
| OMNITRACE_ROCTRACER_HSA_API             | Enable HSA API tracing support          |
| OMNITRACE_ROCTRACER_HSA_API_TYPES       | HSA API type to collect                 |
| OMNITRACE_ROCTRACER_TIMELINE_PROFILE    | Create unique entries for every kern... |
| OMNITRACE_SAMPLING_DELAY                | Number of seconds to delay activatin... |
| OMNITRACE_SAMPLING_FREQ                 | Number of software interrupts per se... |
| OMNITRACE_SCIENTIFIC                    | Set the global numerical reporting t... |
| OMNITRACE_SETTINGS_DESC                 | Provide descriptions when printing s... |
| OMNITRACE_SHMEM_SIZE_HINT_KB            | Hint for shared-memory buffer size i... |
| OMNITRACE_SUPPRESS_CONFIG               | Disable processing of setting config... |
| OMNITRACE_SUPPRESS_PARSING              | Disable parsing environment             |
| OMNITRACE_TEXT_OUTPUT                   | Write text output files                 |
| OMNITRACE_TIMELINE_PROFILE              | Set the label hierarchy mode to defa... |
| OMNITRACE_TIMELINE_SAMPLING             | Create unique entries for every samp... |
| OMNITRACE_TIMEMORY_COMPONENTS           | List of components to collect via ti... |
| OMNITRACE_TIME_FORMAT                   | Customize the folder generation when... |
| OMNITRACE_TIME_OUTPUT                   | Output data to subfolder w/ a timest... |
| OMNITRACE_TIMING_PRECISION              | Set the precision for components wit... |
| OMNITRACE_TIMING_SCIENTIFIC             | Set the numerical reporting format f... |
| OMNITRACE_TIMING_UNITS                  | Set the units for components with 'u... |
| OMNITRACE_TIMING_WIDTH                  | Set the output width for components ... |
| OMNITRACE_TREE_OUTPUT                   | Write hierarchical json output files    |
| OMNITRACE_USE_KOKKOSP                   | Enable support for Kokkos Tools         |
| OMNITRACE_USE_PERFETTO                  | Enable perfetto backend                 |
| OMNITRACE_USE_PID                       | Enable tagging filenames with proces... |
| OMNITRACE_USE_ROCM_SMI                  | Enable sampling GPU power, temp, uti... |
| OMNITRACE_USE_ROCTRACER                 | Enable ROCM tracing                     |
| OMNITRACE_USE_SAMPLING                  | Enable statistical sampling of call-... |
| OMNITRACE_USE_TIMEMORY                  | Enable timemory backend                 |
| OMNITRACE_VERBOSE                       | Verbosity level                         |
| OMNITRACE_WIDTH                         | Set the global output width for comp... |
|-----------------------------------------|-----------------------------------------|
```

#### Components

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

#### Hardware Counters

```console
$ omnitrace-avail -H -bd
|---------------------|-------------------------------------------------|
|  HARDWARE COUNTER   |                   DESCRIPTION                   |
|---------------------|-------------------------------------------------|
|         CPU         |                                                 |
|---------------------|-------------------------------------------------|
| PAPI_L1_DCM         | Level 1 data cache misses                       |
| PAPI_L1_ICM         | Level 1 instruction cache misses                |
| PAPI_L2_DCM         | Level 2 data cache misses                       |
| PAPI_L2_ICM         | Level 2 instruction cache misses                |
| PAPI_L3_DCM         | Level 3 data cache misses                       |
| PAPI_L3_ICM         | Level 3 instruction cache misses                |
| PAPI_L1_TCM         | Level 1 cache misses                            |
| PAPI_L2_TCM         | Level 2 cache misses                            |
| PAPI_L3_TCM         | Level 3 cache misses                            |
| PAPI_CA_SNP         | Requests for a snoop                            |
| PAPI_CA_SHR         | Requests for exclusive access to shared cach... |
| PAPI_CA_CLN         | Requests for exclusive access to clean cache... |
| PAPI_CA_INV         | Requests for cache line invalidation            |
| PAPI_CA_ITV         | Requests for cache line intervention            |
| PAPI_L3_LDM         | Level 3 load misses                             |
| PAPI_L3_STM         | Level 3 store misses                            |
| PAPI_BRU_IDL        | Cycles branch units are idle                    |
| PAPI_FXU_IDL        | Cycles integer units are idle                   |
| PAPI_FPU_IDL        | Cycles floating point units are idle            |
| PAPI_LSU_IDL        | Cycles load/store units are idle                |
| PAPI_TLB_DM         | Data translation lookaside buffer misses        |
| PAPI_TLB_IM         | Instruction translation lookaside buffer misses |
| PAPI_TLB_TL         | Total translation lookaside buffer misses       |
| PAPI_L1_LDM         | Level 1 load misses                             |
| PAPI_L1_STM         | Level 1 store misses                            |
| PAPI_L2_LDM         | Level 2 load misses                             |
| PAPI_L2_STM         | Level 2 store misses                            |
| PAPI_BTAC_M         | Branch target address cache misses              |
| PAPI_PRF_DM         | Data prefetch cache misses                      |
| PAPI_L3_DCH         | Level 3 data cache hits                         |
| PAPI_TLB_SD         | Translation lookaside buffer shootdowns         |
| PAPI_CSR_FAL        | Failed store conditional instructions           |
| PAPI_CSR_SUC        | Successful store conditional instructions       |
| PAPI_CSR_TOT        | Total store conditional instructions            |
| PAPI_MEM_SCY        | Cycles Stalled Waiting for memory accesses      |
| PAPI_MEM_RCY        | Cycles Stalled Waiting for memory reads         |
| PAPI_MEM_WCY        | Cycles Stalled Waiting for memory writes        |
| PAPI_STL_ICY        | Cycles with no instruction issue                |
| PAPI_FUL_ICY        | Cycles with maximum instruction issue           |
| PAPI_STL_CCY        | Cycles with no instructions completed           |
| PAPI_FUL_CCY        | Cycles with maximum instructions completed      |
| PAPI_HW_INT         | Hardware interrupts                             |
| PAPI_BR_UCN         | Unconditional branch instructions               |
| PAPI_BR_CN          | Conditional branch instructions                 |
| PAPI_BR_TKN         | Conditional branch instructions taken           |
| PAPI_BR_NTK         | Conditional branch instructions not taken       |
| PAPI_BR_MSP         | Conditional branch instructions mispredicted    |
| PAPI_BR_PRC         | Conditional branch instructions correctly pr... |
| PAPI_FMA_INS        | FMA instructions completed                      |
| PAPI_TOT_IIS        | Instructions issued                             |
| PAPI_TOT_INS        | Instructions completed                          |
| PAPI_INT_INS        | Integer instructions                            |
| PAPI_FP_INS         | Floating point instructions                     |
| PAPI_LD_INS         | Load instructions                               |
| PAPI_SR_INS         | Store instructions                              |
| PAPI_BR_INS         | Branch instructions                             |
| PAPI_VEC_INS        | Vector/SIMD instructions (could include inte... |
| PAPI_RES_STL        | Cycles stalled on any resource                  |
| PAPI_FP_STAL        | Cycles the FP unit(s) are stalled               |
| PAPI_TOT_CYC        | Total cycles                                    |
| PAPI_LST_INS        | Load/store instructions completed               |
| PAPI_SYC_INS        | Synchronization instructions completed          |
| PAPI_L1_DCH         | Level 1 data cache hits                         |
| PAPI_L2_DCH         | Level 2 data cache hits                         |
| PAPI_L1_DCA         | Level 1 data cache accesses                     |
| PAPI_L2_DCA         | Level 2 data cache accesses                     |
| PAPI_L3_DCA         | Level 3 data cache accesses                     |
| PAPI_L1_DCR         | Level 1 data cache reads                        |
| PAPI_L2_DCR         | Level 2 data cache reads                        |
| PAPI_L3_DCR         | Level 3 data cache reads                        |
| PAPI_L1_DCW         | Level 1 data cache writes                       |
| PAPI_L2_DCW         | Level 2 data cache writes                       |
| PAPI_L3_DCW         | Level 3 data cache writes                       |
| PAPI_L1_ICH         | Level 1 instruction cache hits                  |
| PAPI_L2_ICH         | Level 2 instruction cache hits                  |
| PAPI_L3_ICH         | Level 3 instruction cache hits                  |
| PAPI_L1_ICA         | Level 1 instruction cache accesses              |
| PAPI_L2_ICA         | Level 2 instruction cache accesses              |
| PAPI_L3_ICA         | Level 3 instruction cache accesses              |
| PAPI_L1_ICR         | Level 1 instruction cache reads                 |
| PAPI_L2_ICR         | Level 2 instruction cache reads                 |
| PAPI_L3_ICR         | Level 3 instruction cache reads                 |
| PAPI_L1_ICW         | Level 1 instruction cache writes                |
| PAPI_L2_ICW         | Level 2 instruction cache writes                |
| PAPI_L3_ICW         | Level 3 instruction cache writes                |
| PAPI_L1_TCH         | Level 1 total cache hits                        |
| PAPI_L2_TCH         | Level 2 total cache hits                        |
| PAPI_L3_TCH         | Level 3 total cache hits                        |
| PAPI_L1_TCA         | Level 1 total cache accesses                    |
| PAPI_L2_TCA         | Level 2 total cache accesses                    |
| PAPI_L3_TCA         | Level 3 total cache accesses                    |
| PAPI_L1_TCR         | Level 1 total cache reads                       |
| PAPI_L2_TCR         | Level 2 total cache reads                       |
| PAPI_L3_TCR         | Level 3 total cache reads                       |
| PAPI_L1_TCW         | Level 1 total cache writes                      |
| PAPI_L2_TCW         | Level 2 total cache writes                      |
| PAPI_L3_TCW         | Level 3 total cache writes                      |
| PAPI_FML_INS        | Floating point multiply instructions            |
| PAPI_FAD_INS        | Floating point add instructions                 |
| PAPI_FDV_INS        | Floating point divide instructions              |
| PAPI_FSQ_INS        | Floating point square root instructions         |
| PAPI_FNV_INS        | Floating point inverse instructions             |
| PAPI_FP_OPS         | Floating point operations                       |
| PAPI_SP_OPS         | Floating point operations; optimized to coun... |
| PAPI_DP_OPS         | Floating point operations; optimized to coun... |
| PAPI_VEC_SP         | Single precision vector/SIMD instructions       |
| PAPI_VEC_DP         | Double precision vector/SIMD instructions       |
| PAPI_REF_CYC        | Reference clock cycles                          |
|---------------------|-------------------------------------------------|
```

## Creating a Configuration File

[Omnitrace](https://github.com/AMDResearch/omnitrace) supports 3 configuration file formats: JSON, XML, and plain text.
Configuration files are specified via the `OMNITRACE_CONFIG_FILE` environment variable
and by default will look for `${HOME}/omnitrace.cfg` and `${HOME}/omnitrace.json`.
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
$USE                            = ON

# use fields
OMNITRACE_USE_PERFETTO          = $USE
OMNITRACE_USE_TIMEMORY          = $USE
OMNITRACE_USE_SAMPLING          = $USE
OMNITRACE_USE_PID               = OFF
OMNITRACE_CRITICAL_TRACE        = OFF

# debug
OMNITRACE_DEBUG                 = OFF
OMNITRACE_VERBOSE               = 1
OMNITRACE_DL_VERBOSE            = 1

# output fields
OMNITRACE_OUTPUT_PREFIX         = %tag%-
OMNITRACE_OUTPUT_PATH           = omnitrace-example-output
OMNITRACE_TIME_OUTPUT           = OFF

# timemory fields
OMNITRACE_PAPI_EVENTS           = PAPI_TOT_INS PAPI_FP_INS
OMNITRACE_TIMEMORY_COMPONENTS   = wall_clock trip_count

# sampling fields
OMNITRACE_SAMPLING_FREQ         = 10

# rocm-smi fields
OMNITRACE_ROCM_SMI_DEVICES      = 1

# misc env variables
OMNITRACE_SAMPLING_KEEP_DYNINST_SUFFIX  = OFF
OMNITRACE_SAMPLING_KEEP_INTERNAL        = OFF
```

### Sample XML Configuration File

The full XML specification for a configuration value contains
a lot of information:

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
                    <value0>--timemory-add-secondary</value0>
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

Howver when writing an XML configuration file, the following is perfectly acceptable
to set `OMNITRACE_ADD_SECONDARY=false`:

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

### Sample JSON Configuration File

The full JSON specification for a configuration value contains the same information as the XML:

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
                    "--timemory-add-secondary"
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

Similarly, the
Howver when writing an XML configuration file, the following is perfectly acceptable
to set `OMNITRACE_ADD_SECONDARY=false`:

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
