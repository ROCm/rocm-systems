.. meta::
   :description: ROCm Systems Profiler runtime options documentation and reference
   :keywords: rocprof-sys, rocprofiler-systems, Omnitrace, ROCm, runtime options, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

****************************************************
Configuring runtime options
****************************************************

The ``rocprof-sys.cfg`` file maintains a list of the
`ROCm Systems Profiler <https://github.com/ROCm/rocprofiler-systems>`_ runtime
options. To create this configuration
file and view the current runtime options, use the ``rocprof-sys-avail`` executable.

The rocprof-sys-avail executable
========================================

The ``rocprof-sys-avail`` executable provides information about the runtime settings,
data collection capabilities, and, when built with PAPI support, the
available hardware counters. The executable is effectively
self-updating. As new capabilities and settings are added to the ROCm Systems Profiler source code, they are
propagated to ``rocprof-sys-avail``. ``rocprof-sys-avail`` should be viewed as the ultimate authority
in the event of any conflicts with this documentation.

It is recommended that you create a default configuration file in
``${HOME}/.rocprof-sys.cfg``. This can be done by
running the command ``rocprof-sys-avail -G ~/.rocprof-sys.cfg``. Alternatively,
use the ``rocprof-sys-avail -G ~/.rocprof-sys.cfg --all`` option
for a verbose configuration file with descriptions, categories, and additional information.

Modify ``${HOME}/.rocprof-sys.cfg`` as required. For example, enable `Perfetto <https://perfetto.dev/>`_,
`Timemory <https://github.com/ROCm/timemory>`_, sampling, and process-level sampling by default
and tweak the default sampling values.

.. code-block:: shell

   # ...
   ROCPROFSYS_TRACE                = true
   ROCPROFSYS_PROFILE              = true
   ROCPROFSYS_USE_SAMPLING         = true
   ROCPROFSYS_USE_PROCESS_SAMPLING = true
   # ...
   ROCPROFSYS_SAMPLING_FREQ        = 50
   ROCPROFSYS_SAMPLING_CPUS        = all
   ROCPROFSYS_SAMPLING_GPUS        = $env:HIP_VISIBLE_DEVICES

Use the configuration file
-----------------------------------

Tell rocprof-systems to use the modified settings by creating the environment variable ``ROCPROFSYS_CONFIG_FILE``

.. code-block:: shell

   export ROCPROFSYS_CONFIG_FILE=~/.rocprof-sys.cfg

See :ref:`creating a configuration file<creating-a-configuration-file>` for more details.

Exploring runtime settings
-----------------------------------

Use the following command to view the list of the available runtime settings, their current values, and descriptions
for each setting:

.. code-block:: shell

   rocprof-sys-avail --description

.. note::

   Use ``--brief`` to suppress printing the current value and/or ``-c 0`` to suppress truncation of the descriptions.

Any Boolean setting (``rocprof-sys-avail --settings --value --brief --filter bool``)
accepts a case insensitive match for nearly all common Boolean logic expressions:
``ON``, ``OFF``, ``YES``, ``NO``, ``TRUE``, ``FALSE``, ``0``, ``1``, etc.

Exploring components
-----------------------------------

ROCm Systems Profiler uses `Timemory <https://github.com/ROCm/timemory>`_ extensively to provide
various capabilities and manage
data and resources. By default, with ``ROCPROFSYS_PROFILE=ON``, ROCm Systems Profiler only collects wall-clock
timing values. However, by modifying the ``ROCPROFSYS_TIMEMORY_COMPONENTS`` setting,
ROCm Systems Profiler can be configured to
collect hardware counters, CPU-clock timers, memory usage, context switches, page faults, network statistics,
and much more. ROCm Systems Profiler can even be used as a dynamic instrumentation vehicle
for other third-party profiling
APIs such as `Caliper <https://github.com/LLNL/Caliper>`_ and `LIKWID <https://github.com/RRZE-HPC/likwid>`_.
To leverage this capability, build ROCm Systems Profiler from source with the CMake
options ``TIMEMORY_USE_CALIPER=ON`` or ``TIMEMORY_USE_LIKWID=ON`` and then add
``caliper_marker``, ``likwid_marker``, or both to ``ROCPROFSYS_TIMEMORY_COMPONENTS``.

To view all possible components and their descriptions:

.. code-block:: shell

   rocprof-sys-avail --components --description

To restrict the output to available components and view the string identifiers for ``ROCPROFSYS_TIMEMORY_COMPONENTS``:

.. code-block:: shell

   rocprof-sys-avail --components --available --string --brief

Exploring hardware counters
-----------------------------------

ROCm Systems Profiler supports hardware counter collection via PAPI and ROCm.
Generally, PAPI is used to collect CPU-based hardware counters and ROCm is used to collect GPU-based hardware
counters. Although it is possible to install PAPI with ROCm support and use it to
collect GPU-based hardware counters, this is not recommended because PAPI
cannot simultaneously collect CPU and GPU hardware counters.

To view all possible hardware counters and their descriptions, use the following command:

.. code-block:: shell

   rocprof-sys-avail --hw-counters --description

Appending the ``-c CPU`` option restricts the list of hardware counters to
those available through PAPI, while ``-c GPU`` limits the list to those available from ROCm.

Enabling hardware counters
-----------------------------------

PAPI Hardware counters are configured with the ``ROCPROFSYS_PAPI_EVENTS`` configuration variable.
ROCm Hardware counters are configured with the ``ROCPROFSYS_ROCM_EVENTS`` configuration variable.
ROCm hardware counters also require the ``ROCPROFSYS_USE_ROCPROFILER`` configuration
variable to be enabled using ``ROCPROFSYS_USE_ROCPROFILER=ON``.

Here is a sample configuration for hardware counters:

.. code-block:: shell

   # using papi identifiers
   ROCPROFSYS_PAPI_EVENTS   = PAPI_TOT_CYC PAPI_TOT_INS

   # using perf identifiers
   ROCPROFSYS_PAPI_EVENTS   = perf::INSTRUCTIONS perf::CACHE-REFERENCES perf::CACHE-MISSES

.. _rocprof-sys_papi_events:

ROCPROFSYS_PAPI_EVENTS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to collect the majority of hardware counters via PAPI, ensure the ``/proc/sys/kernel/perf_event_paranoid``
has a value <= 2. If you have ``sudo`` access, use the following command to modify the value:

.. code-block:: shell

   echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid

However this value is not retained upon reboot.
Use the following command to preserve this setting after a reboot:

.. code-block:: shell

   echo 'kernel.perf_event_paranoid=0' | sudo tee -a /etc/sysctl.conf

PAPI events use a concept similar to a namespace. All specified hardware
counters must be from the same namespace.
For hardware counters starting with the ``PAPI_`` prefix, these are high-level
aggregates of multiple hardware counters.
Otherwise, most events use two or three colons (``::`` or ``:::``) between the
component name and the counter name, for example,
``amd64_rapl::RAPL_ENERGY_PKG`` and ``perf::PERF_COUNT_HW_CPU_CYCLES``.

For example, the following is a valid configuration:

.. code-block:: shell

   ROCPROFSYS_PAPI_EVENTS = perf::INSTRUCTIONS  perf::CACHE-REFERENCES  perf::CACHE-MISSES

However, the following specification of a roughly equivalent set of hardware counters is an incorrect configuration because it mixes
PAPI components from different namespaces:

.. code-block:: shell

   ROCPROFSYS_PAPI_EVENTS = PAPI_TOT_INS        perf::CACHE-REFERENCES  perf::CACHE-MISSES

.. note::

   If ROCm Systems Profiler was configured with the default ``ROCPROFSYS_BUILD_PAPI=ON`` setting,
   standard PAPI command-line tools such as
   ``papi_avail`` and ``papi_event_chooser`` are not able to provide information
   about the PAPI library used by ROCm Systems Profiler
   (because ROCm Systems Profiler statically links to ``libpapi``). However, all of these tools are
   installed with the prefix ``rocprof-sys-`` with
   underscores replaced with hyphens, for example ``papi_avail`` becomes ``rocprof-sys-papi-avail``.

ROCPROFSYS_ROCM_EVENTS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ROCm Systems Profiler reads the ROCm events from the ``${ROCM_PATH}/lib/rocprofiler/metrics.xml``
file. Use the ``ROCP_METRICS`` environment
variable to point ROCm Systems Profiler to a different XML metrics file, for example,
``export ROCP_METRICS=${PWD}/custom_metrics.xml``.
``rocprof-sys-avail -H -c GPU`` shows event names with a suffix of ``:device=N``
where ``N`` is the device number.
For example, if you have two devices, the output is:

.. code-block:: shell

   | Wavefronts:device=0                   | Derived counter: SQ_WAVES             |
   ...
   | Wavefronts:device=1                   | Derived counter: SQ_WAVES             |

To collect the event on all devices, specify the event,
such as ``Wavefronts``, without the ``:device=`` suffix.
To collect the event only on specific devices, use the ``:device=`` suffix.

The following example:

* Records the percentage of time the GPU was busy on all devices
* Counts the number of waves sent to SQs on device 0
* Counts the number of VALU instructions issued on device 1

.. code-block:: shell

   ROCPROFSYS_ROCM_EVENTS = GPUBusy     SQ_WAVES:device=0    SQ_INSTS_VALU:device=1

ROCPROFSYS_USE_RCCLP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the setting ``ROCPROFSYS_USE_RCCLP = ON`` to enable profiling and tracing of
ROCm Communication Collectives Library (RCCL, also pronounced as 'Rickle'). When this setting is enabled,
ROCm Systems Profiler will trace the RCCL API calls and collect performance metrics related to collective operations.

The image below shows an example of a Perfetto trace with RCCL communication data and API tracing enabled:

.. image:: ../data/rccl-comm-recv.png
   :alt: Perfetto tracks with RCCL Communication Data and API tracing

.. note::
   There is a known issue which causes the application to exit with an error. However, the trace data can still be found in the output directory.
   This issue is being tracked internally.

Exploring GPU Metrics
---------------------

ROCm Systems Profiler supports GPU metrics collection, sampling, and API tracing via `ROCprofiler-SDK <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/index.html>`_ and `AMD-SMI <https://rocm.docs.amd.com/projects/amdsmi/en/latest/>`_.
ROCprofiler-SDK supports application tracing to provide a big picture of the GPU application execution and kernel profiling to provide low-level hardware details from the performance counters.
The AMD-SMI library offers a unified tool for managing, monitoring, and retrieving information about the system's drivers and GPUs.

Sampling GPU metrics like utilization, temperature, power consumption, memory usage, etc., can be configured with ``ROCPROFSYS_AMD_SMI_METRICS``.
The ``ROCPROFSYS_USE_AMD_SMI`` setting should be enabled for GPU metric collection.

For example, the following is a valid configuration:

.. code-block:: shell

   ROCPROFSYS_AMD_SMI_METRICS=busy,temp,power,vcn_activity,mem_usage

Supported values for ``ROCPROFSYS_AMD_SMI_METRICS`` are: ``busy``, ``temp``, ``power``, ``vcn_activity``, ``mem_usage``, ``jpeg_activity``.

API tracing is configured with the ``ROCPROFSYS_ROCM_DOMAINS`` setting. The domains are used to filter the events that are captured during profiling.
Supported values for this setting are those supported by ROCprofiler-SDK, which are returned by the API ``get_callback_tracing_names()`` and ``get_buffer_tracing_names()``. See the `ROCprofiler-SDK developer API documentation <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/_doxygen/rocprofiler-sdk/html/>`_ to learn more about ROCprofiler-SDK APIs.
Use the following command to view the available domains:

.. code-block:: shell

   rocprof-sys-avail -bd -r ROCM_DOMAINS

.. note::

Some  settings can enable tracing for multiple domains, such as ``hip_api`` which will enable both ``hip_runtime_api`` and ``hip_compiler_api``.
And ``hsa_api`` which will enable all hsa domains, ``hsa_core_api``, ``hsa_amd_ext_api``, ``hsa_image_exit_api``, ``hsa_finalize_ext_api``.
The setting ``marker_api`` or ``roctx`` can be used to enable the roctx marker API tracing.

For example, the following is a valid configuration:

.. code-block:: shell

   ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,rocdecode_api,rocjpeg_api

rocprof-sys-avail examples
-----------------------------------

The following examples demonstrate how to use ``rocprof-sys-avail`` to perform several common
configuration tasks.

Generating a default configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

   $ rocprof-sys-avail -G ~/.rocprof-sys.cfg
   [rocprof-sys-avail] Outputting text configuration file '/home/user/.rocprof-sys.cfg'...
   $ cat ~/.rocprof-sys.cfg
   # auto-generated by rocprof-sys-avail (version 1.2.0) on 2022-06-27 @ 19:15

   ROCPROFSYS_CONFIG_FILE                              =
   ROCPROFSYS_MODE                                     = trace
   ROCPROFSYS_TRACE                                    = true
   ROCPROFSYS_PROFILE                                  = false
   ROCPROFSYS_USE_SAMPLING                             = false
   ROCPROFSYS_USE_PROCESS_SAMPLING                     = true
   ROCPROFSYS_USE_ROCM                                 = true
   ROCPROFSYS_USE_AMD_SMI                              = true
   ROCPROFSYS_USE_KOKKOSP                              = false
   ROCPROFSYS_USE_CODE_COVERAGE                        = false
   ROCPROFSYS_USE_PID                                  = true
   ROCPROFSYS_OUTPUT_PATH                              = rocprof-sys-%tag%-output
   ROCPROFSYS_OUTPUT_PREFIX                            =
   ROCPROFSYS_CI                                       = false
   ROCPROFSYS_THREAD_POOL_SIZE                         = 8
   ROCPROFSYS_DEBUG                                    = false
   ROCPROFSYS_DL_VERBOSE                               = 0
   ROCPROFSYS_INSTRUMENTATION_INTERVAL                 = 1
   ROCPROFSYS_KOKKOSP_KERNEL_LOGGER                    = false
   ROCPROFSYS_PAPI_EVENTS                              = PAPI_TOT_CYC
   ROCPROFSYS_PERFETTO_BACKEND                         = inprocess
   ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB                  = 1024000
   ROCPROFSYS_PERFETTO_COMBINE_TRACES                  = false
   ROCPROFSYS_PERFETTO_FILE                            = perfetto-trace.proto
   ROCPROFSYS_PERFETTO_FILL_POLICY                     = discard
   ROCPROFSYS_PERFETTO_SHMEM_SIZE_HINT_KB              = 4096
   ROCPROFSYS_SAMPLING_CPUS                            =
   ROCPROFSYS_SAMPLING_DELAY                           = 0.5
   ROCPROFSYS_SAMPLING_FREQ                            = 10
   ROCPROFSYS_SAMPLING_GPUS                            = all
   ROCPROFSYS_TIME_OUTPUT                              = true
   ROCPROFSYS_TIMEMORY_COMPONENTS                      = wall_clock
   ROCPROFSYS_TRACE_THREAD_LOCKS                       = false
   ROCPROFSYS_VERBOSE                                  = 0
   ROCPROFSYS_COLLAPSE_PROCESSES                       = false
   ROCPROFSYS_COLLAPSE_THREADS                         = false
   ROCPROFSYS_COUT_OUTPUT                              = false
   ROCPROFSYS_CPU_AFFINITY                             = false
   ROCPROFSYS_DIFF_OUTPUT                              = false
   ROCPROFSYS_ENABLE_SIGNAL_HANDLER                    = true
   ROCPROFSYS_ENABLED                                  = true
   ROCPROFSYS_FILE_OUTPUT                              = true
   ROCPROFSYS_FLAT_PROFILE                             = false
   ROCPROFSYS_INPUT_EXTENSIONS                         = json,xml
   ROCPROFSYS_INPUT_PATH                               =
   ROCPROFSYS_INPUT_PREFIX                             =
   ROCPROFSYS_JSON_OUTPUT                              = true
   ROCPROFSYS_MAX_DEPTH                                = 65535
   ROCPROFSYS_MAX_WIDTH                                = 120
   ROCPROFSYS_MEMORY_PRECISION                         = -1
   ROCPROFSYS_MEMORY_SCIENTIFIC                        = false
   ROCPROFSYS_MEMORY_UNITS                             = MB
   ROCPROFSYS_MEMORY_WIDTH                             = -1
   ROCPROFSYS_NETWORK_INTERFACE                        =
   ROCPROFSYS_NODE_COUNT                               = 0
   ROCPROFSYS_PAPI_FAIL_ON_ERROR                       = false
   ROCPROFSYS_PAPI_MULTIPLEXING                        = false
   ROCPROFSYS_PAPI_OVERFLOW                            = 0
   ROCPROFSYS_PAPI_QUIET                               = false
   ROCPROFSYS_PAPI_THREADING                           = true
   ROCPROFSYS_PRECISION                                = -1
   ROCPROFSYS_SCIENTIFIC                               = false
   ROCPROFSYS_STRICT_CONFIG                            = true
   ROCPROFSYS_SUPPRESS_CONFIG                          = true
   ROCPROFSYS_SUPPRESS_PARSING                         = true
   ROCPROFSYS_TEXT_OUTPUT                              = true
   ROCPROFSYS_TIME_FORMAT                              = %F_%H.%M
   ROCPROFSYS_TIMELINE_PROFILE                         = false
   ROCPROFSYS_TIMING_PRECISION                         = 6
   ROCPROFSYS_TIMING_SCIENTIFIC                        = false
   ROCPROFSYS_TIMING_UNITS                             = sec
   ROCPROFSYS_TIMING_WIDTH                             = -1
   ROCPROFSYS_TREE_OUTPUT                              = true
   ROCPROFSYS_WIDTH                                    = -1

When creating a new configuration file, the following recommendations apply:

* Use the ``--all`` option to view all descriptions, choices, and other information in the configuration file.
* To create a new configuration without inheriting from an existing ``${HOME}/.rocprof-sys.cfg`` file,
  set ``ROCPROFSYS_SUPPRESS_CONFIG=ON`` in the environment beforehand.
* To create a new configuration that makes minor changes to an existing configuration,
  set ``ROCPROFSYS_CONFIG_FILE=/path/to/existing/file`` and define the changes as environment
  variables before generating it.

Viewing the setting descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

   $ rocprof-sys-avail -S -bd
   |-----------------------------------------|-----------------------------------------|
   |          ENVIRONMENT VARIABLE           |               DESCRIPTION               |
   |-----------------------------------------|-----------------------------------------|
   | ROCPROFSYS_CI                            | Enable some runtime validation check... |
   | ROCPROFSYS_ADD_SECONDARY                 | Enable/disable components adding sec... |
   | ROCPROFSYS_COLLAPSE_PROCESSES            | Enable/disable combining process-spe... |
   | ROCPROFSYS_COLLAPSE_THREADS              | Enable/disable combining thread-spec... |
   | ROCPROFSYS_CONFIG_FILE                   | Configuration file for rocprof-sys      |
   | ROCPROFSYS_COUT_OUTPUT                   | Write output to stdout                  |
   | ROCPROFSYS_CPU_AFFINITY                  | Enable pinning threads to CPUs (Linu... |
   | ROCPROFSYS_THREAD_POOL_SIZE              | Number of threads to use when genera... |
   | ROCPROFSYS_DEBUG                         | Enable debug output                     |
   | ROCPROFSYS_DIFF_OUTPUT                   | Generate a difference output vs. a p... |
   | ROCPROFSYS_DL_VERBOSE                    | Verbosity within the rocprof-sys-dl ... |
   | ROCPROFSYS_ENABLED                       | Activation state of timemory            |
   | ROCPROFSYS_ENABLE_SIGNAL_HANDLER         | Enable signals in timemory_init         |
   | ROCPROFSYS_FILE_OUTPUT                   | Write output to files                   |
   | ROCPROFSYS_FLAT_PROFILE                  | Set the label hierarchy mode to defa... |
   | ROCPROFSYS_INPUT_EXTENSIONS              | File extensions used when searching ... |
   | ROCPROFSYS_INPUT_PATH                    | Explicitly specify the input folder ... |
   | ROCPROFSYS_INPUT_PREFIX                  | Explicitly specify the prefix for in... |
   | ROCPROFSYS_INSTRUMENTATION_INTERVAL      | Instrumentation only takes measureme... |
   | ROCPROFSYS_JSON_OUTPUT                   | Write json output files                 |
   | ROCPROFSYS_KOKKOSP_KERNEL_LOGGER         | Enables kernel logging                  |
   | ROCPROFSYS_MAX_DEPTH                     | Set the maximum depth of label hiera... |
   | ROCPROFSYS_MAX_THREAD_BOOKMARKS          | Maximum number of times a worker thr... |
   | ROCPROFSYS_MAX_WIDTH                     | Set the maximum width for component ... |
   | ROCPROFSYS_MEMORY_PRECISION              | Set the precision for components wit... |
   | ROCPROFSYS_MEMORY_SCIENTIFIC             | Set the numerical reporting format f... |
   | ROCPROFSYS_MEMORY_UNITS                  | Set the units for components with u...  |
   | ROCPROFSYS_MEMORY_WIDTH                  | Set the output width for components ... |
   | ROCPROFSYS_NETWORK_INTERFACE             | Default network interface               |
   | ROCPROFSYS_NODE_COUNT                    | Total number of nodes used in applic... |
   | ROCPROFSYS_OUTPUT_FILE                   | Perfetto filename                       |
   | ROCPROFSYS_OUTPUT_PATH                   | Explicitly specify the output folder... |
   | ROCPROFSYS_OUTPUT_PREFIX                 | Explicitly specify a prefix for all ... |
   | ROCPROFSYS_PAPI_EVENTS                   | PAPI presets and events to collect (... |
   | ROCPROFSYS_PAPI_FAIL_ON_ERROR            | Configure PAPI errors to trigger a r... |
   | ROCPROFSYS_PAPI_MULTIPLEXING             | Enable multiplexing when using PAPI     |
   | ROCPROFSYS_PAPI_OVERFLOW                 | Value at which PAPI hw counters trig... |
   | ROCPROFSYS_PAPI_QUIET                    | Configure suppression of reporting P... |
   | ROCPROFSYS_PAPI_THREADING                | Enable multithreading support when u... |
   | ROCPROFSYS_PERFETTO_BACKEND              | Specify the perfetto backend to acti... |
   | ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB       | Size of perfetto buffer (in KB)         |
   | ROCPROFSYS_PERFETTO_COMBINE_TRACES       | Combine Perfetto traces. If not expl... |
   | ROCPROFSYS_PERFETTO_FILL_POLICY          | Behavior when perfetto buffer is ful... |
   | ROCPROFSYS_PERFETTO_SHMEM_SIZE_HINT_KB   | Hint for shared-memory buffer size i... |
   | ROCPROFSYS_PRECISION                     | Set the global output precision for ... |
   | ROCPROFSYS_SAMPLING_CPUS                 | CPUs to collect frequency informatio... |
   | ROCPROFSYS_SAMPLING_DELAY                | Number of seconds to wait before the... |
   | ROCPROFSYS_SAMPLING_FREQ                 | Number of software interrupts per se... |
   | ROCPROFSYS_SAMPLING_GPUS                 | Devices to query when ROCPROFSYS_USE... |
   | ROCPROFSYS_SCIENTIFIC                    | Set the global numerical reporting t... |
   | ROCPROFSYS_STRICT_CONFIG                 | Throw errors for unknown setting nam... |
   | ROCPROFSYS_SUPPRESS_CONFIG               | Disable processing of setting config... |
   | ROCPROFSYS_SUPPRESS_PARSING              | Disable parsing environment             |
   | ROCPROFSYS_TEXT_OUTPUT                   | Write text output files                 |
   | ROCPROFSYS_TIMELINE_PROFILE              | Set the label hierarchy mode to defa... |
   | ROCPROFSYS_TIMEMORY_COMPONENTS           | List of components to collect via ti... |
   | ROCPROFSYS_TIME_FORMAT                   | Customize the folder generation when... |
   | ROCPROFSYS_TIME_OUTPUT                   | Output data to subfolder w/ a timest... |
   | ROCPROFSYS_TIMING_PRECISION              | Set the precision for components wit... |
   | ROCPROFSYS_TIMING_SCIENTIFIC             | Set the numerical reporting format f... |
   | ROCPROFSYS_TIMING_UNITS                  | Set the units for components with u...  |
   | ROCPROFSYS_TIMING_WIDTH                  | Set the output width for components ... |
   | ROCPROFSYS_TRACE_THREAD_LOCKS            | Enable tracking calls to pthread_mut... |
   | ROCPROFSYS_TREE_OUTPUT                   | Write hierarchical json output files    |
   | ROCPROFSYS_USE_CODE_COVERAGE             | Enable support for code coverage        |
   | ROCPROFSYS_USE_KOKKOSP                   | Enable support for Kokkos Tools         |
   | ROCPROFSYS_USE_OMPT                      | Enable support for OpenMP-Tools         |
   | ROCPROFSYS_TRACE                         | Enable perfetto backend                 |
   | ROCPROFSYS_USE_PID                       | Enable tagging filenames with proces... |
   | ROCPROFSYS_USE_AMD_SMI                   | Enable sampling GPU power, temp, uti... |
   | ROCPROFSYS_USE_ROCM                      | Enable ROCM tracing                     |
   | ROCPROFSYS_USE_SAMPLING                  | Enable statistical sampling of call-... |
   | ROCPROFSYS_USE_PROCESS_SAMPLING          | Enable a background thread which sam... |
   | ROCPROFSYS_PROFILE                       | Enable timemory backend                 |
   | ROCPROFSYS_VERBOSE                       | Verbosity level                         |
   | ROCPROFSYS_WIDTH                         | Set the global output width for comp... |
   |------------------------------------------|-----------------------------------------|

Viewing components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

   $ rocprof-sys-avail -C -bd
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
   | rocprof-sys                       | Invokes instrumentation functions rocprof... |
   | sampling_wall_clock               | Wall-clock timing. Derived from statistic... |
   | sampling_cpu_clock                | CPU-clock timing. Derived from statistica... |
   | sampling_percent                  | Fraction of wall-clock time spent in func... |
   | sampling_gpu_power                | GPU Power Usage via AMD-SMI. Derived from... |
   | sampling_gpu_temp                 | GPU Temperature via AMD-SMI. Derived from... |
   | sampling_gpu_busy                 | GPU Utilization (% busy) via AMD-SMI. Der... |
   | sampling_gpu_vcn                  | GPU VCN Utilization (% activity) via AMD ... |
   | sampling_gpu_jpeg                 | GPU JPEG Utilization (% activity) via AMD... |
   | sampling_gpu_memory_usage         | GPU Memory Usage via AMD-SMI. Derived fro... |
   |-----------------------------------|----------------------------------------------|

Viewing hardware counters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

   $ rocprof-sys-avail -H -bd
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

.. _creating-a-configuration-file:

Creating a configuration file
========================================

ROCm Systems Profiler supports three configuration file formats: JSON, XML, and plain text.
Use ``rocprof-sys-avail -G <filename> -F txt json xml`` to generate default
configuration files in each format. Optionally
include the ``--all`` flag to include full descriptions and other information.
Configuration files are specified by the ``ROCPROFSYS_CONFIG_FILE`` environment variable
which by default looks for ``${HOME}/.rocprof-sys.cfg`` and ``${HOME}/.rocprof-sys.json``.
Multiple configuration files can be concatenated using the ``:`` symbol, for example:

.. code-block:: shell

   export ROCPROFSYS_CONFIG_FILE=~/.config/rocprof-sys.cfg:~/.config/rocprof-sys.json

If a configuration variable is specified in both a configuration file and in the environment,
the environment variable takes precedence.

Sample text configuration file
-----------------------------------

Text files support very basic variables and are case insensitive.
Variables are created when an lvalue starts with a ``$`` and are
de-referenced when they appear as rvalues.

Entries in the text configuration file which do not match a known setting
in ``rocprof-sys-avail`` but are prefixed with ``ROCPROFSYS_`` are interpreted as
environment variables. They are exported via ``setenv``
but do not override an existing value for the environment variable.

.. code-block:: shell

   # lvals starting with $ are variables
   $ENABLE                         = ON
   $SAMPLE                         = OFF

   # use fields
   ROCPROFSYS_TRACE                 = $ENABLE
   ROCPROFSYS_PROFILE               = $ENABLE
   ROCPROFSYS_USE_SAMPLING          = $SAMPLE
   ROCPROFSYS_USE_PROCESS_SAMPLING  = $SAMPLE

   # debug
   ROCPROFSYS_DEBUG                 = OFF
   ROCPROFSYS_VERBOSE               = 1

   # output fields
   ROCPROFSYS_OUTPUT_PATH           = rocprof-sys-output
   ROCPROFSYS_OUTPUT_PREFIX         = %tag%/
   ROCPROFSYS_TIME_OUTPUT           = OFF
   ROCPROFSYS_USE_PID               = OFF

   # timemory fields
   ROCPROFSYS_PAPI_EVENTS           = PAPI_TOT_INS PAPI_FP_INS
   ROCPROFSYS_TIMEMORY_COMPONENTS   = wall_clock peak_rss trip_count
   ROCPROFSYS_MEMORY_UNITS          = MB
   ROCPROFSYS_TIMING_UNITS          = sec

   # sampling fields
   ROCPROFSYS_SAMPLING_FREQ         = 50
   ROCPROFSYS_SAMPLING_DELAY        = 0.1
   ROCPROFSYS_SAMPLING_CPUS         = 0-3
   ROCPROFSYS_SAMPLING_GPUS         = $env:HIP_VISIBLE_DEVICES

   # misc env variables (see metadata JSON file after run)
   $env:ROCPROFSYS_SAMPLING_KEEP_DYNINST_SUFFIX  = OFF

Sample JSON configuration file
-----------------------------------

The full JSON specification for a configuration value contains a lot of information:

.. code-block:: json

   {
      "rocprof-sys": {
         "settings": {
               "ROCPROFSYS_ADD_SECONDARY": {
                  "count": -1,
                  "name": "add_secondary",
                  "data_type": "bool",
                  "initial": true,
                  "value": true,
                  "max_count": 1,
                  "cmdline": [
                     "--rocprof-sys-add-secondary"
                  ],
                  "environ": "ROCPROFSYS_ADD_SECONDARY",
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

However when writing an JSON configuration file, the following example is minimally acceptable
for ``ROCPROFSYS_ADD_SECONDARY``:

.. code-block:: json

   {
      "rocprof-sys": {
         "settings": {
               "ROCPROFSYS_ADD_SECONDARY": {
                  "value": true
               }
         }
      }
   }

Sample XML configuration file
-----------------------------------

The full XML specification for a configuration value contains the same information as the JSON specification:

.. code-block:: xml

   <?xml version="1.0" encoding="utf-8"?>
   <timemory_xml>
      <rocprofiler-systems>
         <settings>
               <cereal_class_version>2</cereal_class_version>
               <!-- Full setting specification -->
               <ROCPROFSYS_ADD_SECONDARY>
                  <cereal_class_version>1</cereal_class_version>
                  <name>add_secondary</name>
                  <environ>ROCPROFSYS_ADD_SECONDARY</environ>
                  <description>...</description>
                  <count>-1</count>
                  <max_count>1</max_count>
                  <cmdline>
                     <value0>--rocprof-sys-add-secondary</value0>
                  </cmdline>
                  <categories>
                     <value0>component</value0>
                     <value1>data</value1>
                     <value2>native</value2>
                  </categories>
                  <data_type>bool</data_type>
                  <initial>true</initial>
                  <value>true</value>
               </ROCPROFSYS_ADD_SECONDARY>
               <!-- etc. -->
         </settings>
      </rocprofiler-systems>
   </timemory_xml>

However, when writing an XML configuration file, it is minimally acceptable
to set ``ROCPROFSYS_ADD_SECONDARY=false``:

.. code-block:: xml

   <?xml version="1.0" encoding="utf-8"?>
   <timemory_xml>
      <rocprofiler-systems>
         <settings>
               <ROCPROFSYS_ADD_SECONDARY>
                  <value>false</value>
               </ROCPROFSYS_ADD_SECONDARY>
         </settings>
      </rocprofiler-systems>
   </timemory_xml>
