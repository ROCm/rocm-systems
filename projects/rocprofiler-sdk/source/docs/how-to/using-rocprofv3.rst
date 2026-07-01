.. meta::
  :description: ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software
  :keywords: ROCprofiler-SDK tool usage, rocprofv3 user manual, rocprofv3 usage, rocprofv3 user guide, using rocprofv3, ROCprofiler-SDK tool user guide, ROCprofiler-SDK tool user manual, using ROCprofiler-SDK tool, ROCprofiler-SDK command-line tool, ROCprofiler-SDK CLI, ROCprofiler-SDK command line tool

.. _using-rocprofv3:

==================================================
Application tracing and profiling using rocprofv3
==================================================

``rocprofv3`` is a CLI tool that helps you optimize applications and analyze the low-level kernel details without requiring any modification in the source code.
It's backward compatible with its predecessor, `rocprof <https://rocm.docs.amd.com/projects/rocprofiler/en/latest/index.html>`_, and provides enhanced features for application profiling with better accuracy.

The following sections demonstrate the use of ``rocprofv3`` for application tracing and kernel counter collection using various command-line options.

``rocprofv3`` is installed with ROCm under ``/opt/rocm/bin``. To use the tool from anywhere in the system, export the ``PATH`` variable:

.. code-block:: bash

   export PATH=$PATH:/opt/rocm/bin

Before tracing or profiling your HIP application using ``rocprofv3``, build it using:

.. code-block:: bash

   cmake -B <build-directory> <source-directory> -DCMAKE_PREFIX_PATH=/opt/rocm
   cmake --build <build-directory> --target all --parallel <N>

.. _gpu-performance-level:

Setting GPU performance level for PMC profiling
---------------------------------------------

On RDNA3 (Navi3x) and RDNA4 (Navi4x) GPUs, the ``AUTO`` performance mode disables PMC profiling in some GPU hardware blocks:
the perfmon clock is gated off, which prevents performance counters from functioning. Setting the performance level to
``STABLE_STD`` turns the perfmon clock back on and enables PMC profiling on all GPU hardware blocks.

This is a hardware feature enablement requirement. Without it, PMC profiling on these GPUs produces no meaningful counter
data in some GPU hardware blocks.

There are two ways to configure the GPU performance level:

**Option 1: Using the** ``power_dpm_force_performance_level`` **sysfs entry**

Set the performance level to ``profile_standard`` via the sysfs interface. Replace ``<N>`` with
the card index (for example, ``0`` for ``card0``):

.. code-block:: bash

   sudo chmod 777 /sys/class/drm/card<N>/device/power_dpm_force_performance_level
   sudo sh -c 'echo profile_standard > /sys/class/drm/card<N>/device/power_dpm_force_performance_level'

To verify the setting:

.. code-block:: bash

   cat /sys/class/drm/card<N>/device/power_dpm_force_performance_level

To restore the default behavior after PMC profiling:

.. code-block:: bash

   sudo sh -c 'echo auto > /sys/class/drm/card<N>/device/power_dpm_force_performance_level'

**Option 2: Using** ``amd-smi``

Alternatively, use the ``amd-smi`` tool installed with ROCm to query and set the performance level.

One advantage of using ``amd-smi`` is that it can be used to query and set the performance level on multiple
GPUs in a single command. For example, to set the performance level to ``STABLE_STD`` on all GPUs in the
system, use:

.. code-block:: shell

   $ sudo /opt/rocm/bin/amd-smi set --perf-level STABLE_STD
   GPU: 0
       PERFLEVEL: Successfully set performance level STABLE_STD
   GPU: 1
       PERFLEVEL: Successfully set performance level STABLE_STD

The following examples show how to query and set the performance level for a specific GPU. Replace ``<N>``
with the card index:

To check the current performance level:

.. code-block:: shell

   $ sudo /opt/rocm/bin/amd-smi metric --gpu <N> --perf-level
   GPU: <N>
       PERF_LEVEL: AMDSMI_DEV_PERF_LEVEL_AUTO

To set the performance level to ``STABLE_STD`` (the ``amd-smi`` name for ``profile_standard``):

.. code-block:: shell

   $ sudo /opt/rocm/bin/amd-smi set --gpu <N> --perf-level STABLE_STD
   GPU: <N>
       PERFLEVEL: Successfully set performance level STABLE_STD

To verify the change:

.. code-block:: shell

   $ sudo /opt/rocm/bin/amd-smi metric --gpu <N> --perf-level
   GPU: <N>
       PERF_LEVEL: AMDSMI_DEV_PERF_LEVEL_STABLE_STD

To restore the default performance level after PMC profiling:

.. code-block:: shell

   $ sudo /opt/rocm/bin/amd-smi set --gpu <N> --perf-level AUTO
   GPU: <N>
       PERFLEVEL: Successfully set performance level AUTO

   $ sudo /opt/rocm/bin/amd-smi metric --gpu <N> --perf-level
   GPU: <N>
       PERF_LEVEL: AMDSMI_DEV_PERF_LEVEL_AUTO

.. _application-tracing:

Application tracing
---------------------

Application tracing provides the big picture of a program’s execution by collecting data on the execution times of API calls and GPU commands, such as kernel execution, async memory copy, and barrier packets. This information can be used as the first step in the profiling process to answer important questions, such as how much percentage of time was spent on memory copy and which kernel took the longest time to execute.

To use ``rocprofv3`` for application tracing, run:

.. code-block:: bash

    rocprofv3 <tracing_option> -- <application_path>


.. note::

  All the tracing examples below use the ``--output-format csv`` option to generate output in CSV format.
  However, the default output format is ``rocpd`` (SQLite3 database). You can simply omit the ``--output-format`` option to generate output in the default format.
  ``rocpd`` format can be converted to other formats such as CSV, OTF2, and PFTrace using the ``rocpd`` module.
  To understand how to convert ``rocpd`` output to other formats, see :ref:`using-rocpd-output-format`.

HIP trace
+++++++++++

HIP trace comprises execution traces for the entire application at the HIP level. This includes HIP API functions and their asynchronous activities at the runtime level. In general, HIP APIs directly interact with the user program. It is easier to analyze HIP traces as you can directly map them to the program.
Unlike previous iterations of ``rocprof``, this does not enable kernel tracing, memory copy tracing, and so on. If you want to enable kernel tracing, memory copy tracing, they need to be provided explicitly.

To trace HIP runtime APIs, use:

.. code-block:: bash

    rocprofv3 --hip-trace --output-format csv -- <application_path>

The preceding command generates a ``hip_api_trace.csv`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 238_hip_api_trace.csv

Here are the contents of ``hip_api_trace.csv`` file:

.. csv-table:: HIP api trace
   :file: /data/hip_trace.csv
   :widths: 10,10,10,10,10,20,20
   :header-rows: 1


``rocprofv3`` provides options to collect traces at more granular level. For HIP, you can collect traces for HIP compile-time APIs and runtime APIs separately.

HIP compile-time API traces
****************************

To collect HIP compile-time API traces, use:

.. code-block:: shell

    rocprofv3 --hip-compiler-trace --output-format csv -- <application_path>

The preceding command generates a ``hip_api_trace.csv`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 208_hip_api_trace.csv

Here are the contents of ``hip_api_trace.csv`` file:

.. csv-table:: HIP compile-time api trace
   :file: /data/hip_compile_trace.csv
   :widths: 10,10,10,10,10,20,20
   :header-rows: 1

HIP runtime API traces
***********************

To collect HIP runtime time API traces, use:

.. code-block:: shell

    rocprofv3 --hip-runtime-trace --output-format csv -- <application_path>

The preceding command generates a ``hip_api_trace.csv`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 208_hip_api_trace.csv

Here are the contents of ``hip_api_trace.csv`` file:

.. csv-table:: HIP runtime api trace
   :file: /data/hip_runtime_trace.csv
   :widths: 10,10,10,10,10,20,20
   :header-rows: 1

For the description of the fields in the output file, see :ref:`output-file-fields`.

HSA trace
+++++++++++++

The HIP runtime library is implemented with the low-level HSA runtime. HSA API tracing is more suited for advanced users who want to understand the application behavior at the lower level. In general, tracing at the HIP level is recommended for most users. You should use HSA trace only if you are familiar with HSA runtime.

HSA trace contains the start and end time of HSA runtime API calls and their asynchronous activities.

.. code-block:: bash

    rocprofv3 --hsa-trace --output-format csv -- <application_path>

The preceding command generates a ``hsa_api_trace.csv`` file prefixed with process ID. Note that the contents of this file have been truncated for demonstration purposes.

.. code-block:: shell

    $ cat 197_hsa_api_trace.csv

Here are the contents of ``hsa_api_trace.csv`` file:

.. csv-table:: HSA api trace
   :file: /data/hsa_api_trace.csv
   :widths: 10,10,10,10,10,20,20
   :header-rows: 1


``rocprofv3`` provides options to collect HSA traces at more granular level. HSA traces can be collected separately for four API domains: ``HSA_AMD_EXT_API``, ``HSA_CORE_API``, ``HSA_IMAGE_EXT_API`` and ``HSA_FINALIZE_EXT_API``.

To collect HSA core API traces, use:

.. code-block:: bash

    rocprofv3 --hsa-core-trace --output-format csv -- <application_path>

The preceding command generates a ``hsa_api_trace.csv`` file prefixed with process ID. Note that the contents of this file have been truncated for demonstration purposes.

.. code-block:: shell

    $ cat 197_hsa_api_trace.csv

Here are the contents of ``hsa_api_trace.csv`` file:

.. csv-table:: HSA core api trace
   :file: /data/hsa_core_api_trace.csv
   :widths: 10,10,10,10,10,20,20
   :header-rows: 1

For the description of the fields in the output file, see :ref:`output-file-fields`.

Marker trace
++++++++++++++

.. note::

  To use ``rocprofv3`` for marker tracing, including and linking to old ``ROCTx`` works but it's recommended to switch to the new ``ROCTx`` to utilize new APIs.
  To use the new ``ROCTx``, include header ``"rocprofiler-sdk-roctx/roctx.h"`` and link your application with ``librocprofiler-sdk-roctx.so``.
  To see the complete list of ``ROCTx`` APIs, see public header file ``"rocprofiler-sdk-roctx/roctx.h"``.

  To see usage of ``ROCTx`` or marker library, see :ref:`using-rocprofiler-sdk-roctx`.

Kokkos trace
++++++++++++++

`Kokkos <https://github.com/kokkos/kokkos>`_ is a C++ library for writing performance portable applications. Kokkos is widely used in scientific applications to write performance-portable code for CPUs, GPUs, and other accelerators.
``rocprofv3`` loads an inbuilt `Kokkos Tools library <https://github.com/kokkos/kokkos-tools>`_, which emits roctx ranges with the labels passed using Kokkos APIs. For example, ``Kokkos::parallel_for(“MyParallelForLabel”, …)`` calls ``roctxRangePush`` internally and enables the kernel renaming option to replace the highly templated kernel names with the Kokkos labels.
To enable the inbuilt marker support, use the ``kokkos-trace`` option. Internally, this option automatically enables ``marker-trace`` and ``kernel-rename``:

.. code-block:: bash

    rocprofv3 --kokkos-trace --output-format csv -- <application_path>

The preceding command generates a ``marker-trace`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 210_marker_api_trace.csv
   "Domain","Function","Process_Id","Thread_Id","Correlation_Id","Start_Timestamp","End_Timestamp"
   "MARKER_CORE_API","Kokkos::Initialization Complete",4069256,4069256,1,56728499773965,56728499773965
   "MARKER_CORE_API","Kokkos::Impl::CombinedFunctorReducer<CountFunctor, Kokkos::Impl::FunctorAnalysis<Kokkos::Impl::FunctorPatternInterface::REDUCE, Kokkos::RangePolicy<Kokkos::Serial>, CountFunctor, long int>::Reducer, void>",4069256,4069256,2,56728501756088,56728501764241
   "MARKER_CORE_API","Kokkos::parallel_reduce: fence due to result being value, not view",4069256,4069256,4,56728501767957,56728501769600
   "MARKER_CORE_API","Kokkos::Finalization Complete",4069256,4069256,6,56728502054554,56728502054554

Kernel trace
++++++++++++++

To trace kernel dispatch traces, use:

.. code-block:: shell

    rocprofv3 --kernel-trace --output-format csv -- <application_path>

The preceding command generates a ``kernel_trace.csv`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 199_kernel_trace.csv

Here are the contents of ``kernel_trace.csv`` file:

.. csv-table:: Kernel trace
   :file: /data/kernel_trace.csv
   :widths: 10,10,10,10,10,10,10,10,10,20,20,10,10,10,10,10,10,10,10,10,10,10
   :header-rows: 1

For the description of the fields in the output file, see :ref:`output-file-fields`.

Memory copy trace
+++++++++++++++++++

Memory copy traces track ``hipMemcpy`` and ``hipMemcpyAsync`` functions, which use the ``hsa_amd_memory_async_copy_on_engine`` HSA functions internally. To trace memory moves across the application, use:

.. code-block:: shell

    rocprofv3 –-memory-copy-trace --output-format csv -- <application_path>

The preceding command generates a ``memory_copy_trace.csv`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 197_memory_copy_trace.csv

Here are the contents of ``memory_copy_trace.csv`` file:

.. csv-table:: Memory copy trace
   :file: /data/memory_copy_trace.csv
   :widths: 10,10,10,10,10,10,20,20
   :header-rows: 1

For the description of the fields in the output file, see :ref:`output-file-fields`.

Memory allocation trace
+++++++++++++++++++++++++

Memory allocation traces track the HSA functions ``hsa_memory_allocate``,
``hsa_amd_memory_pool_allocate``, and ``hsa_amd_vmem_handle_create```. The function
``hipMalloc`` calls these underlying HSA functions allowing memory allocations to be
tracked.

In addition to the HSA memory allocation functions listed above, the corresponding HSA
free functions ``hsa_memory_free``, ``hsa_amd_memory_pool_free``, and ``hsa_amd_vmem_handle_release``
are also tracked. Unlike the allocation functions, however, only the address of the freed memory
is recorded. As such, the agent id and size of the freed memory are recorded as 0 in the CSV and
JSON outputs. It should be noted that it is possible for some free functions to records a null
pointer address of 0x0. This situation can occur when some HIP functions such as hipStreamDestroy
call underlying HSA free functions with null pointers, even if the user never explicitly calls
free memory functions with null pointer addresses.

To trace memory allocations during the application run, use:

.. code-block:: shell

    rocprofv3 –-memory-allocation-trace --output-format csv -- <application_path>

The preceding command generates a ``memory_allocation_trace.csv`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 6489_memory_allocation_trace.csv

Here are the contents of ``memory_allocation_trace.csv`` file:

.. csv-table:: Memory allocation trace
   :file: /data/memory_allocation_trace.csv
   :widths: 10,10,10,10,10,10,20,20
   :header-rows: 1

For the description of the fields in the output file, see :ref:`output-file-fields`.

Runtime trace
+++++++++++++++

This is a shorthand option that targets the most relevant tracing options for a standard user by
excluding traces for HSA runtime API and HIP compiler API.

The HSA runtime API is excluded because it is a lower-level API upon which HIP and OpenMP target are built and
thus, tends to be an implementation detail irrelevant to most users. Similarly, the HIP compiler API is also excluded for being an implementation detail as these functions are automatically inserted during HIP compilation.

``--runtime-trace`` traces the HIP runtime API, marker API, kernel dispatches, and
memory operations (copies, allocations, and scratch).

.. code-block:: shell

    rocprofv3 –-runtime-trace --output-format csv -- <application_path>

Running the preceding command generates ``hip_api_trace.csv``, ``kernel_trace.csv``, ``memory_copy_trace.csv``, ``scratch_memory_trace.csv``, ``memory_allocation_trace.csv``, and ``marker_api_trace.csv`` (if ``ROCTx`` APIs are specified in the application) files prefixed with the process ID.

System trace
++++++++++++++

This is an all-inclusive option to collect HIP, HSA, kernel, memory copy, memory allocation, and marker trace (if ``ROCTx`` APIs are specified in the application).

.. code-block:: shell

    rocprofv3 –-sys-trace --output-format csv -- <application_path>

Running the preceding command generates ``hip_api_trace.csv``, ``hsa_api_trace.csv``, ``kernel_trace.csv``, ``memory_copy_trace.csv``, ``scratch_memory_trace.csv``, ``memory_allocation_trace.csv``, and ``marker_api_trace.csv`` if ``ROCTx`` APIs are specified in the application.

Scratch memory trace
++++++++++++++++++++++

This option collects scratch memory operation traces. Scratch is an address space on AMD GPUs roughly equivalent to the local memory in NVIDIA CUDA. The local memory in CUDA is a thread-local global memory with interleaved addressing, which is used for register spills or stack space. This option helps to trace when the ``rocr`` runtime allocates, frees, and tries to reclaim scratch memory.

To trace scratch memory allocations during the application run, use:

.. code-block:: shell

    rocprofv3 –-scratch-memory-trace --output-format csv -- <application_path>

The preceding command generates a ``scratch_memory_trace.csv`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 100_scratch_memory_trace.csv

Here are the contents of ``scratch_memory_trace.csv`` file:

.. csv-table:: Scratch memory trace
   :file: /data/scratch_memory_trace.csv
   :widths: 10,10,10,10,10,10,20,20,20
   :header-rows: 1

For the description of the fields in the output file, see :ref:`output-file-fields`.

RCCL trace
++++++++++++

This section demonstrates how to trace `RCCL` (Rickle) collective communication routines using rocprofv3. `RCCL <https://github.com/ROCm/rccl>`_ (pronounced "Rickle") is a stand-alone library that provides standard collective communication operations for GPUs.
The trace output is captured in a rocpd database file and can be converted to pftrace format for visualization in the Perfetto UI. This approach is useful for analyzing GPU communication performance and identifying bottlenecks in collective operations.

.. code-block:: shell

   rocprofv3 --rccl-trace --sys-trace -- <application_path>

The preceding command generates a rocpd database file prefixed with the process ID, which can be converted into PFTrace for visualization in the Perfetto UI.

.. code-block:: shell

   $ /opt/rocm/bin/rocpd2pftrace -i 163852_results.db

The following image visualizes the ``RCCL`` trace for the referenced `allreduce_rccl sample application <https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-systems/examples/rccl/rccl-tests/src/all_reduce.cpp>`_ using the Perfetto UI.
The host thread track and select compute streams are pinned in the visualization to enhance readability.
This enables clear observation of the ``RCCL`` compute kernels launched during ``ncclAllReduce`` operations on the host thread.

.. image:: /data/perfetto_rccl.png
   :width: 100%
   :align: center

rocDecode trace
++++++++++++++++

`rocDecode <https://github.com/ROCm/rocDecode>`_ is a high-performance video decode SDK for AMD GPUs. This option traces the rocDecode API.

.. code-block:: shell

    rocprofv3 --rocdecode-trace --output-format csv -- <application_path>

The above command generates a ``rocdecode_api_trace`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 41688_rocdecode_api_trace.csv

Here are the contents of ``rocdecode_api_trace.csv`` file:

.. csv-table:: rocDecode trace
   :file: /data/rocdecode_api_trace.csv
   :widths: 10,10,10,10,10,20,20
   :header-rows: 1

Perfetto will also show rocDecode API arguments. Pointers will not be dereferenced and only the address will be displayed.

rocJPEG trace
+++++++++++++++

`rocJPEG <https://github.com/ROCm/rocJPEG>`_ is a high-performance jpeg decode SDK for decoding jpeg images. This option traces the rocJPEG API.

.. code-block:: shell

    rocprofv3 --rocjpeg-trace --output-format csv -- <application_path>

The above command generates a ``rocjpeg_api_trace`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 41688_rocjpeg_api_trace.csv

Here are the contents of ``rocjpeg_api_trace.csv`` file:

.. csv-table:: rocJPEG trace
   :file: /data/rocjpeg_api_trace.csv
   :widths: 10,10,10,10,10,20,20
   :header-rows: 1

Dynamic process attachment
+++++++++++++++++++++++++++

To profile applications dynamically without requiring to restart the application,``rocprofv3`` provides dynamic process attachment. This is particularly useful for profiling long-running applications, services, or applications in a specific state.

Dynamic process attachment uses the ``-p``, ``--pid``, or ``--attach`` options (all equivalent) followed by the target process ID. The profiler instruments the target process and collects the specified tracing or counter data for the configured duration.

For more information, see :ref:`rocprofv3-process-attachment`.

Post-processing tracing options
++++++++++++++++++++++++++++++++

``rocprofv3`` provides options to collect tracing summary or statistics after conclusion of a tracing session. These options are described here.

Stats
######

This option collects statistics for the enabled tracing types. For example, it collects statistics of HIP APIs, when HIP trace is enabled.
The statistics help to determine the API or function that took the most amount of time.

.. code-block:: shell

    rocprofv3 --stats --hip-trace --output-format csv -- <application_path>

The preceding command generates a ``hip_api_stats.csv``, ``domain_stats.csv`` and ``hip_api_trace.csv`` file prefixed with the process ID.

.. code-block:: shell

    $ cat hip_api_stats.csv

Here are the contents of ``hip_api_stats.csv`` file:

.. csv-table:: HIP stats
   :file: /data/hip_api_stats.csv
   :widths: 10,10,20,20,10,10,10,10
   :header-rows: 1

Here are the contents of ``domain_stats.csv`` file:

.. csv-table:: Domain stats
   :file: /data/hip_domain_stats.csv
   :widths: 10,10,20,20,10,10,10,10
   :header-rows: 1

For the description of the fields in the output file, see :ref:`output-file-fields`.

Summary
########

This option displays a summary of tracing data for the enabled tracing type, after conclusion of the profiling session.

.. code-block:: shell

   rocprofv3 -S --hip-trace -- <application_path>

.. image:: /data/rocprofv3_summary.png

Summary per domain
###################

This option displays a summary of each tracing domain for the enabled tracing type, after conclusion of the profiling session.

.. code-block:: shell

    rocprofv3 -D --hsa-trace --hip-trace --output-format csv  -- <application_path>

The preceding command generates a ``hip_trace.csv`` and ``hsa_trace.csv`` file prefixed with the process ID along with displaying the summary of each domain.

Summary groups
###############

This option displays a summary of multiple domains for the domain names specified on the command line. The summary groups can be separated using a pipe ( | ) symbol.

To see a summary for ``MEMORY_COPY`` domains, use:

.. code-block:: shell

   rocprofv3 --summary-groups MEMORY_COPY --sys-trace  -- <application_path>

.. image:: /data/rocprofv3_memcpy_summary.png

To see a summary for ``MEMORY_COPY`` and ``HIP_API`` domains, use:

.. code-block:: shell

   rocprofv3 --summary-groups 'MEMORY_COPY|HIP_API' --sys-trace -- <application_path>

.. image:: /data/rocprofv3_hip_memcpy_summary.png

Summary output file
######################

This option specifies the output file for the summary. By default, the summary is displayed on ``stderr``. To specify another output file for summary, use:

.. code-block:: shell

   rocprofv3 -S -D --summary-output-file filename --sys-trace -- <application_path>

The preceding command generates an output file named "filename" consisting of the summary for each domain. This also generates the files for the enabled tracing types under ``-sys-trace`` option.

.. include:: /data/summary.txt
   :literal:

Configuration output
+++++++++++++++++++++++

The ``--output-config`` option generates a comprehensive configuration output file that contains all resolved ``rocprofv3`` settings and options used during a profiling session. This feature is essential for debugging, reproducibility, and configuration validation.

To generate a configuration output file during profiling, use:

.. code-block:: bash

    rocprofv3 --output-config --hip-trace -- <application_path>

This command generates a configuration file (typically ``<process_id>_config.json``) alongside the regular profiling output files.


The generated JSON configuration file contains detailed information about the profiling session and is structured with a ``rocprofiler-sdk-tool`` array containing comprehensive metadata and configuration details.

The metadata section includes essential session information such as process ID (``pid``), initialization and finalization timestamps (``init_time``, ``fini_time``), the exact command executed, and detailed build specifications. The build specification contains version information, compiler details, git revision, system architecture, and kernel version, providing complete context for reproducing the environment.

The config section is the most comprehensive part, containing all profiling options with their resolved boolean and numerical values. This includes tracing options like ``hip_runtime_api_trace``, ``hip_compiler_api_trace``, ``kernel_trace``, ``hsa_core_api_trace``, ``memory_copy_trace``, and many others. It also shows advanced configuration like PC sampling settings (``pc_sampling_method``, ``pc_sampling_interval``), filtering options (``kernel_filter_include``, ``kernel_filter_exclude``), output formatting choices (``csv_output``, ``json_output``, ``pftrace_output``), and performance tuning parameters.

The environment section captures all environment variables active during the profiling session, including system variables such as ``SHELL``, ``COLORTERM``, ``HOSTNAME``, and ROCm-specific variables, providing complete environmental context for reproduction.

Sample configuration output structure:

.. code-block:: json

    {
      "rocprofiler-sdk-tool": [
        {
          "metadata": {
            "pid": 213524,
            "init_time": 682678344984459,
            "fini_time": 682678842290172,
            "config": {
              "hip_runtime_api_trace": true,
              "hip_compiler_api_trace": true,
              "kernel_trace": false,
              "hsa_core_api_trace": false,
              "memory_copy_trace": false,
              "counter_collection": false,
			  "kernel_filter_include": ".*",
              "demangle": true,
              "minimum_output_bytes": 0,
              "csv_output": true,
              "json_output": false,
              "output_path": "out",
              "output_file": "1a2b3c4d5e6f/213524"
            },
            "command": ["./MatrixTranspose"],
            "build_spec": {
              "version_major": 1,
              "version_minor": 0,
              "compiler_id": "GNU",
              "compiler_version": "11.4.0",
              "git_revision": "a1b2c3d4e5f6789012345678901234567890abcd",
              "system_name": "Linux",
              "system_processor": "x86_64"
            },
            "environment": {
              "SHELL": "/bin/bash",
              "COLORTERM": "truecolor",
              "HOSTNAME": "1a2b3c4d5e6f",
              "ROCM_ROOT": "/opt/rocm-6.4.2",
              "ROCM_VERSION": "6.4.2",
              "BUILD_NUM": "12345",
              "ROCPROF_OUTPUT_PATH": "out",
              "ROCPROF_OUTPUT_CONFIG_FILE": "1",
              "ROCPROF_OUTPUT_FORMAT": "csv",
              "ROCPROF_HIP_COMPILER_API_TRACE": "1",
              "ROCPROF_HIP_RUNTIME_API_TRACE": "1",
               ".... Output truncated for brevity ...."
            }
          }
        }
      ]
    }

The configuration output file provides complete transparency into ``rocprofv3`` operation, documenting all settings, defaults, and environmental context required for profiling sessions.

Collecting traces using input file
++++++++++++++++++++++++++++++++++++

The preceding sections describe how to collect traces by specifying the desired tracing type on the command line. You can also specify the desired tracing types in an input file in YAML (.yaml/.yml), or JSON (.json) format. You can supply any command-line option for tracing in the input file.

Here is a sample input.yaml file for collecting tracing summary:

.. code-block:: yaml

   jobs:
     - output_directory: "@CMAKE_CURRENT_BINARY_DIR@/%env{ARBITRARY_ENV_VARIABLE}%"
       output_file: out
       output_format: [pftrace, json, otf2]
       log_level: env
       runtime_trace: true
       kernel_rename: true
       summary: true
       summary_per_domain: true
       summary_groups: ["KERNEL_DISPATCH|MEMORY_COPY"]
       summary_output_file: "summary"

Here is a sample input.json file for collecting tracing summary:

.. code-block:: json

  {
    "jobs": [
      {
        "output_directory": "out-directory",
        "output_file": "out",
        "output_format": ["pftrace", "json", "otf2"],
        "log_level": "env",
        "runtime_trace": true,
        "kernel_rename": true,
        "summary": true,
        "summary_per_domain": true,
        "summary_groups": ["KERNEL_DISPATCH|MEMORY_COPY"],
        "summary_output_file": "summary"
      }
    ]
  }

Here is the input schema (properties) of JSON or YAML input files:

-  **jobs** *(array)*: ``rocprofv3`` input data per application run.

   -  **Items** *(object)*: Data for ``rocprofv3``

      -  **hip_trace** *(boolean)*
      -  **hip_runtime_trace** *(boolean)*
      -  **hip_compiler_trace** *(boolean)*
      -  **marker_trace** *(boolean)*
      -  **kernel_trace** *(boolean)*
      -  **memory_copy_trace** *(boolean)*
      -  **memory_allocation_trace** *(boolean)*
      -  **scratch_memory_trace** *(boolean)*
      -  **stats** *(boolean)*
      -  **hsa_trace** *(boolean)*
      -  **hsa_core_trace** *(boolean)*
      -  **hsa_amd_trace** *(boolean)*
      -  **hsa_finalize_trace** *(boolean)*
      -  **hsa_image_trace** *(boolean)*
      -  **sys_trace** *(boolean)*
      -  **minimum-output-data** *(integer)*
      -  **disable-signal-handlers** *(boolean)*
      -  **mangled_kernels** *(boolean)*
      -  **truncate_kernels** *(boolean)*
      -  **output_file** *(string)*
      -  **output_directory** *(string)*
      -  **output_format** *(array)*
      -  **log_level** *(string)*
      -  **preload** *(array)*

For description of the options specified under job items, see :ref:`cli-options`.

To supply the input file for collecting traces, use:

.. code-block:: shell

   rocprofv3 -i input.yaml -- <application_path>

Please note that input file format must be a valid YAML or JSON file.

Disabling specific tracing options
++++++++++++++++++++++++++++++++++++

When using aggregate tracing options like ``--runtime-trace`` or ``--sys-trace``, you can disable specific tracing options by setting them to ``False``. This allows fine-grained control over the traces to be collected.

.. code-block:: shell

   rocprofv3 --runtime-trace --scratch-memory-trace=False -- <application_path>

The preceding command enables all traces included in ``--runtime-trace`` except for scratch memory tracing.

Similarly, for ``--sys-trace``:

.. code-block:: shell

   rocprofv3 --sys-trace --hsa-trace=False -- <application_path>

The preceding command enables all traces included in ``--sys-trace`` except for HSA API tracing.

To disable multiple specific tracing options, use:

.. code-block:: shell

   rocprofv3 --sys-trace --hsa-trace=False --scratch-memory-trace=False -- <application_path>

This feature is particularly useful to collect most traces excluding specific ones that might be unnecessary for your analysis or that generate excessive data.

.. _kernel-counter-collection:

Kernel counter collection
--------------------------

The application tracing functionality allows you to evaluate the duration of kernel execution but is of little help in providing insight into kernel execution details. The kernel counter collection functionality allows you to select kernels for profiling and choose the basic counters or derived metrics to be collected for each kernel execution, thus providing a greater insight into kernel execution.

AMDGPUs are equipped with hardware performance counters that can be used to measure specific values during kernel execution, which are then exported from the GPU and written into the output files at the end of the kernel execution. These performance counters vary according to the GPU. Therefore, it is recommended to examine the hardware counters that can be collected before running the profile.

There are two types of data available for profiling: hardware basic counters and derived metrics.

The derived metrics are the counters derived from the basic counters using mathematical expressions. Note that the basic counters and derived metrics are collectively referred as counters in this document.

To see the counters available on the GPU, use:

.. code-block:: shell

   rocprofv3 --list-avail

Sample output for the list-avail command:

.. include:: /data/list-avail.txt
   :width: 100%
   :align: center

You can also customize the counters according to the requirement. Such counters are named :ref:`extra-counters`.

For a comprehensive list of counters available on MI200, see `MI200 performance counters and metrics <https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html>`_.

.. note::

   **Counter dimension collection:** When collecting counters with multiple dimensions or instances, such as ``TCC_MISS`` with ``DIMENSION_INSTANCE[0:15]``, individual dimension values can't be collected separately using bracket notation, such as ``TCC_MISS[0]`` or ``TCC_MISS[15]`` in the input files.

   **To collect aggregated values:** Specify the counter name without dimension specifiers, such as ``pmc: TCC_MISS``. The ``rocprofv3`` tool automatically collects accumulated values across all instances.

   **To collect values per instance:** Use JSON output format, which includes detailed dimension information for individual counter instances.

Counter collection using input file
+++++++++++++++++++++++++++++++++++++

Input files can be in text (.txt), YAML (.yaml/.yml), or JSON (.json) format to specify the the desired counters for collection.

When using input file in text format, the line consisting of the counter names must begin with ``pmc``. The number of counters that can be collected in one profiling run are limited by the GPU hardware resources. If too many counters are selected, the kernels need to be executed multiple times(multi-pass execution) to collect all the counters. For multi-pass execution, include multiple ``pmc`` rows in the input file. Counters in each ``pmc`` row can be collected in each application run.

Here is a sample input.txt file for specifying counters for collection:

.. code-block:: shell

   $ cat input.txt

   pmc: GPUBusy SQ_WAVES
   pmc: GRBM_GUI_ACTIVE

While the input file in text format can only be used for counter collection, JSON and YAML formats support all the command-line options for profiling. The input file in YAML or JSON format has an array of profiling configurations called jobs. Each job is used to configure profiling for an application execution.

Here is the input schema (properties) of JSON or YAML input files:

-  **jobs** *(array)*: ``rocprofv3`` input data per application run

   -  **Items** *(object)*: Data for ``rocprofv3``

      -  **pmc** *(array)*: list of counters for collection
      -  **kernel_include_regex** *(string)*
      -  **kernel_exclude_regex** *(string)*
      -  **kernel_iteration_range** *(string)*
      -  **mangled_kernels** *(boolean)*
      -  **truncate_kernels** *(boolean)*
      -  **output_file** *(string)*
      -  **output_directory** *(string)*
      -  **output_format** *(array)*
      -  **list_avail** *(boolean)*
      -  **log_level** *(string)*
      -  **preload** *(array)*
      -  **minimum-output-data** *(integer)*
      -  **disable-signal-handlers** *(boolean)*
      -  **pc_sampling_unit** *(string)*
      -  **pc_sampling_method** *(string)*
      -  **pc_sampling_interval** *(integer)*
      -  **pc_sampling_beta_enabled** *(boolean)*

For description of the options specified under job items, see :ref:`cli-options`.

Here is a sample input.json file for specifying counters for collection along with the options to filter and control the output:

.. code-block:: shell

    $ cat input.json

    {
      "jobs": [
         {
            "pmc": ["SQ_WAVES", "GRBM_COUNT", "GRBM_GUI_ACTIVE"]
         },
         {
            "pmc": ["FETCH_SIZE", "SQ_WAVE_CYCLES"],
            "kernel_include_regex": ".*_kernel",
            "kernel_exclude_regex": "multiply",
            "kernel_iteration_range": "[1-2],[3-4]",
            "output_file": "out",
            "output_format": [
               "csv",
               "json"
            ],
            "truncate_kernels": true
         }
      ]
    }

Here is a sample input.yaml file for counter collection:

.. code-block:: yaml

  jobs:
    - pmc: ["SQ_WAVES", "GRBM_COUNT", "GRBM_GUI_ACTIVE"]
    - pmc: ["SQ_WAVE_CYCLES", "WRITE_SIZE"]
      kernel_include_regex: ".*_kernel"
      kernel_exclude_regex: "multiply"
      kernel_iteration_range: "[1-2],[3-4]"
      output_file: "out"
      output_format:
        - "csv"
        - "json"
      truncate_kernels: true

To supply the input file for kernel counter collection, use:

.. code-block:: bash

   rocprofv3 -i input.yaml -- <application_path>

Counter collection using command line
++++++++++++++++++++++++++++++++++++++

You can also collect the desired counters by directly specifying them in the command line instead of using an input file.

To supply the counters in the command line, use:

.. code-block:: shell

   rocprofv3 --pmc SQ_WAVES GRBM_COUNT GRBM_GUI_ACTIVE -- <application_path>

.. note::

   - When specifying more than one counter, separate them using space or a comma.
   - Job fails if the entire set of counters can't be collected in a single pass.

Multi-pass counter collection
++++++++++++++++++++++++++++++

When counters can't be collected simultaneously due to hardware limitations, you can use multi-pass counter collection. This helps you collect different sets of counters across multiple profiling passes of the same application.

**Using multiple** ``--pmc`` **flags:**

You can specify multiple ``--pmc`` flags to define different counter groups. Each ``--pmc`` flag represents a separate profiling pass:

.. code-block:: shell

   rocprofv3 --pmc SQ_WAVES SQ_WAVE_CYCLES --pmc GRBM_COUNT GRBM_GUI_ACTIVE -- <application_path>

The preceding command creates two profiling passes:

- Pass 1: Collects ``SQ_WAVES`` and ``SQ_WAVE_CYCLES``.

- Pass 2: Collects ``GRBM_COUNT`` and ``GRBM_GUI_ACTIVE``.

**Combining CLI and input file:**

You can combine ``--pmc`` flag with an input file. The counters specified in CLI and input file are combined, creating separate passes for each counter:

.. code-block:: shell

   rocprofv3 -i input.txt --pmc GRBM_COUNT --pmc SQ_WAVES -- <application_path>

If ``input.txt`` contains:

.. code-block:: text

   pmc: FETCH_SIZE SQ_WAVES
   pmc: GRBM_GUI_ACTIVE

The preceding command creates four profiling passes:

- Pass 1: ``GRBM_COUNT`` (from CLI).

- Pass 2: ``SQ_WAVES`` (from CLI).

- Pass 3: ``FETCH_SIZE SQ_WAVE_CYCLES`` (from input file).

- Pass 4: ``GRBM_GUI_ACTIVE`` (from input file).

**Output organization:**

In multi-pass counter collection, each pass generates its output in a separate ``pass_n`` subdirectory:

.. code-block:: text

   output_directory/
   ├── pass_1/
   │   └── counter_collection.csv
   ├── pass_2/
   │   └── counter_collection.csv
   ├── pass_3/
   │   └── counter_collection.csv
   └── pass_4/
       └── counter_collection.csv

.. note::

   - Multi-pass counter collection is not compatible with attach mode (``--pid``).

   - Multi-pass counter collection is not compatible with ``--collection-period``.

   - Each pass runs the application from start to finish.

.. _extra-counters:

Extra counters
++++++++++++++++

While the basic counters and derived metrics are available for collection by default, you can also define counters as per requirement. These user-defined counters with custom definitions are named extra counters.

You can define the extra counters in a YAML file as shown:

.. code-block:: yaml

    rocprofiler-sdk:
      counters-schema-version: 1
      counters:
        - name: GRBM_GUI_ACTIVE_SUM
          description: "Unit: cycles"
          properties: []
          definitions:
            - architectures:
                - gfx10
                - gfx1010
                - gfx1030
                - gfx1031
                - gfx1032
                - gfx11
                - gfx1100
                - gfx1101
                - gfx1102
                - gfx9
                - gfx906
                - gfx908
                - gfx90a
                - gfx942
              expression: reduce(GRBM_GUI_ACTIVE,max)*CU_NUM
        - name: CPC_CPC_STAT_BUSY
          description: CPC Busy.
          properties: []
          definitions:
            - architectures:
                - gfx940
                - gfx941
              block: CPC
              event: 25

Please note, the above sample uses the ``CPC_CPC_STAT_BUSY`` counter definition for the ``gfx940``
and ``gfx941`` architectures to demonstrate the YAML schema when counters have different
architecture-specific definitions.

If this YAML is placed in a ``extra_counters.yaml`` file, to collect the extra counters defined
in the ``extra_counters.yaml`` file, use the ``-E`` / ``--extra-counters`` option:

.. code-block:: shell

   rocprofv3 -E <path-to-extra_counters.yaml> --pmc GRBM_GUI_ACTIVE_SUM --output-format csv -- <application_path>

Where the option ``--pmc`` is used to specify the extra counters to be collected.

Kernel counter collection output
+++++++++++++++++++++++++++++++++

Using ``rocprofv3`` for counter collection using input file or command line generates a ``./pmc_n/counter_collection.csv`` file prefixed with the process ID. For each ``pmc`` row, a directory ``pmc_n`` containing a ``counter_collection.csv`` file is generated, where n = 1 for the first row and so on.

When using input file in JSON or YAML format, for each job, a directory ``pass_n`` containing a ``counter_collection.csv`` file is generated, where n = 1 for the first job and so on.

Each row of the CSV file is an instance of kernel execution. Here is a truncated version of the output file from ``pmc_1``:

.. code-block:: shell

    $ cat pmc_1/218_counter_collection.csv

Here are the contents of ``counter_collection.csv`` file:

.. csv-table:: Counter collection
   :file: /data/counter_collection.csv
   :widths: 10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10
   :header-rows: 1

For the description of the fields in the output file, see :ref:`output-file-fields`.

Iteration based counter multiplexing
++++++++++++++++++++++++++++++++++++

Counter multiplexing allows a single run of the program to collect groups of counters. This is useful when the counters you want to collect exceed the hardware limits and you cannot run the program multiple times for collection.

This feature is available when using YAML (.yaml/.yml) or JSON (.json) input formats. Two new fields are introduced,  ``pmc_groups`` and ``pmc_group_interval``. The ``pmc_groups`` field is used to specify the groups of counters to be collected in each run. The ``pmc_group_interval`` field is used to specify the interval between each group of counters. Interval is per-device and increments per dispatch on the device (i.e. dispatch_id). When the interval is reached the next group is selected.

Here is a sample input.yaml file for specifying counter multiplexing:

.. code-block:: yaml

   jobs:
   - pmc_groups: [["SQ_WAVES", "GRBM_COUNT"], ["GRBM_GUI_ACTIVE"]]
      pmc_group_interval: 4

This sample input will collect the first group of counters (``SQ_WAVES``, ``GRBM_COUNT``) for the first 4 kernel executions on the device, then the second group of counters (``GRBM_GUI_ACTIVE``) for the next 4 kernel executions on the device, and so on.

An example of the interval period for this input is given below:

.. code-block:: shell

    Device 1, <Kernel A>, Collect SQ_WAVES, GRBM_COUNT
    Device 1, <Kernel A>, Collect SQ_WAVES, GRBM_COUNT
    Device 1, <Kernel B>, Collect SQ_WAVES, GRBM_COUNT
    Device 1, <Kernel C>, Collect SQ_WAVES, GRBM_COUNT
    <Interval reached on Device 1, Switching Counters>
    Device 1, <Kernel D>, Collect GRBM_GUI_ACTIVE

Here is the same sample in JSON format:

.. code-block:: shell

   {
      "jobs": [
         {
               "pmc_groups": [["SQ_WAVES", "GRBM_COUNT"], ["GRBM_GUI_ACTIVE"]],
               "pmc_group_interval": 4
         }
      ]
   }

Perfetto visualization
-----------------------

`Perfetto <https://perfetto.dev/>`_ is an open-source tracing tool that provides a detailed view of system performance. You can use Perfetto to visualize traces and performance counter data as explained in the following sections.

Perfetto visualization for traces
+++++++++++++++++++++++++++++++++++++++++++++

Perfetto helps you to visualize the collected traces in Perfetto viewer, which is a user-friendly interface that makes it easier to analyze and understand the performance characteristics of your application.

To generate a Perfetto trace file, use the ``--output-format pftrace`` option along with the desired tracing options. For example, to collect system traces and generate a Perfetto trace file, use:

.. code-block:: bash

  rocprofv3 --sys-trace --output-format pftrace -- <application_path>

The generated Perfetto trace file can be opened in the `Perfetto UI <https://ui.perfetto.dev/>`_.

**Figure 1:** Generic perfetto visualization

.. image:: /data/perfetto_generic.png
   :width: 100%
   :align: center

**Figure 2:** Visualization of ROCm flow data in Perfetto

.. image:: /data/perfetto_flow.png
   :width: 100%
   :align: center

Perfetto visualization for counter collection
+++++++++++++++++++++++++++++++++++++++++++++

When collecting performance counter data, you can visualize the counter tracks per agent in the Perfetto viewer by using the PFTrace output format. This helps you see how counter values change over time during kernel execution.

To generate a Perfetto trace file with counter data, use:

.. code-block:: shell

    rocprofv3 --pmc SQ_WAVES GRBM_COUNT --output-format pftrace -- <application_path>

The generated Perfetto trace file can be opened in the `Perfetto UI <https://ui.perfetto.dev/>`_. In the viewer, performance counters will appear as counter tracks organized by agent, allowing you to visualize counter values changing over time alongside kernel executions and other traced activities.

You can also combine this with the system trace option to get a more comprehensive view of the system's performance. For example, you can use the following command to collect both system trace and performance counter data:

.. code-block:: bash

  rocprofv3 --pmc SQ_WAVES GRBM_COUNT --sys-trace --output-format pftrace -- <application_path>

.. image:: /data/perfetto_counters.png
   :width: 100%
   :align: center

Scratch Memory Visualization in Perfetto
+++++++++++++++++++++++++++++++++++++++++++++

When using the ``--scratch-memory-trace`` option with Perfetto output format, ROCProfiler SDK creates visualization tracks for scratch memory usage. Scratch memory operations are displayed as counter tracks organized by agent (GPU), allowing you to monitor the scratch memory allocation patterns during kernel execution.

To generate a Perfetto trace file that includes scratch memory visualization:

.. code-block:: bash

  rocprofv3 --scratch-memory-trace --output-format pftrace -- <application_path>

In the Perfetto UI, scratch memory appears as counter tracks that show:

- **Allocation peaks**: Each peak represents scratch memory allocation for a kernel execution
- **Memory usage over time**: The height of each peak indicates the amount of memory allocated (typically in KB)
- **Allocation/deallocation pattern**: You can observe when memory is allocated at kernel start and freed at kernel end

For applications with multiple kernel iterations, you'll see multiple peaks in the scratch memory track, with each peak corresponding to a kernel execution. This visualization helps identify scratch memory usage patterns and potential optimization opportunities.

.. image:: /data/perfetto_scratch_memory.png
   :width: 100%
   :align: center

For comprehensive GPU execution insights, combine scratch memory tracing with kernel tracing:

.. code-block:: bash

  rocprofv3 --kernel-trace --scratch-memory-trace --output-format pftrace -- <application_path>

This allows you to correlate scratch memory allocation patterns with specific kernel executions in the Perfetto visualization.

Agent info
-----------

.. note::
  All tracing and counter collection options generate an additional ``agent_info.csv`` file prefixed with the process ID.

The ``agent_info.csv`` file contains information about the CPU or GPU the kernel runs on.

.. code-block:: shell

    $ cat 238_agent_info.csv

    "Node_Id","Logical_Node_Id","Agent_Type","Cpu_Cores_Count","Simd_Count","Cpu_Core_Id_Base","Simd_Id_Base","Max_Waves_Per_Simd","Lds_Size_In_Kb","Gds_Size_In_Kb","Num_Gws","Wave_Front_Size","Num_Xcc","Cu_Count","Array_Count","Num_Shader_Banks","Simd_Arrays_Per_Engine","Cu_Per_Simd_Array","Simd_Per_Cu","Max_Slots_Scratch_Cu","Gfx_Target_Version","Vendor_Id","Device_Id","Location_Id","Domain","Drm_Render_Minor","Num_Sdma_Engines","Num_Sdma_Xgmi_Engines","Num_Sdma_Queues_Per_Engine","Num_Cp_Queues","Max_Engine_Clk_Ccompute","Max_Engine_Clk_Fcompute","Sdma_Fw_Version","Fw_Version","Capability","Cu_Per_Engine","Max_Waves_Per_Cu","Family_Id","Workgroup_Max_Size","Grid_Max_Size","Local_Mem_Size","Hive_Id","Gpu_Id","Workgroup_Max_Dim_X","Workgroup_Max_Dim_Y","Workgroup_Max_Dim_Z","Grid_Max_Dim_X","Grid_Max_Dim_Y","Grid_Max_Dim_Z","Name","Vendor_Name","Product_Name","Model_Name"
    0,0,"CPU",24,0,0,0,0,0,0,0,0,1,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3800,0,0,0,0,0,0,23,0,0,0,0,0,0,0,0,0,0,0,"AMD Ryzen 9 3900X 12-Core Processor","CPU","AMD Ryzen 9 3900X 12-Core Processor",""
    1,1,"GPU",0,256,0,2147487744,10,64,0,64,64,1,64,4,4,1,16,4,32,90000,4098,26751,12032,0,128,2,0,2,24,3800,1630,432,440,138420864,16,40,141,1024,4294967295,0,0,64700,1024,1024,1024,4294967295,4294967295,4294967295,"gfx900","AMD","Radeon RX Vega","vega10"

.. _output-file-fields:

Output file fields
-------------------

The following table lists the various fields or the columns in the output CSV files generated for application tracing and kernel counter collection:

.. raw:: html

   <div class="pst-scrollable-table-container">
      <table id="rocprov3-output-fields" class="table">
         <thead>
            <tr>
               <th>Information type</th>
               <th>Field</th>
               <th>Description</th>
            </tr>
         </thead>
         <colgroup>
            <col span="1">
            <col span="1">
         </colgroup>
         <tbody class="output-fields">
            <tr>
               <th rowspan="7">Dispatch information</th>
               <td>Agent_Id</td>
               <td>GPU identifier to which the kernel was submitted.</td>
            </tr>
            <tr>
               <td>Correlation_Id</td>
               <td>Unique identifier for correlation between HIP and HSA async calls during activity tracing.</td>
            </tr>
            <tr>
               <td>Dispatch_Id</td>
               <td>Dispatch identifier</td>
            </tr>
            <tr>
               <td>Process_Id</td>
               <td>Process identifier</td>
            </tr>
            <tr>
               <td>Thread_Id</td>
               <td>Thread identifier</td>
            </tr>
            <tr>
               <td>Queue_Id</td>
               <td>ROCm queue unique identifier to which the kernel was submitted.</td>
            </tr>
            <tr>
               <td>Stream_Id</td>
               <td>Identifies HIP stream ID to which kernel or memory copy operation was submitted. Defaults to 0 if the hip-stream-display option is not enabled</td>
            </tr>
            <tr>
               <th rowspan="8">Kernel information</th>
               <td>Grid_Size</td>
               <td>The total number of work-items (or, threads) launched as a part of the kernel dispatch. In HIP, this is equivalent to the total grid size multiplied by the total workgroup (or, block) size.</td>
            </tr>
            <tr>
               <td>Grid_Size_n</td>
               <td>Number of work-items (or, threads) in the nth dimension required to launch the kernel, where n = X, Y, or Z.</td>
            </tr>
            <tr>
               <td>Kernel_Id</td>
               <td>Kernel identifier</td>
            </tr>
            <tr>
               <td>Kernel_Name</td>
               <td>Kernel name</td>
            </tr>
            <tr>
               <td>Workgroup_Size</td>
               <td>The total number of work-items (or, threads) in each workgroup (or, block) launched as part of the kernel dispatch. In HIP, this is equivalent to the total block size.</td>
            </tr>
            <tr>
               <td>Workgroup_Size_n</td>
               <td>Size of the workgroup in the nth dimension as declared by the compute shader, where n = X, Y, or Z.</td>
            </tr>
            <tr>
               <td>Private_Segment_Size</td>
               <td>The amount of memory required in bytes for the combined private, spill, and arg segments for a work item.</td>
            </tr>
            <tr>
               <td>Group_Segment_Size</td>
               <td>The group segment memory required by a workgroup in bytes. This does not include any dynamically allocated group segment memory that may be added when the kernel is dispatched.</td>
            </tr>
            <tr>
               <th rowspan="5">Resource usage</th>
               <td>LDS_Block_Size</td>
               <td>Thread block size for the kernel’s Local Data Share (LDS) memory (shared memory per work-group).</td>
            </tr>
            <tr>
               <td>Scratch_Size</td>
               <td>Kernel’s scratch memory (private memory per work-item) size.</td>
            </tr>
            <tr>
               <td>SGPR_Count</td>
               <td>Kernel’s Scalar General Purpose Register (SGPR) count.</td>
            </tr>
            <tr>
               <td>VGPR_Count</td>
               <td>Kernel’s Architected Vector General Purpose Register (VGPR) count.</td>
            </tr>
            <tr>
               <td>Accum_VGPR_Count</td>
               <td>Kernel’s Accumulation Vector General Purpose Register (Accum_VGPR/AGPR) count.</td>
            </tr>
            <tr>
               <th rowspan="4">Counter data</td>
               <td>Counter_Name</td>
               <td>Name of the counter</td>
            </tr>
            <tr>
               <td>Counter_Value</td>
               <td>The numeric value measured by a specific hardware performance counter during a kernel dispatch</td>
            </tr>
            <tr>
               <td>Start_Timestamp</td>
               <td>Begin time in nanoseconds (ns) when the kernel begins execution.</td>
            </tr>
            <tr>
               <td>End_Timestamp</td>
               <td>End time in ns when the kernel finishes execution.</td>
            </tr>
         </tbody>
      </table>
   </div>

Output formats
----------------

- rocpd (SQLite3 Database (Default))
- CSV
- JSON (Custom format for programmatic analysis only)
- PFTrace (Perfetto trace for visualization with Perfetto)
- OTF2 (Open Trace Format for visualization with compatible third-party tools)


The default output format is ``rocpd``. To know more about the rocpd format, see :ref:`using-rocpd-output-format`.
To specify the particular output format, use the ``--output-format`` option followed by the desired format.

.. code-block::

   rocprofv3 -i input.txt --output-format json -- <application_path>

Format selection is case-insensitive and multiple output formats are supported. While ``--output-format json`` exclusively enables JSON output, ``--output-format csv json pftrace otf2, rocpd`` enables all four output formats for the run.

For PFTrace trace visualization, use the PFTrace format and open the trace in `ui.perfetto.dev <https://ui.perfetto.dev/>`_.

For OTF2 trace visualization, open the trace in `vampir.eu <https://vampir.eu/>`_ or any supported visualizer.

.. note::
  For large trace files (> 10GB), it's recommended to use OTF2 format.

JSON output schema
++++++++++++++++++++

``rocprofv3`` supports a custom JSON output format designed for programmatic analysis and **NOT** for visualization.
The schema is optimized for size while factoring in usability.

.. note::

   Perfetto UI doesn't accept this JSON output format.

To generate the JSON output, use ``--output-format json`` command-line option.

Properties
###########

Here are the properties of the JSON output schema:

- **rocprofiler-sdk-tool** `(array)`: rocprofv3 data per process (each element represents a process).
   - **Items** `(object)`: Data for rocprofv3.
      - **metadata** `(object, required)`: Metadata related to the profiler session.
         - **pid** `(integer, required)`: Process ID.
         - **init_time** `(integer, required)`: Initialization time in nanoseconds.
         - **fini_time** `(integer, required)`: Finalization time in nanoseconds.
      - **agents** `(array, required)`: List of agents.
         - **Items** `(object)`: Data for an agent.
            - **size** `(integer, required)`: Size of the agent data.
            - **id** `(object, required)`: Identifier for the agent.
               - **handle** `(integer, required)`: Handle for the agent.
            - **type** `(integer, required)`: Type of the agent.
            - **cpu_cores_count** `(integer)`: Number of CPU cores.
            - **simd_count** `(integer)`: Number of SIMD units.
            - **mem_banks_count** `(integer)`: Number of memory banks.
            - **caches_count** `(integer)`: Number of caches.
            - **io_links_count** `(integer)`: Number of I/O links.
            - **cpu_core_id_base** `(integer)`: Base ID for CPU cores.
            - **simd_id_base** `(integer)`: Base ID for SIMD units.
            - **max_waves_per_simd** `(integer)`: Maximum waves per SIMD.
            - **lds_size_in_kb** `(integer)`: Size of LDS in KB.
            - **gds_size_in_kb** `(integer)`: Size of GDS in KB.
            - **num_gws** `(integer)`: Number of GWS (global work size).
            - **wave_front_size** `(integer)`: Size of the wave front.
            - **num_xcc** `(integer)`: Number of XCC (execution compute units).
            - **cu_count** `(integer)`: Number of compute units (CUs).
            - **array_count** `(integer)`: Number of arrays.
            - **num_shader_banks** `(integer)`: Number of shader banks.
            - **simd_arrays_per_engine** `(integer)`: SIMD arrays per engine.
            - **cu_per_simd_array** `(integer)`: CUs per SIMD array.
            - **simd_per_cu** `(integer)`: SIMDs per CU.
            - **max_slots_scratch_cu** `(integer)`: Maximum slots for scratch CU.
            - **gfx_target_version** `(integer)`: GFX target version.
            - **vendor_id** `(integer)`: Vendor ID.
            - **device_id** `(integer)`: Device ID.
            - **location_id** `(integer)`: Location ID.
            - **domain** `(integer)`: Domain identifier.
            - **drm_render_minor** `(integer)`: DRM render minor version.
            - **num_sdma_engines** `(integer)`: Number of SDMA engines.
            - **num_sdma_xgmi_engines** `(integer)`: Number of SDMA XGMI engines.
            - **num_sdma_queues_per_engine** `(integer)`: Number of SDMA queues per engine.
            - **num_cp_queues** `(integer)`: Number of CP queues.
            - **max_engine_clk_ccompute** `(integer)`: Maximum engine clock for compute.
            - **max_engine_clk_fcompute** `(integer)`: Maximum engine clock for F compute.
            - **sdma_fw_version** `(object)`: SDMA firmware version.
               - **uCodeSDMA** `(integer, required)`: SDMA microcode version.
               - **uCodeRes** `(integer, required)`: Reserved microcode version.
            - **fw_version** `(object)`: Firmware version.
               - **uCode** `(integer, required)`: Microcode version.
               - **Major** `(integer, required)`: Major version.
               - **Minor** `(integer, required)`: Minor version.
               - **Stepping** `(integer, required)`: Stepping version.
            - **capability** `(object, required)`: Agent capability flags.
               - **HotPluggable** `(integer, required)`: Hot pluggable capability.
               - **HSAMMUPresent** `(integer, required)`: HSAMMU present capability.
               - **SharedWithGraphics** `(integer, required)`: Shared with graphics capability.
               - **QueueSizePowerOfTwo** `(integer, required)`: Queue size is power of two.
               - **QueueSize32bit** `(integer, required)`: Queue size is 32-bit.
               - **QueueIdleEvent** `(integer, required)`: Queue idle event.
               - **VALimit** `(integer, required)`: VA limit.
               - **WatchPointsSupported** `(integer, required)`: Watch points supported.
               - **WatchPointsTotalBits** `(integer, required)`: Total bits for watch points.
               - **DoorbellType** `(integer, required)`: Doorbell type.
               - **AQLQueueDoubleMap** `(integer, required)`: AQL queue double map.
               - **DebugTrapSupported** `(integer, required)`: Debug trap supported.
               - **WaveLaunchTrapOverrideSupported** `(integer, required)`: Wave launch trap override supported.
               - **WaveLaunchModeSupported** `(integer, required)`: Wave launch mode supported.
               - **PreciseMemoryOperationsSupported** `(integer, required)`: Precise memory operations supported.
               - **DEPRECATED_SRAM_EDCSupport** `(integer, required)`: Deprecated SRAM EDC support.
               - **Mem_EDCSupport** `(integer, required)`: Memory EDC support.
               - **RASEventNotify** `(integer, required)`: RAS event notify.
               - **ASICRevision** `(integer, required)`: ASIC revision.
               - **SRAM_EDCSupport** `(integer, required)`: SRAM EDC support.
               - **SVMAPISupported** `(integer, required)`: SVM API supported.
               - **CoherentHostAccess** `(integer, required)`: Coherent host access.
               - **DebugSupportedFirmware** `(integer, required)`: Debug supported firmware.
               - **Reserved** `(integer, required)`: Reserved field.
      - **counters** `(array, required)`: Array of counter objects.
         - **Items** `(object)`
            - **agent_id** *(object, required)*: Agent ID information.
               - **handle** *(integer, required)*: Handle of the agent.
            - **id** *(object, required)*: Counter ID information.
               - **handle** *(integer, required)*: Handle of the counter.
            - **is_constant** *(integer, required)*: Indicator if the counter value is constant.
            - **is_derived** *(integer, required)*: Indicator if the counter value is derived.
            - **name** *(string, required)*: Name of the counter.
            - **description** *(string, required)*: Description of the counter.
            - **block** *(string, required)*: Block information of the counter.
            - **expression** *(string, required)*: Expression of the counter.
            - **dimension_ids** *(array, required)*: Array of dimension IDs.
               - **Items** *(integer)*: Dimension ID.
      - **strings** *(object, required)*: String records.
         - **callback_records** *(array)*: Callback records.
            - **Items** *(object)*
               - **kind** *(string, required)*: Kind of the record.
               - **operations** *(array, required)*: Array of operations.
                  - **Items** *(string)*: Operation.
         - **buffer_records** *(array)*: Buffer records.
            - **Items** *(object)*
               - **kind** *(string, required)*: Kind of the record.
               - **operations** *(array, required)*: Array of operations.
                  - **Items** *(string)*: Operation.
         - **marker_api** *(array)*: Marker API records.
            - **Items** *(object)*
               - **key** *(integer, required)*: Key of the record.
               - **value** *(string, required)*: Value of the record.
         - **counters** *(object)*: Counter records.
            - **dimension_ids** *(array, required)*: Array of dimension IDs.
               - **Items** *(object)*
                  - **id** *(integer, required)*: Dimension ID.
                  - **instance_size** *(integer, required)*: Size of the instance.
                  - **name** *(string, required)*: Name of the dimension.
         -  **pc_sample_instructions** *(array)*: Array of decoded
            instructions matching sampled PCs from pc_sample_host_trap
            section.
         -  **pc_sample_comments** *(array)*: Comments matching
            assembly instructions from pc_sample_instructions array. If
            debug symbols are available, comments provide instructions
            to source-line mapping. Otherwise, a comment is an empty
            string.
      - **code_objects** *(array, required)*: Code object records.
         - **Items** *(object)*
            - **size** *(integer, required)*: Size of the code object.
            - **code_object_id** *(integer, required)*: ID of the code object.
            - **rocp_agent** *(object, required)*: ROCP agent information.
               - **handle** *(integer, required)*: Handle of the ROCP agent.
            - **hsa_agent** *(object, required)*: HSA agent information.
               - **handle** *(integer, required)*: Handle of the HSA agent.
            - **uri** *(string, required)*: URI of the code object.
            - **load_base** *(integer, required)*: Base address for loading.
            - **load_size** *(integer, required)*: Size for loading.
            - **load_delta** *(integer, required)*: Delta for loading.
            - **storage_type** *(integer, required)*: Type of storage.
            - **memory_base** *(integer, required)*: Base address for memory.
            - **memory_size** *(integer, required)*: Size of memory.
      - **kernel_symbols** *(array, required)*: Kernel symbol records.
         - **Items** *(object)*
            - **size** *(integer, required)*: Size of the kernel symbol.
            - **kernel_id** *(integer, required)*: ID of the kernel.
            - **code_object_id** *(integer, required)*: ID of the code object.
            - **kernel_name** *(string, required)*: Name of the kernel.
            - **kernel_object** *(integer, required)*: Object of the kernel.
            - **kernarg_segment_size** *(integer, required)*: Size of the kernarg segment.
            - **kernarg_segment_alignment** *(integer, required)*: Alignment of the kernarg segment.
            - **group_segment_size** *(integer, required)*: Size of the group segment.
            - **private_segment_size** *(integer, required)*: Size of the private segment.
            - **formatted_kernel_name** *(string, required)*: Formatted name of the kernel.
            - **demangled_kernel_name** *(string, required)*: Demangled name of the kernel.
            - **truncated_kernel_name** *(string, required)*: Truncated name of the kernel.
      - **callback_records** *(object, required)*: Callback record details.
         - **counter_collection** *(array)*: Counter collection records.
            - **Items** *(object)*
               - **dispatch_data** *(object, required)*: Dispatch data details.
                  - **size** *(integer, required)*: Size of the dispatch data.
                  - **correlation_id** *(object, required)*: Correlation ID information.
                     - **internal** *(integer, required)*: Internal correlation ID.
                     - **external** *(integer, required)*: External correlation ID.
                  - **dispatch_info** *(object, required)*: Dispatch information details.
                     - **size** *(integer, required)*: Size of the dispatch information.
                     - **agent_id** *(object, required)*: Agent ID information.
                        - **handle** *(integer, required)*: Handle of the agent.
                     - **queue_id** *(object, required)*: Queue ID information.
                        - **handle** *(integer, required)*: Handle of the queue.
                     - **kernel_id** *(integer, required)*: ID of the kernel.
                     - **dispatch_id** *(integer, required)*: ID of the dispatch.
                     - **private_segment_size** *(integer, required)*: Size of the private segment.
                     - **group_segment_size** *(integer, required)*: Size of the group segment.
                     - **workgroup_size** *(object, required)*: Workgroup size information.
                        - **x** *(integer, required)*: X dimension.
                        - **y** *(integer, required)*: Y dimension.
                        - **z** *(integer, required)*: Z dimension.
                     - **grid_size** *(object, required)*: Grid size information.
                        - **x** *(integer, required)*: X dimension.
                        - **y** *(integer, required)*: Y dimension.
                        - **z** *(integer, required)*: Z dimension.
               - **records** *(array, required)*: Records.
                  - **Items** *(object)*
                     - **counter_id** *(object, required)*: Counter ID information.
                        - **handle** *(integer, required)*: Handle of the counter.
                     - **value** *(number, required)*: Value of the counter.
               - **thread_id** *(integer, required)*: Thread ID.
               - **arch_vgpr_count** *(integer, required)*: Count of Architected VGPRs.
               - **accum_vgpr_count** *(integer, required)*: Count of Accumulation VGPRs.
               - **sgpr_count** *(integer, required)*: Count of SGPRs.
               - **lds_block_size_v** *(integer, required)*: Size of LDS block.
      -  **pc_sample_host_trap** *(array)*: Host Trap PC Sampling records.
            - **Items** *(object)*
               - **hw_id** *(object)*: Describes hardware part on which sampled wave was running.
                  -  **chiplet** *(integer)*: Chiplet index.
                  -  **wave_id** *(integer)*: Wave slot index.
                  -  **simd_id** *(integer)*: SIMD index.
                  -  **pipe_id** *(integer)*: Pipe index.
                  -  **cu_or_wgp_id** *(integer)*: Index of compute unit or workgroup processor.
                  -  **shader_array_id** *(integer)*: Shader array index.
                  -  **shader_engine_id** *(integer)*: Shader engine
                     index.
                  -  **workgroup_id** *(integer)*: Workgroup position in the 3D.
                  -  **vm_id** *(integer)*: Virtual memory ID.
                  -  **queue_id** *(integer)*: Queue id.
                  -  **microengine_id** *(integer)*: ACE
                     (microengine) index.
               -  **pc** *(object)*: Encapsulates information about
                  sampled PC.
                  -  **code_object_id** *(integer)*: Code object id.
                  -  **code_object_offset** *(integer)*: Offset within the object if the latter is known. Otherwise, virtual address of the PC.
               -  **exec_mask** *(integer)*: Execution mask indicating active SIMD lanes of sampled wave.
               -  **timestamp** *(integer)*: Timestamp.
               -  **dispatch_id** *(integer)*: Dispatch id.
               -  **correlation_id** *(object)*: Correlation ID information.
                  -  **internal** *(integer)*: Internal correlation ID.
                  -  **external** *(integer)*: External correlation ID.
               - **rocprofiler_dim3_t** *(object)*: Position of the workgroup in 3D grid.
                  -  **x** *(integer)*: Dimension x.
                  -  **y** *(integer)*: Dimension y.
                  -  **z** *(integer)*: Dimension z.
               -  **wave_in_group** *(integer)*: Wave position within the workgroup (0-31).
      - **buffer_records** *(object, required)*: Buffer record details.
         - **kernel_dispatch** *(array)*: Kernel dispatch records.
            - **Items** *(object)*
               - **size** *(integer, required)*: Size of the dispatch.
               - **kind** *(integer, required)*: Kind of the dispatch.
               - **operation** *(integer, required)*: Operation of the dispatch.
               - **thread_id** *(integer, required)*: Thread ID.
               - **correlation_id** *(object, required)*: Correlation ID information.
                  - **internal** *(integer, required)*: Internal correlation ID.
                  - **external** *(integer, required)*: External correlation ID.
               - **start_timestamp** *(integer, required)*: Start timestamp.
               - **end_timestamp** *(integer, required)*: End timestamp.
               - **dispatch_info** *(object, required)*: Dispatch information details.
                  - **size** *(integer, required)*: Size of the dispatch information.
                  - **agent_id** *(object, required)*: Agent ID information.
                     - **handle** *(integer, required)*: Handle of the agent.
                  - **queue_id** *(object, required)*: Queue ID information.
                     - **handle** *(integer, required)*: Handle of the queue.
                  - **kernel_id** *(integer, required)*: ID of the kernel.
                  - **dispatch_id** *(integer, required)*: ID of the dispatch.
                  - **private_segment_size** *(integer, required)*: Size of the private segment.
                  - **group_segment_size** *(integer, required)*: Size of the group segment.
                  - **workgroup_size** *(object, required)*: Workgroup size information.
                     - **x** *(integer, required)*: X dimension.
                     - **y** *(integer, required)*: Y dimension.
                     - **z** *(integer, required)*: Z dimension.
                  - **grid_size** *(object, required)*: Grid size information.
                     - **x** *(integer, required)*: X dimension.
                     - **y** *(integer, required)*: Y dimension.
                     - **z** *(integer, required)*: Z dimension.
         - **hip_api** *(array)*: HIP API records.
            - **Items** *(object)*
               - **size** *(integer, required)*: Size of the HIP API record.
               - **kind** *(integer, required)*: Kind of the HIP API.
               - **operation** *(integer, required)*: Operation of the HIP API.
               - **correlation_id** *(object, required)*: Correlation ID information.
                  - **internal** *(integer, required)*: Internal correlation ID.
                  - **external** *(integer, required)*: External correlation ID.
               - **start_timestamp** *(integer, required)*: Start timestamp.
               - **end_timestamp** *(integer, required)*: End timestamp.
               - **thread_id** *(integer, required)*: Thread ID.
         - **hsa_api** *(array)*: HSA API records.
            - **Items** *(object)*
               - **size** *(integer, required)*: Size of the HSA API record.
               - **kind** *(integer, required)*: Kind of the HSA API.
               - **operation** *(integer, required)*: Operation of the HSA API.
               - **correlation_id** *(object, required)*: Correlation ID information.
                  - **internal** *(integer, required)*: Internal correlation ID.
                  - **external** *(integer, required)*: External correlation ID.
               - **start_timestamp** *(integer, required)*: Start timestamp.
               - **end_timestamp** *(integer, required)*: End timestamp.
               - **thread_id** *(integer, required)*: Thread ID.
         - **marker_api** *(array)*: Marker (ROCTx) API records.
            - **Items** *(object)*
               - **size** *(integer, required)*: Size of the Marker API record.
               - **kind** *(integer, required)*: Kind of the Marker API.
               - **operation** *(integer, required)*: Operation of the Marker API.
               - **correlation_id** *(object, required)*: Correlation ID information.
                  - **internal** *(integer, required)*: Internal correlation ID.
                  - **external** *(integer, required)*: External correlation ID.
               - **start_timestamp** *(integer, required)*: Start timestamp.
               - **end_timestamp** *(integer, required)*: End timestamp.
               - **thread_id** *(integer, required)*: Thread ID.
         - **memory_copy** *(array)*: Async memory copy records.
            - **Items** *(object)*
               - **size** *(integer, required)*: Size of the Marker API record.
               - **kind** *(integer, required)*: Kind of the Marker API.
               - **operation** *(integer, required)*: Operation of the Marker API.
               - **correlation_id** *(object, required)*: Correlation ID information.
                  - **internal** *(integer, required)*: Internal correlation ID.
                  - **external** *(integer, required)*: External correlation ID.
               - **start_timestamp** *(integer, required)*: Start timestamp.
               - **end_timestamp** *(integer, required)*: End timestamp.
               - **thread_id** *(integer, required)*: Thread ID.
               - **dst_agent_id** *(object, required)*: Destination Agent ID.
                  - **handle** *(integer, required)*: Handle of the agent.
               - **src_agent_id** *(object, required)*: Source Agent ID.
                  - **handle** *(integer, required)*: Handle of the agent.
               - **bytes** *(integer, required)*: Bytes copied.
         - **memory_allocation** *(array)*: Memory allocation records.
            - **Items** *(object)*
               - **size** *(integer, required)*: Size of the Marker API record.
               - **kind** *(integer, required)*: Kind of the Marker API.
               - **operation** *(integer, required)*: Operation of the Marker API.
               - **correlation_id** *(object, required)*: Correlation ID information.
                  - **internal** *(integer, required)*: Internal correlation ID.
                  - **external** *(integer, required)*: External correlation ID.
               - **start_timestamp** *(integer, required)*: Start timestamp.
               - **end_timestamp** *(integer, required)*: End timestamp.
               - **thread_id** *(integer, required)*: Thread ID.
               - **agent_id** *(object, required)*: Agent ID.
                  - **handle** *(integer, required)*: Handle of the agent.
               - **address** *(string, required)*: Starting address of allocation.
               - **allocation_size** *(integer, required)*: Size of allocation.
         - **rocDecode_api** *(array)*: rocDecode API records.
            - **Items** *(object)*
               - **size** *(integer, required)*: Size of the rocDecode API record.
               - **kind** *(integer, required)*: Kind of the rocDecode API.
               - **operation** *(integer, required)*: Operation of the rocDecode API.
               - **correlation_id** *(object, required)*: Correlation ID information.
                  - **internal** *(integer, required)*: Internal correlation ID.
                  - **external** *(integer, required)*: External correlation ID.
               - **start_timestamp** *(integer, required)*: Start timestamp.
               - **end_timestamp** *(integer, required)*: End timestamp.
               - **thread_id** *(integer, required)*: Thread ID.
