.. meta::
  :description: Documentation of the installation, configuration, use of the ROCprofiler-SDK, and rocprofv3 command-line tool
  :keywords: ROCprofiler-SDK tool, ROCprofiler-SDK library, rocprofv3, rocprofv3 tool usage, Using rocprofv3, ROCprofiler-SDK command line tool, ROCprofiler-SDK CLI

.. _using-rocprofv3:

======================
Using rocprofv3
======================

``rocprofv3`` is a CLI tool that helps you quickly optimize applications and understand the low-level kernel details without requiring any modification in the source code.
It's backward compatible with its predecessor, ``rocprof``, and provides more features for application profiling with better accuracy.

The following sections demonstrate the use of ``rocprofv3`` for application tracing and kernel profiling using various command-line options.

``rocprofv3`` is installed with ROCm under ``/opt/rocm/bin``. To use the tool from anywhere in the system, export ``PATH`` variable:

.. code-block:: bash

    export PATH=$PATH:/opt/rocm/bin

Before you start tracing or profiling your HIP application using ``rocprofv3``, build the application using:

.. code-block:: bash

    cmake -B <build-directory> <source-directory> -DCMAKE_PREFIX_PATH=/opt/rocm
    cmake --build <build-directory> --target all --parallel <N>

Options
---------

Here is the sample of commonly used ``rocprofv3`` command-line options. Some options are used for application tracing and some for kernel profiling while the output control options control the presentation and redirection of the generated output.

.. list-table:: rocprofv3 options
  :header-rows: 1

  * - Option
    - Description
    - Use

  * - ``-i`` \| ``--input``
    - Specifies the input file. JSON and YAML formats support configuration of all command-line options whereas the text format only supports specifying HW counters.
    - Run Configuration

  * - ``-d`` \| ``--output-directory``
    - Specifies the path for the output files. Supports special keys: ``%hostname%``, ``%pid%``, ``%rank%``, etc.
    - Output control

  * - ``-o`` \| ``--output-file``
    - Specifies the name of the output file. Note that this name is appended to the default names (_api_trace or counter_collection.csv) of the generated files'. Supports special keys: ``%hostname%``, ``%pid%``, ``%rank%``, etc.
    - Output control

  * - ``--output-format``
    - For adding output format (supported formats: csv, json, pftrace)
    - Output control

  * - ``-r`` \| ``--runtime-trace``
    - Collects HIP (runtime), memory copy, memory allocation, marker, scratch memory, rocDecode, and kernel dispatch traces.
    - Application Tracing

  * - ``-s`` \| ``--sys-trace``
    - Collects HIP, HSA, memory copy, memory allocation, marker, scratch memory, rocDecode, and kernel dispatch traces.
    - Application Tracing

  * - ``--hip-trace``
    - Collects HIP runtime and compiler traces.
    - Application tracing

  * - ``--kernel-trace``
    - Collects kernel dispatch traces.
    - Application tracing

  * - ``--marker-trace``
    - Collects marker (ROC-TX) traces.
    - Application tracing

  * - ``--memory-copy-trace``
    - Collects memory copy traces.
    - Application tracing

  * - ``--memory-allocation-trace``
    - Collects memory allocation traces.
    - Application tracing

  * - ``--scratch-memory-trace``
    - Collects scratch memory operations traces.
    - Application tracing

  * - ``--rocdecode-trace``
    - Collects rocDecode API traces.
    - Application tracing

  * - ``--hsa-trace``
    - Collects HSA API traces.
    - Application tracing

  * - ``--hip-runtime-trace``
    - Collects HIP runtime API traces.
    - Application tracing

  * - ``--hsa-core-trace``
    - Collects HSA API traces (core API).
    - Application tracing

  * - ``--hsa-amd-trace``
    - Collects HSA API traces (AMD-extension API).
    - Application tracing

  * - ``--stats``
    - For Collecting statistics of enabled tracing types
    - Application tracing

  * - ``-p`` \| ``--summary``
    - Display summary of collected data
    - Application tracing

  * - ``--kernel-include-regex``
    - Include the kernels matching this filter.
    - Kernel Dispatch Counter Collection

  * - ``--kernel-exclude-regex``
    - Exclude the kernels matching this filter.
    - Kernel Dispatch Counter Collection

  * - ``--kernel-iteration-range``
    - Iteration range for each kernel that match the filter [start-stop].
    - Kernel Dispatch Counter Collection

  * - ``-L`` \| ``--list-avail``
    - List metrics for counter collection
    - List supported PC sampling configurations.

  * - ``-E`` \| ``--extra_counters``
    - Specifies the path to a YAML file containing extra counter definitions.
    - Kernel Dispatch Counter Collection

  * - ``-M`` \| ``--mangled-kernels``
    - Overrides the default demangling of kernel names.
    - Output control

  * - ``-T`` \| ``--truncate-kernels``
    - Truncates the demangled kernel names for improved readability.
    - Output control

  * - ``--output-format``
    - For adding output format (supported formats: csv, json, pftrace, otf2)
    - Output control

  * - ``--preload``
    - Libraries to prepend to LD_PRELOAD (usually for sanitizers)
    - Extension

  * - ``--perfetto-backend {inprocess,system}``
    - Perfetto data collection backend. 'system' mode requires starting traced and perfetto daemons
    - Extension

  * - ``--perfetto-buffer-size KB``
    - Size of buffer for perfetto output in KB. default: 1 GB
    - Extension

  * - ``--perfetto-buffer-fill-policy {discard,ring_buffer}``
    - Policy for handling new records when perfetto has reached the buffer limit
    - Extension

  * - ``--perfetto-shmem-size-hint KB``
    - Perfetto shared memory size hint in KB. default: 64 KB
    - Extension
    
  * - ``--pc-sampling-beta-enabled``
    - pc sampling support is in beta version
    - This flag set the ROCPROFILER_PC_SAMPLING_BETA_ENABLED environment variable

  * - ``--pc-sampling-method``
    - Type of PC Sampling, currently only host trap method is supported
    - PC Sampling Configurations

  * - ``--pc-sampling-unit``
    - The unit appropriate to the PC sampling type/method, currently only time unit is supported
    - PC Sampling Configurations

  * - ``--pc-sampling-interval``
    - Frequency at which PC samples are generated
    - PC Sampling Configurations

  * - ``--collection-period \| -p [(START_DELAY_TIME):(COLLECTION_TIME):(REPEAT), ...]``
    - The times are specified in seconds by default, but the unit can be changed using the `--collection-period-unit` or `-pu` option. Start Delay Time is the time in seconds before the collection begins, Collection Time is the duration in seconds for which data is collected, and Rate is the number of times the cycle is repeated. A repeat of 0 indicates that the cycle will repeat indefinitely. Users can specify multiple configurations, each defined by a triplet in the format `start_delay:collection_time:repeat`. For example, the command `-p 10:10:1 5:3:0` specifies two configurations: the first with a start delay of 10 seconds, a collection time of 10 seconds, and a repeat of 1 (the cycle will repeat once); the second with a start delay of 5 seconds, a collection time of 3 seconds, and a repeat of 0 (the cycle will repeat indefinitely).
    - Filtering Options

  * - ``--collection-period-unit {hour,min,sec,msec,usec,nsec}``
    - To change the unit used in `--collection-period` or `-p`, you can specify the desired unit using the `--collection-period-unit` option. The available units are `hour` for hours, `min` for minutes, `sec` for seconds, `msec` for milliseconds, `usec` for microseconds, and `nsec` for nanoseconds.
    - Filtering Options

To see exhaustive list of ``rocprofv3`` options, run:

.. code-block:: bash

    rocprofv3 --help

Application tracing
---------------------

Application tracing provides the big picture of a program’s execution by collecting data on the execution times of API calls and GPU commands, such as kernel execution, async memory copy, and barrier packets. This information can be used as the first step in the profiling process to answer important questions, such as how much percentage of time was spent on memory copy and which kernel took the longest time to execute.

To use ``rocprofv3`` for application tracing, run:

.. code-block:: bash

    rocprofv3 <tracing_option> -- <application_path>

HIP trace
+++++++++++

HIP trace comprises execution traces for the entire application at the HIP level. This includes HIP API functions and their asynchronous activities at the runtime level. In general, HIP APIs directly interact with the user program. It is easier to analyze HIP traces as you can directly map them to the program.

To trace HIP runtime APIs, use:

.. code-block:: bash

    rocprofv3 --hip-trace -- <application_path>

The above command generates a ``hip_api_trace.csv`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 238_hip_api_trace.csv

Here are the contents of ``hip_api_trace.csv`` file:

.. csv-table:: HIP runtime api trace
   :file: /data/hip_compile_trace.csv
   :widths: 10,10,10,10,10,20,20
   :header-rows: 1

To trace HIP compile time APIs, use:

.. code-block:: shell

    rocprofv3 --hip-compiler-trace -- <application_path>

The above command generates a ``hip_api_trace.csv`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 208_hip_api_trace.csv

Here are the contents of ``hip_api_trace.csv`` file:

.. csv-table:: HIP compile time api trace
   :file: /data/hip_compile_trace.csv
   :widths: 10,10,10,10,10,20,20
   :header-rows: 1

For the description of the fields in the output file, see :ref:`output-file-fields`.

HSA trace
+++++++++++++

The HIP runtime library is implemented with the low-level HSA runtime. HSA API tracing is more suited for advanced users who want to understand the application behavior at the lower level. In general, tracing at the HIP level is recommended for most users. You should use HSA trace only if you are familiar with HSA runtime.

HSA trace contains the start and end time of HSA runtime API calls and their asynchronous activities.

.. code-block:: bash

    rocprofv3 --hsa-trace -- <application_path>

The above command generates a ``hsa_api_trace.csv`` file prefixed with process ID. Note that the contents of this file have been truncated for demonstration purposes.

.. code-block:: shell

    $ cat 197_hsa_api_trace.csv

Here are the contents of ``hsa_api_trace.csv`` file:

.. csv-table:: HSA api trace
   :file: /data/hsa_api_trace.csv
   :widths: 10,10,10,10,10,20,20
   :header-rows: 1

For the description of the fields in the output file, see :ref:`output-file-fields`.

Marker trace
++++++++++++++

In certain situations, such as debugging performance issues in large-scale GPU programs, API-level tracing might be too fine-grained to provide a big picture of the program execution. In such cases, it is helpful to define specific tasks to be traced.

To specify the tasks for tracing, enclose the respective source code with the API calls provided by the ``ROCTx`` library. This process is also known as instrumentation. As the scope of code for instrumentation is defined using the enclosing API calls, it is called a range. A range is a programmer-defined task that has a well-defined start and end code scope. You can also refine the scope specified within a range using further nested ranges. ``rocprofv3`` also reports the timelines for these nested ranges.

Here is a list of useful APIs for code instrumentation.

- ``roctxMark``: Inserts a marker in the code with a message. Creating marks help you see when a line of code is executed.
- ``roctxRangeStart``: Starts a range. Different threads can start ranges.
- ``roctxRangePush``: Starts a new nested range.
- ``roctxRangePop``: Stops the current nested range.
- ``roctxRangeStop``: Stops the given range.
- ``roctxProfilerPause``: Request any currently running profiling tool that it should stop collecting data.
- ``roctxProfilerResume``: Request any currently running profiling tool that it should resume collecting data.
- ``roctxGetThreadId``: Retrieve a id value for the current thread which will be identical to the id value a profiling tool gets via `rocprofiler_get_thread_id(rocprofiler_thread_id_t*)`.
- ``roctxNameOsThread``: Current CPU OS thread to be labeled by the provided name in the output of the profiling tool.
- ``roctxNameHsaAgent``: Given HSA agent to be labeled by the provided name in the output of the profiling tool.
- ``roctxNameHipDevice``: Given HIP device id to be labeled by the provided name in the output of the profiling tool.
- ``roctxNameHipStream``: Given HIP stream to be labeled by the provided name in the output of the profiling tool.


.. note::
  To use ``rocprofv3`` for marker tracing, including and linking to old ROCTx works but it is recommended to switch to new ROCTx because
  it has been extended with new APIs.
  To use new ROCTx, please include header ``"rocprofiler-sdk-roctx/roctx.h"`` and link your application with ``librocprofiler-sdk-roctx.so``.
  Above list of APIs is not exhaustive. See public header file ``"rocprofiler-sdk-roctx/roctx.h"`` for full list.

See how to use ``ROCTx`` APIs in the MatrixTranspose application below:

.. code-block:: bash

    #include <rocprofiler-sdk-roctx/roctx.h>

    roctxMark("before hipLaunchKernel");
    int rangeId = roctxRangeStart("hipLaunchKernel range");
    roctxRangePush("hipLaunchKernel");

    // Launching kernel from host
    hipLaunchKernelGGL(matrixTranspose, dim3(WIDTH/THREADS_PER_BLOCK_X, WIDTH/THREADS_PER_BLOCK_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0,0,gpuTransposeMatrix,gpuMatrix, WIDTH);

    roctxMark("after hipLaunchKernel");

    // Memory transfer from device to host
    roctxRangePush("hipMemcpy");

    hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost);

    roctxRangePop();  // for "hipMemcpy"
    roctxRangePop();  // for "hipLaunchKernel"
    roctxRangeStop(rangeId);

To trace the API calls enclosed within the range, use:

.. code-block:: bash

    rocprofv3 --marker-trace -- <application_path>

Running the preceding command generates a ``marker_api_trace.csv`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 210_marker_api_trace.csv

Here are the contents of ``marker_api_trace.csv`` file:

.. csv-table:: Marker api trace
   :file: /data/marker_api_trace.csv
   :widths: 10,10,10,10,10,20,20
   :header-rows: 1

For the description of the fields in the output file, see :ref:`output-file-fields`.

``roctxProfilerPause`` and ``roctxProfilerResume`` can be used to hide the calls between them. This is useful when you want to hide the calls that are not relevant to your profiling session.

.. code-block:: bash

    #include <rocprofiler-sdk-roctx/roctx.h>

    // Memory transfer from host to device
    HIP_API_CALL(hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice));

    auto tid = roctx_thread_id_t{};
    roctxGetThreadId(&tid);
    roctxProfilerPause(tid);
    // Memory transfer that should be hidden by profiling tool
    HIP_API_CALL(
        hipMemcpy(gpuTransposeMatrix, gpuMatrix, NUM * sizeof(float), hipMemcpyDeviceToDevice));
    roctxProfilerResume(tid);

    // Lauching kernel from host
    hipLaunchKernelGGL(matrixTranspose,
                       dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                       0,
                       0,
                       gpuTransposeMatrix,
                       gpuMatrix,
                       WIDTH);

    // Memory transfer from device to host
    HIP_API_CALL(
        hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost));

.. code-block:: shell

    rocprofv3 --marker-trace --hip-trace -- <application_path>

    The above command generates a ``hip_api_trace.csv`` file prefixed with the process ID, which has only 2  `hipMemcpy` calls and the in between ``hipMemcpyDeviceToHost`` is hidden .

.. code-block:: shell

   "Domain","Function","Process_Id","Thread_Id","Correlation_Id","Start_Timestamp","End_Timestamp"
   "HIP_COMPILER_API","__hipRegisterFatBinary",1643920,1643920,1,320301257609216,320301257636427
   "HIP_COMPILER_API","__hipRegisterFunction",1643920,1643920,2,320301257650707,320301257678857
   "HIP_RUNTIME_API","hipGetDevicePropertiesR0600",1643920,1643920,4,320301258114239,320301337764472
   "HIP_RUNTIME_API","hipMalloc",1643920,1643920,5,320301338073823,320301338247374
   "HIP_RUNTIME_API","hipMalloc",1643920,1643920,6,320301338248284,320301338399595
   "HIP_RUNTIME_API","hipMemcpy",1643920,1643920,7,320301338410995,320301631549262
   "HIP_COMPILER_API","__hipPushCallConfiguration",1643920,1643920,10,320301632131175,320301632134215
   "HIP_COMPILER_API","__hipPopCallConfiguration",1643920,1643920,11,320301632137745,320301632139735
   "HIP_RUNTIME_API","hipLaunchKernel",1643920,1643920,12,320301632142615,320301632898289
   "HIP_RUNTIME_API","hipMemcpy",1643920,1643920,14,320301632901249,320301633934395
   "HIP_RUNTIME_API","hipFree",1643920,1643920,15,320301643320908,320301643511479
   "HIP_RUNTIME_API","hipFree",1643920,1643920,16,320301643512629,320301643585639

Kernel Rename
++++++++++++++

To rename kernels with their enclosing roctxRangePush/roctxRangePop message. Known as --roctx-rename in earlier rocprof versions.

See how to use ``--kernel-rename`` option with help of below code snippet:

.. code-block:: bash

    #include <rocprofiler-sdk-roctx/roctx.h>

    roctxRangePush("HIP_Kernel-1");

    // Launching kernel from host
    hipLaunchKernelGGL(matrixTranspose, dim3(WIDTH/THREADS_PER_BLOCK_X, WIDTH/THREADS_PER_BLOCK_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0,0,gpuTransposeMatrix,gpuMatrix, WIDTH);

    // Memory transfer from device to host
    roctxRangePush("hipMemCpy-DeviceToHost");

    hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost);

    roctxRangePop();  // for "hipMemcpy"
    roctxRangePop();  // for "hipLaunchKernel"
    roctxRangeStop(rangeId);

To rename the kernel , use:

.. code-block:: bash

    rocprofv3 --marker-trace --kernel-rename -- <application_path>

The above command generates a ``marker-trace`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 210_marker_api_trace.csv
   "Domain","Function","Process_Id","Thread_Id","Correlation_Id","Start_Timestamp","End_Timestamp"
   "MARKER_CORE_API","roctxGetThreadId",315155,315155,2,58378843928406,58378843930247
   "MARKER_CONTROL_API","roctxProfilerPause",315155,315155,3,58378844627184,58378844627502
   "MARKER_CONTROL_API","roctxProfilerResume",315155,315155,4,58378844638601,58378844639267
   "MARKER_CORE_API","pre-kernel-launch",315155,315155,5,58378844641787,58378844641787
   "MARKER_CORE_API","post-kernel-launch",315155,315155,6,58378844936586,58378844936586
   "MARKER_CORE_API","memCopyDth",315155,315155,7,58378844938371,58378851383270
   "MARKER_CORE_API","HIP_Kernel-1",315155,315155,1,58378526575735,58378851384485


Kokkos Trace
++++++++++++++

rocprofv3 has a built-in `Kokkos Tools library <https://github.com/kokkos/kokkos-tools>`_ support to trace Kokkos API calls. `Kokkos <https://github.com/kokkos/kokkos>`_ is a C++ library for writing performance portable applications. It is used in many scientific applications to write performance portable code that can run on CPUs, GPUs, and other accelerators.
rocprofv3 loads a built-in Kokkos tools library which emits roctx ranges with the labels passed through the API, e.g. Kokkos::parallel_for(“MyParallelForLabel”, …); will internally calls for roctxRangePush and enables the kernel renaming option so that the highly templated kernel names are replaced by the Kokkos labels.
To enable built-in marker support, use the ``kokkos-trace`` option. Internally this option enables ``marker-trace`` and ``kernel-rename``.:

.. code-block:: bash

    rocprofv3 --kokkos-trace -- <application_path>

The above command generates a ``marker-trace`` file prefixed with the process ID.

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

    rocprofv3 --kernel-trace -- <application_path>

The above command generates a ``kernel_trace.csv`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 199_kernel_trace.csv

Here are the contents of ``kernel_trace.csv`` file:

.. csv-table:: Kernel trace
   :file: /data/kernel_trace.csv
   :widths: 10,10,10,10,10,10,10,10,20,20,10,10,10,10,10,10,10,10
   :header-rows: 1

For the description of the fields in the output file, see :ref:`output-file-fields`.

Memory copy trace
+++++++++++++++++++

To trace memory moves across the application, use:

.. code-block:: shell

    rocprofv3 –-memory-copy-trace -- <application_path>

The above command generates a ``memory_copy_trace.csv`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 197_memory_copy_trace.csv

Here are the contents of ``memory_copy_trace.csv`` file:

.. csv-table:: Memory copy trace
   :file: /data/memory_copy_trace.csv
   :widths: 10,10,10,10,10,20,20
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
JSON outputs.

To trace memory allocations during the application run, use:

.. code-block:: shell

    rocprofv3 –-memory-allocation-trace -- < app_path >

The above command generates a ``memory_allocation_trace.csv`` file prefixed with the process ID.

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

This is a short-hand option which attempts to target the most relevant tracing options for a standard user by
excluding tracing the HSA runtime API and HIP compiler API.

The HSA runtime API is excluded because it is a lower-level API upon which HIP and OpenMP target are built and
thus, tends to be an implementation detail not relevant to most users. The HIP compiler API is excluded
because these are functions which are automatically inserted during HIP compilation and thus, also tend to be
implementation details which are not relevant to most users.

At present, `--runtime-trace` enables tracing the HIP runtime API, the marker API, kernel dispatches, and
memory operations (copies and scratch).

.. code-block:: shell

    rocprofv3 –-runtime-trace -- <application_path>

Running the above command generates ``hip_api_trace.csv``, ``kernel_trace.csv``, ``memory_copy_trace.csv``, ``scratch_memory_trace.csv``, ``memory_allocation_trace.csv``, and ``marker_api_trace.csv`` (if ``ROCTx`` APIs are specified in the application) files prefixed with the process ID.

System trace
++++++++++++++

This is an all-inclusive option to collect all the above-mentioned traces.

.. code-block:: shell

    rocprofv3 –-sys-trace -- <application_path>

Running the above command generates ``hip_api_trace.csv``, ``hsa_api_trace.csv``, ``kernel_trace.csv``, ``memory_copy_trace.csv``, ``memory_allocation_trace.csv``, and ``marker_api_trace.csv`` (if ``ROCTx`` APIs are specified in the application) files prefixed with the process ID.

Scratch memory trace
++++++++++++++++++++++

This option collects scratch memory operation's traces. Scratch is an address space on AMD GPUs, which is roughly equivalent to the `local memory` in NVIDIA CUDA. The `local memory` in CUDA is a thread-local global memory with interleaved addressing, which is used for register spills or stack space. With this option, you can trace when the ``rocr`` runtime allocates, frees, and tries to reclaim scratch memory.

.. code-block:: shell

    rocprofv3 --scratch-memory-trace -- <application_path>


RCCL trace
++++++++++++

`RCCL <https://github.com/ROCm/rccl>`_ (pronounced "Rickle") is a stand-alone library of standard collective communication routines for GPUs. This option traces those communication routines.

.. code-block:: shell

    rocprofv3 --rccl-trace -- <application_path>

The above command generates a ``rccl_api_trace`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 197_rccl_api_trace.csv

Here are the contents of ``rccl_api_trace.csv`` file:

.. csv-table:: RCCL trace
   :file: /data/rccl_trace.csv
   :widths: 10,10,10,10,10,20,20
   :header-rows: 1

rocDecode trace
++++++++++++++++

`rocDecode <https://github.com/ROCm/rocDecode>`_ is a high-performance video decode SDK for AMD GPUs. This option traces the rocDecode API.

.. code-block:: shell

    rocprofv3 --rocdecode-trace -- <application_path>

The above command generates a ``rocdecode_api_trace`` file prefixed with the process ID.

.. code-block:: shell

    $ cat 41688_rocdecode_api_trace.csv

Here are the contents of ``rocdecode_api_trace.csv`` file:

.. csv-table:: rocDecode trace
   :file: /data/rocdecode_api_trace.csv
   :widths: 10,10,10,10,10,20,20
   :header-rows: 1

Post-processing tracing options
++++++++++++++++++++++++++++++++

1. Stats
+++++++++

This option collects statistics for the enabled tracing types. For example, to collect statistics of HIP APIs, when HIP trace is enabled.
A higher percentage in statistics can help user focus on the API/function that has taken the most time:

.. code-block:: shell

    rocprofv3 --stats --hip-trace  -- <application_path>

The above command generates a ``hip_api_stats.csv``, ``domain_stats.csv`` and ``hip_api_trace.csv`` file prefixed with the process ID.

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

2. Summary
+++++++++++

Output single summary of tracing data at the conclusion of the profiling session

.. code-block:: shell

   rocprofv3 -S --hip-trace -- <application_path>

.. image:: /data/rocprofv3_summary.png


2.1 Summary per domain
++++++++++++++++++++++

Outputs the summary of each tracing domain at the end of profiling session.

.. code-block:: shell

    rocprofv3 -D --hsa-trace --hip-trace  -- <application_path>

The above command generates a ``hip_trace.csv``, ``hsa_trace.csv`` file prefixed with the process ID along with the summary of each domain at the terminal.

2.2 Summary groups
+++++++++++++++++++

Users can create a summary of multiple domains by specifying the domain names in the command line. The summary groups are separated by a pipe (|) symbol.
To create a summary for ``MEMORY_COPY`` domains, use:

.. code-block:: shell

   rocprofv3 --summary-groups MEMORY_COPY --sys-trace  -- <application_path>

.. image:: /data/rocprofv3_memcpy_summary.png


To create a summary for ``MEMORY_COPY`` and ``HIP_API`` domains, use:

.. code-block:: shell

   rocprofv3 --summary-groups 'MEMORY_COPY|HIP_API' --sys-trace -- <application_path>

.. image:: /data/rocprofv3_hip_memcpy_summary.png


Kernel profiling
-------------------

The application tracing functionality allows you to evaluate the duration of kernel execution but is of little help in providing insight into kernel execution details. The kernel profiling functionality allows you to select kernels for profiling and choose the basic counters or derived metrics to be collected for each kernel execution, thus providing a greater insight into kernel execution.

For a comprehensive list of counters available on MI200, see `MI200 performance counters and metrics <https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html>`_.

Input file
++++++++++++

To collect the desired basic counters or derived metrics or tracing, mention them in an input file. The input file could be in text (.txt), yaml (.yaml/.yml), or JSON (.json) format.

In the input text file, the line consisting of the counter or metric names must begin with ``pmc``.
The number of basic counters or derived metrics that can be collected in one run of profiling are limited by the GPU hardware resources. If too many counters or metrics are selected, the kernels need to be executed multiple times to collect them. For multi-pass execution, include multiple ``pmc`` rows in the input file. Counters or metrics in each ``pmc`` row can be collected in each application run.

The JSON and YAML files supports all the command line options and it can be used to configure both tracing and profiling. The input file has an array of profiling/tracing configurations called jobs. Each job is used to configure profiling/tracing for an application execution. The input schema of these files is given below.

Properties
++++++++++++

-  **``jobs``** *(array)*: rocprofv3 input data per application run.

   -  **Items** *(object)*: data for rocprofv3.

      -  **``pmc``** *(array)*: list of counters to collect.
      -  **``kernel_include_regex``** *(string)*: Include the kernels
         matching this filter.
      -  **``kernel_exclude_regex``** *(string)*: Exclude the kernels
         matching this filter.
      -  **``kernel_iteration_range``** *(string)*: Iteration range for
         each kernel that match the filter [start-stop].
      -  **``hip_trace``** *(boolean)*: For Collecting HIP Traces
         (runtime + compiler).
      -  **``hip_runtime_trace``** *(boolean)*: For Collecting HIP
         Runtime API Traces.
      -  **``hip_compiler_trace``** *(boolean)*: For Collecting HIP
         Compiler generated code Traces.
      -  **``marker_trace``** *(boolean)*: For Collecting Marker (ROCTx)
         Traces.
      -  **``kernel_trace``** *(boolean)*: For Collecting Kernel
         Dispatch Traces.
      -  **``memory_copy_trace``** *(boolean)*: For Collecting Memory
         Copy Traces.
      -  **``memory_allocation_trace``** *(boolean)*: For Collecting Memory
         Allocation Traces.
      -  **``scratch_memory_trace``** *(boolean)*: For Collecting
         Scratch Memory operations Traces.
      -  **``stats``** *(boolean)*: For Collecting statistics of enabled
         tracing types.
      -  **``hsa_trace``** *(boolean)*: For Collecting HSA Traces (core
         + amd + image + finalizer).
      -  **``hsa_core_trace``** *(boolean)*: For Collecting HSA API
         Traces (core API).
      -  **``hsa_amd_trace``** *(boolean)*: For Collecting HSA API
         Traces (AMD-extension API).
      -  **``hsa_finalize_trace``** *(boolean)*: For Collecting HSA API
         Traces (Finalizer-extension API).
      -  **``hsa_image_trace``** *(boolean)*: For Collecting HSA API
         Traces (Image-extension API).
      -  **``sys_trace``** *(boolean)*: For Collecting HIP, HSA, Marker
         (ROCTx), Memory copy, Memory allocation, Scratch memory, and
         Kernel dispatch traces.
      -  **``mangled_kernels``** *(boolean)*: Do not demangle the kernel
         names.
      -  **``truncate_kernels``** *(boolean)*: Truncate the demangled
         kernel names.
      -  **``output_file``** *(string)*: For the output file name.
      -  **``output_directory``** *(string)*: For adding output path
         where the output files will be saved.
      -  **``output_format``** *(array)*: For adding output format
         (supported formats: csv, json, pftrace, otf2).
      -  **``list_metrics``** *(boolean)*: List the metrics.
      -  **``log_level``** *(string)*: fatal, error, warning, info,
         trace.
      -  **``preload``** *(array)*: Libraries to prepend to LD_PRELOAD
         (usually for sanitizers).
      -  **``pc_sampling_unit``** *(string)*: pc sampling unit.
      -  **``pc_sampling_method``** *(string)*: pc sampling method.
      -  **``pc_sampling_interval``** *(integer)*: pc sampling interval.
      -  **``pc-sampling-beta-enabled``** *(boolean)*: enable pc
         sampling support; beta version.

.. code-block:: shell

    $ cat input.txt

    pmc: GPUBusy SQ_WAVES
    pmc: GRBM_GUI_ACTIVE

.. code-block:: shell

    $ cat input.json

    {
      "jobs": [
         {
            "pmc": ["SQ_WAVES", "GRBM_COUNT", "GRBM_GUI_ACTIVE"]
         },
         {
            "pmc": ["FETCH_SIZE", "WRITE_SIZE"],
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

.. code-block:: shell

    $ cat input.yaml

  jobs:
    - pmc:
        - SQ_WAVES
        - GRBM_COUNT
        - GRBM_GUI_ACTIVE
    - pmc:
        - FETCH_SIZE
        - WRITE_SIZE


Command-line
+++++++++++++

Desired counters can now be collected as ``command-line`` option as well.

To supply the counters via ``command-line`` options, use:

.. code-block:: shell

   rocprofv3 --pmc SQ_WAVES GRBM_COUNT GRBM_GUI_ACTIVE -- <application_path>

.. note::
   1. Please note that more than 1 counters should be separated by a space or a comma.
   2. Job will fail if entire set of counters cannot be collected in single pass

Extra-counters
++++++++++++++++

Counters with custom definitions can be defined through an extra_counters.yaml
file using the ``command-line`` option.

To supply the extra counters via ``command-line`` options, use:

.. code-block:: shell

   rocprofv3 -E <path-to-extra_counters.yaml> --pmc <custom_metric> -- <app_relative_path>

Kernel profiling output
+++++++++++++++++++++++++

To supply the input file for kernel profiling, use:

.. code-block:: shell

    rocprofv3 -i input.txt -- <application_path>

Running the above command generates a ``./pmc_n/counter_collection.csv`` file prefixed with the process ID. For each ``pmc`` row, a directory ``pmc_n`` containing a ``counter_collection.csv`` file is generated, where n = 1 for the first row and so on.

In case of JSON or YAML input file, for each job, a directory ``pass_n`` containing a ``counter_collection.csv`` file is generated where n = 1...N jobs.

Each row of the CSV file is an instance of kernel execution. Here is a truncated version of the output file from ``pmc_1``:

.. code-block:: shell

    $ cat pmc_1/218_counter_collection.csv

Here are the contents of ``counter_collection.csv`` file:

.. csv-table:: Counter collection
   :file: /data/counter_collection.csv
   :widths: 10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10
   :header-rows: 1

For the description of the fields in the output file, see :ref:`output-file-fields`.

Kernel filtering
+++++++++++++++++

rocprofv3 supports kernel filtering in case of profiling. A kernel filter is a set of a regex string (to include the kernels matching this filter), a regex string (to exclude the kernels matching this filter),
and an iteration range (set of iterations of the included kernels). If the iteration range is not provided then all iterations of the included kernels are profiled.

.. code-block:: shell

    $ cat input.yml
    jobs:
        - pmc: [SQ_WAVES]
        kernel_include_regex: "divide"
        kernel_exclude_regex: ""
        kernel_iteration_range: "[1, 2, [5-8]]"

Agent info
++++++++++++

.. note::
  All tracing and counter collection options generate an additional ``agent_info.csv`` file prefixed with the process ID.

The ``agent_info.csv`` file contains information about the CPU or GPU the kernel runs on.

.. code-block:: shell

    $ cat 238_agent_info.csv

    "Node_Id","Logical_Node_Id","Agent_Type","Cpu_Cores_Count","Simd_Count","Cpu_Core_Id_Base","Simd_Id_Base","Max_Waves_Per_Simd","Lds_Size_In_Kb","Gds_Size_In_Kb","Num_Gws","Wave_Front_Size","Num_Xcc","Cu_Count","Array_Count","Num_Shader_Banks","Simd_Arrays_Per_Engine","Cu_Per_Simd_Array","Simd_Per_Cu","Max_Slots_Scratch_Cu","Gfx_Target_Version","Vendor_Id","Device_Id","Location_Id","Domain","Drm_Render_Minor","Num_Sdma_Engines","Num_Sdma_Xgmi_Engines","Num_Sdma_Queues_Per_Engine","Num_Cp_Queues","Max_Engine_Clk_Ccompute","Max_Engine_Clk_Fcompute","Sdma_Fw_Version","Fw_Version","Capability","Cu_Per_Engine","Max_Waves_Per_Cu","Family_Id","Workgroup_Max_Size","Grid_Max_Size","Local_Mem_Size","Hive_Id","Gpu_Id","Workgroup_Max_Dim_X","Workgroup_Max_Dim_Y","Workgroup_Max_Dim_Z","Grid_Max_Dim_X","Grid_Max_Dim_Y","Grid_Max_Dim_Z","Name","Vendor_Name","Product_Name","Model_Name"
    0,0,"CPU",24,0,0,0,0,0,0,0,0,1,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3800,0,0,0,0,0,0,23,0,0,0,0,0,0,0,0,0,0,0,"AMD Ryzen 9 3900X 12-Core Processor","CPU","AMD Ryzen 9 3900X 12-Core Processor",""
    1,1,"GPU",0,256,0,2147487744,10,64,0,64,64,1,64,4,4,1,16,4,32,90000,4098,26751,12032,0,128,2,0,2,24,3800,1630,432,440,138420864,16,40,141,1024,4294967295,0,0,64700,1024,1024,1024,4294967295,4294967295,4294967295,"gfx900","AMD","Radeon RX Vega","vega10"

Kernel filtering
+++++++++++++++++

Kernel filtering allows you to filter the kernel profiling output based on the kernel name by specifying regex strings in the input file. To include kernel names matching the regex string in the kernel profiling output, use ``kernel_include_regex``. To exclude the kernel names matching the regex string from the kernel profiling output, use ``kernel_exclude_regex``.
You can also specify an iteration range for set of iterations of the included kernels. If the iteration range is not specified, then all iterations of the included kernels are profiled.

Here is an input file with kernel filters:

.. code-block:: shell

    $ cat input.yml
    jobs:
        - pmc: [SQ_WAVES]
        kernel_include_regex: "divide"
        kernel_exclude_regex: ""

To collect counters for the kernels matching the filters specified in the preceding input file, run:

.. code-block:: shell

    rocprofv3 -i input.yml -- <application_path>

    $ cat pass_1/312_counter_collection.csv
    "Correlation_Id","Dispatch_Id","Agent_Id","Queue_Id","Process_Id","Thread_Id","Grid_Size","Kernel_Name","Workgroup_Size","LDS_Block_Size","Scratch_Size","VGPR_Count","SGPR_Count","Counter_Name","Counter_Value","Start_Timestamp","End_Timestamp"
    4,4,1,1,36499,36499,1048576,"divide_kernel(float*, float const*, float const*, int, int)",64,0,0,12,16,"SQ_WAVES",16384,2228955885095594,2228955885119754
    8,8,1,2,36499,36499,1048576,"divide_kernel(float*, float const*, float const*, int, int)",64,0,0,12,16,"SQ_WAVES",16384,2228955885095594,2228955885119754
    12,12,1,3,36499,36499,1048576,"divide_kernel(float*, float const*, float const*, int, int)",64,0,0,12,16,"SQ_WAVES",16384,2228955892986914,2228955893006114
    16,16,1,4,36499,36499,1048576,"divide_kernel(float*, float const*, float const*, int, int)",64,0,0,12,16,"SQ_WAVES",16384,2228955892986914,2228955893006114

.. _output-file-fields:

Output file fields
-----------------------

The following table lists the various fields or the columns in the output CSV files generated for application tracing and kernel profiling:

.. list-table:: output file fields
  :header-rows: 1

  * - Field
    - Description

  * - Agent_Id
    - GPU identifier to which the kernel was submitted.

  * - Correlation_Id
    - Unique identifier for correlation between HIP and HSA async calls during activity tracing.

  * - Start_Timestamp
    - Begin time in nanoseconds (ns) when the kernel begins execution.

  * - End_Timestamp
    - End time in ns when the kernel finishes execution.

  * - Queue_Id
    - ROCm queue unique identifier to which the kernel was submitted.

  * - Private_Segment_Size
    - The amount of memory required in bytes for the combined private, spill, and arg segments for a work item.

  * - Group_Segment_Size
    - The group segment memory required by a workgroup in bytes. This does not include any dynamically allocated group segment memory that may be added when the kernel is dispatched.

  * - Workgroup_Size
    - Size of the workgroup as declared by the compute shader.

  * - Workgroup_Size_n
    - Size of the workgroup in the nth dimension as declared by the compute shader, where n = X, Y, or Z.

  * - Grid_Size
    - Number of thread blocks required to launch the kernel.

  * - Grid_Size_n
    - Number of thread blocks in the nth dimension required to launch the kernel, where n = X, Y, or Z.

  * - LDS_Block_Size
    - Thread block size for the kernel's Local Data Share (LDS) memory.

  * - Scratch_Size
    - Kernel’s scratch memory size.

  * - SGPR_Count
    - Kernel's Scalar General Purpose Register (SGPR) count.

  * - VGPR_Count
    - Kernel's Vector General Purpose Register (VGPR) count.

Output formats
----------------

``rocprofv3`` supports the following output formats:

- CSV (Default)
- JSON (Custom format for programmatic analysis only)
- PFTrace (Perfetto trace for visualization with Perfetto)
- OTF2 (Open Trace Format for visualization with compatible third party tools)

You can specify the output format using the ``--output-format`` command-line option. Format selection is case-insensitive
and multiple output formats are supported. For example: ``--output-format json`` enables JSON output exclusively whereas
``--output-format csv json pftrace otf2`` enables all four output formats for the run.

For .pftrace trace visualization, use the PFTrace format and open the trace in `ui.perfetto.dev <https://ui.perfetto.dev/>`_.

For .otf2 trace visualization, open the trace in `vampir.eu <https://vampir.eu/>`_ or any supported visualizer.

.. note::
  For large trace files(> 10GB), its recommended to use otf2 format.

JSON output schema
++++++++++++++++++++

``rocprofv3`` supports a **custom** JSON output format designed for programmatic analysis and **NOT** for visualization.
The schema is optimized for size while factoring in usability. The Perfetto UI does not accept this JSON output format produced by rocprofv3.
Perfetto is dropping support for the JSON Chrome tracing format in favor of the binary Perfetto protobuf format (.pftrace extension), which is supported by rocprofv3.
You can generate the JSON output using ``--output-format json`` command-line option.

Properties
++++++++++++

- **`rocprofiler-sdk-tool`** `(array)`: rocprofv3 data per process (each element represents a process).
   - **Items** `(object)`: Data for rocprofv3.
      - **`metadata`** `(object, required)`: Metadata related to the profiler session.
         - **`pid`** `(integer, required)`: Process ID.
         - **`init_time`** `(integer, required)`: Initialization time in nanoseconds.
         - **`fini_time`** `(integer, required)`: Finalization time in nanoseconds.
      - **`agents`** `(array, required)`: List of agents.
         - **Items** `(object)`: Data for an agent.
            - **`size`** `(integer, required)`: Size of the agent data.
            - **`id`** `(object, required)`: Identifier for the agent.
               - **`handle`** `(integer, required)`: Handle for the agent.
            - **`type`** `(integer, required)`: Type of the agent.
            - **`cpu_cores_count`** `(integer)`: Number of CPU cores.
            - **`simd_count`** `(integer)`: Number of SIMD units.
            - **`mem_banks_count`** `(integer)`: Number of memory banks.
            - **`caches_count`** `(integer)`: Number of caches.
            - **`io_links_count`** `(integer)`: Number of I/O links.
            - **`cpu_core_id_base`** `(integer)`: Base ID for CPU cores.
            - **`simd_id_base`** `(integer)`: Base ID for SIMD units.
            - **`max_waves_per_simd`** `(integer)`: Maximum waves per SIMD.
            - **`lds_size_in_kb`** `(integer)`: Size of LDS in KB.
            - **`gds_size_in_kb`** `(integer)`: Size of GDS in KB.
            - **`num_gws`** `(integer)`: Number of GWS (global work size).
            - **`wave_front_size`** `(integer)`: Size of the wave front.
            - **`num_xcc`** `(integer)`: Number of XCC (execution compute units).
            - **`cu_count`** `(integer)`: Number of compute units (CUs).
            - **`array_count`** `(integer)`: Number of arrays.
            - **`num_shader_banks`** `(integer)`: Number of shader banks.
            - **`simd_arrays_per_engine`** `(integer)`: SIMD arrays per engine.
            - **`cu_per_simd_array`** `(integer)`: CUs per SIMD array.
            - **`simd_per_cu`** `(integer)`: SIMDs per CU.
            - **`max_slots_scratch_cu`** `(integer)`: Maximum slots for scratch CU.
            - **`gfx_target_version`** `(integer)`: GFX target version.
            - **`vendor_id`** `(integer)`: Vendor ID.
            - **`device_id`** `(integer)`: Device ID.
            - **`location_id`** `(integer)`: Location ID.
            - **`domain`** `(integer)`: Domain identifier.
            - **`drm_render_minor`** `(integer)`: DRM render minor version.
            - **`num_sdma_engines`** `(integer)`: Number of SDMA engines.
            - **`num_sdma_xgmi_engines`** `(integer)`: Number of SDMA XGMI engines.
            - **`num_sdma_queues_per_engine`** `(integer)`: Number of SDMA queues per engine.
            - **`num_cp_queues`** `(integer)`: Number of CP queues.
            - **`max_engine_clk_ccompute`** `(integer)`: Maximum engine clock for compute.
            - **`max_engine_clk_fcompute`** `(integer)`: Maximum engine clock for F compute.
            - **`sdma_fw_version`** `(object)`: SDMA firmware version.
               - **`uCodeSDMA`** `(integer, required)`: SDMA microcode version.
               - **`uCodeRes`** `(integer, required)`: Reserved microcode version.
            - **`fw_version`** `(object)`: Firmware version.
               - **`uCode`** `(integer, required)`: Microcode version.
               - **`Major`** `(integer, required)`: Major version.
               - **`Minor`** `(integer, required)`: Minor version.
               - **`Stepping`** `(integer, required)`: Stepping version.
            - **`capability`** `(object, required)`: Agent capability flags.
               - **`HotPluggable`** `(integer, required)`: Hot pluggable capability.
               - **`HSAMMUPresent`** `(integer, required)`: HSAMMU present capability.
               - **`SharedWithGraphics`** `(integer, required)`: Shared with graphics capability.
               - **`QueueSizePowerOfTwo`** `(integer, required)`: Queue size is power of two.
               - **`QueueSize32bit`** `(integer, required)`: Queue size is 32-bit.
               - **`QueueIdleEvent`** `(integer, required)`: Queue idle event.
               - **`VALimit`** `(integer, required)`: VA limit.
               - **`WatchPointsSupported`** `(integer, required)`: Watch points supported.
               - **`WatchPointsTotalBits`** `(integer, required)`: Total bits for watch points.
               - **`DoorbellType`** `(integer, required)`: Doorbell type.
               - **`AQLQueueDoubleMap`** `(integer, required)`: AQL queue double map.
               - **`DebugTrapSupported`** `(integer, required)`: Debug trap supported.
               - **`WaveLaunchTrapOverrideSupported`** `(integer, required)`: Wave launch trap override supported.
               - **`WaveLaunchModeSupported`** `(integer, required)`: Wave launch mode supported.
               - **`PreciseMemoryOperationsSupported`** `(integer, required)`: Precise memory operations supported.
               - **`DEPRECATED_SRAM_EDCSupport`** `(integer, required)`: Deprecated SRAM EDC support.
               - **`Mem_EDCSupport`** `(integer, required)`: Memory EDC support.
               - **`RASEventNotify`** `(integer, required)`: RAS event notify.
               - **`ASICRevision`** `(integer, required)`: ASIC revision.
               - **`SRAM_EDCSupport`** `(integer, required)`: SRAM EDC support.
               - **`SVMAPISupported`** `(integer, required)`: SVM API supported.
               - **`CoherentHostAccess`** `(integer, required)`: Coherent host access.
               - **`DebugSupportedFirmware`** `(integer, required)`: Debug supported firmware.
               - **`Reserved`** `(integer, required)`: Reserved field.
      - **`counters`** `(array, required)`: Array of counter objects.
         - **Items** `(object)`
            - **`agent_id`** *(object, required)*: Agent ID information.
               - **`handle`** *(integer, required)*: Handle of the agent.
            - **`id`** *(object, required)*: Counter ID information.
               - **`handle`** *(integer, required)*: Handle of the counter.
            - **`is_constant`** *(integer, required)*: Indicator if the counter value is constant.
            - **`is_derived`** *(integer, required)*: Indicator if the counter value is derived.
            - **`name`** *(string, required)*: Name of the counter.
            - **`description`** *(string, required)*: Description of the counter.
            - **`block`** *(string, required)*: Block information of the counter.
            - **`expression`** *(string, required)*: Expression of the counter.
            - **`dimension_ids`** *(array, required)*: Array of dimension IDs.
               - **Items** *(integer)*: Dimension ID.
      - **`strings`** *(object, required)*: String records.
         - **`callback_records`** *(array)*: Callback records.
            - **Items** *(object)*
               - **`kind`** *(string, required)*: Kind of the record.
               - **`operations`** *(array, required)*: Array of operations.
                  - **Items** *(string)*: Operation.
         - **`buffer_records`** *(array)*: Buffer records.
            - **Items** *(object)*
               - **`kind`** *(string, required)*: Kind of the record.
               - **`operations`** *(array, required)*: Array of operations.
                  - **Items** *(string)*: Operation.
         - **`marker_api`** *(array)*: Marker API records.
            - **Items** *(object)*
               - **`key`** *(integer, required)*: Key of the record.
               - **`value`** *(string, required)*: Value of the record.
         - **`counters`** *(object)*: Counter records.
            - **`dimension_ids`** *(array, required)*: Array of dimension IDs.
               - **Items** *(object)*
                  - **`id`** *(integer, required)*: Dimension ID.
                  - **`instance_size`** *(integer, required)*: Size of the instance.
                  - **`name`** *(string, required)*: Name of the dimension.
         -  **``pc_sample_instructions``** *(array)*: Array of decoded
            instructions matching sampled PCs from pc_sample_host_trap
            section.
         -  **``pc_sample_comments``** *(array)*: Comments matching
            assembly instructions from pc_sample_instructions array. If
            debug symbols are available, comments provide instructions
            to source-line mapping. Otherwise, a comment is an empty
            string.
      - **`code_objects`** *(array, required)*: Code object records.
         - **Items** *(object)*
            - **`size`** *(integer, required)*: Size of the code object.
            - **`code_object_id`** *(integer, required)*: ID of the code object.
            - **`rocp_agent`** *(object, required)*: ROCP agent information.
               - **`handle`** *(integer, required)*: Handle of the ROCP agent.
            - **`hsa_agent`** *(object, required)*: HSA agent information.
               - **`handle`** *(integer, required)*: Handle of the HSA agent.
            - **`uri`** *(string, required)*: URI of the code object.
            - **`load_base`** *(integer, required)*: Base address for loading.
            - **`load_size`** *(integer, required)*: Size for loading.
            - **`load_delta`** *(integer, required)*: Delta for loading.
            - **`storage_type`** *(integer, required)*: Type of storage.
            - **`memory_base`** *(integer, required)*: Base address for memory.
            - **`memory_size`** *(integer, required)*: Size of memory.
      - **`kernel_symbols`** *(array, required)*: Kernel symbol records.
         - **Items** *(object)*
            - **`size`** *(integer, required)*: Size of the kernel symbol.
            - **`kernel_id`** *(integer, required)*: ID of the kernel.
            - **`code_object_id`** *(integer, required)*: ID of the code object.
            - **`kernel_name`** *(string, required)*: Name of the kernel.
            - **`kernel_object`** *(integer, required)*: Object of the kernel.
            - **`kernarg_segment_size`** *(integer, required)*: Size of the kernarg segment.
            - **`kernarg_segment_alignment`** *(integer, required)*: Alignment of the kernarg segment.
            - **`group_segment_size`** *(integer, required)*: Size of the group segment.
            - **`private_segment_size`** *(integer, required)*: Size of the private segment.
            - **`formatted_kernel_name`** *(string, required)*: Formatted name of the kernel.
            - **`demangled_kernel_name`** *(string, required)*: Demangled name of the kernel.
            - **`truncated_kernel_name`** *(string, required)*: Truncated name of the kernel.
      - **`callback_records`** *(object, required)*: Callback record details.
         - **`counter_collection`** *(array)*: Counter collection records.
            - **Items** *(object)*
               - **`dispatch_data`** *(object, required)*: Dispatch data details.
                  - **`size`** *(integer, required)*: Size of the dispatch data.
                  - **`correlation_id`** *(object, required)*: Correlation ID information.
                     - **`internal`** *(integer, required)*: Internal correlation ID.
                     - **`external`** *(integer, required)*: External correlation ID.
                  - **`dispatch_info`** *(object, required)*: Dispatch information details.
                     - **`size`** *(integer, required)*: Size of the dispatch information.
                     - **`agent_id`** *(object, required)*: Agent ID information.
                        - **`handle`** *(integer, required)*: Handle of the agent.
                     - **`queue_id`** *(object, required)*: Queue ID information.
                        - **`handle`** *(integer, required)*: Handle of the queue.
                     - **`kernel_id`** *(integer, required)*: ID of the kernel.
                     - **`dispatch_id`** *(integer, required)*: ID of the dispatch.
                     - **`private_segment_size`** *(integer, required)*: Size of the private segment.
                     - **`group_segment_size`** *(integer, required)*: Size of the group segment.
                     - **`workgroup_size`** *(object, required)*: Workgroup size information.
                        - **`x`** *(integer, required)*: X dimension.
                        - **`y`** *(integer, required)*: Y dimension.
                        - **`z`** *(integer, required)*: Z dimension.
                     - **`grid_size`** *(object, required)*: Grid size information.
                        - **`x`** *(integer, required)*: X dimension.
                        - **`y`** *(integer, required)*: Y dimension.
                        - **`z`** *(integer, required)*: Z dimension.
               - **`records`** *(array, required)*: Records.
                  - **Items** *(object)*
                     - **`counter_id`** *(object, required)*: Counter ID information.
                        - **`handle`** *(integer, required)*: Handle of the counter.
                     - **`value`** *(number, required)*: Value of the counter.
               - **`thread_id`** *(integer, required)*: Thread ID.
               - **`arch_vgpr_count`** *(integer, required)*: Count of VGPRs.
               - **`sgpr_count`** *(integer, required)*: Count of SGPRs.
               - **`lds_block_size_v`** *(integer, required)*: Size of LDS block.
      -  **``pc_sample_host_trap``** *(array)*: Host Trap PC Sampling records.
            - **Items** *(object)*
               - **``hw_id``** *(object)*: Describes hardware part on which sampled wave was running.
                  -  **``chiplet``** *(integer)*: Chiplet index.
                  -  **``wave_id``** *(integer)*: Wave slot index.
                  -  **``simd_id``** *(integer)*: SIMD index.
                  -  **``pipe_id``** *(integer)*: Pipe index.
                  -  **``cu_or_wgp_id``** *(integer)*: Index of compute unit or workgroup processer.
                  -  **``shader_array_id``** *(integer)*: Shader array index.
                  -  **``shader_engine_id``** *(integer)*: Shader engine
                     index.
                  -  **``workgroup_id``** *(integer)*: Workgroup position in the 3D.
                  -  **``vm_id``** *(integer)*: Virtual memory ID.
                  -  **``queue_id``** *(integer)*: Queue id.
                  -  **``microengine_id``** *(integer)*: ACE
                     (microengine) index.
               -  **``pc``** *(object)*: Encapsulates information about
                  sampled PC.
                  -  **``code_object_id``** *(integer)*: Code object id.
                  -  **``code_object_offset``** *(integer)*: Offset within the object if the latter is known. Otherwise, virtual address of the PC.
               -  **``exec_mask``** *(integer)*: Execution mask indicating active SIMD lanes of sampled wave.
               -  **``timestamp``** *(integer)*: Timestamp.
               -  **``dispatch_id``** *(integer)*: Dispatch id.
               -  **``correlation_id``** *(object)*: Correlation ID information.
                  -  **``internal``** *(integer)*: Internal correlation ID.
                  -  **``external``** *(integer)*: External correlation ID.
               - **``rocprofiler_dim3_t``** *(object)*: Position of the workgroup in 3D grid.
                  -  **``x``** *(integer)*: Dimension x.
                  -  **``y``** *(integer)*: Dimension y.
                  -  **``z``** *(integer)*: Dimension z.
               -  **``wave_in_group``** *(integer)*: Wave position within the workgroup (0-31).
      - **`buffer_records`** *(object, required)*: Buffer record details.
         - **`kernel_dispatch`** *(array)*: Kernel dispatch records.
            - **Items** *(object)*
               - **`size`** *(integer, required)*: Size of the dispatch.
               - **`kind`** *(integer, required)*: Kind of the dispatch.
               - **`operation`** *(integer, required)*: Operation of the dispatch.
               - **`thread_id`** *(integer, required)*: Thread ID.
               - **`correlation_id`** *(object, required)*: Correlation ID information.
                  - **`internal`** *(integer, required)*: Internal correlation ID.
                  - **`external`** *(integer, required)*: External correlation ID.
               - **`start_timestamp`** *(integer, required)*: Start timestamp.
               - **`end_timestamp`** *(integer, required)*: End timestamp.
               - **`dispatch_info`** *(object, required)*: Dispatch information details.
                  - **`size`** *(integer, required)*: Size of the dispatch information.
                  - **`agent_id`** *(object, required)*: Agent ID information.
                     - **`handle`** *(integer, required)*: Handle of the agent.
                  - **`queue_id`** *(object, required)*: Queue ID information.
                     - **`handle`** *(integer, required)*: Handle of the queue.
                  - **`kernel_id`** *(integer, required)*: ID of the kernel.
                  - **`dispatch_id`** *(integer, required)*: ID of the dispatch.
                  - **`private_segment_size`** *(integer, required)*: Size of the private segment.
                  - **`group_segment_size`** *(integer, required)*: Size of the group segment.
                  - **`workgroup_size`** *(object, required)*: Workgroup size information.
                     - **`x`** *(integer, required)*: X dimension.
                     - **`y`** *(integer, required)*: Y dimension.
                     - **`z`** *(integer, required)*: Z dimension.
                  - **`grid_size`** *(object, required)*: Grid size information.
                     - **`x`** *(integer, required)*: X dimension.
                     - **`y`** *(integer, required)*: Y dimension.
                     - **`z`** *(integer, required)*: Z dimension.
         - **`hip_api`** *(array)*: HIP API records.
            - **Items** *(object)*
               - **`size`** *(integer, required)*: Size of the HIP API record.
               - **`kind`** *(integer, required)*: Kind of the HIP API.
               - **`operation`** *(integer, required)*: Operation of the HIP API.
               - **`correlation_id`** *(object, required)*: Correlation ID information.
                  - **`internal`** *(integer, required)*: Internal correlation ID.
                  - **`external`** *(integer, required)*: External correlation ID.
               - **`start_timestamp`** *(integer, required)*: Start timestamp.
               - **`end_timestamp`** *(integer, required)*: End timestamp.
               - **`thread_id`** *(integer, required)*: Thread ID.
         - **`hsa_api`** *(array)*: HSA API records.
            - **Items** *(object)*
               - **`size`** *(integer, required)*: Size of the HSA API record.
               - **`kind`** *(integer, required)*: Kind of the HSA API.
               - **`operation`** *(integer, required)*: Operation of the HSA API.
               - **`correlation_id`** *(object, required)*: Correlation ID information.
                  - **`internal`** *(integer, required)*: Internal correlation ID.
                  - **`external`** *(integer, required)*: External correlation ID.
               - **`start_timestamp`** *(integer, required)*: Start timestamp.
               - **`end_timestamp`** *(integer, required)*: End timestamp.
               - **`thread_id`** *(integer, required)*: Thread ID.
         - **`marker_api`** *(array)*: Marker (ROCTx) API records.
            - **Items** *(object)*
               - **`size`** *(integer, required)*: Size of the Marker API record.
               - **`kind`** *(integer, required)*: Kind of the Marker API.
               - **`operation`** *(integer, required)*: Operation of the Marker API.
               - **`correlation_id`** *(object, required)*: Correlation ID information.
                  - **`internal`** *(integer, required)*: Internal correlation ID.
                  - **`external`** *(integer, required)*: External correlation ID.
               - **`start_timestamp`** *(integer, required)*: Start timestamp.
               - **`end_timestamp`** *(integer, required)*: End timestamp.
               - **`thread_id`** *(integer, required)*: Thread ID.
         - **`memory_copy`** *(array)*: Async memory copy records.
            - **Items** *(object)*
               - **`size`** *(integer, required)*: Size of the Marker API record.
               - **`kind`** *(integer, required)*: Kind of the Marker API.
               - **`operation`** *(integer, required)*: Operation of the Marker API.
               - **`correlation_id`** *(object, required)*: Correlation ID information.
                  - **`internal`** *(integer, required)*: Internal correlation ID.
                  - **`external`** *(integer, required)*: External correlation ID.
               - **`start_timestamp`** *(integer, required)*: Start timestamp.
               - **`end_timestamp`** *(integer, required)*: End timestamp.
               - **`thread_id`** *(integer, required)*: Thread ID.
               - **`dst_agent_id`** *(object, required)*: Destination Agent ID.
                  - **`handle`** *(integer, required)*: Handle of the agent.
               - **`src_agent_id`** *(object, required)*: Source Agent ID.
                  - **`handle`** *(integer, required)*: Handle of the agent.
               - **`bytes`** *(integer, required)*: Bytes copied.
         - **`memory_allocation`** *(array)*: Memory allocation records.
            - **Items** *(object)*
               - **`size`** *(integer, required)*: Size of the Marker API record.
               - **`kind`** *(integer, required)*: Kind of the Marker API.
               - **`operation`** *(integer, required)*: Operation of the Marker API.
               - **`correlation_id`** *(object, required)*: Correlation ID information.
                  - **`internal`** *(integer, required)*: Internal correlation ID.
                  - **`external`** *(integer, required)*: External correlation ID.
               - **`start_timestamp`** *(integer, required)*: Start timestamp.
               - **`end_timestamp`** *(integer, required)*: End timestamp.
               - **`thread_id`** *(integer, required)*: Thread ID.
               - **`agent_id`** *(object, required)*: Agent ID.
                  - **`handle`** *(integer, required)*: Handle of the agent.
               - **`address`** *(string, required)*: Starting address of allocation.
               - **`allocation_size`** *(integer, required)*: Size of allocation.
         - **`rocDecode_api`** *(array)*: rocDecode API records.
            - **Items** *(object)*
               - **`size`** *(integer, required)*: Size of the rocDecode API record.
               - **`kind`** *(integer, required)*: Kind of the rocDecode API.
               - **`operation`** *(integer, required)*: Operation of the rocDecode API.
               - **`correlation_id`** *(object, required)*: Correlation ID information.
                  - **`internal`** *(integer, required)*: Internal correlation ID.
                  - **`external`** *(integer, required)*: External correlation ID.
               - **`start_timestamp`** *(integer, required)*: Start timestamp.
               - **`end_timestamp`** *(integer, required)*: End timestamp.
               - **`thread_id`** *(integer, required)*: Thread ID.

