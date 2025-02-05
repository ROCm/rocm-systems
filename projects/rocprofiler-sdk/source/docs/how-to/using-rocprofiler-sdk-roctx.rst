.. meta::
  :description: Documentation for the usage of rocprofiler-sdk-roctx library
  :keywords: ROCprofiler-SDK tool, using-rocprofiler-sdk-roctx library, roctx, markers, ranges, rocprofv3, rocprofv3 tool usage, Using rocprofv3, ROCprofiler-SDK command line tool, marker-trace

.. _using-rocprofiler-sdk-roctx:

=============================================
Using ``ROCTx`` (AMD Tools Extension Library)
=============================================

``ROCtx`` is AMD's cross platform API for annotating code with markers and ranges. The ``ROCTx`` API is written in C++.
In certain situations, such as debugging performance issues in large-scale GPU programs, API-level tracing might be too fine-grained to provide a big picture of the program execution. 
In such cases, it is helpful to define specific tasks to be traced.To specify the tasks for tracing, enclose the respective source code with the API calls provided by the ``ROCTx`` library. 
This process is also known as instrumentation.

What kinds of annotation does ``ROCTx`` provide?
+++++++++++++++++++++++++++++++++++++++++++++++++
``ROCTx`` provides two types of annotations: markers and ranges.

Markers:
========
Helps you inserts a marker in the code with a message. Creating markers help you see when a line of code is executed.

Ranges:
=======
The scope of code for instrumentation is defined using the enclosing API calls, it is called a range. 
A range is a programmer-defined task that has a well-defined start and end code scope. 
You can also refine the scope specified within a range using further nested ranges. ``rocprofv3`` also reports the timelines for these nested ranges.
There are two types of ranges:

1. **Push/Pop ranges**:
   - These can be nested to form a stack.
   - The Pop call is automatically associated with a prior Push call on the same thread.

2. **Start/End ranges**:
   - These may overlap with other ranges arbitrarily.
   - The Start call returns a handle which must be passed to the End call.
   - These ranges can start and end on different threads.

List of APIs supported by ``ROCTx``
===================================
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


How to use ``ROCTx`` in your application?
===========================================

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

    // Launching kernel from host
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

Resource Naming:
++++++++++++++++

OS Thread:
==========

:code:`roctxNameOsThread(const char*)` function Current CPU OS thread to be labeled by the provided name in the output of the profiling tool.

Indicate to a profiling tool that, where possible, you would like the current CPU OS thread to be labeled by the provided name in the output of the profiling tool.
Rocprofiler does not provide any explicit support for how profiling tools handle this request:
- support for this capability is tool specific.
- ROCTx does NOT rename the thread via `pthread_setname_np`.

HIP Runtime Resources:
======================

:code:`roctxNameHipDevice(const char* name, int device_id)` and :code:`roctxNameHipStream(const char* name, const struct ihipStream_t* stream)` functions indicate to a profiling tool that, where possible, you would like the given HIP device id and HIP stream-id to be labeled by the provided name in the output of the profiling tool. Rocprofiler does not provide any explicit support for how profiling tools handle this request:
- support for this capability is tool specific.

HSA Runtime Resources:
======================

:code:`roctxNameHsaAgent(const char* name, const struct hsa_agent_s*)` function indicates to a profiling tool that, where possible, you would like the given HSA agent to be labeled by the provided name in the output of the profiling tool.
- support for this capability is tool specific.
