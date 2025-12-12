.. meta::
  :description: ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software
  :keywords: ROCprofiler-SDK tool usage, rocprofv3 user manual, rocprofv3 usage, rocprofv3 user guide, using rocprofv3, ROCprofiler-SDK tool user guide, ROCprofiler-SDK tool user manual, using ROCprofiler-SDK tool, ROCprofiler-SDK command-line tool, ROCprofiler-SDK CLI, ROCprofiler-SDK command line tool

.. _kernel-naming-filtering:

============================================
Kernel naming and filtering using rocprofv3
============================================

``rocprofv3`` provides the following functionalities to configure the kernel name in the output file or to filter the kernels based on the requirement.

Kernel name mangling
---------------------

In ``rocprofv3`` output, by default, the kernel names are demangled to exclude the kernel arguments. This improves readability of the collected output.

To see the mangled kernel names, disable this feature by using the ``--mangled-kernels`` option.

Here is an example of kernel trace by default:

.. code-block:: shell

    $ cat 123_kernel_trace.csv

    "Kind","Agent_Id","Queue_Id","Stream_Id","Thread_Id","Dispatch_Id","Kernel_Id","Kernel_Name","Correlation_Id","Start_Timestamp","End_Timestamp","LDS_Block_Size","Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count","Workgroup_Size_X","Workgroup_Size_Y","Workgroup_Size_Z","Grid_Size_X","Grid_Size_Y","Grid_Size_Z"
    "KERNEL_DISPATCH","Agent 4",1,1,852831,1,10,"void addition_kernel<float>(float*, float const*, float const*, int, int)",1,1551874061244694,1551874061255734,0,0,8,0,16,64,1,1,1024,1024,1
    "KERNEL_DISPATCH","Agent 4",1,1,852831,2,13,"subtract_kernel(float*, float const*, float const*, int, int)",2,1551874061259214,1551874061270254,0,0,8,0,16,64,1,1,1024,1024,1
    "KERNEL_DISPATCH","Agent 4",1,1,852831,3,12,"multiply_kernel(float*, float const*, float const*, int, int)",3,1551874061270254,1551874061279974,0,0,8,0,16,64,1,1,1024,1024,1
    "KERNEL_DISPATCH","Agent 4",2,2,852831,8,11,"divide_kernel(float*, float const*, float const*, int, int)",8,1551874061326294,1551874061335454,0,0,12,4,16,64,1,1,1024,1024,1

To disable kernel name demangling, use:

.. code-block:: shell

   rocprofv3 --mangled-kernels --kernel-trace --output-format csv -- <application_path>

The preceding command generates the following ``kernel_trace.csv`` file with mangled kernel names:

.. code-block:: shell

    $ cat 123_kernel_trace.csv

    "Kind","Agent_Id","Queue_Id","Stream_Id","Thread_Id","Dispatch_Id","Kernel_Id","Kernel_Name","Correlation_Id","Start_Timestamp","End_Timestamp","LDS_Block_Size","Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count","Workgroup_Size_X","Workgroup_Size_Y","Workgroup_Size_Z","Grid_Size_X","Grid_Size_Y","Grid_Size_Z"
    "KERNEL_DISPATCH","Agent 4",1,1,850334,1,10,"_Z15addition_kernelIfEvPT_PKfS3_ii.kd",1,1551636841670446,1551636841681606,0,0,8,0,16,64,1,1,1024,1024,1
    "KERNEL_DISPATCH","Agent 4",1,1,850334,2,13,"_Z15subtract_kernelPfPKfS1_ii.kd",2,1551636841686726,1551636841697606,0,0,8,0,16,64,1,1,1024,1024,1
    "KERNEL_DISPATCH","Agent 4",1,1,850334,3,12,"_Z15multiply_kernelPfPKfS1_ii.kd",3,1551636841701926,1551636841712806,0,0,8,0,16,64,1,1,1024,1024,1
    "KERNEL_DISPATCH","Agent 4",2,2,850334,8,11,"_Z13divide_kernelPfPKfS1_ii.kd",8,1551636841762926,1551636841774646,0,0,12,4,16,64,1,1,1024,1024,1

Kernel name truncation
-----------------------

The kernel name truncation feature allows you to limit the kernel name length in the output files. This is useful when dealing with long kernel names that can make the output files difficult to read.

To enable kernel name truncation, use the ``--truncate-kernels`` option:

.. code-block:: shell

    rocprofv3 --truncate-kernels --kernel-trace --output-format csv -- <application_path>

The preceding command generates the following ``kernel_trace.csv`` file with truncated kernel names:

.. csv-table:: Kernel trace truncated
   :file: /data/kernel_trace_truncated.csv
   :widths: 10,10,10,10,10,10,10,10,10,20,20,10,10,10,10,10,10,10,10,10,10,10
   :header-rows: 1

.. _kernel-filtering:

Kernel filtering
-----------------

Kernel filtering helps to include or exclude the kernels for profiling by specifying a filter using a regex string. You can also specify an iteration range for profiling the included kernels. If the iteration range is not provided, then all iterations of the included kernels are profiled.

Here is an input file with kernel filters:

.. code-block:: shell

    $ cat input.yml
    jobs:
        - pmc: [SQ_WAVES]
        kernel_include_regex: "divide"
        kernel_exclude_regex: ""
        kernel_iteration_range: "[1, 2, [5-8]]"

To collect counters for the kernels matching the filters specified in the preceding input file, run:

.. code-block:: shell

    rocprofv3 -i input.yml --output-format csv -- <application_path>

    $ cat pass_1/312_counter_collection.csv
    "Correlation_Id","Dispatch_Id","Agent_Id","Queue_Id","Process_Id","Thread_Id","Grid_Size","Kernel_Id","Kernel_Name","Workgroup_Size","LDS_Block_Size","Scratch_Size","VGPR_Count","Accum_VGPR_Count","SGPR_Count","Counter_Name","Counter_Value","Start_Timestamp","End_Timestamp"
    1,1,4,1,225049,225049,1048576,10,"void addition_kernel<float>(float*, float const*, float const*, int, int)",64,0,0,8,0,16,"SQ_WAVES",16384.000000,317095766765717,317095766775957
    2,2,4,1,225049,225049,1048576,13,"subtract_kernel(float*, float const*, float const*, int, int)",64,0,0,8,0,16,"SQ_WAVES",16384.000000,317095767013157,317095767022957
    3,3,4,1,225049,225049,1048576,11,"multiply_kernel(float*, float const*, float const*, int, int)",64,0,0,8,0,16,"SQ_WAVES",16384.000000,317095767176998,317095767186678
    4,4,4,1,225049,225049,1048576,12,"divide_kernel(float*, float const*, float const*, int, int)",64,0,0,12,4,16,"SQ_WAVES",16384.000000,317095767380718,317095767390878

Kernel rename
--------------

The ``roctxRangePush`` and ``roctxRangePop`` also let you rename the enclosed kernel with the supplied message. In the legacy ``rocprof``, this functionality was known as ``--roctx-rename``.

See how to use ``roctxRangePush`` and ``roctxRangePop`` for renaming the enclosed kernel:

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

To rename the kernel, use:

.. code-block:: bash

    rocprofv3 --marker-trace --kernel-rename --output-format csv -- <application_path>

The preceding command generates the following ``marker-trace`` file prefixed with the process ID:

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
