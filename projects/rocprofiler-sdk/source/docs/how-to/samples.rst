.. meta::
    :description: "ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software."
    :keywords: "ROCprofiler-SDK, ROCProfiler-SDK samples"

.. _rocprofiler-sdk-samples:

ROCprofiler-SDK samples
========================

The samples are provided to help you see the profiler in action.

Finding samples
---------------

The ROCm installation provides sample programs and ``rocprofv3`` tool.

- Sample programs are installed here:

.. code-block:: bash
    
    /opt/rocm/share/rocprofiler-sdk/samples

- ``rocprofv3`` tool is installed here:

.. code-block:: bash
    
    /opt/rocm/bin

Building Samples
----------------

To build samples from any directory, run:

.. code-block:: bash

    cmake -B build-rocprofiler-sdk-samples /opt/rocm/share/rocprofiler-sdk/samples -DCMAKE_PREFIX_PATH=/opt/rocm
    cmake --build build-rocprofiler-sdk-samples --target all --parallel 8


Running samples
---------------

To run the built samples, ``cd`` into the ``build-rocprofiler-sdk-samples`` directory and run:

.. code-block:: bash
    
    ctest -V

The `-V` option enables verbose output, providing detailed information about the test execution.
