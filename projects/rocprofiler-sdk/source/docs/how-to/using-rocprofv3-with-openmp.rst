.. meta::
  :description: Documentation for using rocprofv3 with OpenMP applications
  :keywords: ROCprofiler-SDK tool, OpenMP, rocprofv3, rocprofv3 tool usage, ROCprofiler-SDK command line tool, ROCprofiler-SDK CLI


.. _using-rocprofv3-with-openmp:

Using rocprofv3 with OpenMP
+++++++++++++++++++++++++++++

For computations offloaded to AMD GPUs using OpenMP (for example, via OpenMP target offload), ``rocprofv3`` can be used to capture and profile GPU activities initiated by these offloaded regions. Note that ``rocprofv3`` doesn't provide native support for profiling CPU-side OpenMP code or parallel regions.

Example: Vector addition using OpenMP offload on AMD GPUs
----------------------------------------------------------

The following example demonstrates how to perform vector addition using OpenMP target offload, enabling workload execution on AMD GPUs.

**Key steps:**

- Initialize input arrays on the host.
- Offload the vector addition computation to the GPU using OpenMP directives.
- Retrieve and verify the results on the host.

.. code-block:: c

    #include <stdio.h>
    #include <omp.h>

    #define N 1024

    int main() {
        float a[N], b[N], c[N];

        // Initialize input arrays
        for (int i = 0; i < N; ++i) {
            a[i] = i * 1.0f;
            b[i] = (N - i) * 1.0f;
        }

        // Offload vector addition to GPU
        #pragma omp target teams distribute parallel for map(to: a[0:N], b[0:N]) map(from: c[0:N])
        for (int i = 0; i < N; ++i) {
            c[i] = a[i] + b[i];
        }

        // Verify results
        int errors = 0;
        for (int i = 0; i < N; ++i) {
            if (c[i] != N * 1.0f) {
                errors++;
            }
        }

        if (errors == 0) {
            printf("Vector addition successful!\\n");
        } else {
            printf("Vector addition failed with %d errors.\\n", errors);
        }

        return 0;
    }

Building the OpenMP offload application
----------------------------------------

To compile the application for AMD GPU offload, use the following command:

.. code-block:: bash

    amdclang++ -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -L/opt/rocm/lib --offload-arch=gfx9xx -o vector_add <application>

Profiling the application with rocprofv3
-----------------------------------------

To profile the GPU activity during execution, run the application with ``rocprofv3``:

.. code-block:: bash

    rocprofv3 -s --output-format csv -- ./vector_add

Upon execution, ``rocprofv3`` generates several CSV trace files, such as:

- <pid>_kernel_trace.csv
- <pid>_hsa_api_trace.csv
- <pid>_memory_copy_trace.csv
- <pid>_memory_allocation_trace.csv
- <pid>_scratch_memory_trace.csv

The preceding files contain detailed profiling information about GPU kernel execution, HSA API calls, memory operations, and more, enabling comprehensive analysis of the offloaded workload.
