.. meta::
  :description: Documentation for using rocprofv3 with OpenMP applications
  :keywords: ROCprofiler-SDK tool, OpenMP, rocprofv3, rocprofv3 tool usage, ROCprofiler-SDK command line tool, ROCprofiler-SDK CLI


.. _using-rocprofv3-with-openmp:

Using rocprofv3 with OpenMP
+++++++++++++++++++++++++++++

`rocprofv3` does not provide native support for profiling CPU-side OpenMP code. However, when OpenMP is used to offload computations to AMD GPUs (for example, via OpenMP target offload), `rocprofv3` can capture and profile GPU activities initiated by these offloaded regions. Note that profiling of CPU-side OpenMP parallel regions is not supported.

Example: Vector Addition Using OpenMP Offload on AMD GPUs
---------------------------------------------------------

The following example demonstrates how to perform vector addition using OpenMP target offload, enabling execution of the workload on AMD GPUs.

**Key Steps:**

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


Building the OpenMP Offload Application
---------------------------------------

To compile the application for AMD GPU offload, use the following command:

.. code-block:: bash

    amdclang++ -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -L/opt/rocm/lib --offload-arch=gfx9xx -o vector_add <application>

Profiling the Application with rocprofv3
----------------------------------------

To profile the GPU activity during execution, run the application with `rocprofv3`:

.. code-block:: bash

    rocprofv3 -s --output-format csv -- ./vector_add

Upon execution, `rocprofv3` will generate several CSV trace files, such as:

- `<pid>_kernel_trace.csv`
- `<pid>_hsa_api_trace.csv`
- `<pid>_memory_copy_trace.csv`
- `<pid>_memory_allocation_trace.csv`
- `<pid>_scratch_memory_trace.csv`

These files contain detailed profiling information about GPU kernel execution, HSA API calls, memory operations, and more, enabling comprehensive analysis of the offloaded workload.
