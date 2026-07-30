.. meta::
  :description: Documentation for using rocprofv3 with OpenMP applications
  :keywords: ROCprofiler-SDK tool, OpenMP, rocprofv3, rocprofv3 tool usage, ROCprofiler-SDK command line tool, ROCprofiler-SDK CLI


.. _using-rocprofv3-with-openmp:

============================
Using rocprofv3 with OpenMP
============================

For computations offloaded to AMD GPUs using OpenMP (for example, via OpenMP target offload), ``rocprofv3`` can be used to capture and profile GPU activities initiated by these offloaded regions. The ``--ompt-trace`` option additionally records host-side OpenMP execution — parallel regions, tasks, work-sharing, and sync — for any application linked against an OMPT-capable OpenMP runtime; see :ref:`tracing-openmp-with-ompt-trace`.

Example: Vector addition using OpenMP offload on AMD GPUs
==========================================================

The following example demonstrates how to perform vector addition using OpenMP target offload, enabling workload execution on AMD GPUs.

**Key steps:**

1. Initialize the input arrays on the host.

2. Offload the vector addition computation to the GPU using OpenMP directives.

3. Retrieve and verify the results on the host.

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
========================================

To compile the application for AMD GPU offload, use:

.. code-block:: bash

    amdclang++ -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -L/opt/rocm/lib --offload-arch=gfx9xx -o vector_add <application>

Profiling the application with rocprofv3
=========================================

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

.. _tracing-openmp-with-ompt-trace:

Tracing OpenMP runtime events with ``--ompt-trace``
====================================================

The flags shown above capture HIP / HSA / kernel-dispatch / memory activity but not OpenMP-level structure. To trace OpenMP itself, pass ``--ompt-trace`` (or ``ROCPROF_OMPT_TRACE=1``; it is also enabled implicitly by ``--sys-trace`` and ``--runtime-trace``). This is a host-side facility: it records CPU-side OpenMP execution (thread lifecycle, parallel regions, work-sharing, tasks, sync regions, mutexes, dispatch) even for applications that never offload, plus host-side target-offload events (``target``, ``target_data_op``, ``target_submit``, ``device_initialize`` / ``device_load`` / ``device_finalize``) for offloading applications.

.. code-block:: bash

    rocprofv3 --ompt-trace --kernel-trace --memory-copy-trace --output-format rocpd -- ./vector_add

OMPT is a rocpd-only trace: records are written to the rocpd database (``rocpd`` is also the default output format) and are **not** emitted by the direct CSV / JSON / Perfetto / OTF2 generators. If ``--ompt-trace`` is combined with another ``--output-format``, ``rocprofv3`` prints a warning and adds ``rocpd`` automatically so OMPT data is not lost. To view OMPT in CSV / Perfetto / OTF2, convert the database with ``rocpd convert`` (see below).

Combined with ``--kernel-trace`` / ``--memory-copy-trace``, each GPU kernel can be correlated with the surrounding ``target_submit`` / ``target_data_op`` region on the host and placed on the same timeline as the enclosing ``parallel`` / ``work`` regions and tasks.

Exporting OMPT to other formats
-------------------------------

Use ``rocpd convert`` to export the rocpd database (including OMPT) to CSV, Perfetto, or OTF2:

.. code-block:: bash

    rocpd convert -i <pid>_results.db --output-format pftrace csv

The Perfetto conversion preserves all OMPT events, including instantaneous ones (for example, ``parallel_begin`` / ``thread_begin``). The CSV and OTF2 conversions currently export ranged OMPT operations only.

Filtering by category
---------------------

By default ``--ompt-trace`` records every OMPT operation. To cut down on high event volume (for example, fine-grained synchronization), pass a space-separated list of categories (``--ompt-trace <cat1> <cat2> ...``):

.. code-block:: bash

    # All OMPT operations (default; same as bare --ompt-trace or --ompt-trace=all)
    rocprofv3 --ompt-trace -- ./vector_add

    # Only host parallel regions, tasks, and target offload
    rocprofv3 --ompt-trace parallel task target -- ./vector_add

    # Only target-offload events, correlated with kernel dispatches
    rocprofv3 --ompt-trace target --kernel-trace -- ./vector_add

Recognised categories are ``thread``, ``parallel``, ``task``, ``sync``, ``mutex``, ``target``, ``device``, ``error``, and ``all``. Each resolves to a fixed set of OMPT operations (for example, ``parallel`` covers ``parallel_begin``/``parallel_end``/``implicit_task``/``work``/``dispatch``/``reduction``/``masked``; ``target`` covers ``target_emi``/``target_data_op_emi``/``target_submit_emi``). The CLI rejects unknown categories, and comma-separated tokens, at parse time.

Equivalently, set ``ROCPROF_OMPT_TRACE_OPERATIONS=parallel,task,target`` in the environment (env vars use commas because they are single strings). When OMPT tracing is enabled, an unrecognised category in this variable is a fatal error.

.. note::

   ``--ompt-trace`` requires an OMPT-capable OpenMP runtime that implements ``ompt_start_tool``. The LLVM-based ``libomp`` shipped with ROCm / AOMP (used by ``amdclang++ -fopenmp`` above) qualifies. GCC's ``libgomp`` does not implement the OMPT interface (see the `GOMP status page <https://www.gnu.org/software/gcc/projects/gomp/>`_), so ``g++ -fopenmp`` binaries do not produce OMPT records.

   Most OMPT events are host-thread events that scale with ``OMP_NUM_THREADS``; ``OMP_NUM_THREADS=1`` suppresses most parallel-region events. Target-offload events also require ``OMP_TARGET_OFFLOAD`` not to be ``DISABLED`` and at least one supported GPU to be visible (for example, via ``ROCR_VISIBLE_DEVICES``).
