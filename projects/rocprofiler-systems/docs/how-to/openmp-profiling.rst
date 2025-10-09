.. meta::
    :description: ROCm Systems Profiler OpenMP performance profiling
    :keywords: rocprof-sys, rocprofiler-systems, ROCm, tips, how to, profiler, tracking, OpenMP, OMPT, AMD

********************************************
OpenMP performance profiling
********************************************

`ROCm Systems Profiler <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-systems>`_ supports OpenMP profiling.
This functionality is built on the OpenMP Tools Interface (OMPT) and requires ROCm version 6.4.0 or higher.

Only a subset of the OMPT callbacks are processed:

.. code-block:: shell

  |----------------------------------+---------------------------|
  |          OpenMP Callback         |        Track Name         |
  |----------------------------------+---------------------------|
  | ompt_callback_cancel             | omp_cancel                |
  | ompt_callback_dependences        | omp_dependences           |
  | ompt_callback_device_finalize    | omp_device_finalize       |
  | ompt_callback_device_initialize  | omp_device_initialize     |
  | ompt_callbback_device_load       | omp_device_load           |
  | ompt_callback_dispatch           | omp_dispatch              |
  | ompt_callback_error              | omp_error                 |
  | ompt_callback_flush              | omp_flush                 |
  | ompt_callback_implicit_task      | omp_implicit_task         |
  | ompt_callback_lock_destroy       | omp_lock_destroy          |
  | ompt_callback_lock_init          | omp_lock_init             |
  | ompt_callback_masked             | omp_masked                |
  | ompt_callback_mutex_acquire      | omp_mutex_acquire         |
  | ompt_callback_mutex_acquired     | omp_mutex_acquired        |
  | ompt_callback_mutex_released     | omp_mutex_released        |
  | ompt_callback_nest_lock          | omp_nest_lock             |
  | ompt_callback_parallel_begin     | omp_parallel              |
  | ompt_callback_parallel_end       | omp_parallel              |
  | ompt_callback_reduction          | omp_reduction             |
  | ompt_callback_sync_region        | omp_sync_region           |
  | ompt_callback_sync_region_wait   | omp_sync_region_wait      |
  | ompt_callback_target_data_op     | omp_target_data_op_emi    |
  | ompt_callback_target_t           | omp_target_emi            |
  | ompt_callback_target_submit      | omp_target_submit_emi     |
  | ompt_callback_task_create        | omp_task_create           |
  | ompt_callback_task_dependence    | omp_task_dependence       |
  | ompt_callback_task_schedule      | omp_task_schedule         |
  | ompt_callback_work               | omp_work                  |
  |----------------------------------+---------------------------|

.. note::
   The ``omp_parallel`` track begins with ``ompt_callback_parallel_begin`` and ends when the corresponding ``ompt_callback_parallel_end`` is encountered.

Configuration
=============

To enable capturing of OMPT events, the following settings are required:

.. code-block:: shell

  ROCPROFSYS_USE_ROCM=ON
  ROCPROFSYS_USE_OMPT=ON

OMPT events will not be traced if a GPU is not detected.

Instrumenting and running a program
===================================

An example rocprof-sys-instrument command is:

.. code-block:: shell

  rocprof-sys-instrument -o jacobi.inst \
    --log-file jacobi_instr.log --verbose --debug
    -- ./jacobi

This command generates an instrumented binary ``jacobi.inst``. Then, run it with the following command:

.. code-block:: shell

  rocprof-sys-run -- ./foo.inst

.. note::
   If ROCPROFSYS_USE_OMPT=ON is not specified, you can append `-I "ompt"`` to `rocprof-sys-run`.

To view the generated ``.proto`` file in the browser, open the `Perfetto UI page <https://ui.perfetto.dev/>`_. Then, clicl on ``Open trace file`` and select the ``.proto`` file.

[ INSERT IMAGE HERE ]


.. - Configuration
..     - ROCPROFSYS_USE_OMPT = ON (collects OMPT events)
..     - Append "hsa_api" to ROCPROFSYS_ROCM_DOMAINS to collect HSA events (underlying implementation)
..     - export LD_LIBRARY_PATH=/opt/rocm/lib/llvm/lib:$LD_LIBRARY_PATH (WILL THIS BE FIXED TO BE DONE AUTOMATICALLY?)

.. - Example using environment variables
..     export ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,marker_api,kernel_dispatch,memory_copy,hsa_api
..     export ROCPROFSYS_USE_OMPT=ON

.. - Using commands
..     rocprof-sys-run -I "ompt"
..     rocprof-sys-run --rocm-domains "ompt" (PREFER -I OMPT as it also adds the offload tracks that the calls connect to)

.. - List of supported callbacks can be seen in the `--rocm-ompt-operations` when using `rocprof-sys-run --help`
.. NOTE: We do not support `omp_thread_begin` nor `omp_thread_end`. Furthermore, disabling only one of, `omp_parallel_begin` or `omp_parallel_end` will break the parallel track. If you need the event disabled, disable both begin and end.
..             - Probably put this in a manual table


.. SECTIONS:

.. - Introduction (Not named, just say OpenMP performance profiling)
.. - Configuration
.. - Instrumenting and running a program
.. - Runtime instrumentation
.. - Offloading (example using Jacobi)

.. For 1_jacobi_usm:
..     - rocprof-sys-run -I "ompt" -- ./jacobi (WORKS but will take a very long time to generate, about 2m25s)

.. For 2_jacobi_targetdata (perfer this is we can)
..     - instrument then run works
..     - rocprof-sys-run -I "ompt" -- ./jacobi (takes 3 seconds wtf)
