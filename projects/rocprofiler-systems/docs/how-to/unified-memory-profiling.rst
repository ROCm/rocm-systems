.. meta::
   :description: ROCm Systems Profiler unified memory profiling how-to guide
   :keywords: rocprof-sys, rocprofiler-systems, ROCm, unified memory, managed memory, KFD, XNACK, page fault, page migration, profiling

****************************************************
Unified memory profiling
****************************************************

ROCm Systems Profiler can generate unified-memory profiling reports from KFD
page-fault and page-migration events. Use this feature when a HIP managed-memory
workload uses ``hipMallocManaged`` or other unified-memory access patterns and
you want to understand page faults, migration triggers, and effective migration
throughput between host and device memory.

Unified memory profiling writes two summary files in addition to the usual trace
or database outputs:

* ``unified_memory.txt``: Human-readable per-GPU migration and page-fault
  summary.
* ``unified_memory.json``: Machine-readable equivalent for validation,
  scripting, and dashboards.

Prerequisites
=============

Unified memory profiling requires:

* An XNACK-capable AMD GPU.
* ``HSA_XNACK=1`` in the target application's environment.
* ROCProfiler-SDK 1.2.2 or later.
* A workload that produces KFD page-fault or page-migration events.

Check XNACK support with ``rocminfo``:

.. code-block:: shell

   rocminfo | grep xnack

If the output contains ``xnack-``, XNACK is available but disabled. Enable it
before launching the profiled application:

.. code-block:: shell

   export HSA_XNACK=1

Quick start
===========

Enable unified memory profiling with
``ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON``:

.. code-block:: shell

   HSA_XNACK=1 \
   ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON \
   ROCPROFSYS_TRACE=true \
   rocprof-sys-run -- ./my_managed_memory_app

The unified-memory setting automatically enables the required KFD tracing
domains for page faults and page migrations. You don't need to add
``kfd_events`` to ``ROCPROFSYS_ROCM_DOMAINS`` separately.

Example workload
================

The repository includes a HIP managed-memory example under
``examples/unified-memory``. It runs access patterns that can trigger
host-to-device migrations, device-to-host migrations, prefetch-triggered
migrations, memory-pressure behavior, and page faults.

Build and run the example:

.. code-block:: shell

   cmake -B build-unified-memory -S examples/unified-memory -DCMAKE_PREFIX_PATH=/opt/rocm
   cmake --build build-unified-memory

   HSA_XNACK=1 \
   ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON \
   ROCPROFSYS_TRACE=true \
   rocprof-sys-run -- ./build-unified-memory/unified-memory -s 32 -p 256 -i 4

Use ``ROCPROFSYS_OUTPUT_PATH`` when you want a predictable output directory:

.. code-block:: shell

   HSA_XNACK=1 \
   ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON \
   ROCPROFSYS_TRACE=true \
   ROCPROFSYS_OUTPUT_PATH=ump-output \
   rocprof-sys-run -- ./build-unified-memory/unified-memory -s 32 -p 256 -i 4

Output files
============

After a successful run, look in the ROCm Systems Profiler output directory for:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``unified_memory.txt``
     - Text summary with per-device migration rows, total page faults, and
       migration trigger counts.
   * - ``unified_memory.json``
     - Structured summary with ``devices`` and ``summary`` objects.
   * - ``perfetto-trace*.proto``
     - Perfetto trace with unified-memory page-fault counters and migration
       throughput counters when migration events are present.
   * - ``rocpd-*.db``
     - Optional ROCpd database output when ``ROCPROFSYS_USE_ROCPD=ON`` is set.

Sample text output
==================

The following sample shows a discrete-memory system where KFD page-migration
events were emitted:

.. code-block:: text

   ==12345== Unified Memory profiling result:
    Device "gfx950 (0)"
       Count  Avg Size  Min Size  Max Size  Total Size  Total Time    Migration Throughput  Name
          36  11.4670MB  2.0000MB  173.4062MB  412.8125MB  827.8592ms       0.52 GB/s  Host To Device
          32   1.9375MB  4.0000KB   2.0000MB   62.0000MB   39.1353ms       1.66 GB/s  Device To Host

    Total Page Faults: 33

    Migration Triggers:
      GPU page fault:         30
      CPU page fault:          2
      Prefetch:                1

``Migration Throughput`` is calculated as migrated bytes divided by KFD
page-migration event duration. It is an end-to-end migration-service metric and
can include page-fault handling, scheduling, replay, and driver overhead. Don't
interpret it as raw PCIe, XGMI, SDMA, HBM, or memory-subsystem bandwidth.

Sample JSON output
==================

The JSON output uses the same information in a stable structure:

.. code-block:: json

   {
     "devices": [
       {
         "device_id": 0,
         "device_name": "gfx950",
         "migrations": {
           "host_to_device": {
             "count": 36,
             "avg_size_bytes": 12024035.55,
             "min_size_bytes": 2097152,
             "max_size_bytes": 181829632,
             "total_size_bytes": 432865280,
             "total_time_ns": 827859200,
             "migration_throughput_gbps": 0.52
           },
           "device_to_host": {
             "count": 32,
             "avg_size_bytes": 2031616,
             "min_size_bytes": 4096,
             "max_size_bytes": 2097152,
             "total_size_bytes": 65011712,
             "total_time_ns": 39135300,
             "migration_throughput_gbps": 1.66
           },
           "device_to_device": {
             "count": 0,
             "avg_size_bytes": 0,
             "min_size_bytes": 0,
             "max_size_bytes": 0,
             "total_size_bytes": 0,
             "total_time_ns": 0,
             "migration_throughput_gbps": 0
           }
         }
       }
     ],
     "summary": {
       "total_page_faults": 33,
       "xnack_enabled": true,
       "migration_triggers": {
         "gpu_page_fault": 30,
         "cpu_page_fault": 2,
         "prefetch": 1,
         "ttm_eviction": 0,
         "unknown": 0
       }
     }
   }

Perfetto trace output
=====================

When ``ROCPROFSYS_TRACE=true`` is set, the Perfetto trace can include:

* ``Unified Memory Page Faults [Device N]`` counter tracks.
* ``Unified Memory Migration Throughput [Device N]`` counter tracks when KFD
  page-migration events are present.

Open the generated ``perfetto-trace*.proto`` file in the Perfetto UI to compare
page-fault activity, migration-throughput samples, HIP API calls, kernels, and
memory copies on one timeline.

Fault-only output on shared-HBM systems
=======================================

On MI300A and other systems where CPU and GPU agents point to the same physical
HBM, page faults can occur without page migrations because there is no separate
CPU memory and GPU memory to migrate between. In that topology, a valid
unified-memory report can be fault-only:

* ``total_page_faults`` can be nonzero.
* ``devices`` can be empty, or migration direction buckets can have zero counts.
* ``Unified Memory Page Faults`` can appear in Perfetto.
* ``Unified Memory Migration Throughput`` is not shown when no migration events
  are emitted.

Example fault-only text output:

.. code-block:: text

   ==12345== Unified Memory profiling result:
    Total Page Faults: 100

Example fault-only JSON output:

.. code-block:: json

   {
     "devices": [],
     "summary": {
       "total_page_faults": 100,
       "xnack_enabled": true,
       "migration_triggers": {
         "gpu_page_fault": 0,
         "cpu_page_fault": 0,
         "prefetch": 0,
         "ttm_eviction": 0,
         "unknown": 0
       }
     }
   }

Related memory profiling options
================================

Unified memory profiling is focused on KFD page-fault and page-migration
events. Combine it with ROCm API domains when you also need the surrounding
memory operations in the timeline or ROCpd database:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Domain
     - Use
   * - ``memory_copy``
     - Traces asynchronous memory-copy operations. This is useful for comparing
       explicit copies with unified-memory page migrations.
   * - ``memory_allocation``
     - Traces ROCm memory allocation and free operations, including virtual
       memory allocation records when supported by the ROCProfiler-SDK version.
   * - ``scratch_memory``
     - Traces kernel scratch-memory allocation activity and, on supported ROCm
       versions, emits scratch allocation-size counter tracks.

For a memory-focused trace, enable the related domains explicitly:

.. code-block:: shell

   HSA_XNACK=1 \
   ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON \
   ROCPROFSYS_TRACE=true \
   ROCPROFSYS_USE_ROCPD=ON \
   ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,memory_allocation,scratch_memory \
   rocprof-sys-run -- ./my_managed_memory_app

Troubleshooting
===============

No ``unified_memory.txt`` or ``unified_memory.json`` file was generated
---------------------------------------------------------------------------

Check that ``ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON`` and ``HSA_XNACK=1``
are set in the environment used to launch the target application. Also confirm
that the workload actually uses managed memory and produces KFD page-fault or
page-migration events.

Only page faults are shown
--------------------------

Fault-only output can be expected on shared-HBM systems such as MI300A. If you
expected migration rows on a discrete-memory system, confirm that XNACK is
enabled and that the workload moves managed-memory pages between CPU and GPU
accesses.

No migration-throughput track appears in Perfetto
-------------------------------------------------

The migration-throughput track is emitted only when KFD page-migration events
are present. Fault-only output can still be valid.

Unexpected overhead
-------------------

KFD event tracing and extra ROCm domains add overhead. Start with
``ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=ON`` and add
``memory_copy``, ``memory_allocation``, and ``scratch_memory`` only when those
events are needed for the analysis.

See also
========

* :doc:`Configuring runtime options <./configuring-runtime-options>` for the
  runtime option reference.
* :doc:`Using preset profiles and domain flags <./using-preset-profiles>` for
  ROCm domain flag usage.
* :doc:`Understanding ROCm Systems Profiler output <./understanding-rocprof-sys-output>`
  for output directory and trace-file details.
