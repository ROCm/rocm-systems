.. meta::
   :description: How to use ROCm Compute Profiler's profile mode
   :keywords: ROCm Compute Profiler, ROCm, profiler, tool, Instinct, accelerator, AMD,
              profiling, profile mode

************
Profile mode
************

The following chapter walks you through ROCm Compute Profiler's core profiling features by
example.

Learn about analysis with ROCm Compute Profiler in :doc:`../analyze/mode`. For an overview of
ROCm Compute Profiler's other modes, see :ref:`modes`.

Profiling
=========

Use the ``rocprof-compute`` executable to acquire all necessary performance monitoring
data through analysis of compute workloads.

Profiling with ROCm Compute Profiler provides the following benefits:

* :ref:`Automate counter collection <profiling-routine>`: ROCm Compute Profiler handles all
  of your profiling via pre-configured input files.

* :ref:`Profiling output format <profiling-output-format>`: ROCm Compute Profiler can control
  the output format of raw performance counter data produced by the underlying
  :doc:`ROCprofiler-SDK <rocprofiler-sdk:index>` backend. Supported output formats are
  ``csv`` and ``rocpd``. The default output format is ``rocpd``.

* :ref:`Filtering <filtering>`: Apply runtime filters to speed up the profiling
  process.

* :ref:`Standalone roofline <standalone-roofline>`: Isolate a subset of built-in
  metrics or build your own profiling configuration.

* :ref:`Iteration multiplexing <iteration-multiplexing>`: Collect a large number of
  performance counters with minimal profiling overhead.

Run ``rocprof-compute profile -h`` for more details. See
:ref:`Basic usage <modes-profile>`.

.. warning::

   **Kernel dispatches are serialized across HIP streams on the same GPU during profiling.**

   ROCm Compute Profiler collects GPU performance counters with kernel
   dispatch association which requires serializing kernel dispatches.
   Kernels launched on separate HIP streams on the same GPU will not execute
   concurrently during profiling. Streams on different GPUs are not
   serialized. This means:

   - Kernel duration and throughput metrics reflect serialized execution, not
     the concurrent behavior that may occur during normal execution.
   - Some metrics may show reduced utilization compared to normal execution
     due to the lack of concurrent kernel execution.

.. _profile-example:

Profiling example
-----------------

The `<https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-compute/sample/vcopy.cpp>`__ repository
includes source code for a sample GPU compute workload, ``vcopy.cpp``. A copy of
this file is available in the ``share/sample`` subdirectory after a normal
ROCm Compute Profiler installation, or via the ``$ROCPROFCOMPUTE_SHARE/sample`` directory when
using the supplied modulefile.

The examples in this section use a compiled version of the ``vcopy`` workload to
demonstrate the use of ROCm Compute Profiler in MI accelerator performance analysis. Unless
otherwise noted, the performance analysis is done on the
:ref:`MI325X platform <def-soc>`.

Workload compilation
^^^^^^^^^^^^^^^^^^^^

The following example demonstrates compilation of ``vcopy``.

.. code-block:: shell-session

   $ hipcc vcopy.cpp -o vcopy
   $ ls
   vcopy   vcopy.cpp
   $ ./vcopy -n 1048576 -b 256
   vcopy testing on GCD 0
   Finished allocating vectors on the CPU
   Finished allocating vectors on the GPU
   Finished copying vectors to the GPU
   sw thinks it moved 1.000000 KB per wave
   Total threads: 1048576, Grid Size: 4096 block Size:256, Wavefronts:16384:
   Launching the  kernel on the GPU
   Finished executing kernel
   Finished copying the output vector from the GPU to the CPU
   Releasing GPU memory
   Releasing CPU memory

The following sample command profiles the ``vcopy`` workload.

.. code-block:: shell-session

   $ rocprof-compute profile --name vcopy -- ./vcopy -n 1048576 -b 256
      INFO 6a57288d55

                                    __                                       _
   _ __ ___   ___ _ __  _ __ ___  / _|       ___ ___  _ __ ___  _ __  _   _| |_ ___
   | '__/ _ \ / __| '_ \| '__/ _ \| |_ _____ / __/ _ \| '_ ` _ \| '_ \| | | | __/ _ \
   | | | (_) | (__| |_) | | | (_) |  _|_____| (_| (_) | | | | | | |_) | |_| | ||  __/
   |_|  \___/ \___| .__/|_|  \___/|_|        \___\___/|_| |_| |_| .__/ \__,_|\__\___|
                  |_|                                           |_|

      INFO Rocprofiler-Compute version: 3.5.0
      INFO Profiler choice: rocprofiler-sdk
      INFO Path: /home/auser/rocm-systems/projects/rocprofiler-compute/workloads/vcopy/MI325X
      INFO Target: MI325X
      INFO Command: ./sample/vcopy -n 1048576 -b 256
      INFO Kernel Selection: None
      INFO Dispatch Selection: None
      INFO Filtered sections: All
      INFO
      INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      INFO Collecting Performance Counters
      INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      INFO
      INFO Using native counter collection tool: /tmp/rocprofiler-compute-tool-xxxxx/librocprofiler-compute-tool.so
      INFO [profiling] Iteration multiplexing: Disabled
      INFO [Run 1/13][Approximate profiling time left: pending first measurement...]
      INFO [profiling] Current input file: /home/auser/rocm-systems/projects/rocprofiler-compute/workloads/vcopy/MI325X/perfmon/SQC_DCACHE_INFLIGHT_LEVEL.txt
      INFO    |-> [rocprofiler-sdk] [rocprofiler-compute] [rocprofiler_configure] (priority=1) is using rocprofiler-sdk v1.1.0 (1.1.0)
      INFO    |-> [rocprofiler-sdk] W20260323 16:43:44.337323 139842239868672 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.250706 sec
      INFO    |-> [rocprofiler-sdk] [rocprofiler-compute] In tool init
      INFO    |-> [rocprofiler-sdk] W20260323 16:43:44.337534 139842239868672 simple_timer.cpp:55] [rocprofv3] './sample/vcopy -n 1048576 -b 256' ::     0.000000 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:43:44.511316 139842239868672 tool.cpp:2422] HSA version 8.21.0 initialized (instance=0)
      INFO    |-> [rocprofiler-sdk] W20260323 16:43:44.611214 139842239868672 simple_timer.cpp:55] [rocprofv3] './sample/vcopy -n 1048576 -b 256' ::     0.273680 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:43:44.626377 139842239868672 generateRocpd.cpp:582] writing SQL database for process 113640 on node 1574819130
      INFO    |-> [rocprofiler-sdk] E20260323 16:43:44.626821 139842239868672 generateRocpd.cpp:605] Opened result file: /home/auser/rocm-systems/projects/rocprofiler-compute/workloads/vcopy/MI325X/out/pmc_1/banff-ccs-aus-g05-05/113640_results.db (UUID=0000001e-7561-7561-8a76-754447167346)
      INFO    |-> [rocprofiler-sdk] W20260323 16:43:44.662283 139842239868672 simple_timer.cpp:55] SQLite3 generation :: rocpd_string             ::     0.008452 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:43:44.662683 139842239868672 simple_timer.cpp:55] SQLite3 generation :: rocpd_info_node          ::     0.000389 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:43:44.663665 139842239868672 simple_timer.cpp:55] SQLite3 generation :: rocpd_info_process       ::     0.000977 sec

   ...

      INFO    |-> [rocprofiler-sdk] Finished executing kernel
      INFO    |-> [rocprofiler-sdk] Finished copying the output vector from the GPU to the CPU
      INFO    |-> [rocprofiler-sdk] Releasing GPU memory
      INFO    |-> [rocprofiler-sdk] Releasing CPU memory
      INFO [Run 13/13][Approximate profiling time left: 0 seconds]...
      INFO [profiling] Current input file: /home/auser/rocm-systems/projects/rocprofiler-compute/workloads/vcopy/MI325X/perfmon/pmc_perf_5.txt
      INFO    |-> [rocprofiler-sdk] [rocprofiler-compute] [rocprofiler_configure] (priority=1) is using rocprofiler-sdk v1.1.0 (1.1.0)
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:43.871622 140315166887680 simple_timer.cpp:55] [rocprofv3] tool initialization ::     0.224905 sec
      INFO    |-> [rocprofiler-sdk] [rocprofiler-compute] In tool init
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:43.871808 140315166887680 simple_timer.cpp:55] [rocprofv3] './sample/vcopy -n 1048576 -b 256' ::     0.000000 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.049909 140315166887680 tool.cpp:2422] HSA version 8.21.0 initialized (instance=0)
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.147303 140315166887680 simple_timer.cpp:55] [rocprofv3] './sample/vcopy -n 1048576 -b 256' ::     0.275496 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.162131 140315166887680 generateRocpd.cpp:582] writing SQL database for process 116379 on node 1574819130
      INFO    |-> [rocprofiler-sdk] E20260323 16:44:44.162543 140315166887680 generateRocpd.cpp:605] Opened result file: /home/auser/rocm-systems/projects/rocprofiler-compute/workloads/vcopy/MI325X/out/pmc_1/banff-ccs-aus-g05-05/116379_results.db (UUID=0000001f-5e13-7e13-9cab-4e867beb6f39)
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.195657 140315166887680 simple_timer.cpp:55] SQLite3 generation :: rocpd_string             ::     0.006754 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.196516 140315166887680 simple_timer.cpp:55] SQLite3 generation :: rocpd_info_node          ::     0.000842 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.197639 140315166887680 simple_timer.cpp:55] SQLite3 generation :: rocpd_info_process       ::     0.001077 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.214007 140315166887680 simple_timer.cpp:55] SQLite3 generation :: rocpd_info_agent         ::     0.015730 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.248623 140315166887680 simple_timer.cpp:55] SQLite3 generation :: rocpd_info_pmc           ::     0.034606 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.249465 140315166887680 simple_timer.cpp:55] SQLite3 generation :: rocpd kernel info        ::     0.000832 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.249471 140315166887680 simple_timer.cpp:55] SQLite3 generation :: rocpd_region             ::     0.000002 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.255495 140315166887680 simple_timer.cpp:55] SQLite3 generation :: rocpd_kernel_dispatch    ::     0.006021 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.255503 140315166887680 simple_timer.cpp:55] SQLite3 generation :: rocpd_pmc_event          ::     0.000000 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.255505 140315166887680 simple_timer.cpp:55] SQLite3 generation :: rocpd_memory_copy        ::     0.000000 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.255507 140315166887680 simple_timer.cpp:55] SQLite3 generation :: rocpd_memory_allocate    ::     0.000001 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.255584 140315166887680 simple_timer.cpp:55] SQLite3 generation :: SQL indexing             ::     0.000076 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.255888 140315166887680 simple_timer.cpp:55] SQLite3 generation :: total                    ::     0.093758 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.257594 140315166887680 simple_timer.cpp:55] [rocprofv3] output generation ::     0.106959 sec
      INFO    |-> [rocprofiler-sdk] W20260323 16:44:44.257624 140315166887680 simple_timer.cpp:55] [rocprofv3] tool finalization ::     0.110297 sec
      INFO    |-> [rocprofiler-sdk] [rocprofiler-compute] In tool fini
      INFO    |-> [rocprofiler-sdk] [rocprofiler-compute] [generate_output] Counter collection data has been written to: /home/auser/rocm-systems/projects/rocprofiler-compute/workloads/vcopy/MI325X/out/pmc_1/116379_native_counter_collection.csv
      INFO    |-> [rocprofiler-sdk] vcopy testing on GCD 0
      INFO    |-> [rocprofiler-sdk] Finished allocating vectors on the CPU
   WARNING PC sampling data collection skipped as block 21 is not specified.
      INFO [roofline] Checking for roofline.csv in /home/auser/rocm-systems/projects/rocprofiler-compute/workloads/vcopy/MI325X
   GPU Device 0 (gfx942) with 304 CUs: Profiling...
   100% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||]
   HBM BW, GPU ID: 0, workgroupSize:256, workgroups:6225920, experiments:100, traffic:25501368320 bytes, duration:5.4 ms, mean:4705.4 GB/sec, stdev:22.1 GB/sec
   100% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||]
   MALL BW, GPU ID: 0, workgroupSize:256, workgroups:38912, experiments:100, traffic:2611340115968 bytes, duration:403.9 ms, mean:6466.1 GB/sec, stdev:26.892396 GB/sec
   100% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||]
   L2 BW, GPU ID: 0, workgroupSize:256, workgroups:38912, experiments:100, traffic:1632087572480 bytes, duration:57.3 ms, mean:28505.3 GB/sec, stdev:27.182463 GB/sec
   100% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||]
   L1 BW, GPU ID: 0, workgroupSize:256, workgroups:38912, experiments:100, traffic:127506841600 bytes, duration:3.4 ms, mean:37782.9 GB/sec, stdev:688.291117 GB/sec

   ...

   100% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||]
   Peak MFMA FLOPs (F32), GPU ID: 0, workgroupSize:256, workgroups:38912, experiments:100, FLOP:1275068416000, duration:8.23 ms, mean:154854.6 GFLOPS, stdev:1782.6 GFLOPS
   100% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||]
   Peak MFMA FLOPs (F64), GPU ID: 0, workgroupSize:256, workgroups:38912, experiments:100, FLOP:637534208000, duration:4.09 ms, mean:155983.3 GFLOPS, stdev:1811.2 GFLOPS
   100% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||]
   Peak MFMA IOPs (I8), GPU ID: 0, workgroupSize:256, workgroups:38912, experiments:100, IOP:10200547328000, duration:4.52 ms, mean:2258954.1 GOPS, stdev:41800.0 GFLOPS
      INFO [roofline] Roofline data saved to /home/auser/rocm-systems/projects/rocprofiler-compute/workloads/vcopy/MI325X/roofline.csv
   Run 'rocprof-compute analyze -p /home/auser/rocm-systems/projects/rocprofiler-compute/workloads/vcopy/MI325X' for charts

.. tip::

   To reduce verbosity of profiling output try the ``--quiet`` flag. This hides
   ``rocprofiler-sdk`` output and activates a progress bar.

.. _profiling-routine:

Notice the two main stages in ROCm Compute Profiler's **default** profiling routine.

1. The first stage collects all the counters needed for ROCm Compute Profiler analysis
   (omitting any filters you have provided).

2. The second stage collects data for the roofline analysis (this stage can be
   disabled using ``--no-roof``).

At the end of profiling, you can find all resulting ``csv`` files in a
:ref:`SoC <def-soc>`-specific target directory; for
example:

* "MI300A" or "MI300X" for the AMD Instinct MI300 Series GPUs
* "MI200" for the AMD Instinct MI200 Series GPUs
* "MI100" for the AMD Instinct MI100 Series GPUs

The SoC names are generated as a part of ROCm Compute Profiler, and do not *always*
distinguish between different GPUs in the same family; for instance,
an Instinct MI210 vs an Instinct MI250.

.. note::

   Additionally, you will notice a few extra files. An SoC parameters file,
   ``sysinfo.csv``, is created to reflect the target device settings. All
   profiling output is stored in ``log.txt``. Roofline-specific benchmark
   results are stored in ``roofline.csv``. To generate roofline HTML plots,
   run ``rocprof-compute analyze`` on the profiling output directory
   (see :doc:`../analyze/mode`).

.. code-block:: shell-session

   $ ls workloads/vcopy/MI325X/
   total 408
   -rw-r--r-- 1 auser agroup   55771 Mar 21 23:49 log.txt
   drwxr-xr-x 1 auser agroup    4096 Mar 21 23:47 perfmon
   -rw-r--r-- 1 auser agroup  348790 Mar 21 23:48 pmc_perf.csv
   -rw-r--r-- 1 auser agroup    1119 Mar 21 23:47 profiling_config.yaml
   -rw-r--r-- 1 auser agroup    1684 Mar 21 23:49 roofline.csv
   -rw-r--r-- 1 auser agroup     899 Mar 21 23:47 sysinfo.csv

Output directory configuration
------------------------------

Profile mode writes results into a workload directory. By default, the output
directory is derived from ``--name`` and the target system information:

* Without MPI rank detection, the default is ``./workloads/<name>/<gpu_model>``.
* With MPI rank detection, the default is ``./workloads/<name>/<rank>``.

You can override the output directory with ``--output-directory``. The
``--path`` (``-p``) argument is deprecated for profile mode. When ``--output-directory`` is
explicitly provided, ``--name`` is ignored.

.. note::

   ``--path`` and ``--subpath`` are deprecated for profile mode and will be
   removed in a future release. Use ``--output-directory`` with parameterized
   placeholders instead.

The output directory can be parameterized with the following keywords:

* ``%hostname%``: Host name
* ``%gpumodel%``: GPU model
* ``%rank%``: MPI process rank (ignored with a warning if no rank is detected)
* ``%env{NAME}%``: Environment variable ``NAME`` (empty string if unset)

If MPI rank is detected and the output directory does not include ``%rank%``,
ROCm Compute Profiler appends ``/<rank>`` to avoid collisions across ranks.

Examples:

* Profiling without MPI:

.. code-block:: shell-session

   $ rocprof-compute profile --name vcopy -- ./vcopy -n 1048576 -b 256

   $ tree workloads/vcopy

   └── MI325X
    ├── log.txt
    ├── perfmon
    │   ├── pmc_perf_0.txt
    │   ├── pmc_perf_0.yaml
    │   ├── pmc_perf_1.txt
    │   ├── pmc_perf_1.yaml
    │   ├── pmc_perf_2.txt
    │   ├── pmc_perf_2.yaml
    │   ├── pmc_perf_3.txt
    │   ├── pmc_perf_3.yaml
    │   ├── pmc_perf_4.txt
    │   ├── pmc_perf_4.yaml
    │   ├── pmc_perf_5.txt
    │   ├── SQC_DCACHE_INFLIGHT_LEVEL.txt
    │   ├── SQC_DCACHE_INFLIGHT_LEVEL.yaml
    │   ├── SQC_ICACHE_INFLIGHT_LEVEL.txt
    │   ├── SQC_ICACHE_INFLIGHT_LEVEL.yaml
    │   ├── SQ_IFETCH_LEVEL.txt
    │   ├── SQ_IFETCH_LEVEL.yaml
    │   ├── SQ_INST_LEVEL_LDS.txt
    │   ├── SQ_INST_LEVEL_LDS.yaml
    │   ├── SQ_INST_LEVEL_SMEM.txt
    │   ├── SQ_INST_LEVEL_SMEM.yaml
    │   ├── SQ_INST_LEVEL_VMEM.txt
    │   ├── SQ_INST_LEVEL_VMEM.yaml
    │   ├── SQ_LEVEL_WAVES.txt
    │   └── SQ_LEVEL_WAVES.yaml
    ├── profiling_config.yaml
    ├── results_pmc_perf_0.csv
    ├── results_pmc_perf_1.csv
    ├── results_pmc_perf_2.csv
    ├── results_SQ_LEVEL_WAVES.csv
    ├── roofline.csv
    └── sysinfo.csv

The output files use the default ``rocpd`` format. See :ref:`profiling-output-format` for details
on available output formats and when the final ``pmc_perf.csv`` is created.

* Profiling with MPI at host ``amd-ryzen``:

.. code-block:: shell-session

   $ mpirun -n 4 rocprof-compute profile --output-directory /tmp/profiles/%hostname%/%rank% -- ./vcopy -n 1048576 -b 256

   $ tree /tmp/profiles/amd-ryzen/0

   └── MI325X
    ├── log.txt
    ├── perfmon
    │   ├── pmc_perf_0.txt
    │   ├── pmc_perf_0.yaml
    │   ├── pmc_perf_1.txt
    │   ├── pmc_perf_1.yaml
    │   ├── pmc_perf_2.txt
    │   ├── pmc_perf_2.yaml
    │   ├── pmc_perf_3.txt
    │   ├── pmc_perf_3.yaml
    │   ├── pmc_perf_4.txt
    │   ├── pmc_perf_4.yaml
    │   ├── pmc_perf_5.txt
    │   ├── SQC_DCACHE_INFLIGHT_LEVEL.txt
    │   ├── SQC_DCACHE_INFLIGHT_LEVEL.yaml
    │   ├── SQC_ICACHE_INFLIGHT_LEVEL.txt
    │   ├── SQC_ICACHE_INFLIGHT_LEVEL.yaml
    │   ├── SQ_IFETCH_LEVEL.txt
    │   ├── SQ_IFETCH_LEVEL.yaml
    │   ├── SQ_INST_LEVEL_LDS.txt
    │   ├── SQ_INST_LEVEL_LDS.yaml
    │   ├── SQ_INST_LEVEL_SMEM.txt
    │   ├── SQ_INST_LEVEL_SMEM.yaml
    │   ├── SQ_INST_LEVEL_VMEM.txt
    │   ├── SQ_INST_LEVEL_VMEM.yaml
    │   ├── SQ_LEVEL_WAVES.txt
    │   └── SQ_LEVEL_WAVES.yaml
    ├── profiling_config.yaml
    ├── results_pmc_perf_0.csv
    ├── results_pmc_perf_1.csv
    ├── results_pmc_perf_2.csv
    ├── results_SQ_LEVEL_WAVES.csv
    ├── roofline.csv
    └── sysinfo.csv

.. _profiling-output-format:

Profiling output format
-----------------------

Use the ``--format-rocprof-output <format>`` profile mode option to specify the output format
of raw performance counter data produced by the underlying
:doc:`ROCprofiler-SDK <rocprofiler-sdk:index>` backend. The following formats are supported:

* ``csv`` format:
   * Instructs ROCprofiler-SDK to write raw performance counter data in CSV format.
   * Generates separate CSV files for each profiling run (``pmc_perf_0.csv``, ``pmc_perf_1.csv``, ``SQ_*.csv``, etc.) in the workload directory.
   * These files are merged into a single ``pmc_perf.csv`` file when running ``rocprof-compute analyze``.

* ``rocpd`` format (default):
   * Instructs ROCprofiler-SDK to write raw performance counter data in rocpd (SQLite) format.
   * The rocpd database files are converted to CSV files (``results_pmc_perf_0.csv``, ``results_SQ_*.csv``, etc.) for each profiling run, after which the database files are removed.
   * These files are merged into a single ``pmc_perf.csv`` file when running ``rocprof-compute analyze``.
   * Use ``--retain-rocpd-output`` to preserve the ``rocpd`` database(s) in the workload folder for custom analysis.


.. _filtering:

Filtering
=========

To reduce profiling time and the counters collected, use profiling filters.
The filters described below apply to the default ``rocprofiler-sdk`` backend.
See :ref:`core-install-rocprof-var` for details on backend selection.

Filtering options
-----------------

``-b``, ``--block <block-id|block-alias|metric-id>``
   Allows system profiling on one or more selected analysis report blocks to speed
   up the profiling process. See :ref:`profiling-hw-component-filtering`.
   Note that this option cannot be used with ``--roof-only`` or ``--set``.

``-k``, ``--kernel <kernel-substr>``
   Allows for kernel filtering. See :ref:`profiling-kernel-filtering`.

``-d``, ``--dispatch <dispatch-id>``
   Allows for dispatch iteration filtering. See :ref:`profiling-dispatch-filtering`.

``--set <metric-set>``
   Allows for single pass counter collection of sets of metrics with minimized profiling overhead.
   Cannot be used with ``--roof-only`` or ``--block``.
   See :ref:`profiling-metric-sets`.

.. tip::

   Be cautious when combining different profiling filters in the same call.
   Conflicting filters may result in error.

   For example, filtering a dispatch, but that dispatch doesn't match your
   kernel name filter.

.. _profiling-hw-component-filtering:

Analysis report block filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can profile specific hardware report blocks to speed up the profiling process.
In ROCm Compute Profiler, the term analysis report block refers to a section of the
analysis report which focuses on metrics associated with a hardware component or
a group of hardware components. All profiling results are accumulated in the same
target directory without overwriting those for other hardware components.
This enables incremental profiling and analysis.

The following example only gathers hardware counters used to calculate metrics
for ``Compute Unit - Instruction Mix`` (block 10) and ``Wavefront Launch Statistics``
(block 7) sections of the analysis report, while skipping over all other hardware counters.

.. code-block:: shell-session

   $ rocprof-compute profile --name vcopy -b 10 7 -- ./vcopy -n 1048576 -b 256

                                    __                                       _
    _ __ ___   ___ _ __  _ __ ___  / _|       ___ ___  _ __ ___  _ __  _   _| |_ ___
   | '__/ _ \ / __| '_ \| '__/ _ \| |_ _____ / __/ _ \| '_ ` _ \| '_ \| | | | __/ _ \
   | | | (_) | (__| |_) | | | (_) |  _|_____| (_| (_) | | | | | | |_) | |_| | ||  __/
   |_|  \___/ \___| .__/|_|  \___/|_|        \___\___/|_| |_| |_| .__/ \__,_|\__\___|
                  |_|                                           |_|

   INFO Rocprofiler-Compute version: 3.5.0
   INFO Profiler choice: rocprofiler-sdk
   INFO Path: /home/auser/rocprofiler-compute/workloads/vcopy/MI325X
   INFO Target: MI325X
   INFO Command: ./vcopy -n 1048576 -b 256
   INFO Kernel Selection: None
   INFO Dispatch Selection: None
   INFO Filtered sections: ['10', '7']
   INFO
   INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   INFO Collecting Performance Counters
   INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ...


It is also possible to collect individual metrics from the analysis report by providing metric ids.
The following example only collects the counters required to calculate ``Total VALU FLOPs`` (metric id 11.1.0) and ``LDS Utilization`` (metric id 12.1.0).

.. code-block:: shell-session

   $ rocprof-compute profile --name vcopy -b 11.1.1 12.1.1 -- ./vcopy -n 1048576 -b 256

                                    __                                       _
    _ __ ___   ___ _ __  _ __ ___  / _|       ___ ___  _ __ ___  _ __  _   _| |_ ___
   | '__/ _ \ / __| '_ \| '__/ _ \| |_ _____ / __/ _ \| '_ ` _ \| '_ \| | | | __/ _ \
   | | | (_) | (__| |_) | | | (_) |  _|_____| (_| (_) | | | | | | |_) | |_| | ||  __/
   |_|  \___/ \___| .__/|_|  \___/|_|        \___\___/|_| |_| |_| .__/ \__,_|\__\___|
                  |_|                                           |_|

   INFO Rocprofiler-Compute version: 3.5.0
   INFO Profiler choice: rocprofiler-sdk
   INFO Path: /home/auser/rocprofiler-compute/workloads/vcopy/MI325X
   INFO Target: MI325X
   INFO Command: ./vcopy -n 1048576 -b 256
   INFO Kernel Selection: None
   INFO Dispatch Selection: None
   INFO Filtered sections: ['11.1.1', '12.1.1']
   INFO
   INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   INFO Collecting Performance Counters
   INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ...


To see a list of available hardware report blocks, use the ``--list-available-metrics`` option.

.. code-block:: shell-session

   $ rocprof-compute profile --list-available-metrics

                                    __                                       _
    _ __ ___   ___ _ __  _ __ ___  / _|       ___ ___  _ __ ___  _ __  _   _| |_ ___
   | '__/ _ \ / __| '_ \| '__/ _ \| |_ _____ / __/ _ \| '_ ` _ \| '_ \| | | | __/ _ \
   | | | (_) | (__| |_) | | | (_) |  _|_____| (_| (_) | | | | | | |_) | |_| | ||  __/
   |_|  \___/ \___| .__/|_|  \___/|_|        \___\___/|_| |_| |_| .__/ \__,_|\__\___|
                  |_|                                           |_|

   0 -> Top Stats
   1 -> System Info
   2 -> System Speed-of-Light
         2.1 -> Speed-of-Light
                  2.1.0 -> VALU FLOPs
                  2.1.1 -> VALU IOPs
                  2.1.2 -> MFMA FLOPs (F8)
   ...
   5 -> Command Processor (CPC/CPF)
         5.1 -> Command Processor Fetcher
                  5.1.0 -> CPF Utilization
                  5.1.1 -> CPF Stall
                  5.1.2 -> CPF-L2 Utilization
         5.2 -> Packet Processor
                  5.2.0 -> CPC Utilization
                  5.2.1 -> CPC Stall Rate
                  5.2.5 -> CPC-UTCL1 Stall
   ...
   6 -> Workgroup Manager (SPI)
         6.1 -> Workgroup Manager Utilizations
                  6.1.0 -> Accelerator Utilization
                  6.1.1 -> Scheduler-Pipe Utilization
                  6.1.2 -> Workgroup Manager Utilization



.. _profiling-kernel-filtering:

Kernel filtering
^^^^^^^^^^^^^^^^

Kernel filtering is based on the name of the kernels you want to isolate. Use a
kernel name substring list to isolate desired kernels.

The following example demonstrates profiling isolating the kernel matching
substring ``vecCopy``.

.. code-block:: shell-session

   $ rocprof-compute profile --name vcopy -k vecCopy -- ./vcopy -n 1048576 -b 256

                                    __                                       _
    _ __ ___   ___ _ __  _ __ ___  / _|       ___ ___  _ __ ___  _ __  _   _| |_ ___
   | '__/ _ \ / __| '_ \| '__/ _ \| |_ _____ / __/ _ \| '_ ` _ \| '_ \| | | | __/ _ \
   | | | (_) | (__| |_) | | | (_) |  _|_____| (_| (_) | | | | | | |_) | |_| | ||  __/
   |_|  \___/ \___| .__/|_|  \___/|_|        \___\___/|_| |_| |_| .__/ \__,_|\__\___|
                  |_|                                           |_|

   INFO Rocprofiler-Compute version: 3.5.0
   INFO Profiler choice: rocprofiler-sdk
   INFO Path: /home/auser/rocprofiler-compute/workloads/vcopy/MI325X
   INFO Target: MI325X
   INFO Command: ./vcopy -n 1048576 -b 256
   INFO Kernel Selection: ['vecCopy']
   INFO Dispatch Selection: None
   INFO Filtered sections: All
   INFO
   INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   INFO Collecting Performance Counters
   INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ...

.. _profiling-dispatch-filtering:

Dispatch filtering
^^^^^^^^^^^^^^^^^^

Dispatch filtering is based on the *global* dispatch index of kernels in a run.

The following example profiles only the first kernel dispatch in the execution
of the application (zero-based indexing).

.. code-block:: shell-session

   $ rocprof-compute profile --name vcopy -d 0 -- ./vcopy -n 1048576 -b 256

                                    __                                       _
    _ __ ___   ___ _ __  _ __ ___  / _|       ___ ___  _ __ ___  _ __  _   _| |_ ___
   | '__/ _ \ / __| '_ \| '__/ _ \| |_ _____ / __/ _ \| '_ ` _ \| '_ \| | | | __/ _ \
   | | | (_) | (__| |_) | | | (_) |  _|_____| (_| (_) | | | | | | |_) | |_| | ||  __/
   |_|  \___/ \___| .__/|_|  \___/|_|        \___\___/|_| |_| |_| .__/ \__,_|\__\___|
                  |_|                                           |_|

   INFO Rocprofiler-Compute version: 3.5.0
   INFO Profiler choice: rocprofiler-sdk
   INFO Path: /home/auser/rocprofiler-compute/workloads/vcopy/MI325X
   INFO Target: MI325X
   INFO Command: ./vcopy -n 1048576 -b 256
   INFO Kernel Selection: None
   INFO Dispatch Selection: ['0']
   INFO Filtered sections: All
   INFO
   INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   INFO Collecting Performance Counters
   INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ...

.. _profiling-metric-sets:

Metric sets filtering
^^^^^^^^^^^^^^^^^^^^^^^

A metrics set contains a subset of metrics that can be collected in a single pass. This filtering option minimizes profiling overhead by only collecting counters of interest.
The `--set` filter option provides a convenient way to group related metrics for common profiling scenarios, eliminating the need to manually specify individual metrics for typical analysis workflows.
This option cannot be used with ``--roof-only`` and ``--block``.

.. code-block:: shell-session

   $ rocprof-compute profile --name vcopy --set compute_thruput_util -- ./vcopy -n 1048576 -b 256

                                    __                                       _
    _ __ ___   ___ _ __  _ __ ___  / _|       ___ ___  _ __ ___  _ __  _   _| |_ ___
   | '__/ _ \ / __| '_ \| '__/ _ \| |_ _____ / __/ _ \| '_ ` _ \| '_ \| | | | __/ _ \
   | | | (_) | (__| |_) | | | (_) |  _|_____| (_| (_) | | | | | | |_) | |_| | ||  __/
   |_|  \___/ \___| .__/|_|  \___/|_|        \___\___/|_| |_| |_| .__/ \__,_|\__\___|
                  |_|                                           |_|

   INFO Rocprofiler-Compute version: 3.5.0
   INFO Profiler choice: rocprofiler-sdk
   INFO Path: /home/auser/rocprofiler-compute/workloads/vcopy/MI325X
   INFO Target: MI325X
   INFO Command: ./vcopy -n 1048576 -b 256
   INFO Kernel Selection: None
   INFO Dispatch Selection: None
   INFO Filtered sections: ['11.2.2', '11.2.3', '11.2.4', '11.2.5']
   INFO
   INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   INFO Collecting Performance Counters
   INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ...


To see a list of available sets, use the ``--list-sets`` option.

.. code-block:: shell-session

   $ rocprof-compute profile --list-sets

                                    __                                       _
    _ __ ___   ___ _ __  _ __ ___  / _|       ___ ___  _ __ ___  _ __  _   _| |_ ___
   | '__/ _ \ / __| '_ \| '__/ _ \| |_ _____ / __/ _ \| '_ ` _ \| '_ \| | | | __/ _ \
   | | | (_) | (__| |_) | | | (_) |  _|_____| (_| (_) | | | | | | |_) | |_| | ||  __/
   |_|  \___/ \___| .__/|_|  \___/|_|        \___\___/|_| |_| |_| .__/ \__,_|\__\___|
                  |_|                                           |_|

   Available Sets:
   ===================================================================================================================
   Set Option                          Set Title                           Metric Name                    Metric ID
   -------------------------------------------------------------------------------------------------------------------
   compute_thruput_util                Compute Throughput Utilization      SALU Utilization               11.2.3
                                                                           VALU Utilization               11.2.4
                                                                           VMEM Utilization               11.2.6
                                                                           Branch Utilization             11.2.7

   ...

   launch_stats                        Launch Stats                        Grid Size                      7.1.0
                                                                           Workgroup Size                 7.1.1
                                                                           Total Wavefronts               7.1.2
                                                                           VGPRs                          7.1.5
                                                                           AGPRs                          7.1.6
                                                                           SGPRs                          7.1.7
                                                                           LDS Allocation                 7.1.8
                                                                           Scratch Allocation             7.1.9

   Usage Examples:
   rocprof-compute profile --set compute_thruput_util  # Profile this set
   rocprof-compute profile --list-sets        # Show this help


.. _standalone-roofline:

Standalone roofline
===================

Roofline analysis occurs on any profile mode run, provided ``--no-roof`` option is not included.
You don't need to include any additional roofline-specific options for roofline analysis.
If you want to focus only on roofline-specific performance data and reduce the time it takes to profile, you can use the ``--roof-only`` option.
This option checks if there is existing roofline benchmark data in the workload directory (``roofline.csv``):

a) If found, skips microbenchmark execution;

b) Otherwise, profile mode runs microbenchmarks and collects roofline performance counters.

.. note::

  ``--roof-only`` cannot be used with ``--block`` or ``--set`` options.

Profile mode generates ``roofline.csv`` containing microbenchmark data. To generate
roofline HTML plots, use ``rocprof-compute analyze`` on the profiling output directory
(see :doc:`../analyze/mode`). Visualization options (``--sort``, ``--mem-level``,
``--roofline-data-type``) are available in analyze mode.

Roofline options (profile)
--------------------------

``--device <gpu_id>``
   Allows you to specify a device ID to collect performance data from when
   running a roofline benchmark on your system.

``-k``, ``--kernel <kernel-substr>``
   Allows for kernel filtering. See :ref:`profiling-kernel-filtering`.

.. note::

  For more information on data types supported based on the GPU architecture, see :doc:`../../conceptual/performance-model`


Roofline only
-------------

The following example demonstrates profiling roofline data only:

.. code-block:: shell-session

   $ rocprof-compute profile --name occupancy --roof-only -- ./tests/occupancy -n 1048576 -b 256
                                    __                                       _
    _ __ ___   ___ _ __  _ __ ___  / _|       ___ ___  _ __ ___  _ __  _   _| |_ ___
   | '__/ _ \ / __| '_ \| '__/ _ \| |_ _____ / __/ _ \| '_ ` _ \| '_ \| | | | __/ _ \
   | | | (_) | (__| |_) | | | (_) |  _|_____| (_| (_) | | | | | | |_) | |_| | ||  __/
   |_|  \___/ \___| .__/|_|  \___/|_|        \___\___/|_| |_| |_| .__/ \__,_|\__\___|
                  |_|                                           |_|
   ...
   INFO [roofline] Generating pmc_perf.csv (roofline counters only).
   INFO Rocprofiler-Compute version: 3.5.0
   INFO Profiler choice: rocprofiler-sdk
   INFO Path: /home/auser/rocprofiler-compute/workloads/occupancy/MI325X
   INFO Target: MI325X
   INFO Command: ./tests/occupancy -n 1048576 -b 256
   INFO Kernel Selection: None
   INFO Dispatch Selection: None
   INFO Filtered sections: ['4']
   INFO
   INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   INFO Collecting Performance Counters (Roofline Only)
   INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   INFO
   INFO [Run 1/3][Approximate profiling time left: pending first measurement...]
   INFO [profiling] Current input file: /home/auser/rocprofiler-compute/workloads/occupancy/MI325X/perfmon/pmc_perf_0.txt
   ...
   INFO [roofline] Checking for roofline.csv in /home/auser/rocprofiler-compute/workloads/occupancy/MI325X
   INFO [roofline] No roofline data found. Generating...
   Empirical Roofline Calculation
   Copyright © 2026  Advanced Micro Devices, Inc. All rights reserved.
   Total detected GPU devices: 8
   GPU Device 0 (gfx942) with 304 CUs: Profiling...
   99% [||||||||||||||||||||||||||||||||||||||||||||||||||||||||||| ]
   ...


An inspection of our workload output folder shows ``roofline.csv`` was generated
successfully.

.. code-block:: shell-session

   $ ls workloads/occupancy/MI325X
   total 48
   drwxr-xr-x 1 auser agroup     0 Mar 21 23:49 perfmon
   -rw-r--r-- 1 auser agroup  1101 Mar 21 23:49 pmc_perf.csv
   -rw-r--r-- 1 auser agroup  1715 Mar 21 23:49 roofline.csv
   -rw-r--r-- 1 auser agroup   650 Mar 21 23:49 sysinfo.csv
   -rw-r--r-- 1 auser agroup   399 Mar 21 23:49 timestamps.csv

To generate roofline HTML plots from this data, see :doc:`../analyze/mode`.

.. _torch-operator-mapping:

Torch operator mapping
========================

ROCm Compute Profiler offers Torch operator mapping functionality to analyze the performance metrics at the PyTorch operator level. This feature maps the performance counters to specific PyTorch operators, enabling detailed performance analysis of
the PyTorch workloads at the operator granularity.

When enabled, this feature instruments your PyTorch application to correlate GPU
kernel executions with their originating PyTorch operators, providing insights into
which operators contribute to specific performance counter values.

.. warning::

   Torch trace is currently an experimental feature. You must pass ``--experimental`` to
   both **profile** and **analyze** command when using the Torch trace related options
   (``--torch-trace`` for profile; ``--list-torch-operators`` and ``--torch-operator``
   for analyze).

.. note::

   **Mapping PyTorch operators to GPU kernels**: PyTorch operators (such as ``conv2d``,
   ``linear``, and ``relu``) are high-level API functions. When executed on GPU, these
   operators may launch multiple low-level GPU kernels (such as
   ``implicit_convolve_sgemm``) that perform the actual computation on the hardware.
   The ``--torch-trace`` option provides operator-level attribution by injecting
   markers that map the collected kernel performance counters to their originating PyTorch
   operators.

Requirements
------------

* Valid PyTorch installation in the profiling environment.
* PyTorch application must be run as a Python script or a Python command.
* Workload’s Python version must match roctx’s Python version.

Usage
-----

To enable Torch operator mapping, use ``--experimental`` with the ``--torch-trace``
option when profiling a PyTorch workload:

.. code-block:: shell-session

   $ rocprof-compute --experimental profile --name mnist_torch --torch-trace -- python train.py

                                    __                                       _
    _ __ ___   ___ _ __  _ __ ___  / _|       ___ ___  _ __ ___  _ __  _   _| |_ ___
   | '__/ _ \ / __| '_ \| '__/ _ \| |_ _____ / __/ _ \| '_ ` _ \| '_ \| | | | __/ _ \
   | | | (_) | (__| |_) | | | (_) |  _|_____| (_| (_) | | | | | | |_) | |_| | ||  __/
   |_|  \___/ \___| .__/|_|  \___/|_|        \___\___/|_| |_| |_| .__/ \__,_|\__\___|
                  |_|                                           |_|

   INFO Rocprofiler-Compute version: 3.5.0
   INFO Profiler choice: rocprofiler-sdk
   INFO Path: /home/auser/workloads/mnist_torch/MI325X
   INFO Target: MI325X
   INFO Command: python train.py
   INFO Torch Trace: Enabled
   INFO Kernel Selection: None
   INFO Dispatch Selection: None
   INFO Hardware Blocks: All
   INFO
   INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   INFO Collecting Performance Counters
   INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   ...

Output
------

When Torch operator mapping is enabled, profiling writes additional CSV files in
the workload directory: **marker_api_trace** and **counter_collection** files with
the ``torch_trace`` prefix. These correlate PyTorch operators
with GPU kernels and performance counters. When you run analyze (e.g. with
``--list-torch-operators`` or ``--torch-operator``), a consolidated CSV is written
to ``torch_trace/consolidated.csv``; the source marker and counter files are
**retained** in the workload directory and are not deleted.

``torch_trace/`` directory
The ``torch_trace/`` directory contains ``consolidated.csv`` with all
operator/kernel data. The columns include:

   * ``Operator_Name``: Full operator hierarchy (e.g. ``nn.Module.Net.forward/nn.Module.Conv2d.forward/torch.nn.functional.relu``, ``nn.Module.ResNet.forward/torch.nn.functional.relu``).
   * ``Context_Id``: Call context (e.g., ``1@__init__.py:231``)
   * ``Counter_Name`` / ``Counter_Value``: Performance counter values
   * ``Start_Timestamp_function`` / ``End_Timestamp_function``: Operator timing
   * ``Start_Timestamp_kernel`` / ``End_Timestamp_kernel``: Kernel timing

The consolidated CSV is generated automatically on the first analysis run that
requires it (``--list-torch-operators`` or ``--torch-operator``) and is reused on
subsequent runs.

Sample rows from ``torch_trace/consolidated.csv`` (from profiling an mnist model).

.. list-table::
   :header-rows: 1
   :widths: 16 14 42 22 12 14 14 14 14

   * - Operator_Name
     - Context_Id
     - Kernel_Name
     - Counter_Name
     - Counter_Value
     - Start_Timestamp_function
     - End_Timestamp_function
     - Start_Timestamp_kernel
     - End_Timestamp_kernel

   * - torch.ones_like
     - 1@__init__.py:231
     - ``void at::native::vectorized_elementwise_kernel<...>(...)``
     - CPC_CPC_STAT_BUSY
     - 23004
     - 6789210204040073
     - 6789210223815845
     - 6789210223810274
     - 6789210223811914

   * - torch.ones_like
     - 1@__init__.py:231
     - ``void at::native::vectorized_elementwise_kernel<...>(...)``
     - CPC_CPC_STAT_IDLE
     - 0
     - 6789210204040073
     - 6789210223815845
     - 6789210223810274
     - 6789210223811914

   * - torch.ones_like
     - 1@__init__.py:231
     - ``void at::native::vectorized_elementwise_kernel<...>(...)``
     - CPC_CPC_STAT_STALL
     - 6715
     - 6789281060081123
     - 6789281079930585
     - 6789281079932564
     - 6789281079934204

Performance counter data file
-----------------------------

The ``pmc_perf.csv`` file contains the standard performance counter data (same as non-torch profiling). This data enables analysis such as:

* Identifying which PyTorch operators executed which GPU kernels
* Aggregating performance counter values by operator
* Correlating operator-level timing with kernel-level hardware metrics
* Tracing the execution flow from high-level PyTorch API to low-level GPU kernels

Limitations
-----------

The Torch trace feature currently has the following limitations:

* Torch trace is experimental. Use ``rocprof-compute --experimental profile ... --torch-trace`` and ``rocprof-compute --experimental analyze ...`` with ``--list-torch-operators`` or ``--torch-operator`` as needed.

* The ``--torch-trace`` option requires the application to be a Python command or Python script.

* A valid PyTorch installation must be available in the environment where the workload runs.

* The workload’s Python version must match the Python version used by ``roctx``.

* This feature adds instrumentation overhead to track operator boundaries. For performance-critical measurements, consider profiling without this option first.


.. _torch-operator-profiling:

Hierarchical operator names
----------------------------

PyTorch operators are captured with full module hierarchy when available (e.g.,
``nn.Module`` and ``torch.nn.functional`` wrappers), so you see where each
operator occurs in your PyTorch application:

.. code-block:: text

   nn.Module.Net.forward/nn.Module.Conv2d.forward/torch.nn.functional.conv2d
   nn.Module.MyModel.forward/nn.Module.Linear.forward
   torch.nn.functional.relu

The ``Operator_Name`` column in ``torch_trace/consolidated.csv`` contains
the full operator hierarchy.

This hierarchical information enables:

* **Context preservation**: See exactly which model layer triggered each kernel.
* **Debugging**: Identify performance issues in specific model components.
* **Optimization**: Focus tuning efforts on bottleneck operators.

Example with hierarchical naming:

.. code-block:: python

   class MyModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.encoder = nn.Linear(512, 1024)
           self.decoder = nn.Linear(1024, 512)

       def forward(self, x):
           x = self.encoder(x)  # Captured as nn.Module.MyModel.forward/nn.Module.Linear.forward
           x = self.decoder(x)  # Same hierarchy; both appear in consolidated.csv under Operator_Name
           return x

**Analyzing captured operators**: After profiling, use the analyze CLI (see
:doc:`../analyze/cli`) to list and filter by operator name. Filtering
(``--torch-operator``) accepts PurePosixPath glob patterns (e.g. ``*conv2d``,
``torch.nn.functional.conv2d``, ``*/*conv2d``). To select all operators, pass
no arguments, ``all``, ``*``, or ``**`` — all four forms are equivalent.

Combining Torch operator with other options
-------------------------------------------

Torch operator mapping can be combined with other profiling options. Use
``--experimental`` with ``--torch-trace`` in all cases:

.. code-block:: shell-session

   # Combine with block filtering for targeted counter collection
   $ rocprof-compute --experimental profile --name mnist --torch-trace -b 11 12 -- python train.py

   # Combine with iteration multiplexing
   $ rocprof-compute --experimental profile --name mnist --torch-trace --iteration-multiplexing kernel -- python train.py

   # Combine with kernel filtering (filters by GPU kernel name)
   $ rocprof-compute --experimental profile --name mnist --torch-trace -k elementwise -- python train.py

.. _iteration-multiplexing:

Iteration multiplexing
========================

To reduce profiling overhead when collecting a large number of performance counters,
ROCm Compute Profiler supports iteration multiplexing. This technique divides the
total set of requested performance counters into smaller subsets that can be collected
over multiple iterations of the kernel execution, thereby preventing the need for
application replay. Each iteration collects a different subset of counters, and the
results are later combined to provide a comprehensive view of the performance metrics.

.. note::

   Iteration multiplexing is most beneficial for large workloads that take a long time to run,
   as it helps reduce profiling overhead by eliminating the need for application replay while
   spreading counter collection across iterations. For small workloads with few kernel dispatches,
   iteration multiplexing may result in incomplete metric calculations due to insufficient kernel
   dispatch counts to cover all counter subsets.

Usage
-----

To enable iteration multiplexing in ROCm Compute Profiler, use the
``--iteration-multiplexing`` option in your profiling command. You can optionally specify
the policy for multiplexing. The available policies are:

* ``kernel``
   The counters are divided based on the kernels being executed. Each kernel call
   for a particular kernel collects a different subset of counters.
* ``kernel_launch_params``
   The counters are divided based on both the kernels and their launch parameters.
   This allows for more granular control over counter collection. Each unique combination of kernel and launch
   parameters collects a different subset of counters.

By default, if no policy is specified, ROCm Compute Profiler uses the ``kernel_launch_params`` policy.

.. note::

   * Iteration multiplexing requires ROCprofiler-SDK from ROCm 7.0.0 or later and is only
     supported with the ``rocprofiler-sdk`` backend.

   * Iteration multiplexing depends on the :ref:`native counter collection tool
     <core-install-native-tool>`. Do not use ``--no-native-tool`` with
     ``--iteration-multiplexing``.

   * Do not use ``--attach-pid`` with ``--iteration-multiplexing``. Ensure that ``--attach-pid`` is not used in your profiling command.

   * Ensure that your workload runs for enough iterations to cover all counter subsets.
     When using iteration multiplexing, the total number of iterations, for each kernel (for ``kernel`` policy)
     or for each unique kernel and launch parameters combination (for ``kernel_launch_params`` policy),
     specified in the workload should be sufficient to cover all subsets of counters. If the number of iterations
     is too low, some counters may not be collected.

   * Launch parameters for ``kernel_launch_params`` policy.
     Launch parameters refer to the following paramaters:

     - Grid size
     - Workgroup size
     - LDS size

The following example demonstrates how to use iteration multiplexing with the
``vcopy`` workload:

.. code-block:: shell-session

   $ rocprof-compute profile --name vcopy --iteration-multiplexing kernel -- ./vcopy -i 20 -n 1048576 -b 256

                                    __                                       _
    _ __ ___   ___ _ __  _ __ ___  / _|       ___ ___  _ __ ___  _ __  _   _| |_ ___
   | '__/ _ \ / __| '_ \| '__/ _ \| |_ _____ / __/ _ \| '_ ` _ \| '_ \| | | | __/ _ \
   | | | (_) | (__| |_) | | | (_) |  _|_____| (_| (_) | | | | | | |_) | |_| | ||  __/
   |_|  \___/ \___| .__/|_|  \___/|_|        \___\___/|_| |_| |_| .__/ \__,_|\__\___|
                  |_|                                           |_|

   INFO Rocprofiler-Compute version: 3.5.0
   INFO Profiler choice: rocprofiler-sdk
   INFO Path: /home/auser/rocprofiler-compute/workloads/vcopy_kernel/MI325X
   INFO Target: MI325X
   INFO Command: ./vcopy -i 20 -n 1048576 -b 256
   INFO Kernel Selection: None
   INFO Dispatch Selection: None
   INFO Filtered sections: All
   INFO
   INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   INFO Collecting Performance Counters
   INFO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   INFO
   INFO Using native counter collection tool: /tmp/rocprofiler-compute-tool-xxxxx/librocprofiler-compute-tool.so
   INFO [profiling] Iteration multiplexing: kernel
   INFO [profiling] Current input files: .../perfmon/SQC_DCACHE_INFLIGHT_LEVEL.txt, .../perfmon/pmc_perf_0.txt, ...
   INFO    |-> [rocprofiler-sdk] [rocprofiler-compute] [rocprofiler_configure] (priority=1) is using rocprofiler-sdk v1.1.0 (1.1.0)
   INFO    |-> [rocprofiler-sdk] [rocprofiler-compute] In tool init
   INFO    |-> [rocprofiler-sdk] vcopy testing on GCD 0
   INFO    |-> [rocprofiler-sdk] Finished allocating vectors on the CPU
   INFO    |-> [rocprofiler-sdk] Finished allocating vectors on the GPU
   INFO    |-> [rocprofiler-sdk] ...
   INFO    |-> [rocprofiler-sdk] [rocprofiler-compute] In tool fini
   ...


Caveats
---------

Iteration multiplexing feature comes with some caveats to be considered when profiling any workload:

* **Accuracy vs speed trade-off**

  Iteration multiplexing provides a trade-off with decreased profiling time by eliminating application replay while sacrificing accuracy since only a handful of counters can be collected per kernel dispatch; while we test for closeness in metric values with and without iteration multiplexing in our automatic test suite, more accurate results can be obtained by not using iteration multiplexing.

* **Minimum number of kernel dispatches required**

  When using iteration multiplexing it is recommended to filter by kernel(s) of interest and make sure these kernels are dispatched enough times (50 recommended) to cover all counter subsets (currently around 15); a warning is thrown for kernels with insufficient dispatch counts to warn the user about missing counter data for those kernels, and it is not possible to calculate some metrics for these kernels.

* **Non-deterministic workloads**

  Workloads which dispatch kernels with non-deterministic names and launch parameters may trigger warnings for insufficient dispatch counts because iteration multiplexing identifies unique kernels by their names and optionally by their launch parameters; this is especially true of large AI workloads that dispatch kernels non-deterministically based on the model layers being used for the current input, and in such cases kernel filtering of common kernels is recommended.

Multi-rank profiling
========================

When profiling MPI workloads, ROCm Compute Profiler can isolate outputs by rank.
If a rank is detected and no rank placeholder is provided, each rank writes to a
subdirectory named by its rank to avoid output collisions.

Example usage
-------------

Some examples of using multi-rank profiling are:

* **With** ``--output-directory`` **option:**

.. code-block:: shell-session

   $ mpirun -n 4 rocprof-compute profile --output-directory /tmp/mpi_profile -- ./laplace_eqn -n 1048576 -b 256

The example above produces:

.. code-block:: shell-session

   $ ls /tmp/mpi_profile
   0  1  2  3

   $ tree /tmp/mpi_profile/0

   └── MI325X
    ├── log.txt
    ├── perfmon
    │   ├── pmc_perf_0.txt
    │   ├── pmc_perf_0.yaml
    │   ├── pmc_perf_1.txt
    │   ├── pmc_perf_1.yaml
    │   ├── pmc_perf_2.txt
    │   ├── pmc_perf_2.yaml
    │   ├── pmc_perf_3.txt
    │   ├── pmc_perf_3.yaml
    │   ├── pmc_perf_4.txt
    │   ├── pmc_perf_4.yaml
    │   ├── pmc_perf_5.txt
    │   ├── SQC_DCACHE_INFLIGHT_LEVEL.txt
    │   ├── SQC_DCACHE_INFLIGHT_LEVEL.yaml
    │   ├── SQC_ICACHE_INFLIGHT_LEVEL.txt
    │   ├── SQC_ICACHE_INFLIGHT_LEVEL.yaml
    │   ├── SQ_IFETCH_LEVEL.txt
    │   ├── SQ_IFETCH_LEVEL.yaml
    │   ├── SQ_INST_LEVEL_LDS.txt
    │   ├── SQ_INST_LEVEL_LDS.yaml
    │   ├── SQ_INST_LEVEL_SMEM.txt
    │   ├── SQ_INST_LEVEL_SMEM.yaml
    │   ├── SQ_INST_LEVEL_VMEM.txt
    │   ├── SQ_INST_LEVEL_VMEM.yaml
    │   ├── SQ_LEVEL_WAVES.txt
    │   └── SQ_LEVEL_WAVES.yaml
    ├── pmc_perf_0.csv
    ├── pmc_perf_1.csv
    ├── pmc_perf_2.csv
    ├── profiling_config.yaml
    ├── roofline.csv
    └── sysinfo.csv

* **With** ``--name`` **option:**

.. code-block:: shell-session

   $ mpirun -n 4 rocprof-compute profile --name laplace -- ./laplace_eqn -n 1048576 -b 256

The example above produces:

.. code-block:: shell-session

   $ ls ./workloads/laplace_eqn
   0  1  2  3

   $ tree ./workloads/laplace_eqn/0

   └── MI325X
    ├── log.txt
    ├── perfmon
    │   ├── pmc_perf_0.txt
    │   ├── pmc_perf_0.yaml
    │   ├── pmc_perf_1.txt
    │   ├── pmc_perf_1.yaml
    │   ├── pmc_perf_2.txt
    │   ├── pmc_perf_2.yaml
    │   ├── pmc_perf_3.txt
    │   ├── pmc_perf_3.yaml
    │   ├── pmc_perf_4.txt
    │   ├── pmc_perf_4.yaml
    │   ├── pmc_perf_5.txt
    │   ├── SQC_DCACHE_INFLIGHT_LEVEL.txt
    │   ├── SQC_DCACHE_INFLIGHT_LEVEL.yaml
    │   ├── SQC_ICACHE_INFLIGHT_LEVEL.txt
    │   ├── SQC_ICACHE_INFLIGHT_LEVEL.yaml
    │   ├── SQ_IFETCH_LEVEL.txt
    │   ├── SQ_IFETCH_LEVEL.yaml
    │   ├── SQ_INST_LEVEL_LDS.txt
    │   ├── SQ_INST_LEVEL_LDS.yaml
    │   ├── SQ_INST_LEVEL_SMEM.txt
    │   ├── SQ_INST_LEVEL_SMEM.yaml
    │   ├── SQ_INST_LEVEL_VMEM.txt
    │   ├── SQ_INST_LEVEL_VMEM.yaml
    │   ├── SQ_LEVEL_WAVES.txt
    │   └── SQ_LEVEL_WAVES.yaml
    ├── pmc_perf_0.csv
    ├── pmc_perf_1.csv
    ├── pmc_perf_2.csv
    ├── profiling_config.yaml
    ├── roofline.csv
    └── sysinfo.csv


To control output placement explicitly, add ``%rank%`` (and other placeholders)
to your output directory. The following example is run on the host ``amd-ryzen``:

.. code-block:: shell-session

   $ mpirun -n 4 rocprof-compute profile --output-directory /tmp/mpi_profile/%hostname%/%rank% -- ./laplace_eqn -n 1048576 -b 256

   $ ls /tmp/mpi_profile/amd-ryzen/
   0  1  2  3

   $ tree /tmp/mpi_profile/amd-ryzen/0

   └── MI325X
    ├── log.txt
    ├── perfmon
    │   ├── pmc_perf_0.txt
    │   ├── pmc_perf_0.yaml
    │   ├── pmc_perf_1.txt
    │   ├── pmc_perf_1.yaml
    │   ├── pmc_perf_2.txt
    │   ├── pmc_perf_2.yaml
    │   ├── pmc_perf_3.txt
    │   ├── pmc_perf_3.yaml
    │   ├── pmc_perf_4.txt
    │   ├── pmc_perf_4.yaml
    │   ├── pmc_perf_5.txt
    │   ├── SQC_DCACHE_INFLIGHT_LEVEL.txt
    │   ├── SQC_DCACHE_INFLIGHT_LEVEL.yaml
    │   ├── SQC_ICACHE_INFLIGHT_LEVEL.txt
    │   ├── SQC_ICACHE_INFLIGHT_LEVEL.yaml
    │   ├── SQ_IFETCH_LEVEL.txt
    │   ├── SQ_IFETCH_LEVEL.yaml
    │   ├── SQ_INST_LEVEL_LDS.txt
    │   ├── SQ_INST_LEVEL_LDS.yaml
    │   ├── SQ_INST_LEVEL_SMEM.txt
    │   ├── SQ_INST_LEVEL_SMEM.yaml
    │   ├── SQ_INST_LEVEL_VMEM.txt
    │   ├── SQ_INST_LEVEL_VMEM.yaml
    │   ├── SQ_LEVEL_WAVES.txt
    │   └── SQ_LEVEL_WAVES.yaml
    ├── pmc_perf_0.csv
    ├── pmc_perf_1.csv
    ├── pmc_perf_2.csv
    ├── profiling_config.yaml
    ├── roofline.csv
    └── sysinfo.csv

ROCm Compute Profiler supports the following libraries, APIs, and job schedulers:

* OpenMPI
* MPICH
* MVAPICH2
* Slurm
* Flux Core
* PMI
* PMIx
* PALS

For other MPI implementations or job schedulers, use the ``%env{NAME}%``
placeholder to include environment variables that identify the rank. For example,
if your MPI implementation sets the ``MY_MPI_RANK`` environment variable, you can
specify the output directory as:

.. code-block:: shell-session

   $ mpirun -n 4 rocprof-compute profile --output-directory /tmp/mpi_profile/%env{MY_MPI_RANK}% -- ./my_mpi_application

Caveats
-----------------------------

When profiling multi-rank applications, be aware of the following caveats:

**MPI launcher placement:**

MPI launchers (``mpirun``, ``mpiexec``, ``srun``, and ``orterun``) must wrap the
``rocprof-compute`` command, not appear after ``--``. The following is incorrect:

.. code-block:: shell-session

   $ rocprof-compute profile --name my_app -- mpirun -n 4 ./my_application   # WRONG

Instead, use the correct form where the MPI launcher wraps ``rocprof-compute``:

.. code-block:: shell-session

   $ mpirun -n 4 rocprof-compute profile --name my_app -- ./my_application   # CORRECT

If you use an MPI launcher after ``--``, an error will be raised with guidance
on the correct usage.

**Application replay mode (default):**

By default, ROCm Compute Profiler uses application replay mode, which runs the
workload multiple times to collect all performance counters. This mode fails
for MPI applications because running the application multiple times results in
multiple ``MPI_Init`` and ``MPI_Finalize`` calls, which is not permitted by the
MPI specification.

**PC Sampling:**

PC sampling (block 21) may fail to collect data for multi-rank applications with
MPI communication due to synchronization requirements.

**Recommended single-pass modes:**

For multi-rank applications with MPI communication, use one of these single-pass
profiling modes:

* ``--iteration-multiplexing``: Collects all counters in a single application run by distributing counter collection across kernel dispatches. Recommended for applications with sufficient kernel dispatch counts.

.. code-block:: shell-session

   $ mpirun -n 4 rocprof-compute profile --name my_mpi_app --iteration-multiplexing -- ./my_mpi_app

* ``--set <name>``: Profiles a predefined counter set that fits in a single pass.

.. code-block:: shell-session

   $ mpirun -n 4 rocprof-compute profile --name my_mpi_app --set compute_thruput_util -- ./my_mpi_app

**Multi-node profiling:**

When profiling across multiple nodes, ensure that:

* Output directories are accessible from all nodes (shared filesystem), or
* Use node-specific output directories with ``%hostname%`` placeholder.

.. code-block:: shell-session

   $ mpirun -n 8 --hostfile hosts.txt rocprof-compute profile \
       --output-directory /shared/profiles/%hostname%/%rank% -- ./my_mpi_app
