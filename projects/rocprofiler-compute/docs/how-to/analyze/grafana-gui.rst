.. meta::
   :description: ROCm Compute Profiler analysis: Grafana GUI
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, profiler, tool,
              Instinct, accelerator, Grafana, panels, GUI, import

********************
Grafana GUI analysis
********************

.. warning::

   Grafana and MongoDB functionality is deprecated and will be removed in a future release.

Find setup instructions in :doc:`../../install/grafana-setup`.

The ROCm Compute Profiler Grafana analysis dashboard GUI supports the following features to
facilitate MI accelerator performance profiling and analysis:

* System and hardware component (hardware block)

* Speed-of-Light (SOL)

* Multiple normalization options

* Baseline comparisons

* Regex-based dispatch ID filtering

* Roofline analysis

* Detailed performance counters and metrics per hardware component, such as:

  * Command Processor - Fetch (CPF) / Command Processor - Controller (CPC)

  * Workgroup Manager (SPI)

  * Shader Sequencer (SQ)

  * Shader Sequencer Controller (SQC)

  * L1 Address Processing Unit, a.k.a. Texture Addresser (TA) / L1 Backend Data
    Processing Unit, a.k.a. Texture Data (TD)

  * L1 Cache (TCP)

  * L2 Cache (TCC) (both aggregated and per-channel perf info)

See the full list of :ref:`ROCm Compute Profiler's analysis panels <panels>`.

.. _analysis-sol:

Speed-of-Light
--------------

Speed-of-Light panels are provided at both the system and per hardware component
level to help diagnosis performance bottlenecks. The performance numbers of the
workload under testing are compared to the theoretical maximum, such as floating
point operations, bandwidth, cache hit rate, etc., to indicate the available
room to further utilize the hardware capability.

.. _analysis-normalizations:

Normalizations
--------------

Multiple performance number normalizations are provided to allow performance
inspection within both hardware and software context. The following
normalizations are available.

* ``per_wave``

* ``per_cycle``

* ``per_kernel``

* ``per_second``

See :ref:`normalization-units` to learn more about ROCm Compute Profiler normalizations.

.. _analysis-baseline-comparison:

Baseline comparison
-------------------

ROCm Compute Profiler enables baseline comparison to allow checking A/B effect. Currently
baseline comparison is limited to the same :ref:`SoC <def-soc>`. Cross
comparison between SoCs is in development.

For both the Current Workload and the Baseline Workload, you can independently
setup the following filters to allow fine grained comparisons:

* Workload Name

* GPU ID filtering (multi-selection)

* Kernel Name filtering (multi-selection)

* Dispatch ID filtering (regex filtering)

* ROCm Compute Profiler Panels (multi-selection)

.. _analysis-regex-dispatch-id:

Regex-based dispatch ID filtering
---------------------------------

ROCm Compute Profiler allows filtering via Regular Expressions (regex), a standard Linux
string matching syntax, based dispatch ID filtering to flexibly choose the
kernel invocations.

For example, to inspect Dispatch Range from 17 to 48, inclusive, the
corresponding regex is : ``(1[7-9]|[23]\d|4[0-8])``.

.. tip::

   Try `Regex Numeric Range Generator <https://3widgets.com/>`_ for help
   generating typical number ranges.

.. _analysis-incremental-profiling:

Incremental profiling
---------------------

ROCm Compute Profiler supports incremental profiling to speed up performance analysis.

Refer to the :ref:`profiling-hw-component-filtering` section for this command.

By default, the entire application is profiled to collect performance counters
for all hardware blocks, giving a complete view of where the workload stands in
terms of performance optimization opportunities and bottlenecks.

You can choose to focus on only a few hardware components -- for example L1
cache or LDS -- to closely check the effect of software optimizations, without
performing application replay for *all* other hardware components. This saves
a lot of compute time. In addition, prior profiling results for other hardware
components are not overwritten; instead, they can be merged during the import to
piece together an overall profile of the system.

.. _analysis-color-coding:

Color coding
------------

Uniform color coding applies to most visualizations -- including bar graphs,
tables, and diagrams -- for easy inspection. As a rule of thumb, *yellow* means
over 50%, while *red* means over 90% percent.

Global variables and configurations
-----------------------------------

.. image:: ../../data/analyze/global_variables.png
   :align: center
   :alt: ROCm Compute Profiler global variables and configurations
   :width: 800

.. _grafana-gui-import:

Grafana GUI import
------------------

The ROCm Compute Profiler database ``--import`` option imports the raw profiling data to
Grafana's backend MongoDB database. This step is only required for Grafana
GUI-based performance analysis.

Default username and password for MongoDB (to be used in database mode) are as
follows:

* **Username**: ``temp``

* **Password**: ``temp123``

Each workload is imported to a separate database with the following naming
convention:

.. code-block:: shell

   rocprofiler-compute_<team>_<database>_<soc>

For example:

.. code-block:: shell

   rocprofiler-compute_asw_vcopy_mi200

When using :ref:`database mode <modes-database>`, be sure to tailor the
connection options to the machine hosting your
:doc:`server-side instance </install/grafana-setup>`. Below is the sample
command to import the *vcopy* profiling data, assuming our host machine is
called ``dummybox``.

.. _grafana-gui-remove:

.. code-block:: shell-session

   $ rocprof-compute database --help
   usage:

   rocprof-compute database <interaction type> [connection options]



   -------------------------------------------------------------------------------

   Examples:

           rocprof-compute database --import -H pavii1 -u temp -t asw -w workloads/vcopy/mi200/

           rocprof-compute database --remove -H pavii1 -u temp -w rocprofiler-compute_asw_sample_mi200

   -------------------------------------------------------------------------------



   Help:
     -h, --help         show this help message and exit

   General Options:
     -v, --version      show program's version number and exit
     -V, --verbose      Increase output verbosity (use multiple times for higher levels)
     -s, --specs        Print system specs.

   Interaction Type:
     -i, --import                                  Import workload to ROCm Compute Profiler DB
     -r, --remove                                  Remove a workload from ROCm Compute Profiler DB

   Connection Options:
     -H , --host                                   Name or IP address of the server host.
     -P , --port                                   TCP/IP Port. (DEFAULT: 27018)
     -u , --username                               Username for authentication.
     -p , --password                               The user's password. (will be requested later if it's not set)
     -t , --team                                   Specify Team prefix.
     -w , --workload                               Specify name of workload (to remove) or path to workload (to import)
     --kernel-verbose              Specify Kernel Name verbose level 1-5. Lower the level, shorter the kernel name. (DEFAULT: 5) (DISABLE: 5)


ROCm Compute Profiler import for vcopy:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell-session

   $ rocprof-compute database --import -H dummybox -u temp -t asw -w workloads/vcopy/mi200/

                                    __                                       _
    _ __ ___   ___ _ __  _ __ ___  / _|       ___ ___  _ __ ___  _ __  _   _| |_ ___
   | '__/ _ \ / __| '_ \| '__/ _ \| |_ _____ / __/ _ \| '_ ` _ \| '_ \| | | | __/ _ \
   | | | (_) | (__| |_) | | | (_) |  _|_____| (_| (_) | | | | | | |_) | |_| | ||  __/
   |_|  \___/ \___| .__/|_|  \___/|_|        \___\___/|_| |_| |_| .__/ \__,_|\__\___|
                  |_|                                           |_|


   Pulling data from  /home/auser/repos/rocprofiler-compute/sample/workloads/vcopy/MI200
   The directory exists
   Found sysinfo file
   KernelName shortening enabled
   Kernel name verbose level: 2
   Password:
   Password received
   -- Conversion & Upload in Progress --
     0%|                                                                                                                                                                                                             | 0/11 [00:00<?, ?it/s]/home/auser/repos/rocprofiler-compute/sample/workloads/vcopy/MI200/SQ_IFETCH_LEVEL.csv
     9%|█████████████████▉                                                                                                                                                                                   | 1/11 [00:00<00:01,  8.53it/s]/home/auser/repos/rocprofiler-compute/sample/workloads/vcopy/MI200/pmc_perf.csv
    18%|███████████████████████████████████▊                                                                                                                                                                 | 2/11 [00:00<00:01,  6.99it/s]/home/auser/repos/rocprofiler-compute/sample/workloads/vcopy/MI200/SQ_INST_LEVEL_SMEM.csv
    27%|█████████████████████████████████████████████████████▋                                                                                                                                               | 3/11 [00:00<00:01,  7.90it/s]/home/auser/repos/rocprofiler-compute/sample/workloads/vcopy/MI200/SQ_LEVEL_WAVES.csv
    36%|███████████████████████████████████████████████████████████████████████▋                                                                                                                             | 4/11 [00:00<00:00,  8.56it/s]/home/auser/repos/rocprofiler-compute/sample/workloads/vcopy/MI200/SQ_INST_LEVEL_LDS.csv
    45%|█████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                           | 5/11 [00:00<00:00,  9.00it/s]/home/auser/repos/rocprofiler-compute/sample/workloads/vcopy/MI200/SQ_INST_LEVEL_VMEM.csv
    55%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                                                         | 6/11 [00:00<00:00,  9.24it/s]/home/auser/repos/rocprofiler-compute/sample/workloads/vcopy/MI200/sysinfo.csv
    64%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                       | 7/11 [00:00<00:00,  9.37it/s]/home/auser/repos/rocprofiler-compute/sample/workloads/vcopy/MI200/roofline.csv
    82%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                   | 9/11 [00:00<00:00, 12.60it/s]/home/auser/repos/rocprofiler-compute/sample/workloads/vcopy/MI200/timestamps.csv
   100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 11/11 [00:00<00:00, 11.05it/s]
   9 collections added.
   Workload name uploaded
   -- Complete! --

.. _panels:

ROCm Compute Profiler panels
----------------------------

There are currently 18 main panel categories available for analyzing the compute
workload performance. Each category contains several panels for close inspection
of the system performance.

- :ref:`Kernel Statistics <grafana-panel-kernel-stats>`

  - Kernel time histogram

  - Top ten bottleneck kernels

- :ref:`System Speed-of-Light <grafana-panel-system-sol>`

  - Speed-of-Light

  - System Info table

- :ref:`Memory Chart Analysis <grafana-panel-memory-chart-analysis>`

- :ref:`Roofline Analysis <grafana-panel-roofline-analysis>`

  - FP32/FP64

  - FP16/INT8

- :ref:`Command Processor <grafana-panel-cp>`

  - Command Processor - Fetch (CPF)

  - Command Processor - Controller (CPC)

- :ref:`Workgroup Manager or Shader Processor Input (SPI) <grafana-panel-spi>`

  - SPI Stats

  - SPI Resource Allocations

- :ref:`Wavefront Launch <grafana-panel-wavefront>`

  - Wavefront Launch Stats

  - Wavefront runtime stats

  - per-SE Wavefront Scheduling performance

- :ref:`Wavefront Lifetime <grafana-panel-wavefront>`

  - Wavefront lifetime breakdown

  - per-SE wavefront life (average)

  - per-SE wavefront life (histogram)

- :ref:`Wavefront Occupancy <grafana-panel-wavefront>`

  - per-SE wavefront occupancy

  - per-CU wavefront occupancy

- :ref:`Compute Unit - Instruction Mix <grafana-panel-cu-instruction-mix>`

  - per-wave Instruction mix

  - per-wave VALU Arithmetic instruction mix

  - per-wave MFMA Arithmetic instruction mix

- :ref:`Compute Unit - Compute Pipeline <grafana-panel-cu-compute-pipeline>`

  - Speed-of-Light: Compute Pipeline

  - Arithmetic OPs count

  - Compute pipeline stats

  - Memory latencies

- :ref:`Local Data Share (LDS) <grafana-panel-lds>`

  - Speed-of-Light: LDS

  - LDS stats

- :ref:`Instruction Cache <grafana-panel-instruction-cache>`

  - Speed-of-Light: Instruction Cache

  - Instruction Cache Accesses

- Constant Cache

  - Speed-of-Light: Constant Cache

  - Constant Cache Accesses

  - Constant Cache - L2 Interface stats

- :ref:`Texture Addresser and Texture Data <grafana-panel-ta>`

  - Texture Addresser (TA)

  - Texture Data (TD)

- L1 Cache

  - Speed-of-Light: L1 Cache

  - L1 Cache Accesses

  - L1 Cache Stalls

  - L1 - L2 Transactions

  - L1 - UTCL1 Interface stats

- :ref:`L2 Cache <grafana-panel-l2-cache>`

  - Speed-of-Light: L2 Cache

  - L2 Cache Accesses

  - L2 - EA Transactions

  - L2 - EA Stalls

- :ref:`L2 Cache Per Channel Performance <grafana-panel-l2-cache-per-channel>`

  - Per-channel L2 Hit rate

  - Per-channel L1-L2 Read requests

  - Per-channel L1-L2 Write Requests

  - Per-channel L1-L2 Atomic Requests

  - Per-channel L2-EA Read requests

  - Per-channel L2-EA Write requests

  - Per-channel L2-EA Atomic requests

  - Per-channel L2-EA Read latency

  - Per-channel L2-EA Write latency

  - Per-channel L2-EA Atomic latency

  - Per-channel L2-EA Read stall (I/O, GMI, HBM)

  - Per-channel L2-EA Write stall (I/O, GMI, HBM, Starve)

Most panels are designed around a specific hardware component block to
thoroughly understand its behavior. Additional panels, including custom panels,
could also be added to aid the performance analysis.

.. _grafana-panel-sys-info:

System Info
^^^^^^^^^^^

.. figure:: ../../data/analyze/grafana/system-info_panel.png
   :align: center
   :alt: System details logged from the host machine
   :width: 800

   System details logged from the host machine.

.. _grafana-panel-kernel-stats:

Kernel Statistics
^^^^^^^^^^^^^^^^^

Kernel Time Histogram
+++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/Kernel_time_histogram.png
   :align: center
   :alt: Kernel time histogram panel in ROCm Compute Profiler Grafana
   :width: 800

   Mapping application kernel launches to execution duration.

Top Bottleneck Kernels
++++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/top-stat_panel.png
   :align: center
   :alt: Top bottleneck kernels panel in ROCm Compute Profiler Grafana
   :width: 800

   Top N kernels and relevant statistics. Sorted by total duration.

Top Bottleneck Dispatches
+++++++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/Top_bottleneck_dispatches.png
   :align: center
   :alt: Top bottleneck dispatches panel in ROCm Compute Profiler Grafana
   :width: 800

   Top N kernel dispatches and relevant statistics. Sorted by total duration.

Current and Baseline Dispatch IDs (Filtered)
++++++++++++++++++++++++++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/Current_and_baseline_dispatch_ids.png
   :align: center
   :alt: Current and baseline dispatch IDs panel in ROCm Compute Profiler Grafana
   :width: 800

   List of all kernel dispatches.

.. _grafana-panel-system-sol:

System Speed-of-Light
^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../data/analyze/grafana/sol_panel.png
   :align: center
   :alt: System Speed-of-Light panel in ROCm Compute Profiler Grafana
   :width: 800

   Key metrics from various sections of ROCm Compute Profiler’s profiling report.

.. tip::

   See :doc:`/conceptual/system-speed-of-light` to learn about reported metrics.

.. _grafana-panel-memory-chart-analysis:

Memory Chart Analysis
^^^^^^^^^^^^^^^^^^^^^

.. note::

   The Memory Chart Analysis support multiple normalizations. Due to limited
   space, all transactions, when normalized to ``per_sec``, default to unit of
   billion transactions per second.

.. figure:: ../../data/analyze/grafana/memory-chart_panel.png
   :align: center
   :alt: Memory Chart Analysis panel in ROCm Compute Profiler Grafana
   :width: 800

   A graphical representation of performance data for memory blocks on the GPU.


.. _grafana-panel-roofline-analysis:

Empirical Roofline Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../../data/analyze/grafana/roofline_panel.png
   :align: center
   :alt: Roofline Analysis panel in ROCm Compute Profiler Grafana
   :width: 800

   Visualize achieved performance relative to a benchmarked peak performance.


.. _grafana-panel-cp:

Command Processor
^^^^^^^^^^^^^^^^^

.. tip::

   See :doc:`/conceptual/command-processor` to learn about reported metrics.

Command Processor Fetcher
+++++++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/cpc_panel.png
   :align: center
   :alt: Command Processor Fetcher panel in ROCm Compute Profiler Grafana
   :width: 800

   Fetches commands out of memory to hand them over to the Command Processor
   Fetcher (CPC) for processing

Command Processor Compute
+++++++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/cpf_panel.png
   :align: center
   :alt: Command Processor Compute panel in ROCm Compute Profiler Grafana
   :width: 800

   The micro-controller running the command processing firmware that decodes the
   fetched commands, and (for kernels) passes them to the Workgroup Managers
   (SPIs) for scheduling.

.. _grafana-panel-spi:

Shader Processor Input (SPI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tip::

   See :ref:`desc-spi` to learn about reported metrics.

SPI Stats
+++++++++

.. figure:: ../../data/analyze/grafana/spi-stats_panel.png
   :align: center
   :alt: SPI Stats panel in ROCm Compute Profiler Grafana
   :width: 800

..
   TODO: Add caption after merge

SPI Resource Allocation
+++++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/spi-resource-allocation_panel.png
   :align: center
   :alt: SPI Resource Allocation panel in ROCm Compute Profiler Grafana
   :width: 800

..
   TODO: Add caption after merge

.. _grafana-panel-wavefront:

Wavefront
^^^^^^^^^

Wavefront Launch Stats
++++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/wavefront-launch-stats_panel.png
   :align: center
   :alt: Wavefront Launch Stats panel in ROCm Compute Profiler Grafana
   :width: 800

   General information about the kernel launch.

.. tip::

   See :ref:`wavefront-launch-stats` to learn about reported metrics.

Wavefront Runtime Stats
+++++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/wavefront-runtime-stats_panel.png
   :align: center
   :alt: Wavefront Runtime Stats panel in ROCm Compute Profiler Grafana.
   :width: 800

   High-level overview of the execution of wavefronts in a kernel.

.. tip::

   See :ref:`wavefront-runtime-stats` to learn about reported metrics.

.. _grafana-panel-cu-instruction-mix:

Compute Unit - Instruction Mix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Instruction Mix
+++++++++++++++

.. figure:: ../../data/analyze/grafana/cu-inst-mix_panel.png
   :align: center
   :alt: Instruction Mix panel in ROCm Compute Profiler Grafana
   :width: 800

   Breakdown of the various types of instructions executed by the user’s kernel,
   and which pipelines on the Compute Unit (CU) they were executed on.

.. tip::

   See :ref:`instruction-mix` to learn about reported metrics.

VALU Arithmetic Instruction Mix
+++++++++++++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/cu-value-arith-instr-mix_panel.png
   :align: center
   :alt: VALU Arithmetic Instruction Mix panel in ROCm Compute Profiler Grafana
   :width: 800

   The various types of vector instructions that were issued to the vector
   arithmetic logic unit (VALU).

.. tip::

   See :ref:`valu-arith-instruction-mix` to learn about reported metrics.

MFMA Arithmetic Instruction Mix
+++++++++++++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/cu-mafma-arith-instr-mix_panel.png
   :align: center
   :alt: MFMA Arithmetic Instruction Mix panel in ROCm Compute Profiler Grafana
   :width: 800

   The types of Matrix Fused Multiply-Add (MFMA) instructions that were issued.

.. tip::

   See :ref:`mfma-instruction-mix` to learn about reported metrics.

VMEM Arithmetic Instruction Mix
+++++++++++++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/cu-vmem-instr-mix_panel.png
   :align: center
   :alt: VMEM Arithmetic Instruction Mix panel in ROCm Compute Profiler Grafana
   :width: 800

   The types of vector memory (VMEM) instructions that were issued.

.. tip::

   See :ref:`vmem-instruction-mix` to learn about reported metrics.

.. _grafana-panel-cu-compute-pipeline:

Compute Unit - Compute Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Speed-of-Light
++++++++++++++

.. figure:: ../../data/analyze/grafana/cu-sol_panel.png
   :align: center
   :alt: Speed-of-Light (CU) panel in ROCm Compute Profiler Grafana
   :width: 800

   The number of floating-point and integer operations executed on the vector
   arithmetic logic unit (VALU) and Matrix Fused Multiply-Add (MFMA) units in
   various precisions.

.. tip::

   See :ref:`compute-speed-of-light` to learn about reported metrics.

Pipeline Stats
++++++++++++++

.. figure:: ../../data/analyze/grafana/cu-pipeline-stats_panel.png
   :align: center
   :alt: Pipeline Stats panel in ROCm Compute Profiler Grafana
   :width: 800

   More detailed metrics to analyze the several independent pipelines found in
   the Compute Unit (CU).

.. tip::

   See :ref:`pipeline-stats` to learn about reported metrics.

Arithmetic Operations
+++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/cu-arith-ops_panel.png
   :align: center
   :alt: Arithmetic Operations panel in ROCm Compute Profiler Grafana
   :width: 800

   The total number of floating-point and integer operations executed in various
   precisions.

.. tip::

   See :ref:`arithmetic-operations` to learn about reported metrics.

.. _grafana-panel-lds:

Local Data Share (LDS)
^^^^^^^^^^^^^^^^^^^^^^

Speed-of-Light
++++++++++++++

.. figure:: ../../data/analyze/grafana/lds-sol_panel.png
   :align: center
   :alt: Speed-of-Light (LDS) panel in ROCm Compute Profiler Grafana
   :width: 800

   Key metrics for the Local Data Share (LDS) as a comparison with the peak
   achievable values of those metrics.

.. tip::

   See :ref:`lds-sol` to learn about reported metrics.

LDS Stats
+++++++++

.. figure:: ../../data/analyze/grafana/lds-stats_panel.png
   :align: center
   :alt: LDS Stats panel in ROCm Compute Profiler Grafana
   :width: 800

   More detailed view of the Local Data Share (LDS) performance.

.. tip::

   See :ref:`lds-stats` to learn about reported metrics.

.. _grafana-panel-instruction-cache:

Instruction Cache
^^^^^^^^^^^^^^^^^

Speed-of-Light
++++++++++++++

.. figure:: ../../data/analyze/grafana/instr-cache-sol_panel.png
   :align: center
   :alt: Speed-of-Light (instruction cache) panel in ROCm Compute Profiler Grafana
   :width: 800

   Key metrics of the L1 Instruction (L1I) cache as a comparison with the peak
   achievable values of those metrics.

.. tip::

   See :ref:`desc-l1i-sol` to learn about reported metrics.

Instruction Cache Stats
+++++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/instr-cache-accesses_panel.png
   :align: center
   :alt: Instruction Cache Stats panel in ROCm Compute Profiler Grafana
   :width: 800

   More detail on the hit/miss statistics of the L1 Instruction (L1I) cache.

.. tip::

   See :ref:`desc-l1i-stats` to learn about reported metrics.

.. _grafana-panel-sl1d-cache:

Scalar L1D Cache
^^^^^^^^^^^^^^^^

.. tip::

   See :ref:`desc-sl1d` to learn about reported metrics.

Speed-of-Light
++++++++++++++

.. figure:: ../../data/analyze/grafana/sl1d-sol_panel.png
   :align: center
   :alt: Speed-of-Light (SL1D) panel in ROCm Compute Profiler Grafana
   :width: 800

   Key metrics of the Scalar L1 Data (sL1D) cache as a comparison with the peak
   achievable values of those metrics.

.. tip::

   See :ref:`desc-sl1d-sol` to learn about reported metrics.

Scalar L1D Cache Accesses
+++++++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/sl1d-cache-accesses_panel.png
   :align: center
   :alt: Scalar L1D Cache Accesses panel in ROCm Compute Profiler Grafana
   :width: 800

   More detail on the types of accesses made to the Scalar L1 Data (sL1D) cache,
   and the hit/miss statistics.

.. tip::

   See :ref:`desc-sl1d-stats` to learn about reported metrics.

Scalar L1D Cache - L2 Interface
+++++++++++++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/sl1d-l12-interface_panel.png
   :align: center
   :alt: Scalar L1D Cache - L2 Interface panel in ROCm Compute Profiler Grafana
   :width: 800

   More detail on the data requested across the Scalar L1 Data (sL1D) cache <->
   L2 interface.

.. tip::

   See :ref:`desc-sl1d-l2-interface` to learn about reported metrics.

.. _grafana-panel-ta:

Texture Address and Texture Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Texture Addresser
+++++++++++++++++

.. figure:: ../../data/analyze/grafana/ta_panel.png
   :align: center
   :alt: Texture Addresser in ROCm Compute Profiler Grafana
   :width: 800

   Metric specific to texture addresser (TA) which receives commands (e.g.,
   instructions) and write/atomic data from the Compute Unit (CU), and coalesces
   them into fewer requests for the cache to process.

.. tip::

   See :ref:`desc-ta` to learn about reported metrics.

.. _grafana-panel-td:

Texture Data
++++++++++++

.. figure:: ../../data/analyze/grafana/td_panel.png
   :align: center
   :alt: Texture Data panel in ROCm Compute Profiler Grafana
   :width: 800

   Metrics specific to texture data (TD) which routes data back to the
   requesting Compute Unit (CU).

.. tip::

   See :ref:`desc-td` to learn about reported metrics.

.. _grafana-panel-vl1d:

Vector L1 Data Cache
^^^^^^^^^^^^^^^^^^^^

Speed-of-Light
++++++++++++++

.. figure:: ../../data/analyze/grafana/vl1d-sol_panel.png
   :align: center
   :alt: Speed-of-Light (VL1D) panel in ROCm Compute Profiler Grafana
   :width: 800

   Key metrics of the vector L1 data (vL1D) cache as a comparison with the peak
   achievable values of those metrics.

.. tip::

   See :ref:`vl1d-sol` to learn about reported metrics.

L1D Cache Stalls
++++++++++++++++

.. figure:: ../../data/analyze/grafana/vl1d-cache-stalls_panel.png
   :align: center
   :alt: L1D Cache Stalls panel in ROCm Compute Profiler Grafana
   :width: 800

   More detail on where vector L1 data (vL1D) cache is stalled in the pipeline,
   which may indicate performance limiters of the cache.

.. tip::

   See :ref:`vl1d-cache-stall-metrics` to learn about reported metrics.

L1D Cache Accesses
++++++++++++++++++

.. figure:: ../../data/analyze/grafana/vl1d-cache-accesses_panel.png
   :align: center
   :alt: L1D Cache Accesses
   :width: 800

   The type of requests incoming from the cache front-end, the number of requests
   that were serviced by the vector L1 data (vL1D) cache, and the number & type
   of outgoing requests to the L2 cache.

.. tip::

   See :ref:`vl1d-cache-access-metrics` to learn about reported metrics.

L1D - L2 Transactions
+++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/vl1d-l2-transactions_panel.png
   :align: center
   :alt: L1D - L2 Transactions in ROCm Compute Profiler Grafana
   :width: 800

   A more granular look at the types of requests made to the L2 cache.

.. tip::

   See :ref:`vl1d-l2-transaction-detail` to learn more.

L1D Addr Translation
++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/vl1d-addr-translation_panel.png
   :align: center
   :alt: L1D Addr Translation panel in ROCm Compute Profiler Grafana
   :width: 800

   After a vector memory instruction has been processed/coalesced by the address
   processing unit of the vector L1 data (vL1D) cache, it must be translated
   from a virtual to physical address. These metrics provide more details on the
   L1 Translation Lookaside Buffer (TLB) which handles this process.

.. tip::

   See :ref:`desc-utcl1` to learn about reported metrics.

.. _grafana-panel-l2-cache:

L2 Cache
^^^^^^^^

.. tip::

   See :doc:`/conceptual/l2-cache` to learn about reported metrics.

Speed-of-Light
++++++++++++++

.. figure:: ../../data/analyze/grafana/l2-sol_panel.png
   :align: center
   :alt: Speed-of-Light (L2 cache) panel in ROCm Compute Profiler Grafana
   :width: 800

   Key metrics about the performance of the L2 cache, aggregated over all the
   L2 channels, as a comparison with the peak achievable values of those
   metrics.

.. tip::

   See :ref:`l2-sol` to learn about reported metrics.

L2 Cache Accesses
+++++++++++++++++

.. figure:: ../../data/analyze/grafana/l2-accesses_panel.png
   :align: center
   :alt: L2 Cache Accesses panel in ROCm Compute Profiler Grafana
   :width: 800

   Incoming requests to the L2 cache from the vector L1 data (vL1D) cache and
   other clients (e.g., the sL1D and L1I caches).

.. tip::

   See :ref:`l2-cache-accesses` to learn about reported metrics.

L2 - Fabric Transactions
++++++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/l2-fabric-transactions_panel.png
   :align: center
   :alt: L2 - Fabric Transactions panel in ROCm Compute Profiler Grafana
   :width: 800

   More detail on the flow of requests through Infinity Fabric™.

.. tip::

   See :ref:`l2-fabric` to learn about reported metrics.

L2 - Fabric Interface Stalls
++++++++++++++++++++++++++++

.. figure:: ../../data/analyze/grafana/l2-fabric-interface-stalls_panel.png
   :align: center
   :alt: L2 - Fabric Interface Stalls panel in ROCm Compute Profiler Grafana
   :width: 800

   A breakdown of what types of requests in a kernel caused a stall
   (e.g., read vs write), and to which locations (e.g., to the accelerator’s
   local memory, or to remote accelerators/CPUs).

.. tip::

   See :ref:`l2-fabric-stalls` to learn about reported metrics.

.. _grafana-panel-l2-cache-per-channel:

L2 Cache Per Channel
^^^^^^^^^^^^^^^^^^^^

.. tip::

   See :ref:`l2-sol` for more information.

Aggregate Stats
+++++++++++++++

.. figure:: ../../data/analyze/grafana/l2-per-channel-agg-stats_panel.png
   :align: center
   :alt: Aggregate Stats (L2 cache per channel) panel in ROCm Compute Profiler Grafana
   :width: 800

   L2 Cache per channel performance at a glance. Metrics are aggregated over all available channels.
