.. meta::
   :description: Build and run RCCL-Tests to benchmark collective communication bandwidth and validate correctness on single-node and multi-node AMD GPU systems.
   :keywords: RCCL, rccl-tests, benchmark, AllReduce, algbw, busbw, MPI, multi-node, ROCm, performance testing, gfx942, MI300X, all_reduce_perf, sendrecv_perf, HIP graph

.. _running-rccl-tests:

**************
Run RCCL-Tests
**************

RCCL-Tests is the standard benchmarking and correctness-validation suite for
RCCL. It produces per-collective executables that measure algorithm bandwidth
(algBw) and bus bandwidth (busBw) across a sweep of message sizes, and
optionally verify that each result is numerically correct. This guide walks you
through building RCCL-Tests, understanding its output, and running it in both
single-node and multi-node configurations.

Prerequisites
=============

Before you begin, ensure the following are in place:

- ROCm is installed. See :doc:`Install RCCL <../install/installation>` for
  more information.
- RCCL is installed, either from a ROCm package (``/opt/rocm``) or built from
  source. See :doc:`Build RCCL from source <../install/building-installing>`.
- CMake 3.16 or later is available (``cmake --version``).
- For multi-node runs: MPI is installed (for example, Open MPI or MPICH) and
  your cluster is configured with passwordless SSH and a working InfiniBand or
  RoCE fabric.

Build RCCL-Tests
================

RCCL-Tests supports both CMake and Makefile build systems. CMake is recommended
for most workflows.

1. Clone the repository. RCCL-Tests is part of the ``rocm-systems`` monorepo:

   .. code-block:: bash

      git clone https://github.com/ROCm/rocm-systems.git
      cd rocm-systems/projects/rccl-tests

2. Build using one of the following options:

   For a single-node build without MPI, run the install script:

   .. code-block:: bash

      ./install.sh

   Executables are placed in ``./build/``.

   For a multi-node build with MPI enabled, use CMake directly:

   .. code-block:: bash

      mkdir build && cd build
      cmake -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_PREFIX_PATH="/path/to/mpi;/opt/rocm" \
            -DUSE_MPI=ON \
            -DGPU_TARGETS="gfx942;gfx950" \
            ..
      make -j$(nproc)

   Replace ``GPU_TARGETS`` with the architecture strings for your GPUs. Common
   values are ``gfx90a`` (MI200), ``gfx942`` (MI300X), and ``gfx950`` (MI350).

   Alternatively, use the Makefile build:

   .. code-block:: bash

      make MPI=1 MPI_HOME=/path/to/mpi HIP_HOME=/opt/rocm NCCL_HOME=/opt/rocm

   The Makefile build also places executables in ``./build/``.

The following executables are produced — one per collective operation:

.. code-block:: text

   build/all_reduce_perf
   build/all_gather_perf
   build/reduce_scatter_perf
   build/broadcast_perf
   build/reduce_perf
   build/gather_perf
   build/scatter_perf
   build/alltoall_perf
   build/alltoallv_perf
   build/sendrecv_perf

Understand the output
=====================

Every executable produces a table with one row per message size. Before reading
results, it is important to understand the two bandwidth metrics reported:

- **Algorithm bandwidth (algBw)** is the application-visible rate; the input data size divided by the elapsed time. This is the number that matters most for estimating the communication overhead in your workload.

- **Bus bandwidth (busBw)** is the actual bytes transferred on the interconnect, divided by time. Because collective algorithms send more data than the input size (for example, a Ring AllReduce with *N* ranks transfers 2(N−1)/N × input size), busBw is always higher than algBw. As N grows large, busBw approaches 2 × algBw for AllReduce. Bus bandwidth is useful for comparing actual utilization against the theoretical hardware limit.

For example, on an 8-GPU MI300X node running a 1 GB AllReduce at 179.90 GB/s
algBw:

.. code-block:: text

   busBw = algBw × 2(N-1)/N = 179.90 × 2(7)/8 = 179.90 × 1.75 = 314.83 GB/s

A sample output row looks like this:

.. code-block:: text

   #        size  count  type  redop  root  time    algbw   busbw  #wrong
   # (B)                               (us)  (GB/s)  (GB/s)
    1073741824  268435456  float  sum    -1  5968.5  179.90  314.83       0

The ``#wrong`` column reports the number of elements that differ from the
expected result by more than the correctness threshold. A value of ``0``
means the collective produced the correct answer.

Command-line options
====================

All executables share the same set of options. The most commonly used are:

.. list-table::
   :header-rows: 1
   :widths: 20 55 25

   * - Option
     - Description
     - Default
   * - ``-g``
     - Number of GPUs to use per process (thread).
     - ``1``
   * - ``-b``
     - Starting message size (bytes or human-readable, e.g. ``1M``).
     - ``32M``
   * - ``-e``
     - Ending message size.
     - ``32M``
   * - ``-f``
     - Multiplicative step factor between sizes.
     - ``1``
   * - ``-i``
     - Additive step increment between sizes (bytes).
     - ``1M``
   * - ``-w``
     - Number of warmup iterations before timing begins.
     - ``5``
   * - ``-n``
     - Number of timed iterations per message size.
     - ``20``
   * - ``-N``
     - Number of run cycles (``0`` = infinite).
     - ``1``
   * - ``-d``
     - Data type: ``float``, ``bfloat16``, ``fp8_e4m3``, etc.
     - ``float``
   * - ``-o``
     - Reduction operator: ``sum``, ``prod``, ``min``, ``max``, ``avg``.
     - ``sum``
   * - ``-r``
     - Root rank index (for Reduce and Broadcast).
     - ``0``
   * - ``-c``
     - Enable correctness checking (``1`` = on, ``0`` = off).
     - ``1``
   * - ``-G``
     - Capture and replay as a HIP graph (number of replays).
     - ``0``
   * - ``-R``
     - Buffer registration mode: ``0`` = none, ``1`` = local, ``2`` = symmetric.
     - ``0``
   * - ``-a``
     - MPI reporting mode: ``0`` = rank 0, ``1`` = average, ``2`` = min, ``3`` = max.
     - ``1``
   * - ``-z``
     - Blocking mode: wait for CPU sync after each collective.
     - ``0``

Run a single-node benchmark
============================

Single-node runs use the ``-g`` flag to specify the number of GPUs to use
within a single process. This is the simplest way to verify RCCL is working
and to establish a bandwidth baseline.

- AllReduce across 8 GPUs, 1 MB to 1 GB, doubling each step:

  .. code-block:: bash

     ./build/all_reduce_perf -g 8 -b 1M -e 1G -f 2

- Broadcast from GPU 4, 1 MB to 32 MB in 2 MB increments:

  .. code-block:: bash

     ./build/broadcast_perf -g 8 -r 4 -b 1M -e 32M -i 2097152

- ReduceScatter with bfloat16, 1 MB to 32 MB:

  .. code-block:: bash

     ./build/reduce_scatter_perf -g 8 -b 1M -e 32M -f 2 -d bfloat16

If you built RCCL from source and want to test against that build rather than
the installed package, prepend it to ``LD_LIBRARY_PATH``:

.. code-block:: bash

   LD_LIBRARY_PATH=/path/to/rccl/build/release:$LD_LIBRARY_PATH \
     ./build/all_reduce_perf -g 8 -b 1M -e 1G -f 2

Run a multi-node benchmark with MPI
=====================================

For multi-node runs, use one MPI process per GPU (``-g 1``) rather than
multiple GPUs per process. This better represents real-world framework usage
(PyTorch ``torch.distributed``, JAX, etc.) and ensures each process has
exclusive use of one GPU.

- AllReduce across 2 nodes, 8 GPUs each (16 ranks total):

  .. code-block:: bash

     mpirun -np 16 \
       --bind-to numa \
       -x NCCL_DEBUG=VERSION \
       -x HSA_NO_SCRATCH_RECLAIM=1 \
       ./build/all_reduce_perf -g 1 -b 1M -e 1G -f 2

- Broadcast across 8 GPUs with one process per GPU:

  .. code-block:: bash

     mpirun -np 8 \
       ./build/broadcast_perf -g 1 -r 4 -b 1M -e 32M -i 2097152

.. note::

   - ``--bind-to numa`` pins each MPI rank to the NUMA node closest to its GPU.
   This prevents cross-NUMA memory traffic that would otherwise inflate latency
   and reduce bandwidth.

   - ``HSA_NO_SCRATCH_RECLAIM=1`` is required on gfx90a (MI200) when using ROCm
   7.13 or later, and is recommended on all AMD Instinct™ GPUs for stable
   benchmark results.

Test specific collective operations
=====================================

The following examples illustrate common single-node test patterns for each
collective. Substitute ``mpirun -np <N>`` and ``-g 1`` for multi-node runs.

- AllGather — gather all GPU contributions into a complete tensor on every GPU:

  .. code-block:: bash

     ./build/all_gather_perf -g 8 -b 1M -e 512M -f 2

- ReduceScatter — reduce and distribute one slice per GPU:

  .. code-block:: bash

     ./build/reduce_scatter_perf -g 8 -b 1M -e 512M -f 2

- AllToAll — exchange distinct data between every pair of GPUs:

  .. code-block:: bash

     ./build/alltoall_perf -g 8 -b 1M -e 256M -f 2

- SendRecv — point-to-point latency and bandwidth:

  .. code-block:: bash

     ./build/sendrecv_perf -g 8 -b 1 -e 1G -f 2

Disable correctness checking for throughput-only runs
------------------------------------------------------

Correctness checking is enabled by default (``-c 1``). On large GPU counts, it
adds significant overhead. Disable it for pure throughput measurement:

.. code-block:: bash

   ./build/all_reduce_perf -g 8 -b 1M -e 1G -f 2 -c 0

Test with HIP graphs
---------------------

HIP graph capture replays a collective without re-launching host-side setup
code. This more closely measures GPU-side execution time and is relevant for
frameworks that use graph mode:

.. code-block:: bash

   ./build/all_reduce_perf -g 8 -b 32M -e 1G -f 2 -G 5

The ``-G 5`` flag captures one iteration as a HIP graph and then replays it 5
times for measurement.

Interpret the results
======================

- Check algBw against hardware limits: For a single MI300X node with 8 GPUs in a fully connected xGMI mesh, the theoretical AllReduce bandwidth approaches ~179 GB/s algBw for large messages. If your results are significantly lower, check that:

  - All 8 GPUs are participating (``-g 8`` or 8 MPI ranks).
  - ``HSA_NO_SCRATCH_RECLAIM=1`` is set (critical on ROCm 7.13+ with gfx90a).
  - The system is not throttled (check GPU clocks with ``rocm-smi``).

- Compare algBw against busBw: For a Ring AllReduce with *N* ranks, the formula is ``busBw = algBw × 2(N-1)/N``. If your measured busBw is close to the peak xGMI or network link bandwidth, communication is saturating the hardware. If it is well below, the bottleneck might be latency (small messages), CPU overhead, or a topology issue.

- The ``#wrong`` column: A non-zero value means the collective produced an incorrect result for at least one element. This should not happen in normal operation. Common causes are mismatched RCCL and ROCm versions, GPU memory errors, or reduced-precision data types hitting floating-point edge cases. Run with ``NCCL_DEBUG=WARN`` to get additional diagnostic output.

- Avg bus bandwidth line: The final line of each test run prints the average bus bandwidth across all tested message sizes. This is useful as a single summary number for comparing runs.

Troubleshoot test failures
===========================

If a test hangs or crashes, the following variables help diagnose the problem:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - When to use it
   * - ``NCCL_DEBUG=INFO``
     - Prints communicator initialization details, algorithm and protocol
       selection, and topology information.
   * - ``NCCL_DEBUG=WARN``
     - Prints only warnings and errors. Less verbose than ``INFO``.
   * - ``NCCL_DEBUG_SUBSYS=INIT,COLL,TUNING``
     - Filters ``NCCL_DEBUG=INFO`` output to specific subsystems.
   * - ``NCCL_TIMEOUT=60``
     - Sets the collective timeout in seconds. Useful for catching hangs.
   * - ``RCCL_PROXYTRACE=1``
     - Enables proxy-thread tracing to diagnose multi-node network hangs.

For a full troubleshooting guide, see
:doc:`Troubleshoot RCCL <./troubleshooting-rccl>`.

Related topics
===============

- :doc:`Build RCCL from source <../install/building-installing>` — build
  options, GPU targets, and install prefixes
- :doc:`RCCL usage tips <./rccl-usage-tips>` — environment variables and
  tuning guidance for production workloads
- :doc:`Collective operations in RCCL <../conceptual/collective-operations>` —
  explains algBw, busBw, and what each collective does
- :doc:`Hardware-specific optimizations <../conceptual/hardware-specific-optimizations>` —
  expected bandwidth numbers per GPU generation
