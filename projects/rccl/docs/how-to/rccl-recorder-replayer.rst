.. meta::
   :description: Use the RCCL Recorder to capture collective call logs and the RCCL Replayer to reproduce and debug RCCL workloads on AMD GPUs.
   :keywords: RCCL, recorder, replayer, RcclReplayer, RCCL_REPLAY_FILE, collective replay, debugging, ROCm, MPI, binary log, JSON, rcclReplayer, RCCL_LOG_LEVEL

.. _rccl-recorder-replayer:

***********************************
Use the RCCL Recorder and Replayer
***********************************

The RCCL Recorder and Replayer are two complementary debugging tools:

- The **Recorder** is built into RCCL. It captures a structured log of every
  collective call made during a run, including all parameters, buffer addresses,
  communicator state, and timing. No application changes are required — you
  enable it with a single environment variable.

- The **Replayer** is a standalone MPI-based tool. It reads the logs produced
  by the Recorder and re-executes the same sequence of RCCL calls using dummy
  data, with no dependency on the original application. This makes it possible
  to reproduce performance issues or failures in isolation, bisect regressions,
  and measure collective bandwidth without running the full workload.

Prerequisites
=============

- ROCm is installed. See :doc:`Install RCCL <../install/installation>` for
  more information.
- RCCL is installed or built from source. See
  :doc:`Build RCCL from source <../install/building-installing>`.
- MPI is installed (Open MPI or MPICH). The Replayer requires the same number
  of MPI processes and the same node layout as the original recorded job.

Record a workload with the RCCL Recorder
=========================================

The Recorder is part of the RCCL library and requires no build step. To enable
it, set ``RCCL_REPLAY_FILE`` to the base path for the output log files before
launching your workload:

.. code-block:: bash

   export RCCL_REPLAY_FILE=/path/to/logs/myrun

The Recorder writes one log file per process. Each file is named using the
pattern:

.. code-block:: text

   <basename>.<PID>.<hostname>[.json]

For example:

.. code-block:: text

   /path/to/logs/myrun.1275.node01.bin
   /path/to/logs/myrun.1276.node01.bin
   /path/to/logs/myrun.1277.node02.bin

If the base filename ends in ``.json``, the Recorder writes human-readable JSON.
Otherwise it writes a compact binary format. Binary files are smaller and faster
to write; JSON files are easier to inspect with standard tools.

.. code-block:: bash

   # Binary output (default)
   export RCCL_REPLAY_FILE=/path/to/logs/myrun

   # JSON output
   export RCCL_REPLAY_FILE=/path/to/logs/myrun.json

Control log verbosity
---------------------

The ``RCCL_LOG_LEVEL`` variable controls which calls are recorded:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - What is logged
   * - ``1`` (default)
     - Essential APIs only: collective operations, communicator lifecycle,
       group calls, memory operations, and user-buffer registration.
   * - ``2`` or higher
     - All APIs, including informational calls such as ``ncclGetVersion``,
       ``ncclGetErrorString``, and ``ncclCommCount``.

.. code-block:: bash

   export RCCL_LOG_LEVEL=1

HIP graph support
-----------------

The Recorder is compatible with HIP graph capture. Calls made inside a graph
capture are deferred using a HIP graph host node callback and written to the log
when the graph is executed, so the log reflects execution order rather than
submission order.

Log file format
---------------

Each record in a binary log is a fixed-size 160-byte ``rcclApiCall`` struct
containing:

- **Implicit context fields:** PID, thread ID, HIP device index, group depth,
  timestamp (milliseconds since epoch), HIP graph ID, and whether the call was
  captured in a graph.
- **Call fields:** call type, operation count, send and receive buffer addresses
  and extents, accumulator pointer, element count, data type, reduction operator,
  root rank, number of ranks, communicator pointer, stream pointer, task index,
  global rank, and communicator ID.

The JSON format uses a custom line-per-call layout:

.. code-block:: text

   CallName : [opCount : <hex>, sendbuff : [addr : <ptr>, base : <ptr>, size : <n>],
               recvbuff : [...], count : <n>, datatype : <n>, op : <n>, root : <n>,
               comm : <ptr>, nranks : <n>, stream : <ptr>, task : <n>,
               globalrank : <n>, context : [time : <ms>, thread : <n>,
               device : <n>, captured : <n>, graphID : <n> ]]

Build the RCCL Replayer
========================

The Replayer is located in the RCCL source tree under
``projects/rccl/tools/RcclReplayer/``. It is not included in the installed
ROCm package and must be built from source.

.. code-block:: bash

   cd /path/to/rocm-systems/projects/rccl/tools/RcclReplayer
   MPI_DIR=/path/to/mpi make

The Makefile also accepts ``ROCM_DIR`` (default ``/opt/rocm``) and ``RCCL_DIR``
(default ``../../build/release``) to point to a locally built RCCL:

.. code-block:: bash

   MPI_DIR=/opt/ompi ROCM_DIR=/opt/rocm RCCL_DIR=/path/to/rccl/build/release make

This produces the ``rcclReplayer`` executable in the current directory.

Replay a recorded workload
===========================

The Replayer must be launched with the same number of MPI processes and the
same node layout as the original job. It automatically discovers the log files
that belong to each rank by matching the hostname and PID encoded in the
filenames.

**Basic replay (single node):**

.. code-block:: bash

   mpirun -np <numProcesses> ./rcclReplayer /path/to/logs/myrun.bin

**Multi-node replay:**

.. code-block:: bash

   mpirun --hostfile /path/to/hostfile.txt \
     -np <numProcesses> \
     ./rcclReplayer /path/to/logs/myrun.bin

Replace ``<numProcesses>`` with the total number of MPI ranks used in the
original run. The Replayer passes the base path (without PID or hostname suffix)
and discovers the per-rank files automatically.

.. note::

   - ``RCCL_REPLAY_FILE`` is automatically unset by the Replayer at startup to
   prevent the replay run from recording itself.

   - Depending on your MPI library, you might need to adjust flags such as
   ``--bind-to numa`` or ``--mca pml ucx``. Check the output of your MPI
   implementation's ``mpirun --help`` for options relevant to your cluster.

What the Replayer does
----------------------

The Replayer reads every call in each rank's log file and
builds:

- A device memory map that tracks each buffer's base address, size, and the
  last log line that uses it — so buffers can be allocated lazily and freed as
  soon as they are no longer needed.
- A stream lifetime map.
- A HIP graph lifecycle map, tracking graph capture depth, start lines, end
  lines, and instantiated graph executables.
- A communicator ID list per rank.

MPI is used to exchange communicator information across all ranks so that
``ncclCommInitRank`` can be called with consistent parameters.

The Replayer then re-executes every call from the log in order,
using dummy send and receive buffers of the correct size and type. It handles
the full RCCL API surface including:

- All collectives: ``ncclAllReduce``, ``ncclAllGather``, ``ncclReduceScatter``,
  ``ncclBroadcast``, ``ncclReduce``, ``ncclGather``, ``ncclScatter``,
  ``ncclAllToAll``, ``ncclAllToAllv``, ``ncclAllReduceWithBias``.
- Point-to-point: ``ncclSend``, ``ncclRecv``.
- Communicator lifecycle: ``ncclCommInitRank``, ``ncclCommInitAll``,
  ``ncclCommSplit``, ``ncclCommDestroy``, ``ncclCommFinalize``,
  ``ncclCommAbort``.
- Memory operations: ``ncclMemAlloc``, ``ncclMemFree``.
- User-buffer registration: ``ncclCommRegister``, ``ncclCommDeregister``.
- HIP graph capture and replay using ``hipStreamBeginCapture``,
  ``hipStreamEndCapture``, ``hipGraphInstantiate``, and ``hipGraphLaunch``.

At the end of the run, the Replayer reports elapsed time and bus bandwidth for
each replayed collective call.

Convert and inspect log files
==============================

The ``replay_log_converter.py`` script (in the same directory as the Replayer)
converts between binary and JSON formats and provides utilities for comparing
logs across runs.

.. code-block:: bash

   # Convert binary log to JSON
   python3 replay_log_converter.py tojson /path/to/logs/myrun

   # Convert JSON log to binary
   python3 replay_log_converter.py tobin /path/to/logs/myrun.json

   # Convert the custom JSON format to standard JSON (parseable by stdlib)
   python3 replay_log_converter.py --standardize /path/to/logs/myrun.json

   # Sanitize a log: remap all pointer values, timestamps, PIDs, and thread IDs
   # to stable canonical values so two logs from different runs can be diffed
   python3 replay_log_converter.py --sanitize /path/to/logs/myrun.json

   # Sanitize and strip timestamps (useful for exact diff comparisons)
   python3 replay_log_converter.py --sanitize --no-timestamp /path/to/logs/myrun.json

The sanitizer remaps all hex pointer values, communicator IDs, timestamps,
thread IDs, and PIDs to stable canonical integers using regex-based substitution.
This lets you diff two sanitized logs from different runs to isolate behavioral
changes.

Typical debugging workflows
============================

Reproduce a hang or incorrect result
-------------------------------------

#. Set ``RCCL_REPLAY_FILE`` and run the failing workload to capture a log.
#. Replay the log with the Replayer on the same hardware.
#. If the problem reproduces, you have an isolated test case that does not
   require the original application, framework, or model weights.
#. Bisect by replaying on different RCCL builds to identify the introducing
   commit.

Compare performance across RCCL versions
-----------------------------------------

#. Record a production workload once with the current RCCL version.
#. Replay the same log against an older or newer ``librccl.so`` by adjusting
   ``RCCL_DIR`` during the Replayer build and ``LD_LIBRARY_PATH`` at runtime.
#. Compare the per-collective bandwidth reported by the Replayer across versions.

Diff two runs for behavioral changes
--------------------------------------

#. Record both runs with ``RCCL_REPLAY_FILE=/path/myrun.json``.
#. Sanitize both logs with ``replay_log_converter.py --sanitize --no-timestamp``.
#. Diff the sanitized outputs to see which calls, parameters, or communicator
   layouts changed between runs.

Environment variable reference
===============================

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Description
   * - ``RCCL_REPLAY_FILE``
     - Enables the Recorder. Set to the base path for output log files.
       Append ``.json`` to the base name to request JSON output instead of
       binary. Unset or empty disables recording.
   * - ``RCCL_LOG_LEVEL``
     - Verbosity level. ``1`` (default) logs essential APIs; ``2`` or higher
       logs all APIs including informational calls.

Related topics
==============

- :doc:`Run RCCL-Tests <./running-rccl-tests>` — benchmark collectives without
  recording a workload first
- :doc:`Troubleshooting <../reference/troubleshooting-rccl>` — collect system
  information and RCCL diagnostics for issue reports
- :doc:`RCCL usage tips <./rccl-usage-tips>` — environment variables and
  performance tuning guidance
- `RcclReplayer source and documentation <https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl/tools/RcclReplayer>`_
