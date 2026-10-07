.. meta::
   :description: Use the RCCL active diagnostics to verify GPU peer-to-peer paths during communicator initialization on AMD GPUs
   :keywords: RCCL, ROCm, AMD, diagnostics, NCCL_RUN_DIAGNOSTICS, P2P, XGMI, troubleshooting

.. _using-rccl-diagnostics:

*********************************************
Verify GPU P2P paths with diagnostics in RCCL
*********************************************

RCCL can check the GPU peer-to-peer (P2P) paths it intends to use while a
communicator is being created. The check writes data between every eligible
pair of GPUs on a node, reads it back, and prints a short report. A passing
report rules out broken intra-node links, peer-memory mapping problems, and
container or IPC isolation problems before the application sends any traffic.
A failing report names the exact GPU pair and path to investigate.

This feature is inherited from NCCL (active diagnostics). Besides the P2P
check, it includes a network bandwidth check that runs only on multi-node
communicators (see `Network bandwidth check`_ for more info). The passive RAS diagnostics
(``NCCL_RUN_RAS_DIAGNOSTICS``), which compare configuration across ranks
without exercising data paths, are a separate feature and are not covered on
this page.

Enable diagnostics
==================

Diagnostics are disabled by default. Set ``NCCL_RUN_DIAGNOSTICS=1`` for every
process of the job:

.. code:: shell

   NCCL_RUN_DIAGNOSTICS=1 <application> [arguments]

For example, with rccl-tests:

.. code:: shell

   NCCL_RUN_DIAGNOSTICS=1 ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8

The check runs inside every communicator initialization, including
``ncclCommInitRank``, ``ncclCommInitAll``, and ``ncclCommSplit``. An
application that creates several communicators prints one report per
communicator.

Diagnostics are informational. A reported problem does not make communicator
initialization fail, and the communicator remains usable after the check.

Read the report
===============

The report is printed to standard output, not to ``NCCL_DEBUG_FILE``. Every
line starts with ``<hostname>:<pid> NCCL DIAG``. The process that hosts rank 0
of the communicator prints the header, the summary lines, and the
``completed`` line. Failed edges are listed by the first rank on the node where they occur,
and a few notices are printed by the rank they concern. Collect the standard
output of all processes, not only of rank 0.

A passing run on an 8-GPU AMD Instinct MI355X node, with one process per GPU,
looks like this:

.. code:: none

   node01:15 NCCL DIAG === NCCL Diagnostics ===
   node01:15 NCCL DIAG [OK] net bw: skipped (single host or no network transport)
   node01:15 NCCL DIAG [OK]   p2p: verified P2P access in both directions between every GPU pair (56 peer accesses)
   node01:15 NCCL DIAG NCCL diagnostics completed in 40.0 ms across 8 ranks

If some edges fail, the summary line reads
``[INFO] p2p: only <passed>/<tested> GPU-to-GPU peer accesses passed verification``
and the failed edges are listed on their own lines.

Result lines use two tags:

* ``[OK]`` means that every tested edge passed, or, on the ``net bw:`` line,
  that the network bandwidth check passed or was skipped.
* ``[INFO]`` marks both notices and problems. A notice, such as the
  peer-access line described in `Single-process and multi-process jobs`_, is
  expected on a healthy system. A problem means that at least one edge failed
  or that a step of the check could not run, and the line identifies the
  affected GPU pair. Use the message text and the summary line to tell them
  apart: when every tested edge passed, the summary line is tagged ``[OK]``.

Which GPU pairs are tested
--------------------------

The check tests directed edges. For each pair of GPUs, A and B, it tests both
A to B and B to A. Only GPUs on the same node are tested, and only pairs that
RCCL's topology detection allows to use P2P. On a node with ``N`` GPUs in the
communicator, the check tests ``N * (N - 1)`` edges when all pairs are
eligible. The summary line counts each tested edge as one peer access. For a
multi-node communicator the edge counts of all nodes are added together. For
example, two nodes with 8 GPUs each report 112 peer accesses.

The P2P check does not test the network between nodes.

Pairs that can only reach each other through an intermediate GPU are not
tested and are not counted in the summary. With ``NCCL_DEBUG=INFO`` and
``NCCL_DEBUG_SUBSYS=INIT``, each such edge is logged with ``reason=indirect``.
On AMD GPUs RCCL does not route P2P through an intermediate GPU, so no edge is
left out for this reason.

.. note::

   If no pair is eligible, the report contains no ``p2p:`` line between the
   header and the ``completed`` line. This happens, for example, with one GPU
   per node, or on a host with an AMD CPU with ``NCCL_P2P_DISABLE=1`` or a
   restrictive ``NCCL_P2P_LEVEL``. On a host with an Intel, Zhaoxin, or Arm CPU,
   RCCL currently replaces these settings with its own P2P level for that CPU, so
   pairs connected through XGMI or a PCIe switch stay eligible and are tested. A
   missing ``p2p:`` line means that nothing was tested, not that everything
   passed.

Single-process and multi-process jobs
-------------------------------------

When one process drives several GPUs, for example ``ncclCommInitAll`` or
rccl-tests with ``-g 8``, the check needs peer access between the devices of
that process. Without HIP virtual memory management (cuMem, see
``NCCL_CUMEM_ENABLE``), it enables context-wide peer access for its duration,
and each rank that newly enables it prints a line similar to the following.
This is expected and is not an error:

.. code:: none

   node01:4242 NCCL DIAG [INFO] p2p: temporarily enabled context-wide CUDA peer access rank=3 cudaDev=3; avoid concurrent CUDA use on this context until diagnostics completes

In this message, "CUDA" refers to the HIP runtime. Do not issue HIP work on
these devices from other threads while the communicator is being initialized.

The line is not printed when peer access between the devices was already
enabled. A missing line does not mean that the check did not run.

.. note::

   On ROCm, the check cannot map the buffers of other devices of the same
   process through cuMem. With cuMem enabled, a single process that drives
   several GPUs reports every edge between its devices as
   ``peer-memory import failed ... handle=DIRECT reason=import``. cuMem is
   enabled by default on gfx1250. Run such jobs with ``NCCL_CUMEM_ENABLE=0``
   when you use the check.

When each GPU is driven by its own process, peer memory is shared through HIP
IPC handles, or through HIP virtual-memory handles when cuMem is enabled, and
no such line is printed.

Network bandwidth check
-----------------------

The network bandwidth check runs before the P2P check, and only when the
communicator spans at least two hosts and uses a network transport between
them. It runs ``ib_write_bw`` from the ``perftest`` package between the nodes,
so the tool must be installed on every node and host names must be resolvable
between the nodes. Rank 0 prints the minimum, median, and maximum bandwidth on
a ``net bw:`` line and lists the ranks that are more than 30 percent off the
median. Like the P2P check, it never makes initialization fail. On a single
host, or without a network transport, the check does not run and rank 0 prints:

.. code:: none

   node01:15 NCCL DIAG [OK] net bw: skipped (single host or no network transport)

Performance impact
------------------

On an 8-GPU MI355X node the check takes about 30 to 50 ms per communicator
initialization, which is small compared with the time of the initialization
itself. It does not change the performance of collectives after
initialization.
Because the check runs at every initialization, enable it while bringing up or
debugging a system, and leave it disabled for jobs that create many
communicators.

Troubleshoot failed edges
=========================

Each failed edge is reported on its own ``[INFO] p2p:`` line. The line starts
with the kind of failure, followed by fields that identify the edge and a
suggested next step. The table lists the kinds of failure.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Message
     - Meaning
   * - ``destination buffer unavailable``
     - The destination rank could not allocate or export its test buffer.
       Look for earlier allocation or initialization errors on that rank.
   * - ``local CUDA setup failed``
     - The source rank could not set up its device, stream, or buffers.
       Look for earlier HIP errors on that rank.
   * - ``peer-memory import failed``
     - The source rank could not map the destination buffer. In a multi-process
       job this usually means that the processes cannot share memory handles
       (HIP IPC or, with cuMem, HIP virtual-memory handles). See
       :ref:`diagnostics-containers` for more info.
   * - ``write mismatch``
     - Data written by the source GPU into the destination GPU memory did not
       arrive intact. The ``expected`` and ``got`` fields show the test pattern
       and the value that was read back.
   * - ``read mismatch``
     - Data read by the source GPU from the destination GPU memory was wrong.
   * - ``topology check failed``
     - RCCL could not determine whether the pair can use P2P. Look for earlier
       topology (``GRAPH``) messages.
   * - ``launch/check failed``
     - A test kernel could not be launched or did not complete. Look for earlier
       HIP or RCCL warnings.

The following lines report that the check itself failed, not a single edge.
They carry a rank and an ``ncclResult_t`` code, or only the code, instead of
edge fields:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Message
     - Meaning
   * - ``p2p: setup failed on rank <r>``
     - The check could not set up on that rank, so its edges were not tested.
   * - ``p2p: resource cleanup failed rank=<r>``
     - Temporary test resources may not have been released on that rank.
   * - ``transport detect returned <n>``
     - The transport scan failed.
   * - ``p2p: check returned <n>``
     - The P2P check stopped before it completed.

This table explains the edge fields:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``srcRank``, ``dstRank``
     - Ranks of the source and destination GPU in the communicator.
   * - ``srcCudaDev``, ``dstCudaDev``
     - HIP device index as seen by the process. This index depends on
       ``HIP_VISIBLE_DEVICES`` and ``ROCR_VISIBLE_DEVICES``.
   * - ``srcNvmlDev``, ``dstNvmlDev``
     - Index of the GPU in ``amd-smi``. Use it with ``amd-smi ... -g <index>``.
   * - ``path``
     - Path type between the two GPUs: ``XGMI`` for a direct XGMI link, or a
       PCIe path type such as ``PIX``, ``PXB``, ``PHB``, or ``SYS``.
   * - ``handle``
     - How the destination memory was shared: ``DIRECT`` (both GPUs in one
       process), ``LEGACY_CUDA_IPC`` (HIP IPC handle between processes), or
       ``CUMEM_OTHER`` (HIP virtual-memory handle between processes, when cuMem
       is enabled). The report format also defines ``CUMEM_POSIX_FD`` and
       ``CUMEM_FABRIC``, but RCCL does not report them on AMD GPUs.

.. note::

   Known limitation: the suggested next step at the end of each line comes
   from NCCL and still refers to NVIDIA tools and terms (``nvidia-smi``,
   ``nvidia-imex-ctl``, NVLink, CUDA IPC). AMD-specific wording in the report
   is tracked separately. Until then, use the commands in the following table.

On AMD GPUs, use the following commands instead:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Path or handle
     - What to check
   * - ``path=XGMI``
     - Check the XGMI link state with ``amd-smi xgmi -l`` and the link type
       between the two GPUs with ``amd-smi topology -t``.
   * - ``path`` is a PCIe type
     - Check peer access and DMA support with ``amd-smi topology -a`` and
       ``amd-smi topology -d``, then check the IOMMU mode and the PCIe ACS
       settings of the host.
   * - ``handle=LEGACY_CUDA_IPC``
     - Check that all processes see the GPUs and can share IPC handles. See
       :ref:`diagnostics-containers` for more info.
   * - ``handle=CUMEM_OTHER``
     - Check that HIP virtual memory is supported and that the processes can
       share memory handles. See :ref:`diagnostics-containers` for more info.
   * - ``handle=DIRECT``
     - Look for earlier peer-access errors on the source rank.

To see a record for every tested edge, including passing ones, add
``NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT``. Each passing edge produces
``Diagnostics P2P import``, ``write``, and ``read`` records with the same
fields:

.. code:: none

   NCCL INFO Diagnostics P2P write srcRank=6 srcCudaDev=6 srcNvmlDev=6 dstRank=5 dstCudaDev=5 dstNvmlDev=5 path=XGMI handle=LEGACY_CUDA_IPC topoRead=0

An edge that fails before the write phase has no ``write`` or ``read``
record. For example, when the import fails, the ``import`` record shows
``import=0`` and a ``reason``, and no ``write`` or ``read`` record follows
for that edge.

.. _diagnostics-containers:

Run in containers
=================

When the ranks of one node run as separate processes, they share GPU memory
through HIP IPC handles, or through HIP virtual-memory handles passed as file
descriptors when cuMem is enabled. Run all ranks of a node in one container
and make all GPUs of the node visible to it, for example with
``--device /dev/kfd --device /dev/dri``.

If the ranks of one node are split over several containers, what the check
reports depends on the containers:

* Containers with separate ``/dev/shm``, for example with a private IPC
  namespace, are treated as isolated. RCCL does not use P2P between them, so
  the check does not test those pairs and the summary counts fewer than
  ``N * (N - 1)`` peer accesses.
* If a pair between the containers is eligible for P2P but its memory cannot be
  shared, the check reports the edge as failed.
  ``destination buffer unavailable ... reason=noDescriptor`` means that the
  destination rank did not provide its buffer.
  ``peer-memory import failed ... reason=import`` means that the source rank
  could not map a buffer that the destination provided. Such failures are
  typical for HIP IPC handles (``handle=LEGACY_CUDA_IPC``). With cuMem enabled,
  the processes pass the memory handles as file descriptors, and the same
  split can pass.

Collect the report
==================

The report is written to standard output. When standard output is redirected
to a file or a pipe, it is buffered, and the report of a process that is killed
(for example by a job time limit) can be lost. To keep the report in that case,
run the application with line-buffered output:

.. code:: shell

   NCCL_RUN_DIAGNOSTICS=1 stdbuf -oL <application> [arguments] > out.log
