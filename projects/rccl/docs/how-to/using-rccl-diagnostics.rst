.. meta::
   :description: How to use the RCCL active diagnostics to verify GPU peer-to-peer paths during communicator initialization on AMD GPUs
   :keywords: RCCL, ROCm, AMD, diagnostics, NCCL_RUN_DIAGNOSTICS, P2P, XGMI, troubleshooting

.. _using-rccl-diagnostics:

*****************************************
Verifying GPU P2P paths with diagnostics
*****************************************

RCCL can check the GPU peer-to-peer (P2P) paths it intends to use while a
communicator is being created. The check writes data between every eligible
pair of GPUs on a node, reads it back, and prints a short report. A passing
report rules out broken intra-node links, peer-memory mapping problems, and
container or IPC isolation problems before the application sends any traffic.
A failing report names the exact GPU pair and path to investigate.

This feature is inherited from NCCL 2.31 (active diagnostics). The only check
available is the P2P check.

Enabling diagnostics
====================

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

Reading the report
==================

The report is printed to standard output (not to ``NCCL_DEBUG_FILE``) by the
process that hosts rank 0 of the communicator. Other ranks do not print the
report. Every line starts with ``<hostname>:<pid> NCCL DIAG``.

A passing run on an 8-GPU AMD Instinct MI355X node, with one process per GPU,
looks like this:

.. code:: none

   node01:15 NCCL DIAG === NCCL Diagnostics ===
   node01:15 NCCL DIAG [OK]   p2p: all 56 directed GPU P2P edges verified
   node01:15 NCCL DIAG NCCL diagnostics completed in 40.0 ms across 8 ranks

Result lines use two tags:

* ``[OK]`` means that every tested edge passed.
* ``[INFO]`` means that at least one edge failed or that a step of the check
  could not run. The lines that follow identify the affected GPU pairs.

Which GPU pairs are tested
--------------------------

The check tests directed edges. For each pair of GPUs A and B it tests both
A to B and B to A. Only GPUs on the same node are tested, and only pairs that
RCCL's topology detection allows to use P2P. On a node with ``N`` GPUs in the
communicator, the check tests ``N * (N - 1)`` edges when all pairs are
eligible. For a multi-node communicator the edge counts of all nodes are added
together. For example, two nodes with 8 GPUs each report 112 edges.

The check does not test the network between nodes.

Pairs that can only reach each other through an intermediate GPU are not
tested. They are listed as ``(skipped indirect=<count>)`` at the end of the
summary line.

.. note::

   If no pair is eligible, the report contains only the header and the
   ``completed`` line, with no ``p2p:`` line. This happens, for example, with
   ``NCCL_P2P_DISABLE=1``, with a restrictive ``NCCL_P2P_LEVEL``, or with one GPU
   per node. A missing ``p2p:`` line means that nothing was tested, not that
   everything passed.

Single-process and multi-process jobs
-------------------------------------

When one process drives several GPUs, for example ``ncclCommInitAll`` or
rccl-tests with ``-g 8``, the check enables peer access between the devices of
that process for its duration. Each rank then prints a line similar to the
following. This is expected and is not an error:

.. code:: none

   node01:4242 NCCL DIAG [INFO] p2p: temporarily enabled context-wide CUDA peer access rank=3 cudaDev=3; avoid concurrent CUDA use on this context until diagnostics completes

In this message, "CUDA" refers to the HIP runtime. Do not issue HIP work on
these devices from other threads while the communicator is being initialized.

When each GPU is driven by its own process, peer memory is shared through HIP
IPC handles and no such line is printed.

Performance impact
------------------

On an 8-GPU MI355X node the check takes about 30 to 50 ms per communicator
initialization, which is small compared with the time of the initialization
itself. It does not change the performance of collectives after
initialization.
Because the check runs at every initialization, enable it while bringing up or
debugging a system, and leave it disabled for jobs that create many
communicators.

Troubleshooting failed edges
============================

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
       job this usually means that the processes cannot share HIP IPC handles,
       see :ref:`diagnostics-containers`.
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

The edge fields have the following meaning:

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
       process) or ``LEGACY_CUDA_IPC`` (HIP IPC handle between processes).

The suggested next step at the end of each line comes from NCCL and refers to
NVIDIA tools (``nvidia-smi``, ``nvidia-imex-ctl``). On AMD GPUs, use the
following commands instead:

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
     - Check that all processes see the GPUs and can share IPC handles, see
       :ref:`diagnostics-containers`.
   * - ``handle=DIRECT``
     - Look for earlier peer-access errors on the source rank.

To see a record for every tested edge, including passing ones, add
``NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT``. Each edge produces
``Diagnostics P2P import``, ``write``, and ``read`` records with the same
fields:

.. code:: none

   NCCL INFO Diagnostics P2P write srcRank=6 srcCudaDev=6 srcNvmlDev=6 dstRank=5 dstCudaDev=5 dstNvmlDev=5 path=XGMI handle=LEGACY_CUDA_IPC topoRead=0

.. _diagnostics-containers:

Running in containers
=====================

When the ranks of one node run as separate processes, they share GPU memory
through HIP IPC handles. If the processes run in containers that cannot share
these handles, the check reports every affected edge as
``destination buffer unavailable ... handle=LEGACY_CUDA_IPC reason=noDescriptor``.

Run all ranks of a node in one container, or start the containers with
``--ipc=host``, and make all GPUs of the node visible to them with
``--device /dev/kfd --device /dev/dri``.

Collecting the report
=====================

The report is written to standard output. When standard output is redirected
to a file or a pipe, it is buffered, and the report of a process that is killed
(for example by a job time limit) can be lost. To keep the report in that case,
run the application with line-buffered output:

.. code:: shell

   NCCL_RUN_DIAGNOSTICS=1 stdbuf -oL <application> [arguments] > out.log

Related information
===================

* :ref:`troubleshooting-rccl`
* :ref:`env-variables`
