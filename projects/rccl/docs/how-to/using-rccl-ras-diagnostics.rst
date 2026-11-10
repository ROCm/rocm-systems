.. meta::
   :description: How to use the RCCL RAS diagnostics to compare GPU, driver, and NCCL configuration across the ranks of a job on AMD GPUs
   :keywords: RCCL, ROCm, AMD, RAS, diagnostics, NCCL_RUN_RAS_DIAGNOSTICS, rcclras, AMD SMI, ECC, XGMI, troubleshooting

.. _using-rccl-ras-diagnostics:

************************************************
Checking job configuration with RAS diagnostics
************************************************

The RAS (reliability, availability, and serviceability) subsystem of RCCL can
compare the configuration that every rank of a communicator reports and print a
short report. The report shows at a glance whether all ranks run with the same
``NCCL_*`` environment, driver version, and GPU configuration, which is a
common cause of hangs and performance differences in large jobs, and whether a
GPU reports memory errors or a down XGMI link.

This feature is inherited from NCCL 2.31 (RAS diagnostics). The checks only
collect and compare information. They do not send data between GPUs and do not
measure the network or the GPU links. To verify the GPU peer-to-peer data paths
on a node, use the active diagnostics (``NCCL_RUN_DIAGNOSTICS``) instead.

The report can be produced in two ways:

* At every communicator initialization, when ``NCCL_RUN_RAS_DIAGNOSTICS=1`` is
  set. See `Running at communicator initialization`_.
* On demand, at any time while the job runs, with the ``rcclras`` client. See
  `Running on demand`_.

Checks on AMD GPUs
==================

The report has one result per check and communicator:

* **GPU inventory:** compares the number of AMD GPUs on each node and the model
  of the GPU used by each rank.
* **HIP driver version:** compares the driver version that the HIP runtime
  reports (``hipDriverGetVersion``) across the ranks.
* **AMD GPU driver version:** compares the version of the ``amdgpu`` kernel
  driver that serves the GPU of each rank, as ``/sys/module/amdgpu/version``
  shows it, across the ranks.
* **ECC:** reports the ranks whose GPU has uncorrectable or deferred ECC
  errors. With ``NCCL_DIAGNOSTICS_ECC_THRESHOLD`` set, it also reports the
  ranks whose GPU has at least that many correctable errors.
* **XGMI:** compares the number of AMD Infinity Fabric (XGMI) links of the GPU
  used by each rank and reports the ranks whose GPU has a link that is down.
  Disabled links are not counted. A system without XGMI links prints no line.
* **NCCL environment:** compares the names and values of all ``NCCL_*``
  environment variables across the ranks and lists the ranks of every value
  that differs.

The GPU inventory, AMD GPU driver version, ECC, and XGMI checks read AMD SMI
(``libamd_smi.so``), the
same source as the ``amd-smi`` tool. With ``NCCL_RUN_RAS_DIAGNOSTICS=1``, RCCL
loads the library when the process creates its first communicator; otherwise,
the first time the diagnostics run. This does not depend on the value of
``RCCL_USE_AMD_SMI_LIB``. The ECC check uses the error totals of all memory
blocks of the GPU, which ``amd-smi metric --ecc`` shows per block. On a GPU
in a compute partition mode such as CPX or DPX, a rank whose partition AMD SMI
does not list reports the model, ECC counters, and XGMI links of the physical
GPU. Use ``amd-smi`` to inspect a GPU that the report names, for example
``amd-smi static --driver``, ``amd-smi metric --ecc``, and ``amd-smi xgmi``.

If AMD SMI cannot be loaded or does not answer for a GPU, the GPU inventory,
AMD GPU driver version, and ECC checks report ``unavailable via AMD SMI``. The
AMD GPU driver version is also unavailable with an ``amdgpu`` driver that is
built into the kernel, which has no ``/sys/module/amdgpu/version``. Such
results are tagged ``[INFO]`` and do not indicate a problem with the system. The XGMI check prints
no line when no GPU answers; a GPU that does not answer while others do counts
as having no links and is reported as a link-count mismatch. Set
``NCCL_DEBUG=INFO`` and ``NCCL_DEBUG_SUBSYS=RAS`` to log the AMD SMI query that
failed, or the AMD SMI functions that an older library does not provide.

The report also contains the results of checks that come from NCCL and are not
described here: ``rdma_topo``, ``IOMMU mode``, ``ATS state``, ``Xid/SXid``, and
``Paths``. A check that finds no data on the system reports ``[INFO]``, as in
the sample report in `Reading the report`_.

Prerequisites
=============

RAS diagnostics need the RAS subsystem, which is enabled by default
(``NCCL_RAS_ENABLE=1``). With ``NCCL_RAS_ENABLE=0``, a communicator created
with ``NCCL_RUN_RAS_DIAGNOSTICS=1`` prints only
``NCCL DIAG INFO RAS diagnostics skipped: NCCL_RAS_ENABLE=0``: no check runs,
and no report header or ``completed`` line is printed.

The RAS threads of all processes of the job listen for client connections on
``localhost``, port ``28028``. Set ``NCCL_RAS_ADDR`` to change the address, for
example when several jobs share a node. See
`Running on demand on a shared node`_.

Running at communicator initialization
======================================

Initialization-time diagnostics are disabled by default. Set
``NCCL_RUN_RAS_DIAGNOSTICS=1`` for every process of the job:

.. code:: shell

   NCCL_RUN_RAS_DIAGNOSTICS=1 <application> [arguments]

For example, with rccl-tests:

.. code:: shell

   NCCL_RUN_RAS_DIAGNOSTICS=1 ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8

The diagnostics run at every communicator initialization, including
``ncclCommInitRank``, ``ncclCommInitAll``, and ``ncclCommSplit``. Each report
covers only the new communicator, and an application that creates several
communicators prints one report per communicator.

The report is printed to the standard output of the process that hosts rank 0
of the communicator, not to ``NCCL_DEBUG_FILE``. That process prints the
header while the communicator is being created. The results and the
``completed`` line follow from the RAS thread shortly after, usually after
the initialization call has returned. If a process exits or destroys the
communicator right after creating it, the report can be incomplete.

Diagnostics are informational. A reported difference does not make
communicator initialization fail, and the communicator remains usable.

Running on demand
=================

Request a report from a running job with the ``-D`` option of the ``rcclras``
client, on any node of the job. The request does not depend on
``NCCL_RUN_RAS_DIAGNOSTICS``:

.. code:: shell

   rcclras -D
   rcclras -h <host> -p <port> -D

``rcclras`` is installed with RCCL, in the ``bin`` directory of the ROCm
installation. Without a client binary, send the ``DIAGNOSTICS`` command of the
RAS text protocol, for example with netcat:

.. code:: shell

   echo diagnostics | nc localhost 28028

The report covers every communicator known to the RAS threads of the job and is
returned over the client connection. It is printed by ``rcclras``, not by the
application. The report is available in text format only: with
``rcclras -D -f json``, the job replies
``ERROR: diagnostics only supports text output``.

Request a report after every rank has finished creating its communicators. An
earlier request covers only the ranks that RAS already knows about, and reports
the others as incomplete or misses them.

Running on demand on a shared node
----------------------------------

RAS processes of the same user share the client port, and so do two unrelated
jobs of the same user: both listen on it, the kernel spreads incoming
connections between them, and ``rcclras`` reaches either job with no error and
no log message. If another user's job on the same node already listens on the
port, the RAS threads of your job cannot listen on it, and ``rcclras`` silently
connects to the other job instead. This can happen when containers use the host
network. Only this cross-user conflict leaves a sign: a ``RAS failed to
establish a client listening socket`` message, which is printed with
``NCCL_DEBUG=INFO``.

To avoid the conflict, choose a free port for the job and pass the same port to
the client:

.. code:: shell

   export NCCL_RAS_ADDR=localhost:<port>
   rcclras -p <port> -D

The client port does not check who connects. Any user who can reach the port
can read the job state that ``rcclras`` prints (hosts, process IDs, GPUs and
communicators), request a diagnostics report, which includes the names and
values of ``NCCL_*`` variables that differ across ranks, and change the
profiler event mask of every process of the job. Keep ``NCCL_RAS_ADDR`` on
``localhost`` unless remote access is required.

Reading the report
==================

Every line starts with ``<hostname>:<pid> NCCL DIAG``. A report consists of a
header, the results grouped by check and communicator, and a ``completed``
line. An initialization-time report on an 8-GPU AMD Instinct MI355X node, with
one process per GPU, looks like this:

.. code:: none

   node01:4242 NCCL DIAG === RAS Diagnostics ===
   node01:4242 NCCL DIAG [OK]   GPU inventory: 8x AMD Instinct MI355X per node consistent across 8 ranks in comm 0x5fa31c27a9e0d1b4
   node01:4242 NCCL DIAG [OK]   HIP driver version: 71526333 consistent across 8 ranks in comm 0x5fa31c27a9e0d1b4
   node01:4242 NCCL DIAG [OK]   ECC: no uncorrected volatile errors across 8 ranks in comm 0x5fa31c27a9e0d1b4
   node01:4242 NCCL DIAG [OK]   XGMI: 7 links per GPU, all active across 8 ranks in comm 0x5fa31c27a9e0d1b4
   node01:4242 NCCL DIAG [OK]   NCCL environment: NCCL_* env vars consistent across 8 ranks in comm 0x5fa31c27a9e0d1b4
   node01:4242 NCCL DIAG [INFO] rdma_topo check: not usable on 8/8 ranks in comm 0x5fa31c27a9e0d1b4 (unavailable)
   node01:4242 NCCL DIAG [INFO] IOMMU mode: unable to identify relevant GPU/NIC pairs on 8/8 ranks in comm 0x5fa31c27a9e0d1b4
   node01:4242 NCCL DIAG [INFO] ATS state: unable to identify relevant NICs on 8/8 ranks in comm 0x5fa31c27a9e0d1b4
   node01:4242 NCCL DIAG [INFO] Xid/SXid: no kernel-log source was available on host node01
   node01:4242 NCCL DIAG [OK]   AMD GPU driver version: 6.14.14 consistent across 8 ranks in comm 0x5fa31c27a9e0d1b4
   node01:4242 NCCL DIAG [OK]   Paths: self+XGMI across 8 ranks in comm 0x5fa31c27a9e0d1b4
   node01:4242 NCCL DIAG RAS diagnostics completed in 64.5 ms across 8 ranks

The ``completed`` line of an initialization-time report counts the ranks of
the communicator. The ``completed`` line of an on-demand report counts the RAS
peers that answered, that is, the processes of the job, for example
``across 16 RAS peers`` for two nodes with one process per GPU.

Result lines use two tags:

* ``[OK]`` means that the check found no difference across the ranks and, for
  the ECC and XGMI checks, no error and no down link.
* ``[INFO]`` marks a difference, an ECC error or down link, incomplete
  information, or a check whose data is unavailable. Read the message text to
  tell them apart.

A check that finds a difference prints several lines that group the ranks by
value. For example, a job where the processes on the second node run with a
different ``NCCL_BUFFSIZE``:

.. code:: none

   node01:4242 NCCL DIAG [INFO] NCCL environment: mismatch across 16 ranks in comm 0x5fa31c27a9e0d1b4 for NCCL_BUFFSIZE
   node01:4242 NCCL DIAG [INFO] NCCL environment: NCCL_BUFFSIZE=(unset) on rank(s) {0,1,2,3,4,5,6,7}
   node01:4242 NCCL DIAG [INFO] NCCL environment: NCCL_BUFFSIZE=8388608 on rank(s) {8,9,10,11,12,13,14,15}
   node01:4242 NCCL DIAG [INFO] NCCL environment: 1 NCCL_* env var(s) differ across ranks in comm 0x5fa31c27a9e0d1b4

Rank lists contain a limited number of entries. A longer list ends with an
ellipsis and the total number of ranks in the group.

Variables that legitimately differ between processes, such as a per-process
``NCCL_DEBUG_FILE``, are reported in the same way. Compare the listed
variables with the configuration you intended.

If not every rank answered, for example because a process stopped responding,
the result reads ``diagnostics incomplete, gathered <n>/<total> ranks`` and
names the communicator.

The ``[INFO]`` results that need attention:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Result
     - What to check
   * - ``GPU inventory: count mismatch``
     - The listed ranks run on nodes with a different number of AMD GPUs, or
       on nodes where AMD SMI could not count the GPUs, which counts as none.
       Compare ``amd-smi list`` on those nodes, for example after a GPU reset.
   * - ``GPU inventory: model mismatch``
     - The listed ranks use a different GPU model than the lowest rank of the
       communicator, whose value is shown in parentheses.
   * - ``HIP driver version: mismatch``
     - The listed ranks run a different driver version. Compare the installed
       ROCm and ``amdgpu`` driver versions on those nodes.
   * - ``AMD GPU driver version: mismatch``
     - The GPUs of the listed ranks are served by a different ``amdgpu``
       driver than the GPU of the lowest rank of the communicator, whose
       version is shown in parentheses. A rank whose version AMD SMI could not
       read differs as well. Compare ``amd-smi static --driver`` on those
       nodes.
   * - ``ECC: uncorrected volatile errors on rank(s)``
     - The GPUs of the listed ranks have uncorrectable or deferred ECC errors.
       ``worst`` is the highest count. Inspect them with ``amd-smi metric --ecc``.
   * - ``ECC: corrected volatile errors at or above threshold``
     - The GPUs of the listed ranks reached ``NCCL_DIAGNOSTICS_ECC_THRESHOLD``
       correctable errors.
   * - ``XGMI: link-count mismatch``
     - The GPUs of the listed ranks have a different number of XGMI links than
       the lowest rank of the communicator, whose value is shown in
       parentheses. A GPU that AMD SMI did not answer for counts as having no
       links, so ``(0)`` means that the lowest rank's GPU did not answer or has
       no XGMI links.
       Inspect the GPUs with ``amd-smi xgmi``.
   * - ``XGMI: inactive link(s) on rank(s)``
     - A link of the GPUs of the listed ranks is down. Inspect them with
       ``amd-smi xgmi``.
   * - ``NCCL environment: mismatch``
     - The ranks run with different values of the named variable. The lines
       that follow list the ranks of each value.
   * - ``NCCL environment: ... comparison may be partial``
     - The ``NCCL_*`` variables of the listed number of ranks exceed the size
       that the check compares, so a difference in the rest can go unreported.
   * - ``diagnostics incomplete``
     - Not every rank answered. Check that the processes of the job are still
       running.
