.. meta::
   :description: ROCm Compute Profiler performance model: Command processor (CP)
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, profiler, tool, Instinct, accelerator, command, processor, fetcher, packet processor, CPF, CPC

**********************
Command processor (CP)
**********************

The command processor (CP) is responsible for interacting with the AMDGPU kernel
driver -- the Linux kernel -- on the CPU and for interacting with user-space
HSA clients when they submit commands to HSA queues. Basic tasks of the CP
include reading commands (such as, corresponding to a kernel launch) out of
:hsa-runtime-pdf:`HSA queues <68>`, scheduling work to subsequent parts of the
scheduler pipeline, and marking kernels complete for synchronization events on
the host.

The command processor consists of two sub-components:

* :ref:`Fetcher <cpf-metrics>` (CPF): Fetches commands out of memory to hand
  them over to the CPC for processing.

* :ref:`Packet processor <cpc-metrics>` (CPC): Micro-controller running the
  command processing firmware that decodes the fetched commands and (for
  kernels) passes them to the :ref:`workgroup processors <desc-spi>` for
  scheduling.

Before scheduling work to the accelerator, the command processor can
first acquire a memory fence to ensure system consistency
(:hsa-runtime-pdf:`Section 2.6.4 <91>`). After the work is complete, the
command processor can apply a memory-release fence. Depending on the AMD CDNA™
accelerator under question, either of these operations *might* initiate a cache
write-back or invalidation.

Analyzing command processor performance is most interesting for kernels
that you suspect to be limited by scheduling or launch rate. The command
processor’s metrics therefore are focused on reporting, for example:

*  Utilization of the fetcher

*  Utilization of the packet processor, and decoding processing packets

*  Stalls in fetching and processing

.. _cpf-metrics:

Command processor fetcher (CPF)
===============================

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - CPF Utilization

     - Percent of total cycles where the CPF was busy actively doing any work.
       The ratio of CPF busy cycles over total cycles counted by the CPF.

     - Percent

   * - CPF Stall

     - Percent of CPF busy cycles where the CPF was stalled for any reason.

     - Percent

   * - CPF-L2 Utilization

     - Percent of total cycles counted by the CPF-:doc:`L2 <l2-cache>` interface
       where the CPF-L2 interface was active doing any work. The ratio of CPF-L2
       busy cycles over total cycles counted by the CPF-L2.

     - Percent

   * - CPF-L2 Stall

     - Percent of CPF-:doc:`L2 <l2-cache>` L2 busy cycles where the CPF-L2
       interface was stalled for any reason.

     - Percent

   * - CPF-UTCL1 Stall

     - Percent of CPF busy cycles where the CPF was stalled by address
       translation.

     - Percent

.. _cpc-metrics:

Command processor packet processor (CPC)
========================================

.. list-table::
   :header-rows: 1

   * - Metric

     - Description

     - Unit

   * - CPC Utilization

     - Percent of total cycles where the CPC was busy actively doing any work.
       The ratio of CPC busy cycles over total cycles counted by the CPC.

     - Percent

   * - CPC Stall

     - Percent of CPC busy cycles where the CPC was stalled for any reason.

     - Percent

   * - CPC Packet Decoding Utilization

     - Percent of CPC busy cycles spent decoding commands for processing.

     - Percent

   * - CPC-Workgroup Manager Utilization

     - Percent of CPC busy cycles spent dispatching workgroups to the
       :ref:`workgroup manager <desc-spi>`.

     - Percent

   * - CPC-L2 Utilization

     - Percent of total cycles counted by the CPC-:doc:`L2 <l2-cache>` interface
       where the CPC-L2 interface was active doing any work.

     - Percent

   * - CPC-UTCL1 Stall

     - Percent of CPC busy cycles where the CPC was stalled by address
       translation.

     - Percent

   * - CPC-UTCL2 Utilization

     - Percent of total cycles counted by the CPC's :doc:`L2 <l2-cache>` address
       translation interface where the CPC was busy doing address translation
       work.

     - Percent
