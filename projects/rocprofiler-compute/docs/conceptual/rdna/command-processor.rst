.. meta::
   :description: Learn about the Command Processor Compute (CPC) metrics in ROCm Compute Profiler, including utilization, interface activity, stalls, and cache behavior on RDNA 3.5 (gfx1151).
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, command processor, CPC

.. _rdna-command-processor:

=========================
Command processor (CP)
=========================

The command processor (CP) connects the host and kernel driver to on-GPU
scheduling. During the process it pulls work from HSA queues, decodes packets, and dispatches the kernel
launches to the front-end (SPI / WGP path). On Instinct GPUs, the profiler
often seperates the metrics into command processor fetcher (CPF) and command processor compute (CPC). The
gfx1151 analysis panels emphasize CPC and ME (Micro Engine) activity, including utilization,
interface utilization, stall cycles, memory requests, and instruction cache.

For the complete CDNA architecture overview and the CPF and CPC metric tabs across MI-series GPUs, see
:doc:`../cdna/command-processor` under CDNA-CDNA4.

Command processor compute (CPC) — gfx1151
===========================================

CPC utilization
---------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-cpc-utilization-gfx1151
         :file: _templates/metrics_table.j2

CPC interface utilization
-------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-cpc-interface-utilization-gfx1151
         :file: _templates/metrics_table.j2

Micro Engine (ME) stall cycles
-------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-mec-stall-cycles-gfx1151
         :file: _templates/metrics_table.j2

CPC memory requests
-------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-cpc-memory-requests-gfx1151
         :file: _templates/metrics_table.j2

Micro Engine (ME) instruction cache
------------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-mec-instruction-cache-gfx1151
         :file: _templates/metrics_table.j2
