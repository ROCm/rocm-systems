.. meta::
   :description: Learn about the Command Processor Compute (CPC) metrics in ROCm Compute Profiler, including utilization, interface activity, stalls, and cache behavior on RDNA 3.5 (gfx11.5).
   :keywords: ROCm Compute Profiler, RDNA, gfx11.5, command processor, CPC

.. _rdna-command-processor:

=========================
Command processor (CP)
=========================

The **command processor (CP)** is the GPU front-end that connects the host and
kernel driver to on-GPU scheduling: it pulls work from HSA queues, decodes
packets, and dispatches kernel launches to the shader-engine SPI / WGP path.
On Instinct GPUs, the profiler often separates the metrics into command processor
fetcher (CPF) and command processor compute (CPC). The gfx11.5 analysis panels
emphasize CPC and ME (Micro Engine) activity, including utilization,
interface utilization, stall cycles, memory requests, and instruction cache.

For the complete CDNA architecture overview and the CPF and CPC metric tabs across MI-series GPUs, see
:doc:`../cdna/command-processor` under CDNA-CDNA4.

Command processor compute (CPC) - gfx11.5
===========================================

CPC utilization
---------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx11.5)
      :selected:

      .. jinja:: rdna115-cpc-utilization-gfx115
         :file: _templates/metrics_table.j2

CPC interface utilization
-------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx11.5)
      :selected:

      .. jinja:: rdna115-cpc-interface-utilization-gfx115
         :file: _templates/metrics_table.j2

Micro Engine (ME) stall cycles
-------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx11.5)
      :selected:

      .. jinja:: rdna115-mec-stall-cycles-gfx115
         :file: _templates/metrics_table.j2

CPC memory requests
-------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx11.5)
      :selected:

      .. jinja:: rdna115-cpc-memory-requests-gfx115
         :file: _templates/metrics_table.j2

Micro Engine (ME) instruction cache
------------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx11.5)
      :selected:

      .. jinja:: rdna115-mec-instruction-cache-gfx115
         :file: _templates/metrics_table.j2
