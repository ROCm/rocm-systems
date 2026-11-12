.. meta::
   :description: Learn about the Command Processor Compute (CPC) metrics in ROCm Compute Profiler, including utilization, interface activity, stalls, and cache behavior on RDNA3.5 (gfx115x).
   :keywords: ROCm Compute Profiler, RDNA, gfx115x, command processor, CPC

.. _rdna-command-processor:
.. _rdna4-command-processor:

=========================
Command processor (CP)
=========================

The command processor (CP) is the GPU front-end that connects the host and
kernel driver to on-GPU scheduling: it pulls work from HSA queues, decodes
packets, and dispatches kernel launches to the Shader engine Workgroup Manager (SPI) or Workgroup processor (WGP) path.
On Instinct GPUs, the profiler often separates the metrics into Command Processor
Fetcher (CPF) and Command Processor Compute (CPC). The gfx115x analysis panels
emphasize CPC and ME (Micro Engine) activity, including utilization,
interface utilization, stall cycles, memory requests, and instruction cache.
gfx1201 has no CPC panel. Select the architecture tab on each table.

For the complete CDNA architecture overview and the CPF and CPC metric tabs across AMD Instinct MI-Series GPUs, see
:doc:`../cdna/command-processor` under CDNA-CDNA4.

Command processor compute (CPC) - gfx115x
===========================================

CPC utilization
---------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-cpc-utilization-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      gfx1201 has no CPC panel.

CPC interface utilization
-------------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-cpc-interface-utilization-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      gfx1201 has no CPC panel.

Micro Engine (ME) stall cycles
-------------------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-mec-stall-cycles-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      gfx1201 has no CPC panel.

CPC memory requests
-------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-cpc-memory-requests-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      gfx1201 has no CPC panel.

Micro Engine (ME) instruction cache
------------------------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-mec-instruction-cache-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      gfx1201 has no CPC panel.
