.. meta::
   :description: Learn about the Command Processor Compute (CPC) panel in ROCm Compute Profiler on RDNA4 (gfx1201).
   :keywords: ROCm Compute Profiler, RDNA, RDNA4, gfx1201, command processor, CPC

.. _rdna4-command-processor:

=========================
Command processor (CP)
=========================

The command processor (CP) is the GPU front-end that connects the host and
kernel driver to on-GPU scheduling: it pulls work from HSA queues, decodes
packets, and dispatches kernel launches to the shader engine Workgroup Manager (SPI).
On Instinct GPUs, the profiler often separates the metrics into Command Processor
Fetcher (CPF) and Command Processor Compute (CPC). The gfx1201 analysis panel
covers CPC.

The CPC (Command Processor - Compute) panel (panel 500) has one table, CPC
Utilizations. That table has metric descriptions in the analysis report. Run
``rocprof-compute analyze`` with ``--block 5`` to show the panel.

For the complete CDNA architecture overview and the CPF and CPC metric tabs across AMD Instinct MI-Series GPUs, see
:doc:`../cdna/command-processor` under CDNA-CDNA4. For RDNA3.5 APUs (gfx115x), see
:doc:`../rdna/command-processor`.
