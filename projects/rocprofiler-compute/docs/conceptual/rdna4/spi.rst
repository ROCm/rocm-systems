.. meta::
   :description: ROCm Compute Profiler - RDNA4 Workgroup Manager (SPI)
   :keywords: ROCm Compute Profiler, RDNA, RDNA4, gfx1201, SPI, Shader Processor Input

.. _rdna4-spi:

================================
Workgroup Manager (SPI)
================================

The Shader Processor Input (SPI) is the RDNA front-end unit that accepts dispatched work from the command processor and schedules wavefronts onto Workgroup Processors (WGPs). In profiler terminology it fills the Workgroup Manager role: it bridges kernel launches to runnable waves.

On gfx1201, the Workgroup Manager (SPI) panel (panel 600) has these tables:

* SPI Utilization

* Wave Dispatch Statistics

* Resource Allocation

* CPC-SPI Interface

SPI Utilization and Wave Dispatch Statistics have metric descriptions.
Resource Allocation and CPC-SPI Interface have no metrics.
Run ``rocprof-compute analyze`` with ``--block 6`` to show the panel.

.. Note::

   For AMD Instinct-centric Shader engine coverage, see :doc:`../cdna/shader-engine`.
   For RDNA3.5 APUs (gfx115x), see :doc:`../rdna/spi`.
