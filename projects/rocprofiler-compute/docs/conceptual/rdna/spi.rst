.. meta::
   :description: ROCm Compute Profiler - RDNA3.5 Workgroup Manager (SPI) metrics
   :keywords: ROCm Compute Profiler, RDNA, gfx115x, SPI, Shader Processor Input

.. _rdna-spi:
.. _rdna4-spi:

================================
Workgroup Manager (SPI)
================================

The Shader Processor Input (SPI) is the RDNA front-end unit that accepts dispatched work from the command processor and schedules wavefronts onto Workgroup processors (WGPs). In profiler terminology it fills the Workgroup Manager role: Bridges the kernel launches to runnable waves and tracks SPI-side utilization.

The sections below list SPI metric descriptions. Select the RDNA3.5 or RDNA4 tab on each table.

.. Note::

   For AMD Instinct-centric Shader engine coverage, see :doc:`../cdna/shader-engine`.

SPI utilization
---------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-spi-utilization-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      SPI Utilization and Wave Dispatch Statistics have metric descriptions on gfx1201. Resource Allocation and CPC-SPI Interface have no metrics. This page does not publish a separate gfx1201 table.

Wave dispatch statistics
------------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-wave-dispatch-statistics-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      SPI Utilization and Wave Dispatch Statistics have metric descriptions on gfx1201. Resource Allocation and CPC-SPI Interface have no metrics. This page does not publish a separate gfx1201 table.
