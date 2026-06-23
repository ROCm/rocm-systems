.. meta::
   :description: ROCm Compute Profiler - RDNA3.5 Workgroup Manager (SPI) metrics
   :keywords: ROCm Compute Profiler, RDNA, gfx115x, SPI, Shader Processor Input

.. _rdna-spi:

================================
Workgroup Manager (SPI)
================================

The **Shader Processor Input (SPI)** is the RDNA front-end unit that accepts dispatched work from the command processor and schedules wavefronts onto WGPs.
In profiler terminology it fills the **Workgroup Manager** role: it bridges kernel launches to runnable waves and tracks SPI-side utilization.

The sections below list RDNA3.5 (gfx115x) metric descriptions for SPI.

.. Note::
   For Instinct-centric shader-engine coverage, see :doc:`../cdna/shader-engine`.

SPI utilization
---------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-spi-utilization-gfx115x
         :file: _templates/metrics_table.j2

Wave dispatch statistics
------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-wave-dispatch-statistics-gfx115x
         :file: _templates/metrics_table.j2
