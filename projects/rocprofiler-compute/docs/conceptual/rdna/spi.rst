.. meta::
   :description: ROCm Compute Profiler - RDNA3.5 Workgroup Manager (SPI) metrics
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, SPI, Shader Processor Input

.. _rdna-spi:

================================
Workgroup Manager (SPI)
================================

The **Shader Processor Input (SPI)** is the RDNA front-end unit that accepts dispatched work from the command processor and schedules wavefronts onto WGPs.
In profiler terminology it fills the **Workgroup Manager** role: it bridges kernel launches to runnable waves and tracks SPI-side utilization.

The sections below list RDNA3.5 (gfx1151) metric descriptions for SPI.

.. Note::
   For Instinct-centric shader-engine coverage, see :doc:`../cdna/shader-engine`.

SPI utilization
---------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-spi-utilization-gfx1151
         :file: _templates/metrics_table.j2

Wave dispatch statistics
------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-wave-dispatch-statistics-gfx1151
         :file: _templates/metrics_table.j2
