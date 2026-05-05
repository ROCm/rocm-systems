.. meta::
   :description: ROCm Compute Profiler performance model: System Speed-of-Light (RDNA)
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, system speed of light

.. _sys-sol-rdna:

=====================
System Speed-of-Light
=====================

This page documents System Speed-of-Light metrics for RDNA3.5 (gfx1151) with the
shipped gfx1151 analysis configuration.
**System Speed-of-Light** is a high-level summary: it highlights the most important
metrics for how your workload is performing on the target GPU, so you can spot
bottlenecks before diving into block-specific panels.

.. Note::
   For AMD Instinct accelerators (CDNA-CDNA4), see
   :doc:`../cdna/system-speed-of-light`.

   Other gfx1151 metric tables follow the RDNA3 hierarchy in :doc:`rdna-performance-model`
   (shader engine: :doc:`spi`, :doc:`wgp`, :doc:`gl0-cache`, :doc:`gl1-cache`; then
   :doc:`gl2-cache`, :doc:`gcea`, :doc:`command-processor`, :doc:`grbm`).

.. warning::

   Theoretical peaks use the maximum clock frequency reported for the GPU (for
   example via ``rocminfo``). That may not match sustained clocks under your
   workload.

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: sys-sol-gfx1151
         :file: _templates/metrics_table.j2
