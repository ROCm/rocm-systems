.. meta::
   :description: ROCm Compute Profiler performance model: System Speed-of-Light (RDNA)
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, system speed of light

.. _sys-sol-rdna:

=====================
System Speed-of-Light
=====================

This page lists System Speed-of-Light metrics for RDNA3.5 (gfx1151) when
you profile with the shipped gfx1151 analysis configuration. The same metric
keys appear in the RDNA3.5 (gfx1151) tab of the analysis report.

.. Note::
   For AMD Instinct accelerators (CDNA-CDNA4), see
   :doc:`../cdna/system-speed-of-light`.

   Other gfx1151 metric tables grouped by hardware block live under :doc:`wgp`,
   :doc:`tcp-cache`, :doc:`gl1-cache`, :doc:`gl2-cache`, :doc:`shader-engine`, and :doc:`command-processor`.

.. warning::

   Theoretical peaks use the maximum clock frequency reported for the GPU (for
   example via ``rocminfo``). That may not match sustained clocks under your
   workload.

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: sys-sol-gfx1151
         :file: _templates/metrics_table.j2
