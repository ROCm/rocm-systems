.. meta::
   :description: ROCm Compute Profiler — RDNA3.5 WGP / roofline metrics
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, WGP, roofline

.. _rdna-wgp:

=========================
Workgroup processor (WGP)
=========================

On RDNA3-class GPUs (including discrete Navi 3x and RDNA3.5 / gfx1151
integrations), shader work is organized into Workgroup Processors (WGPs). A WGP
pairs two Compute Units (CUs) that share resources; compute kernels are typically
tracked with wave32-oriented waves in the gfx1151 panel set. The profiler's
WGP panel is the analogue of the Instinct compute unit chapter: occupancy,
dispatch, instruction mix, and local caches at that granularity.

For VALU / VMEM / MFMA-style pipeline tables and MI-series diagrams, see
:doc:`../cdna/compute-unit`.

The sections below list RDNA3.5 (gfx1151) metric descriptions for this chapter.

Roofline
========

Roofline Performance Rates
--------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-roofline-performance-rates-gfx1151
         :file: _templates/metrics_table.j2

Roofline Plot Points
--------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-roofline-plot-points-gfx1151
         :file: _templates/metrics_table.j2

WGP block metrics
=================

WGP Utilization
---------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-wgp-utilization-gfx1151
         :file: _templates/metrics_table.j2

Wavefront Launch Stats
----------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-wavefront-launch-stats-gfx1151
         :file: _templates/metrics_table.j2

Wave Dispatch
-------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-wave-dispatch-gfx1151
         :file: _templates/metrics_table.j2

Wave Life
---------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-wave-life-gfx1151
         :file: _templates/metrics_table.j2

Wave Instruction Mix
--------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-wave-instruction-mix-gfx1151
         :file: _templates/metrics_table.j2

VMEM Instruction Mix
--------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-vmem-instruction-mix-gfx1151
         :file: _templates/metrics_table.j2

LDS Instruction Mix
-------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-lds-instruction-mix-gfx1151
         :file: _templates/metrics_table.j2

Wait State Analysis
-------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-wait-state-analysis-gfx1151
         :file: _templates/metrics_table.j2

WGP Instruction Cache
---------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-wgp-instruction-cache-gfx1151
         :file: _templates/metrics_table.j2

WGP Scalar Data Cache
---------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-wgp-scalar-data-cache-gfx1151
         :file: _templates/metrics_table.j2
