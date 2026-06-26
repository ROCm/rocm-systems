.. meta::
   :description: ROCm Compute Profiler - RDNA3.5 WGP / roofline metrics
   :keywords: ROCm Compute Profiler, RDNA, gfx115x, WGP, roofline

.. _rdna-wgp:

=========================
Workgroup processor (WGP)
=========================

Within each shader engine, Workgroup Processors (WGPs) pair two Compute Units (CUs) that share resources and execute dispatched waves after the Workgroup Manager (SPI) hands off work.
On RDNA3.5 architecture-based GPUs/APUs, compute kernels are typically tracked with wave32-oriented waves; the gfx115x WGP panels cover occupancy, dispatch, instruction mix, and local caches at that WGP/CU-pair granularity.

The sections below list RDNA3.5 (gfx115x) metric descriptions.

.. Note::
   The CDNA architecture-based AMD Instinct GPUs use a different execution hierarchy and panel grouping.
   For Instinct-only pipeline metrics (for example, VALU / VMEM / MFMA-style tables), see :doc:`../cdna/cdna-performance-model`-without assuming RDNA WGPs or CUs map directly to those layouts.

Roofline
========

Roofline performance rates
--------------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-roofline-performance-rates-gfx115x
         :file: _templates/metrics_table.j2

Roofline plot points
--------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-roofline-plot-points-gfx115x
         :file: _templates/metrics_table.j2

WGP block metrics
=================

WGP utilization
---------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-wgp-utilization-gfx115x
         :file: _templates/metrics_table.j2

Wavefront launch stats
----------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-wavefront-launch-stats-gfx115x
         :file: _templates/metrics_table.j2

Wave dispatch
-------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-wave-dispatch-gfx115x
         :file: _templates/metrics_table.j2

Wave life
---------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-wave-life-gfx115x
         :file: _templates/metrics_table.j2

Wave instruction mix
--------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-wave-instruction-mix-gfx115x
         :file: _templates/metrics_table.j2

VMEM instruction mix
--------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-vmem-instruction-mix-gfx115x
         :file: _templates/metrics_table.j2

LDS instruction mix
-------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-lds-instruction-mix-gfx115x
         :file: _templates/metrics_table.j2

Wait state analysis
-------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-wait-state-analysis-gfx115x
         :file: _templates/metrics_table.j2

WGP instruction cache
---------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-wgp-instruction-cache-gfx115x
         :file: _templates/metrics_table.j2

WGP scalar data cache
---------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-wgp-scalar-data-cache-gfx115x
         :file: _templates/metrics_table.j2
