.. meta::
   :description: ROCm Compute Profiler - RDNA3.5 WGP / roofline metrics
   :keywords: ROCm Compute Profiler, RDNA, gfx11.5, WGP, roofline

.. _rdna-wgp:

=========================
Workgroup processor (WGP)
=========================

Within each shader engine, **Workgroup Processors (WGPs)** pair two **Compute Units (CUs)** that share resources and execute dispatched waves after the SPI workgroup manager hands off work.
On RDNA3-class GPUs (including discrete Ryzen APU 3x and RDNA3.5 / gfx11.5 integrations), compute kernels are typically tracked with wave32-oriented waves; the gfx11.5 **WGP** panels cover occupancy, dispatch, instruction mix, and local caches at that WGP/CU-pair granularity.

The sections below list RDNA3.5 (gfx11.5) metric descriptions.

.. Note::
   AMD Instinct (CDNA) GPUs use a different execution hierarchy and panel grouping.
   For Instinct-only pipeline metrics (for example, VALU / VMEM / MFMA-style tables), see :doc:`../cdna/cdna-performance-model`-without assuming RDNA WGPs or CUs map directly to those layouts.

Roofline
========

Roofline performance rates
--------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx11.5)
      :selected:

      .. jinja:: rdna115-roofline-performance-rates-gfx115
         :file: _templates/metrics_table.j2

Roofline plot points
--------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx11.5)
      :selected:

      .. jinja:: rdna115-roofline-plot-points-gfx115
         :file: _templates/metrics_table.j2

WGP block metrics
=================

WGP utilization
---------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx11.5)
      :selected:

      .. jinja:: rdna115-wgp-utilization-gfx115
         :file: _templates/metrics_table.j2

Wavefront launch stats
----------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx11.5)
      :selected:

      .. jinja:: rdna115-wavefront-launch-stats-gfx115
         :file: _templates/metrics_table.j2

Wave dispatch
-------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx11.5)
      :selected:

      .. jinja:: rdna115-wave-dispatch-gfx115
         :file: _templates/metrics_table.j2

Wave life
---------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx11.5)
      :selected:

      .. jinja:: rdna115-wave-life-gfx115
         :file: _templates/metrics_table.j2

Wave instruction mix
--------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx11.5)
      :selected:

      .. jinja:: rdna115-wave-instruction-mix-gfx115
         :file: _templates/metrics_table.j2

VMEM instruction mix
--------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx11.5)
      :selected:

      .. jinja:: rdna115-vmem-instruction-mix-gfx115
         :file: _templates/metrics_table.j2

LDS instruction mix
-------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx11.5)
      :selected:

      .. jinja:: rdna115-lds-instruction-mix-gfx115
         :file: _templates/metrics_table.j2

Wait state analysis
-------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx11.5)
      :selected:

      .. jinja:: rdna115-wait-state-analysis-gfx115
         :file: _templates/metrics_table.j2

WGP instruction cache
---------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx11.5)
      :selected:

      .. jinja:: rdna115-wgp-instruction-cache-gfx115
         :file: _templates/metrics_table.j2

WGP scalar data cache
---------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx11.5)
      :selected:

      .. jinja:: rdna115-wgp-scalar-data-cache-gfx115
         :file: _templates/metrics_table.j2
