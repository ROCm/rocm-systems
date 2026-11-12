.. meta::
   :description: Learn about GL2 Cache metrics in ROCm Compute Profiler, including cache performance, bandwidth, and request statistics on RDNA3.5 (gfx115x).
   :keywords: ROCm Compute Profiler, RDNA, gfx115x, GL2, GL2 Cache, GL2C

.. _rdna-gl2:
.. _rdna4-gl2:

=========
GL2 cache
=========

GL2 cache (RDNA naming for what Instinct documentation refers to as L2/TCC) is the last-level GFX on-chip cache for most clients before traffic reaches the memory system.
On gfx115x, traffic then proceeds toward system memory through GCEA. On gfx1201, Navi 48 has an 8 MB GL2, modeled as 32 channels, and traffic then goes through the Efficiency Arbiter toward the Infinity Cache and DRAM. Those panels are documented under :doc:`gcea`.

For Instinct L2 (TCC) coherence, channel hashing, and fabric metrics on CDNA
architecture-based AMD Instinct MI-Series GPUs, see :doc:`../cdna/l2-cache` under
CDNA-CDNA4.

.. note::

   The GL2 Cache is also referred to as GL2C in some contexts. Hardware counter
   names (for example, ``GL2C_HIT_sum``) retain the GL2C prefix.

GL2 cache panels
================

GL2 cache performance
---------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-gl2-cache-performance-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: navi48-gl2-lookup-gfx1201
         :file: _templates/metrics_table.j2

GL2 cache request statistics
----------------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-gl2-cache-request-statistics-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: navi48-gl2-utilization-gfx1201
         :file: _templates/metrics_table.j2

GL2 cache bandwidth
-------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-gl2-cache-bandwidth-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      gfx1201 has no GL2 cache bandwidth table.

Memory chart: GL2 cache
=======================

The following Memory Chart table aligns with the on-screen flow through GL2.

Memory chart - GL2 cache
------------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-memory-chart-gl2-cache-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-memory-chart-gl2-cache-gfx1201
         :file: _templates/metrics_table.j2
