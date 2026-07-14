.. meta::
   :description: Learn about GL2 Cache metrics in ROCm Compute Profiler, including cache performance, bandwidth, and request statistics on RDNA3.5 (gfx115x).
   :keywords: ROCm Compute Profiler, RDNA, gfx115x, GL2, GL2 Cache, GL2C

.. _rdna-gl2:

=========
GL2 cache
=========

On gfx115x, GL2 cache (RDNA naming for what Instinct documentation refers to as
L2/TCC) is the last-level GFX on-chip cache for most clients before traffic reaches the memory system.

After GL2, traffic proceeds toward DRAM through GCEA and related interfaces; those panels are documented under :doc:`gcea`.

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
      :selected:

      .. jinja:: rdna115x-gl2-cache-performance-gfx115x
         :file: _templates/metrics_table.j2

GL2 cache request statistics
----------------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-gl2-cache-request-statistics-gfx115x
         :file: _templates/metrics_table.j2

GL2 cache bandwidth
-------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-gl2-cache-bandwidth-gfx115x
         :file: _templates/metrics_table.j2

Memory chart: GL2 cache
=======================

The following Memory Chart table aligns with the on-screen flow through GL2.

Memory chart - GL2 cache
------------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-memory-chart-gl2-cache-gfx115x
         :file: _templates/metrics_table.j2
