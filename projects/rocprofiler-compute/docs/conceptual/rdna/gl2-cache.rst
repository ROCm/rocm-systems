.. meta::
   :description: Learn about GL2 Cache metrics in ROCm Compute Profiler, including cache performance, bandwidth, and request statistics on RDNA3.5 (gfx115x).
   :keywords: ROCm Compute Profiler, RDNA, gfx115x, GL2, GL2 Cache, GL2C

.. _rdna-gl2:

=========
GL2 cache
=========

GL2 cache (RDNA naming for what Instinct documentation refers to as L2/TCC) is the final GFX on-chip cache for most clients before traffic reaches the memory system.
On gfx115x, traffic then proceeds toward system memory through GCEA. On gfx1201, the GL2 size depends on the board (for example 8 MB on the Radeon RX 9070 and RX 9070 XT), modeled as 32 channels, and traffic then goes through GCEA toward the Infinity Cache and VRAM. Those panels are documented under :doc:`gcea`.

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
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      .. jinja:: rdna115x-gl2-cache-performance-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      .. jinja:: rdna4-gl2-cache-performance-gfx1201
         :file: _templates/metrics_table.j2

GL2 cache request statistics
----------------------------

.. tab-set::
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      .. jinja:: rdna115x-gl2-cache-request-statistics-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      .. jinja:: rdna4-gl2-request-statistics-gfx1201
         :file: _templates/metrics_table.j2

GL2 cache bandwidth
-------------------

.. tab-set::
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      .. jinja:: rdna115x-gl2-cache-bandwidth-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      gfx1201 has no GL2 cache bandwidth table.

Memory chart: GL2 cache
=======================

The following Memory Chart table aligns with the on-screen flow through GL2.

Memory chart - GL2 cache
------------------------

.. tab-set::
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      .. jinja:: rdna115x-memory-chart-gl2-cache-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      .. jinja:: rdna4-memory-chart-gl2-cache-gfx1201
         :file: _templates/metrics_table.j2
