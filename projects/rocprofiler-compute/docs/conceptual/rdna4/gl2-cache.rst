.. meta::
   :description: Learn about the GL2 cache in ROCm Compute Profiler on RDNA4 (gfx1201).
   :keywords: ROCm Compute Profiler, RDNA, RDNA4, gfx1201, GL2, GL2 Cache, GL2C

.. _rdna4-gl2:

=========
GL2 cache
=========

On gfx1201, GL2 (RDNA naming for what Instinct documentation refers to as L2/TCC) is
the read/write cache with atomic units after the GL1 write-combining cache. Navi 48 has an 8 MB GL2. The RDNA4
ISA shows one L2 read/write cache per memory channel. The tool models GL2 as 32 channels.

After GL2, traffic goes through the Efficiency Arbiter (EA) toward the 64 MB Infinity
Cache and DRAM. Those panels are documented under :doc:`gcea`.

For Instinct L2 (TCC) coherence, channel hashing, and fabric metrics on CDNA
architecture-based AMD Instinct MI-Series GPUs, see :doc:`../cdna/l2-cache` under
CDNA-CDNA4.

.. note::

   The GL2 Cache is also referred to as GL2C in some contexts. Hardware counter
   names (for example, ``GL2C_HIT_sum``) retain the GL2C prefix.

Memory chart: GL2
=================

Memory chart - GL2 cache
------------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-memory-chart-gl2-cache-gfx1201
         :file: _templates/metrics_table.j2

GL2 cache panel
===============

The GL2 Cache (L2) panel (panel 1300) has tables for request statistics, cache
performance, and bandwidth. These tables have metric descriptions in the
analysis report. Run ``rocprof-compute analyze`` with ``--block 13`` to show
the panel.
