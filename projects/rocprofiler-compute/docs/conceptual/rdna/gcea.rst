.. meta::
   :description: ROCm Compute Profiler - RDNA3.5 GCEA and DRAM interface metrics (gfx115x)
   :keywords: ROCm Compute Profiler, RDNA, gfx115x, GCEA, DRAM

.. _rdna-gcea:

====
GCEA
====

**GCEA** names the stage where traffic leaves the on-chip cache hierarchy toward **DRAM**, covering memory-channel interfaces, the system arbiter (SARB), and return paths.
Use these panels when analyzing bandwidth limits or memory-controller behavior after GL2.

The sections below list RDNA3.5 (gfx115x) GCEA / DRAM interface metric descriptions.

For on-chip GL2 cache tables, see :doc:`gl2-cache`.

.. note::

   Panel YAMLs label DRAM read/write, SARB, and return-interface metrics together under this memory hierarchy stage.

DRAM read interface
-------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-dram-read-interface-gfx115x
         :file: _templates/metrics_table.j2

DRAM write interface
--------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-dram-write-interface-gfx115x
         :file: _templates/metrics_table.j2

System Arbiter (SARB)
---------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-system-arbiter-sarb-gfx115x
         :file: _templates/metrics_table.j2

Return interface
----------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-return-interface-gfx115x
         :file: _templates/metrics_table.j2

Memory chart: GCEA to system memory
===================================

The following Memory Chart table aligns with the on-screen flow from GCEA out
to system memory.

Memory chart - GCEA to system memory
------------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-memory-chart-gcea-to-system-memory-gfx115x
         :file: _templates/metrics_table.j2
