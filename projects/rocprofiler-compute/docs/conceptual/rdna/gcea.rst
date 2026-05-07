.. meta::
   :description: ROCm Compute Profiler - RDNA3.5 GCEA and DRAM interface metrics (gfx1151)
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, GCEA, DRAM

.. _rdna-gcea:

====
GCEA
====

**GCEA** names the stage where traffic leaves the on-chip cache hierarchy toward **DRAM**, covering memory-channel interfaces, the system arbiter (SARB), and return paths.
Use these panels when analyzing bandwidth limits or memory-controller behavior after GL2.

The sections below list RDNA3.5 (gfx1151) GCEA / DRAM interface metric descriptions.

For on-chip GL2 cache tables, see :doc:`gl2-cache`.

.. note::

   Panel YAMLs label DRAM read/write, SARB, and return-interface metrics together under this memory hierarchy stage.

DRAM read interface
-------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-dram-read-interface-gfx1151
         :file: _templates/metrics_table.j2

DRAM write interface
--------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-dram-write-interface-gfx1151
         :file: _templates/metrics_table.j2

System Arbiter (SARB)
---------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-system-arbiter-sarb-gfx1151
         :file: _templates/metrics_table.j2

Return interface
----------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-return-interface-gfx1151
         :file: _templates/metrics_table.j2

Memory chart: GCEA to system memory
===================================

The following Memory Chart table aligns with the on-screen flow from GCEA out
to system memory.

Memory chart - GCEA to system memory
------------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-memory-chart-gcea-to-system-memory-gfx1151
         :file: _templates/metrics_table.j2
