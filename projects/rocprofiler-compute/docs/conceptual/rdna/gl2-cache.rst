.. meta::
   :description: ROCm Compute Profiler — RDNA3.5 GL2C and memory path metrics
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, GL2C, GL2, GCEA

.. _rdna-gl2:

=======================
GL2 cache (GL2C) and EA
=======================

On **gfx1151**, the last-level on-chip cache seen by most clients is **GL2C** (RDNA
naming for what Instinct documentation calls **L2 / TCC**). Traffic that leaves GL2
heads toward **GCEA** and DRAM through the **DRAM read/write**, **SARB**, and
**return** interfaces in the panel YAMLs.

For Instinct **L2 (TCC)** coherence, channel hashing, and fabric metrics, see
:doc:`../cdna/l2-cache`.

GL1C → GL2C and GL2C panels
===========================

GL1C GL1C-GL2 Interface (panel)
---------------------------------

.. jinja:: rdna1151-gl1c-gl1c-gl2-interface-gfx1151
   :file: _templates/metrics_table.j2

GL2C Cache Performance
----------------------

.. jinja:: rdna1151-gl2c-cache-performance-gfx1151
   :file: _templates/metrics_table.j2

GL2C Request Statistics
-----------------------

.. jinja:: rdna1151-gl2c-request-statistics-gfx1151
   :file: _templates/metrics_table.j2

GL2C Bandwidth
--------------

.. jinja:: rdna1151-gl2c-bandwidth-gfx1151
   :file: _templates/metrics_table.j2

GCEA and DRAM interfaces
=========================

DRAM Read Interface
-------------------

.. jinja:: rdna1151-dram-read-interface-gfx1151
   :file: _templates/metrics_table.j2

DRAM Write Interface
--------------------

.. jinja:: rdna1151-dram-write-interface-gfx1151
   :file: _templates/metrics_table.j2

System Arbiter (SARB)
---------------------

.. jinja:: rdna1151-system-arbiter-sarb-gfx1151
   :file: _templates/metrics_table.j2

Return Interface
----------------

.. jinja:: rdna1151-return-interface-gfx1151
   :file: _templates/metrics_table.j2

Memory Chart — GL2 and system memory
======================================

Memory chart — GL1C-GL2 Interface
---------------------------------

.. jinja:: rdna1151-memory-chart-gl1c-gl2-interface-gfx1151
   :file: _templates/metrics_table.j2

Memory chart — GL2C Cache (L2)
------------------------------

.. jinja:: rdna1151-memory-chart-gl2c-cache-l2-gfx1151
   :file: _templates/metrics_table.j2

Memory chart — GCEA to System Memory
------------------------------------

.. jinja:: rdna1151-memory-chart-gcea-to-system-memory-gfx1151
   :file: _templates/metrics_table.j2

Return to :doc:`rdna-performance-model` or the top-level :doc:`../performance-model`.
