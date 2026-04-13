.. meta::
   :description: Learn about GL2C metrics in ROCm Compute Profiler, including the GL1C–GL2 interface, cache performance, bandwidth, GCEA, and DRAM interfaces on RDNA 3.5 (gfx1151).
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, GL2C, GL2, GCEA

.. _rdna-gl2:

===============================
GL2 cache (GL2C) and GCEA
===============================

On gfx1151, the last-level on-chip cache seen by most clients is GL2C (RDNA
naming for what Instinct documentation calls L2 / TCC). Traffic that leaves GL2
heads toward GCEA and DRAM through the DRAM read/write, SARB, and
return interfaces in the panel YAMLs.

For Instinct L2 (TCC) coherence, channel hashing, and fabric metrics, see
:doc:`../cdna/l2-cache`.

GL1C to GL2C and GL2C panels
=============================

GL1C GL1C-GL2 Interface (panel)
---------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl1c-gl1c-gl2-interface-gfx1151
         :file: _templates/metrics_table.j2

GL2C Cache Performance
----------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl2c-cache-performance-gfx1151
         :file: _templates/metrics_table.j2

GL2C Request Statistics
-----------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl2c-request-statistics-gfx1151
         :file: _templates/metrics_table.j2

GL2C Bandwidth
--------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl2c-bandwidth-gfx1151
         :file: _templates/metrics_table.j2

GCEA and DRAM interfaces
=========================

DRAM Read Interface
-------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-dram-read-interface-gfx1151
         :file: _templates/metrics_table.j2

DRAM Write Interface
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

Return Interface
----------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-return-interface-gfx1151
         :file: _templates/metrics_table.j2
