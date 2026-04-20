.. meta::
   :description: Learn about GL2C metrics in ROCm Compute Profiler, including cache performance, bandwidth, GCEA, and DRAM interfaces on RDNA 3.5 (gfx1151).
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, GL2C, GL2, GCEA

.. _rdna-gl2:

===
GL2
===

On gfx1151, GL2C (RDNA naming for what Instinct documentation refers to as L2/TCC) is the last-level on-chip cache for most clients.

Traffic leaving GL2
heads toward GCEA and DRAM through the DRAM read/write, SARB, and
return interfaces in the panel YAMLs.

For Instinct L2 (TCC) coherence, channel hashing, and fabric metrics on CDNA architecture across MI-series GPUs, see
:doc:`../cdna/l2-cache` under CDNA-CDNA4.

GL2C panels
===========

GL2C cache performance
----------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl2c-cache-performance-gfx1151
         :file: _templates/metrics_table.j2

GL2C request statistics
-----------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl2c-request-statistics-gfx1151
         :file: _templates/metrics_table.j2

GL2C bandwidth
--------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl2c-bandwidth-gfx1151
         :file: _templates/metrics_table.j2

GCEA and DRAM interfaces
=========================

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
