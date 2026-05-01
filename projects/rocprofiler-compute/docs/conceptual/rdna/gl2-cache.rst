.. meta::
   :description: Learn about GL2 Cache metrics in ROCm Compute Profiler, including cache performance, bandwidth, GCEA, and DRAM interfaces on RDNA 3.5 (gfx1151).
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, GL2, GL2 Cache, GL2C, GCEA

.. _rdna-gl2:

===
GL2
===

On gfx1151, GL2 Cache (RDNA naming for what Instinct documentation refers to as
L2/TCC) is the last-level on-chip cache for most clients.

Traffic leaving GL2 heads toward GCEA and DRAM through the DRAM read/write,
SARB, and return interfaces in the panel YAMLs.

For Instinct L2 (TCC) coherence, channel hashing, and fabric metrics on CDNA
architecture across MI-series GPUs, see :doc:`../cdna/l2-cache` under
CDNA-CDNA4.

.. note::

   The GL2 Cache is also referred to as GL2C in some contexts. Hardware counter
   names (for example, ``GL2C_HIT_sum``) retain the GL2C prefix.

GL2 Cache panels
================

GL2 Cache performance
---------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl2-cache-performance-gfx1151
         :file: _templates/metrics_table.j2

GL2 Cache request statistics
----------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl2-cache-request-statistics-gfx1151
         :file: _templates/metrics_table.j2

GL2 Cache bandwidth
-------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl2-cache-bandwidth-gfx1151
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
