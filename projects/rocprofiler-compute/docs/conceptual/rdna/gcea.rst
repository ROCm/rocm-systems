.. meta::
   :description: ROCm Compute Profiler - RDNA3.5 GCEA and DRAM interface metrics (gfx115x)
   :keywords: ROCm Compute Profiler, RDNA, gfx115x, GCEA, DRAM

.. _rdna-gcea:
.. _rdna4-gcea:

========================
Efficiency Arbiter (EA)
========================

The Efficiency Arbiter (EA), also called GCEA, is the stage where traffic leaves the on-chip cache hierarchy after GL2.
On gfx115x, the published tables are the DRAM read and write interfaces, the system arbiter (SARB), and the return path toward system memory.
On gfx1201, EA sends requests toward the 64 MB Infinity Cache (MALL) and DRAM.
Use these panels when analyzing bandwidth limits or memory-controller behavior after GL2.

The sections below list the metric tables for both generations. Select the architecture tab on each table.

For on-chip GL2 cache tables, see :doc:`gl2-cache`.

.. note::

   Panel YAMLs label DRAM read/write, SARB, and return-interface metrics together under this memory hierarchy stage.

DRAM read interface
-------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-dram-read-interface-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      gfx1201 has no Efficiency Arbiter panel. Its EA-to-DRAM traffic is in the Memory Chart table below.

DRAM write interface
--------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-dram-write-interface-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      gfx1201 has no Efficiency Arbiter panel. Its EA-to-DRAM traffic is in the Memory Chart table below.

System arbiter (SARB)
---------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-system-arbiter-sarb-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      gfx1201 has no Efficiency Arbiter panel. Its EA-to-DRAM traffic is in the Memory Chart table below.

Return interface
----------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-return-interface-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      gfx1201 has no Efficiency Arbiter panel. Its EA-to-DRAM traffic is in the Memory Chart table below.

Memory chart: GCEA to system memory
===================================

The following memory chart table aligns with the on-screen flow out of this stage.
gfx115x leaves toward system memory. The RDNA4 tab is the EA-to-DRAM row.

Memory chart - GCEA to system memory
------------------------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-memory-chart-gcea-to-system-memory-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-memory-chart-ea-to-dram-gfx1201
         :file: _templates/metrics_table.j2

``DRAM Read Bandwidth`` and ``DRAM Write Bandwidth`` count requests that L2 sends
toward memory, including requests served or absorbed by the Infinity Cache. They
do not measure the bytes that physically reach GDDR6. A value above the GDDR6
peak does not by itself indicate an invalid counter.

Memory chart - Infinity Cache and GDDR6
---------------------------------------

``Infinity Cache Hit Rate``, ``GDDR6 Read Bandwidth``, and ``GDDR6 Write Bandwidth``
display ``N/A`` because this configuration does not collect the counters needed to
measure them. Do not infer those values from memory-destined traffic.

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      gfx115x does not publish an Infinity Cache or GDDR6 table. Device memory on
      these APUs is system memory, in the table above.

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: navi48-memory-chart-infinity-cache-gfx1201
         :file: _templates/metrics_table.j2
