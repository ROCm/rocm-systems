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

      gfx1201 reports EA traffic in the utilization, bandwidth, and EA-SDP tables below. The DRAM read, DRAM write, SARB, and return-interface tables are the gfx115x tables.

DRAM write interface
--------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-dram-write-interface-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      gfx1201 reports EA traffic in the utilization, bandwidth, and EA-SDP tables below. The DRAM read, DRAM write, SARB, and return-interface tables are the gfx115x tables.

System arbiter (SARB)
---------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-system-arbiter-sarb-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      gfx1201 reports EA traffic in the utilization, bandwidth, and EA-SDP tables below. The DRAM read, DRAM write, SARB, and return-interface tables are the gfx115x tables.

Return interface
----------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-return-interface-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      gfx1201 reports EA traffic in the utilization, bandwidth, and EA-SDP tables below. The DRAM read, DRAM write, SARB, and return-interface tables are the gfx115x tables.

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

Efficiency Arbiter panel
========================

On gfx1201, EA sends requests toward MALL and DRAM after GL2. A counter whose
name mentions DRAM still describes this interface. It does not establish the
number of transfers that reach physical GDDR6. No counter separates MALL hits
from DRAM traffic, so DRAM bandwidth in these tables can exceed the 640 GB/s
DRAM peak. These rates use the configured 32, 64, 128, and 256-byte request
categories. Separate compressed request and data counters are not included.
The measured MALL bandwidth is the roofline MALL roof; see :ref:`rdna4-roofline`.


EA utilization
--------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      gfx115x reports this stage in the DRAM, SARB, and return-interface tables above.

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-ea-utilization-gfx1201
         :file: _templates/metrics_table.j2

EA traffic bandwidth
--------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      gfx115x reports this stage in the DRAM, SARB, and return-interface tables above.

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-ea-traffic-bandwidth-gfx1201
         :file: _templates/metrics_table.j2

EA-SDP interface - outstanding requests
---------------------------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      gfx115x does not publish this EA-SDP table.

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-ea-sdp-outstanding-requests-gfx1201
         :file: _templates/metrics_table.j2

EA-SDP interface - stalls
-------------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      gfx115x does not publish this EA-SDP table.

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-ea-sdp-stalls-gfx1201
         :file: _templates/metrics_table.j2
