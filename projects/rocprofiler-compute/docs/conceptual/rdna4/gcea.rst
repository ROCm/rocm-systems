.. meta::
   :description: ROCm Compute Profiler - RDNA4 Efficiency Arbiter (EA) and DRAM interface metrics (gfx1201)
   :keywords: ROCm Compute Profiler, RDNA, RDNA4, gfx1201, EA, GCEA, DRAM, Infinity Cache

.. _rdna4-gcea:

=========================================
Efficiency Arbiter (EA)
=========================================

The Efficiency Arbiter (EA), also called GCEA, is the stage where traffic leaves the
on-chip cache hierarchy after GL2. EA sends requests toward MALL and DRAM, and IO
requests, through the SDP interface and receives the returns.

On Navi 48 (gfx1201), MALL is the 64 MB Infinity Cache between EA and the 256-bit
GDDR6 memory at 20 Gbps (640 GB/s). DRAM-destined requests can hit in MALL before
they reach DRAM. No counter separates MALL hits from DRAM traffic, so DRAM bandwidth
in these tables can exceed the 640 GB/s DRAM peak. The measured MALL bandwidth is
the roofline MALL roof; see :ref:`rdna4-roofline`.

Use these panels when analyzing bandwidth limits or memory-controller behavior after GL2.

For on-chip GL2 cache tables, see :doc:`gl2-cache`.

Memory chart: EA to DRAM
========================

The following memory chart table aligns with the on-screen flow from EA out to DRAM.

Memory chart - EA to DRAM
-------------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-memory-chart-ea-to-dram-gfx1201
         :file: _templates/metrics_table.j2

Efficiency Arbiter panel
========================

EA utilization
--------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-ea-utilization-gfx1201
         :file: _templates/metrics_table.j2

EA traffic bandwidth
--------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-ea-traffic-bandwidth-gfx1201
         :file: _templates/metrics_table.j2

EA-SDP interface - outstanding requests
---------------------------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-ea-sdp-outstanding-requests-gfx1201
         :file: _templates/metrics_table.j2

EA-SDP interface - stalls
-------------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-ea-sdp-stalls-gfx1201
         :file: _templates/metrics_table.j2
