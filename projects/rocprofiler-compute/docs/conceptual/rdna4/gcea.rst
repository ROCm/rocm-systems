.. meta::
   :description: ROCm Compute Profiler - RDNA4 Graphics Core Efficiency Arbiter (GCEA) and DRAM interface metrics (gfx1201)
   :keywords: ROCm Compute Profiler, RDNA, RDNA4, gfx1201, GCEA, DRAM, Infinity Cache

.. _rdna4-gcea:

=======================================
Graphics Core Efficiency Arbiter (GCEA)
=======================================

The Graphics Core Efficiency Arbiter (GCEA) is the stage where traffic leaves the
on-chip cache hierarchy after GL2. GCEA sends requests toward MALL and DRAM, and IO
requests, through the SDP interface and receives the returns.

On Navi 48 (gfx1201), MALL is the 64 MB Infinity Cache between GCEA and the 256-bit
GDDR6 memory at 20 Gbps (640 GB/s). DRAM-destined requests can hit in MALL before
they reach DRAM. No counter separates MALL hits from DRAM traffic, so DRAM bandwidth
in these tables can exceed the 640 GB/s DRAM peak. The measured MALL bandwidth is
the roofline MALL roof; see :ref:`rdna4-roofline`.

Use these panels when analyzing bandwidth limits or memory-controller behavior after GL2.

For on-chip GL2 cache tables, see :doc:`gl2-cache`.

Memory chart: GCEA to DRAM
==========================

The following memory chart table aligns with the on-screen flow from GCEA out to DRAM.

Memory chart - GCEA to DRAM
---------------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-memory-chart-gcea-to-dram-gfx1201
         :file: _templates/metrics_table.j2

Graphics Core Efficiency Arbiter panel
======================================

GCEA utilization
----------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-gcea-utilization-gfx1201
         :file: _templates/metrics_table.j2

GCEA traffic bandwidth
----------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-gcea-traffic-bandwidth-gfx1201
         :file: _templates/metrics_table.j2

GCEA-SDP interface - outstanding requests
-----------------------------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-gcea-sdp-outstanding-requests-gfx1201
         :file: _templates/metrics_table.j2

GCEA-SDP interface - stalls
---------------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-gcea-sdp-stalls-gfx1201
         :file: _templates/metrics_table.j2
