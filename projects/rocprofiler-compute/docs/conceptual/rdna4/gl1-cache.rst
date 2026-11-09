.. meta::
   :description: Learn about the GL1 write-combining cache and the GL1-GL2 interface in ROCm Compute Profiler on RDNA4 (gfx1201).
   :keywords: ROCm Compute Profiler, RDNA, RDNA4, gfx1201, GL1, GL1C

.. _rdna4-gl1:

===
GL1
===

GL1 is the write-combining cache between the WGPs and GL2. The RDNA4 ISA
describes it as "the first-level (GL1) for write-combining cache (collect scatter
and store operations and combine them to provide good access patterns to memory)".
Figure 2 of the ISA labels this cache GL1 Buffer. The tool's panel and metric
names call it GL1 Cache. On gfx1201, the tool counts one GL1 instance per shader array.

GL1 has no hit or miss counter on gfx1201, so the System Speed-of-Light
``GL1 Cache Hit Rate`` row shows N/A.

Requests from GL1 to GL2 are counted on the GL1-GL2 interface, the same stage as
RDNA3.5. Counter names keep the ``GL1C_*`` prefix.

For GL0 and the GL0-GL1 interface, see :doc:`gl0-cache`. For GL2, see
:doc:`gl2-cache`; for the Efficiency Arbiter (EA) beyond GL2, see :doc:`gcea`.

.. note::

   The GL1 write-combining cache is also referred to as GL1C in some contexts. Hardware counter
   names (for example, ``GL1C_REQ_READ_128B_sum``) retain the GL1C prefix.

Memory chart: GL1 and the GL1-GL2 interface
===========================================

The Memory Chart panel reports GL1 utilization and stalls in the GL1 cache table,
and GL1-GL2 bandwidth in the interface table.

Memory chart - GL1 cache
------------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-memory-chart-gl1-cache-gfx1201
         :file: _templates/metrics_table.j2

Memory chart - GL1-GL2 interface
--------------------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-memory-chart-gl1-gl2-interface-gfx1201
         :file: _templates/metrics_table.j2

GL1 Cache panel
===============

The GL1 Cache (L1) panel (panel 1100) has tables for utilization, request
statistics, request bandwidth, and the GL1-GL2 interface. The interface table
counts GL2 read requests, GL2 read 128-byte requests, and GL2 write requests.
None of the panel 1100 metrics have descriptions yet, so this section has no
metric tables. Run ``rocprof-compute analyze`` with ``--block 11`` to show the
panel.
