.. meta::
   :description: Learn about the GL1 write-combining cache and the GL1-GL2 arbiter (GLARB) in ROCm Compute Profiler on RDNA4 (gfx1201).
   :keywords: ROCm Compute Profiler, RDNA, RDNA4, gfx1201, GL1, GL1C, GLARB

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

Requests from GL1 to GL2 pass through the Graphics Level Arbiter (GLARB).

For GL0 and the GL0-GL1 interface, see :doc:`gl0-cache`. For GL2, see
:doc:`gl2-cache`; for the Efficiency Arbiter (EA) beyond GL2, see :doc:`gcea`.

.. note::

   The GL1 write-combining cache is also referred to as GL1C in some contexts. Hardware counter
   names (for example, ``GL1C_REQ_READ_128B_sum``) retain the GL1C prefix.

Memory chart: GLARB
===================

The Memory Chart panel reports GL1 utilization, stalls, and GL1-GL2 bandwidth in
its GLARB table.

Memory chart - GLARB (GL1-GL2 arbiter)
--------------------------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-memory-chart-glarb-gfx1201
         :file: _templates/metrics_table.j2

GL1 Cache panel
================

The GL1 Cache panel (panel 1100) has tables for GL1A processing, GL1C utilization,
requests, and request bandwidth. Its GL1C-GLARB Interface table has three metrics:
GL2 read requests, GL2 read 128-byte requests, and GL2 write requests. None of the
panel 1100 metrics have descriptions yet, so this section has no metric tables.
Run ``rocprof-compute analyze`` with ``--block 11`` to show the panel.

The GLARB panel (panel 1200) is in the configuration. Its tables, including
GLARBC-GL2 Interface, have no metrics yet.
