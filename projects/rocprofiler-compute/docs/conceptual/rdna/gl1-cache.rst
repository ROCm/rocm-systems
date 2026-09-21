.. meta::
   :description: Learn about the GL1 Cache metrics in ROCm Compute Profiler, including utilization, request statistics, cache performance, the GL1-GL2 interface, and stalls on RDNA 3.5 (gfx115x).
   :keywords: ROCm Compute Profiler, RDNA, gfx115x, GL1, GL1 Cache, GL1C

.. _rdna-gl1:

===
GL1
===

**GL1 Cache** is the shared L1 vector cache inside each shader array on gfx115x (one GL1 per shader array), supplied by GL0 (TCP) and forwarding misses toward GL2.
For GL0 panels and Memory Chart rows through the TCP-GL1 boundary, see
:doc:`gl0-cache`. For downstream GL2 panels after GL1, see :doc:`gl2-cache`; for DRAM / GCEA interfaces beyond GL2, see :doc:`gcea`.

.. note::

   The GL1 Cache is also referred to as GL1C in some contexts. Hardware counter
   names (for example, ``GL1C_REQ_sum``) retain the GL1C prefix.

GL1 Cache panels
================

GL1 Cache utilization
---------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-gl1-cache-utilization-gfx115x
         :file: _templates/metrics_table.j2

GL1 Cache request statistics
----------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-gl1-cache-request-statistics-gfx115x
         :file: _templates/metrics_table.j2

GL1 Cache performance
---------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-gl1-cache-performance-gfx115x
         :file: _templates/metrics_table.j2

GL1-GL2 interface
-----------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-gl1-gl2-interface-gfx115x
         :file: _templates/metrics_table.j2

GL1 Cache stalls
----------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-gl1-cache-stalls-gfx115x
         :file: _templates/metrics_table.j2

Memory chart: GL1 cache and GL1-GL2 interface
=============================================

The following Memory Chart tables align with the on-screen flow through GL1
and the GL1-GL2 interface.

Memory chart - GL1 cache
------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-memory-chart-gl1-cache-gfx115x
         :file: _templates/metrics_table.j2

Memory chart - GL1-GL2 interface
--------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-memory-chart-gl1-gl2-interface-gfx115x
         :file: _templates/metrics_table.j2
