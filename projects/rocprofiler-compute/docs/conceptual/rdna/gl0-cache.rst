.. meta::
   :description: ROCm Compute Profiler RDNA3.5 GL0 (TCP vector cache) metrics
   :keywords: ROCm Compute Profiler, RDNA, gfx115x, GL0, TCP, L0

.. _rdna-gl0:

=========================
GL0 (TCP Vector Cache)
=========================

**GL0** is the vector-side TCP cache immediately in front of GL1 inside each shader engine datapath (hardware counters keep the ``TCP_*`` prefix on gfx115x).

For GL1 panels and the GL1 Cache Memory Chart table, see :doc:`gl1-cache`.
The handoff toward GL2 cache is under :doc:`gl2-cache`.

.. note::

   On RDNA3.5, GL0 and TCP refer to the same cache. Hardware counter names
   (for example, ``TCP_REQ_sum``) retain the TCP prefix.

GL0 cache panels
================

GL0 utilization
---------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-gl0-utilization-gfx115x
         :file: _templates/metrics_table.j2

GL0 request statistics
----------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-gl0-request-statistics-gfx115x
         :file: _templates/metrics_table.j2

GL0 cache performance
---------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-gl0-cache-performance-gfx115x
         :file: _templates/metrics_table.j2

GL0-GL1 interface
-----------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-gl0-gl1-interface-gfx115x
         :file: _templates/metrics_table.j2

GL0 stalls
----------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-gl0-stalls-gfx115x
         :file: _templates/metrics_table.j2

Memory chart: path up to GL1
============================

The following Memory Chart tables align with the on-screen flow through instruction
and scalar paths, GL0 (TCP), LDS, and the TCP-GL1 interface.

Memory chart - instruction cache
---------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-memory-chart-instruction-cache-gfx115x
         :file: _templates/metrics_table.j2

Memory chart - scalar data cache
---------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-memory-chart-scalar-data-cache-gfx115x
         :file: _templates/metrics_table.j2

Memory chart - TCP cache (GL0 vector cache)
--------------------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-memory-chart-tcp-cache-gfx115x
         :file: _templates/metrics_table.j2

Memory chart - LDS (local data share)
--------------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-memory-chart-lds-local-data-share-gfx115x
         :file: _templates/metrics_table.j2

Memory chart - TCP-GL1 interface
---------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx115x)
      :selected:

      .. jinja:: rdna115x-memory-chart-tcp-gl1-interface-gfx115x
         :file: _templates/metrics_table.j2
