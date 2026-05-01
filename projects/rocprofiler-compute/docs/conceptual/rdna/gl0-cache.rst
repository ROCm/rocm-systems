.. meta::
   :description: ROCm Compute Profiler RDNA3.5 GL0 (TCP) metrics
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, GL0, TCP, L0

.. _rdna-gl0:

===
GL0
===

On gfx1151, GL0 is the vector L1 data cache (also referred to as TCP) in front of
the GL1 cache. For GL1 panels and the GL1 Cache Memory Chart table, see
:doc:`gl1-cache`. The handoff toward GL2 cache is under :doc:`gl2-cache`.

.. note::

   GL0 is the same as TCP on the RDNA3.5 architecture. Hardware counter names
   (for example, ``TCP_REQ_sum``) retain the TCP prefix.

GL0 cache panels
================

GL0 utilization
---------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl0-utilization-gfx1151
         :file: _templates/metrics_table.j2

GL0 request statistics
----------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl0-request-statistics-gfx1151
         :file: _templates/metrics_table.j2

GL0 cache performance
---------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl0-cache-performance-gfx1151
         :file: _templates/metrics_table.j2

GL0-GL1 interface
-----------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl0-gl1-interface-gfx1151
         :file: _templates/metrics_table.j2

GL0 stalls
----------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl0-stalls-gfx1151
         :file: _templates/metrics_table.j2

Memory chart: path up to GL1
============================

The following Memory Chart tables align with the on-screen flow through instruction
and scalar paths, GL0 (TCP), LDS, and the TCP-GL1 interface.

Memory chart - instruction cache
---------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-memory-chart-instruction-cache-gfx1151
         :file: _templates/metrics_table.j2

Memory chart - scalar data cache
---------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-memory-chart-scalar-data-cache-gfx1151
         :file: _templates/metrics_table.j2

Memory chart - TCP cache (GL0 vector cache)
--------------------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-memory-chart-tcp-cache-gfx1151
         :file: _templates/metrics_table.j2

Memory chart - LDS (local data share)
--------------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-memory-chart-lds-local-data-share-gfx1151
         :file: _templates/metrics_table.j2

Memory chart - TCP-GL1 interface
---------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-memory-chart-tcp-gl1-interface-gfx1151
         :file: _templates/metrics_table.j2
