.. meta::
   :description: ROCm Compute Profiler RDNA3.5 TCP (GL0) metrics
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, TCP, GL0, L0

.. _rdna-tcp:

===
GL0
===

On gfx1151, TCP is the vector L1 data cache (RDNA GL0) in front of GL1C. For GL1C
panels and the GL1C Memory Chart table, see :doc:`gl1-cache`. The handoff toward
GL2C is under :doc:`gl2-cache`.

**GL0 is the same as TCP on current architecture. We keep using the name "TCP" on this
page for consistency with the counters' names.**

TCP cache (panel)
=================

TCP Utilization
---------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-tcp-utilization-gfx1151
         :file: _templates/metrics_table.j2

TCP Request Statistics
----------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-tcp-request-statistics-gfx1151
         :file: _templates/metrics_table.j2

TCP Cache Performance
---------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-tcp-cache-performance-gfx1151
         :file: _templates/metrics_table.j2

TCP-GL1 interface
-----------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-tcp-tcp-gl1-interface-gfx1151
         :file: _templates/metrics_table.j2

TCP Stalls
----------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-tcp-stalls-gfx1151
         :file: _templates/metrics_table.j2

Memory Chart: path up to GL1
============================

The following Memory Chart tables align with the on-screen flow through instruction
and scalar paths, TCP (GL0), LDS, and the TCP-GL1 interface.

Memory chart - Instruction Cache
---------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-memory-chart-instruction-cache-gfx1151
         :file: _templates/metrics_table.j2

Memory chart - Scalar Data Cache
---------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-memory-chart-scalar-data-cache-gfx1151
         :file: _templates/metrics_table.j2

Memory chart - TCP Cache (Vector Data Cache)
---------------------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-memory-chart-tcp-cache-vector-l0-gfx1151
         :file: _templates/metrics_table.j2

Memory chart - LDS (Local Data Share)
--------------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-memory-chart-lds-local-data-share-gfx1151
         :file: _templates/metrics_table.j2

Memory chart - TCP-GL1 Interface
---------------------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-memory-chart-tcp-gl1-interface-gfx1151
         :file: _templates/metrics_table.j2
