.. meta::
   :description: Learn about the GL1C metrics in ROCm Compute Profiler, including utilization, request statistics, cache performance, the GL1C-GL2 interface, and stalls on RDNA 3.5 (gfx1151).
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, GL1C, GL1

.. _rdna-gl1:

===
GL1
===

GL1C is the shared L1 cache layer on gfx1151, supplied by TCP (GL0).
For TCP panels and Memory Chart rows through the TCP-GL1 boundary, see
:doc:`tcp-cache`. For downstream GL2C, GCEA, and DRAM panels after GL1C,
see :doc:`gl2-cache`.

GL1C panels
===========

GL1C utilization
----------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl1c-utilization-gfx1151
         :file: _templates/metrics_table.j2

GL1C request statistics
-----------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl1c-request-statistics-gfx1151
         :file: _templates/metrics_table.j2

GL1C cache performance
----------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl1c-cache-performance-gfx1151
         :file: _templates/metrics_table.j2

GL1C-GL2 interface
------------------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl1c-gl1c-gl2-interface-gfx1151
         :file: _templates/metrics_table.j2

GL1C stalls
-----------

.. tab-set::

   .. tab-item:: RDNA 3.5 (gfx1151)
      :selected:

      .. jinja:: rdna1151-gl1c-stalls-gfx1151
         :file: _templates/metrics_table.j2
