.. meta::
   :description: ROCm Compute Profiler RDNA3.5 TCP and GL1C metrics
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, TCP, GL1C

.. _rdna-gl1:

===========================
GL1 hierarchy: TCP and GL1C
===========================

RDNA3 and RDNA3.5 expose TCP as the vector L0 and GL1C as the L1 layer ahead of GL2C.
This page lists gfx1151 metric tables for TCP and GL1C panels and Memory Chart rows
through GL1C. The GL1C to GL2 handoff is documented under :doc:`gl2-cache`.

TCP cache, vector L0
====================

TCP Utilization
...............

.. jinja:: rdna1151-tcp-utilization-gfx1151
   :file: _templates/metrics_table.j2

TCP Request Statistics
......................

.. jinja:: rdna1151-tcp-request-statistics-gfx1151
   :file: _templates/metrics_table.j2

TCP Cache Performance
.....................

.. jinja:: rdna1151-tcp-cache-performance-gfx1151
   :file: _templates/metrics_table.j2

TCP TCP-GL1 Interface
.....................

.. jinja:: rdna1151-tcp-tcp-gl1-interface-gfx1151
   :file: _templates/metrics_table.j2

TCP Stalls
..........

.. jinja:: rdna1151-tcp-stalls-gfx1151
   :file: _templates/metrics_table.j2

GL1C (L1) cache
===============

GL1C Utilization
................

.. jinja:: rdna1151-gl1c-utilization-gfx1151
   :file: _templates/metrics_table.j2

GL1C Request Statistics
.......................

.. jinja:: rdna1151-gl1c-request-statistics-gfx1151
   :file: _templates/metrics_table.j2

GL1C Cache Performance
......................

.. jinja:: rdna1151-gl1c-cache-performance-gfx1151
   :file: _templates/metrics_table.j2

GL1C Stalls
...........

.. jinja:: rdna1151-gl1c-stalls-gfx1151
   :file: _templates/metrics_table.j2

Memory Chart: path into and through GL1
=======================================

The following Memory Chart tables align with the on-screen flow from instruction and
scalar paths through TCP, LDS, and GL1C.

Memory chart: Instruction Cache
...............................

.. jinja:: rdna1151-memory-chart-instruction-cache-gfx1151
   :file: _templates/metrics_table.j2

Memory chart: Scalar Data Cache
...............................

.. jinja:: rdna1151-memory-chart-scalar-data-cache-gfx1151
   :file: _templates/metrics_table.j2

Memory chart: TCP Cache (Vector L0)
...................................

.. jinja:: rdna1151-memory-chart-tcp-cache-vector-l0-gfx1151
   :file: _templates/metrics_table.j2

Memory chart: LDS (Local Data Share)
....................................

.. jinja:: rdna1151-memory-chart-lds-local-data-share-gfx1151
   :file: _templates/metrics_table.j2

Memory chart: TCP-GL1 Interface
...............................

.. jinja:: rdna1151-memory-chart-tcp-gl1-interface-gfx1151
   :file: _templates/metrics_table.j2

Memory chart: GL1C Cache (L1)
.............................

.. jinja:: rdna1151-memory-chart-gl1c-cache-l1-gfx1151
   :file: _templates/metrics_table.j2

Return to :doc:`rdna-performance-model` or the top-level :doc:`../performance-model`.
