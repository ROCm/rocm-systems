.. meta::
   :description: ROCm Compute Profiler RDNA3.5 TCP (GL0 / vector L0) metrics
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, TCP, GL0, L0

.. _rdna-tcp:

=========
TCP (GL0)
=========

On **gfx1151**, **TCP** is the vector ``L0`` cache (RDNA **GL0**) in front of **GL1C**. Panel metrics
below come from ``0800_TCP_Cache.yaml``. For **GL1C** (L1) panels and the GL1C Memory
Chart table, see :doc:`gl1-cache`. The handoff toward **GL2C** is under
:doc:`gl2-cache`.

TCP cache (panel)
=================

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

TCP–GL1 interface
.................

.. jinja:: rdna1151-tcp-tcp-gl1-interface-gfx1151
   :file: _templates/metrics_table.j2

TCP Stalls
..........

.. jinja:: rdna1151-tcp-stalls-gfx1151
   :file: _templates/metrics_table.j2

Memory Chart: path up to GL1
============================

The following Memory Chart tables align with the on-screen flow through instruction
and scalar paths, **TCP** (vector ``L0``), **LDS**, and the **TCP–GL1** interface.

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

Memory chart: TCP–GL1 Interface
...............................

.. jinja:: rdna1151-memory-chart-tcp-gl1-interface-gfx1151
   :file: _templates/metrics_table.j2

Return to :doc:`rdna-performance-model` or the top-level :doc:`../performance-model`.
Next: :doc:`gl1-cache` (GL1C / L1).
