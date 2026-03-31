.. meta::
   :description: ROCm Compute Profiler — RDNA3.5 command processor / CPC metrics
   :keywords: ROCm Compute Profiler, RDNA, gfx1151, command processor, CPC

.. _rdna-command-processor:

=========================
Command processor (CP)
=========================

The **command processor (CP)** connects the host and kernel driver to on-GPU
scheduling: it pulls work from **HSA queues**, decodes packets, and hands kernel
launches to the front-end (**SPI** / **WGP** path). On **Instinct** GPUs the profiler
often splits metrics into **CPF** (fetcher) and **CPC** (packet processor); the
shipped **gfx1151** analysis panels emphasize **CPC** and **MEC** activity (utilization,
interface, stalls, memory requests, instruction cache).

For the full CDNA narrative and **CPF / CPC** metric tabs across MI-series arches, see
:doc:`../cdna/command-processor`.

Command Processor Compute (CPC) — gfx1151
===========================================

CPC Utilization
---------------

.. jinja:: rdna1151-cpc-utilization-gfx1151
   :file: _templates/metrics_table.j2

CPC Interface Utilization
-------------------------

.. jinja:: rdna1151-cpc-interface-utilization-gfx1151
   :file: _templates/metrics_table.j2

MEC Stall Cycles
----------------

.. jinja:: rdna1151-mec-stall-cycles-gfx1151
   :file: _templates/metrics_table.j2

CPC Memory Requests
-------------------

.. jinja:: rdna1151-cpc-memory-requests-gfx1151
   :file: _templates/metrics_table.j2

MEC Instruction Cache
-----------------------

.. jinja:: rdna1151-mec-instruction-cache-gfx1151
   :file: _templates/metrics_table.j2

Return to :doc:`rdna-performance-model` or the top-level :doc:`../performance-model`.
