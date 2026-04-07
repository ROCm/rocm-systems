.. meta::
   :description: ROCm Compute Profiler — RDNA3.5 (gfx1151) metric reference
   :keywords: ROCm Compute Profiler, gfx1151, RDNA3, metrics, ROCm

.. _gfx1151-metrics-ref:

**********************************
RDNA3.5 (gfx1151) metric reference
**********************************

This page lists **metric descriptions** for the RDNA3.5 analysis configuration
(``gfx1151``) shipped with ROCm Compute Profiler — for example AMD Ryzen AI Max+ 395
(Strix Halo) class integrated graphics. Descriptions are generated from the same
``metrics_description`` fields used by the analysis panels.

For hardware context, narrative, and the same metric tables organized by block (WGP,
GL1, GL2, shader engine, command processor), see :doc:`/conceptual/rdna/rdna-performance-model`
and its sub-pages. This reference page keeps a **single flat outline** of every table.

.. note::

   Panels without entries here either omit ``metrics_description`` content or only
   expose metrics without documentation text yet. The analysis YAMLs under
   ``src/rocprof_compute_soc/analysis_configs/gfx1151/`` remain the source of
   truth for formulas and counter bindings.

System Speed-of-Light
=====================

System Speed-of-Light
---------------------

.. jinja:: sys-sol-gfx1151
   :file: _templates/metrics_table.j2


Roofline
========

Roofline Performance Rates
--------------------------

.. jinja:: rdna1151-roofline-performance-rates-gfx1151
   :file: _templates/metrics_table.j2

Roofline Plot Points
--------------------

.. jinja:: rdna1151-roofline-plot-points-gfx1151
   :file: _templates/metrics_table.j2


Graphics Register Bus Manager (GRBM)
====================================

GPU Utilization
---------------

.. jinja:: rdna1151-gpu-utilization-gfx1151
   :file: _templates/metrics_table.j2

Shader Engine Utilization
-------------------------

.. jinja:: rdna1151-shader-engine-utilization-gfx1151
   :file: _templates/metrics_table.j2


Command Processor Compute (CPC)
===============================

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
---------------------

.. jinja:: rdna1151-mec-instruction-cache-gfx1151
   :file: _templates/metrics_table.j2


Shader Processor Input (SPI)
============================

SPI Utilization
---------------

.. jinja:: rdna1151-spi-utilization-gfx1151
   :file: _templates/metrics_table.j2

Wave Dispatch Statistics
------------------------

.. jinja:: rdna1151-wave-dispatch-statistics-gfx1151
   :file: _templates/metrics_table.j2


Workgroup Processor (WGP)
=========================

WGP Utilization
---------------

.. jinja:: rdna1151-wgp-utilization-gfx1151
   :file: _templates/metrics_table.j2

Wavefront Launch Stats
----------------------

.. jinja:: rdna1151-wavefront-launch-stats-gfx1151
   :file: _templates/metrics_table.j2

Wave Dispatch
-------------

.. jinja:: rdna1151-wave-dispatch-gfx1151
   :file: _templates/metrics_table.j2

Wave Life
---------

.. jinja:: rdna1151-wave-life-gfx1151
   :file: _templates/metrics_table.j2

Wave Instruction Mix
--------------------

.. jinja:: rdna1151-wave-instruction-mix-gfx1151
   :file: _templates/metrics_table.j2

VMEM Instruction Mix
--------------------

.. jinja:: rdna1151-vmem-instruction-mix-gfx1151
   :file: _templates/metrics_table.j2

LDS Instruction Mix
-------------------

.. jinja:: rdna1151-lds-instruction-mix-gfx1151
   :file: _templates/metrics_table.j2

Wait State Analysis
-------------------

.. jinja:: rdna1151-wait-state-analysis-gfx1151
   :file: _templates/metrics_table.j2

WGP Instruction Cache
---------------------

.. jinja:: rdna1151-wgp-instruction-cache-gfx1151
   :file: _templates/metrics_table.j2

WGP Scalar Data Cache
---------------------

.. jinja:: rdna1151-wgp-scalar-data-cache-gfx1151
   :file: _templates/metrics_table.j2


TCP cache (vector L0)
=====================

TCP Utilization
---------------

.. jinja:: rdna1151-tcp-utilization-gfx1151
   :file: _templates/metrics_table.j2

TCP Request Statistics
----------------------

.. jinja:: rdna1151-tcp-request-statistics-gfx1151
   :file: _templates/metrics_table.j2

TCP Cache Performance
---------------------

.. jinja:: rdna1151-tcp-cache-performance-gfx1151
   :file: _templates/metrics_table.j2

TCP TCP-GL1 Interface
---------------------

.. jinja:: rdna1151-tcp-tcp-gl1-interface-gfx1151
   :file: _templates/metrics_table.j2

TCP Stalls
----------

.. jinja:: rdna1151-tcp-stalls-gfx1151
   :file: _templates/metrics_table.j2


GL1 cache (L1)
==============

GL1C Utilization
----------------

.. jinja:: rdna1151-gl1c-utilization-gfx1151
   :file: _templates/metrics_table.j2

GL1C Request Statistics
-----------------------

.. jinja:: rdna1151-gl1c-request-statistics-gfx1151
   :file: _templates/metrics_table.j2

GL1C Cache Performance
----------------------

.. jinja:: rdna1151-gl1c-cache-performance-gfx1151
   :file: _templates/metrics_table.j2

GL1C GL1C-GL2 Interface
-----------------------

.. jinja:: rdna1151-gl1c-gl1c-gl2-interface-gfx1151
   :file: _templates/metrics_table.j2

GL1C Stalls
-----------

.. jinja:: rdna1151-gl1c-stalls-gfx1151
   :file: _templates/metrics_table.j2


GL2 cache (L2)
==============

GL2C Cache Performance
----------------------

.. jinja:: rdna1151-gl2c-cache-performance-gfx1151
   :file: _templates/metrics_table.j2

GL2C Request Statistics
-----------------------

.. jinja:: rdna1151-gl2c-request-statistics-gfx1151
   :file: _templates/metrics_table.j2

GL2C Bandwidth
--------------

.. jinja:: rdna1151-gl2c-bandwidth-gfx1151
   :file: _templates/metrics_table.j2


Graphics Core Efficiency Arbiter (GCEA) and memory
==================================================

DRAM Read Interface
-------------------

.. jinja:: rdna1151-dram-read-interface-gfx1151
   :file: _templates/metrics_table.j2

DRAM Write Interface
--------------------

.. jinja:: rdna1151-dram-write-interface-gfx1151
   :file: _templates/metrics_table.j2

System Arbiter (SARB)
---------------------

.. jinja:: rdna1151-system-arbiter-sarb-gfx1151
   :file: _templates/metrics_table.j2

Return Interface
----------------

.. jinja:: rdna1151-return-interface-gfx1151
   :file: _templates/metrics_table.j2


RDNA3.5 Memory Chart (panel overview)
=====================================

Memory chart — Instruction Cache
--------------------------------

.. jinja:: rdna1151-memory-chart-instruction-cache-gfx1151
   :file: _templates/metrics_table.j2

Memory chart — Scalar Data Cache
--------------------------------

.. jinja:: rdna1151-memory-chart-scalar-data-cache-gfx1151
   :file: _templates/metrics_table.j2

Memory chart — TCP Cache (Vector L0)
------------------------------------

.. jinja:: rdna1151-memory-chart-tcp-cache-vector-l0-gfx1151
   :file: _templates/metrics_table.j2

Memory chart — LDS (Local Data Share)
-------------------------------------

.. jinja:: rdna1151-memory-chart-lds-local-data-share-gfx1151
   :file: _templates/metrics_table.j2

Memory chart — TCP-GL1 Interface
--------------------------------

.. jinja:: rdna1151-memory-chart-tcp-gl1-interface-gfx1151
   :file: _templates/metrics_table.j2

Memory chart — GL1C Cache (L1)
------------------------------

.. jinja:: rdna1151-memory-chart-gl1c-cache-l1-gfx1151
   :file: _templates/metrics_table.j2

Memory chart — GL1C-GL2 Interface
---------------------------------

.. jinja:: rdna1151-memory-chart-gl1c-gl2-interface-gfx1151
   :file: _templates/metrics_table.j2

Memory chart — GL2C Cache (L2)
------------------------------

.. jinja:: rdna1151-memory-chart-gl2c-cache-l2-gfx1151
   :file: _templates/metrics_table.j2

Memory chart — GCEA to System Memory
------------------------------------

.. jinja:: rdna1151-memory-chart-gcea-to-system-memory-gfx1151
   :file: _templates/metrics_table.j2

