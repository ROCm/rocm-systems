.. meta::
   :description: ROCm Compute Profiler - RDNA4 WGP and roofline metrics
   :keywords: ROCm Compute Profiler, RDNA, RDNA4, gfx1201, WGP, roofline, WMMA

.. _rdna4-wgp:

=========================
Workgroup processor (WGP)
=========================

Within each shader engine, Workgroup Processors (WGPs) pair two Compute Units (CUs) that share resources and execute dispatched waves after the Workgroup Manager (SPI) hands off work.
Navi 48 (gfx1201) has 32 WGPs and 64 CUs. Each WGP has an instruction cache, a scalar
data cache, and a 128 KB LDS shared by its two CUs. Each CU has two SIMD32 units.
Compute kernels typically run as wave32.

RDNA4 runs matrix operations as Wave Matrix Multiply Accumulate (WMMA) instructions on
the SIMDs. RDNA4 has no MFMA instructions.

The sections below list RDNA4 (gfx1201) metric descriptions for the Roofline panel and the
WGP rows of the Memory Chart panel.

.. Note::
   The CDNA architecture-based AMD Instinct GPUs use a different execution hierarchy and panel grouping.
   For Instinct-only pipeline metrics (for example, VALU / VMEM / MFMA-style tables), see :doc:`../cdna/cdna-performance-model`-without assuming RDNA WGPs or CUs map directly to those layouts.

.. _rdna4-roofline:

Roofline
========

The roofline peaks are measured on the GPU by the gfx1201 micro-benchmarks. The
bandwidth roofs are DRAM, MALL (the 64 MB Infinity Cache), GL2, the GL1 write-combining cache, GL0, and LDS.
The matrix roofs are FP8, FP16, and BF16.
No counter separates MALL hits from DRAM traffic, so the DRAM-destined traffic is drawn
against both the DRAM roof and the MALL roof.

GL0 bytes count 128 bytes per ``TCP_REQ`` cache-line request. This is an upper bound,
because a request does not always use the full line.

Roofline performance rates
--------------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-roofline-performance-rates-gfx1201
         :file: _templates/metrics_table.j2

Roofline plot points
--------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-roofline-plot-points-gfx1201
         :file: _templates/metrics_table.j2

Memory chart: WGP
=================

The following memory chart tables cover the instruction cache, the scalar data cache,
and the WGP launch summary.

Memory chart - Instruction cache
--------------------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-memory-chart-instruction-cache-gfx1201
         :file: _templates/metrics_table.j2

Memory chart - Scalar data cache
--------------------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-memory-chart-scalar-data-cache-gfx1201
         :file: _templates/metrics_table.j2

Memory chart - Workgroup processors (WGPs)
------------------------------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-memory-chart-wgps-gfx1201
         :file: _templates/metrics_table.j2

WGP block panel
===============

The Workgroup Processor (WGP) panel (panel 700) has tables for utilization,
wavefront launch, dispatch, and life, the wavefront, VMEM, LDS, and VALU instruction
mixes, WMMA FLOPs and IOPs counting, wait state analysis, and the instruction
and scalar data caches. Those tables have metric descriptions in the analysis
report. Run ``rocprof-compute analyze`` with ``--block 7`` to show the panel.
