.. meta::
   :description: ROCm Compute Profiler - RDNA3.5 WGP / roofline metrics
   :keywords: ROCm Compute Profiler, RDNA, gfx115x, WGP, roofline

.. _rdna-wgp:
.. _rdna4-wgp:

=========================
Workgroup processor (WGP)
=========================

Within each shader engine, Workgroup Processors (WGPs) pair two Compute Units (CUs) that share resources and execute dispatched waves after the Workgroup Manager (SPI) hands off work.
On RDNA3.5 architecture-based GPUs/APUs, compute kernels are typically tracked with wave32-oriented waves; the gfx115x WGP panels cover occupancy, dispatch, instruction mix, and local caches at that WGP/CU-pair granularity.

The sections below list WGP metric descriptions for RDNA3.5 (gfx115x) and RDNA4 (gfx1201). Select the architecture tab on each table. Navi 48 has 32 WGPs and 64 CUs. RDNA4 runs matrix operations as WMMA instructions and has no MFMA instructions.

.. Note::
   The CDNA architecture-based AMD Instinct GPUs use a different execution hierarchy and panel grouping.
   For Instinct-only pipeline metrics (for example, VALU / VMEM / MFMA-style tables), see :doc:`../cdna/cdna-performance-model`-without assuming RDNA WGPs or CUs map directly to those layouts.

.. _rdna4-roofline:
.. _navi48-roofline:

Roofline
========

Profile mode runs the roofline benchmark unless you pass ``--no-roof``. Set
``STABLE_STD`` first and use the same performance state for every run you
compare. See :ref:`profiling power-gating requirements <profile-rdna-power-gating>`. The benchmark measures the roofs in the
performance state that is active, and ``STABLE_STD`` fixes lower clocks than
the default state, so its roofs are lower. Do not compare ``roofline.csv``
files or roofline percentages across performance states.

On gfx1201, the measured roofs are VRAM, MALL (the Infinity Cache), GL2,
the GL1 write-combining cache, GL0, and LDS, plus VALU (FP16, FP32, FP64, I32)
and WMMA (FP8, BF8, FP16, BF16). BF8 is ``V_WMMA_F32_16X16X16_BF8_BF8``,
the same 16x16x16 shape as FP8, so it counts 8192 FLOPs per instruction.

* ``VRAM`` (column ``VRAMBw``) is the measured device-memory bandwidth. The
  benchmark streams a dataset far larger than the Infinity Cache.
* ``MALL`` is the measured bandwidth of the Infinity Cache, from a buffer of
  half its reported size that misses in L2 and fits in the Infinity Cache.
* ``L2``, ``L1``, and ``L0`` are measured with working sets that stay resident
  in GL2 and GL0. GL1 has no tags, so the ``L1`` roof is the GL2-hit return
  path for a set that misses GL0. gfx1201 has no ``AI L1`` or ``AI L2`` point,
  so the ``L1`` and ``L2`` roofs have no kernel point.

The ``AI VRAM`` point counts 32-byte GL2C/EA requests destined for memory.
Infinity Cache can serve those requests, and no counter separates its hits from
traffic to physical VRAM. The plot marks the point once, at the ``VRAM`` level.
Compare its performance with both the ``VRAM`` and ``MALL`` roofs. A working set
that fits in the Infinity Cache can sit above the ``VRAM`` roof and below the
``MALL`` roof, so such a kernel is not limited by VRAM. The "Limited by" mode
and the hover text consider the ``VRAM`` level only, and never name ``MALL``.
The analysis database stores this point in its ``hbm_cache_data`` column, which
MI architectures fill with ``AI HBM``.

The VALU rows subtract the WMMA instruction count from the aggregate VALU
instruction count and multiply the result by the wave size. That subtraction is
validated only for the fixed wave32 FP16 16x16x16 diagnostic. Integer and
floating-point instructions both count, and an FMA counts as one issue while the
measured FP32 peak counts it as two FLOPs. These rows and the arithmetic
intensity built from them are estimates, not precision-specific FLOP counts.
One WMMA counter covers FP16, BF16, FP8, and BF8 and counts per vec32, so the
WMMA row assumes 16x16x16 FP16 inputs with FP32 accumulation and compares with
the FP16 WMMA peak.

GL0 bytes count 128 bytes per ``TCP_REQ`` cache-line request. This is an upper
bound, because a request does not always use the full line, so ``AI L0`` reads
low. The ``L1`` roof is measured on the same GL2-hit return path as the ``L2``
roof, so the two can differ by a few percent.

The WMMA counter needs a ROCprofiler-SDK and aqlprofile installation that
supports SQ event 233. Check profiling diagnostics before relying on WMMA
points. WGP ``SQ_INST_CYCLES_VALU`` counts the cycles needed to execute VALU
operations. It is not elapsed GPU time, a FLOP count, or a utilization
percentage. Measured wave counts can exceed the application launch geometry.


Roofline performance rates
--------------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-roofline-performance-rates-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-roofline-performance-rates-gfx1201
         :file: _templates/metrics_table.j2

Roofline plot points
--------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-roofline-plot-points-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-roofline-plot-points-gfx1201
         :file: _templates/metrics_table.j2

WGP block metrics
=================

WGP utilization
---------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-wgp-utilization-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      The gfx1201 WGP panel describes utilization, wavefront launch, dispatch, life, instruction mix, and wait states in the analysis report. This page publishes the Roofline tables and the Memory Chart rows for gfx1201.

Wavefront launch stats
----------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-wavefront-launch-stats-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      The gfx1201 WGP panel describes utilization, wavefront launch, dispatch, life, instruction mix, and wait states in the analysis report. This page publishes the Roofline tables and the Memory Chart rows for gfx1201.

Wave dispatch
-------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-wave-dispatch-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      The gfx1201 WGP panel describes utilization, wavefront launch, dispatch, life, instruction mix, and wait states in the analysis report. This page publishes the Roofline tables and the Memory Chart rows for gfx1201.

Wave life
---------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-wave-life-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      The gfx1201 WGP panel describes utilization, wavefront launch, dispatch, life, instruction mix, and wait states in the analysis report. This page publishes the Roofline tables and the Memory Chart rows for gfx1201.

Wave instruction mix
--------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-wave-instruction-mix-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      The gfx1201 WGP panel describes utilization, wavefront launch, dispatch, life, instruction mix, and wait states in the analysis report. This page publishes the Roofline tables and the Memory Chart rows for gfx1201.

.. note::

   The **Instructions - Dual VALU (VOPD)** row counts ``SQ_INSTS_DUAL_VALU_WAVE32`` —
   the number of VOPD-encoded wave32 instructions issued per normalization unit.
   Each unit corresponds to one dual-issue VALU slot consumed; kernels that use VOPD
   heavily typically show a large value relative to the **Instructions - VALU** row.
   The exact ratio depends on ISA counting semantics for paired operations, so treat
   the two rows as complementary signals rather than a fixed ceiling relationship.
   For peak-rate theory and FLOPs ceilings see :ref:`rdna-dual-issue-valu`.

VMEM instruction mix
--------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-vmem-instruction-mix-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      The gfx1201 WGP panel describes utilization, wavefront launch, dispatch, life, instruction mix, and wait states in the analysis report. This page publishes the Roofline tables and the Memory Chart rows for gfx1201.

LDS instruction mix
-------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-lds-instruction-mix-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      The gfx1201 WGP panel describes utilization, wavefront launch, dispatch, life, instruction mix, and wait states in the analysis report. This page publishes the Roofline tables and the Memory Chart rows for gfx1201.

Wait state analysis
-------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-wait-state-analysis-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      The gfx1201 WGP panel describes utilization, wavefront launch, dispatch, life, instruction mix, and wait states in the analysis report. This page publishes the Roofline tables and the Memory Chart rows for gfx1201.

WGP instruction cache
---------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-wgp-instruction-cache-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      The gfx1201 WGP panel describes utilization, wavefront launch, dispatch, life, instruction mix, and wait states in the analysis report. This page publishes the Roofline tables and the Memory Chart rows for gfx1201.

WGP scalar data cache
---------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      .. jinja:: rdna115x-wgp-scalar-data-cache-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      The gfx1201 WGP panel describes utilization, wavefront launch, dispatch, life, instruction mix, and wait states in the analysis report. This page publishes the Roofline tables and the Memory Chart rows for gfx1201.

Memory chart: WGP
=================

The following memory chart tables cover the instruction cache, the scalar data cache,
and the WGP launch summary on gfx1201.

Memory chart - Instruction cache
--------------------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      The gfx115x instruction-cache rows are in the WGP instruction cache table above and on the GL0 cache page.

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-memory-chart-instruction-cache-gfx1201
         :file: _templates/metrics_table.j2

Memory chart - Scalar data cache
--------------------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      The gfx115x scalar data cache rows are in the WGP scalar data cache table above and on the GL0 cache page.

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-memory-chart-scalar-data-cache-gfx1201
         :file: _templates/metrics_table.j2

Memory chart - Workgroup processors (WGPs)
------------------------------------------

.. tab-set::

   .. tab-item:: RDNA3.5 (gfx115x)

      gfx115x does not publish this Memory Chart table on the WGP page.

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-memory-chart-wgps-gfx1201
         :file: _templates/metrics_table.j2
