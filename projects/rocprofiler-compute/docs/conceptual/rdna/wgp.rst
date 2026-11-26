.. meta::
   :description: ROCm Compute Profiler - RDNA3.5 WGP / roofline metrics
   :keywords: ROCm Compute Profiler, RDNA, gfx115x, WGP, roofline

.. _rdna-wgp:

=========================
Workgroup processor (WGP)
=========================

Within each shader engine, Workgroup Processors (WGPs) pair two Compute Units (CUs) that share resources and execute dispatched waves after the Workgroup Manager (SPI) hands off work.
On RDNA3.5 architecture-based GPUs/APUs, compute kernels are typically tracked with wave32-oriented waves; the gfx115x WGP panels cover occupancy, dispatch, instruction mix, and local caches at that WGP/CU-pair granularity.

The sections below list WGP metric descriptions for RDNA3.5 (gfx115x) and RDNA4 (gfx1201). Select the architecture tab on each table. The full Navi 48 die (Radeon RX 9070 XT) has 32 WGPs and 64 CUs. The Radeon RX 9070 has 56 CUs. RDNA4 runs matrix operations as WMMA instructions and has no MFMA instructions.

.. Note::
   The CDNA architecture-based AMD Instinct GPUs use a different execution hierarchy and panel grouping.
   For Instinct-only pipeline metrics (for example, VALU / VMEM / MFMA-style tables), see :doc:`../cdna/cdna-performance-model`-without assuming RDNA WGPs or CUs map directly to those layouts.

.. _rdna4-roofline:

Roofline
========

Profile mode runs the roofline benchmark unless you select blocks with ``-b`` or ``--set`` that do not include block 4. Set
``STABLE_STD`` first and use the same performance state for every run you
compare. See :ref:`profiling power-gating requirements <profile-rdna-power-gating>`. The benchmark measures the roofs in the
performance state that is active, and ``STABLE_STD`` fixes lower clocks than
the default state, so its roofs are lower. Do not compare ``roofline.csv``
files or roofline percentages across performance states.

On gfx1201, the benchmark measures three bandwidth roofs: VRAM, GL0, and LDS. It
also measures VALU roofs (FP16, FP32, FP64, I32) and WMMA roofs (FP8, FP16,
BF16). The ``roofline.csv`` columns ``L1Bw``, ``L2Bw``, and ``MALLBw`` are 0,
because gfx1201 has no kernel point for those levels.

* ``VRAM`` (column ``VRAMBw``) is the measured device-memory bandwidth. The
  benchmark streams a dataset far larger than the Infinity Cache.
* ``L0`` and ``LDS`` are measured with working sets that stay resident in GL0
  and LDS.
* There is no ``L1`` roof, because GL1 is a write-combining buffer, not a cache.
* There is no ``L2`` roof and no ``MALL`` (Infinity Cache) roof.

The ``AI VRAM`` point counts 32-byte GL2C/EA requests destined for memory.
The Infinity Cache can serve those requests, and no counter separates its hits
from traffic to physical VRAM. The plot marks the point once, at the ``VRAM``
level. A working set that fits in the Infinity Cache can therefore sit above the
``VRAM`` roof, and such a kernel is not limited by VRAM. The "Limited by" mode
and the hover text consider the ``VRAM`` level only.

The VALU metrics subtract the WMMA instruction count from the aggregate VALU
instruction count and multiply the result by the wave size. Integer and
floating-point instructions both count, and an FMA counts as one issue while the
measured FP32 peak counts it as two FLOPs. These metrics and the arithmetic
intensity built from them are estimates, not precision-specific FLOP counts. They
are most reliable for wave32 FP16 16x16x16 WMMA kernels.
One WMMA counter covers FP16, BF16, FP8, and BF8 and counts per vec32, so the
WMMA metric assumes 16x16x16 FP16 inputs with FP32 accumulation and compares with
the FP16 WMMA peak.

GL0 bytes count 128 bytes per ``TCP_REQ`` cache-line request. This is an upper
bound, because a request does not always use the full line, so ``AI L0`` reads
low.

The WMMA counter needs a ROCprofiler-SDK and aqlprofile installation that
provides the ``SQ_WMMA_VALU_INSTS`` counter. Check profiling diagnostics before
relying on WMMA points. WGP ``SQ_INST_CYCLES_VALU`` counts the cycles needed to execute VALU
operations. It is not elapsed GPU time, a FLOP count, or a utilization
percentage. Measured wave counts can exceed the application launch geometry.


Roofline performance rates
--------------------------

.. tab-set::
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      .. jinja:: rdna115x-roofline-performance-rates-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      .. jinja:: rdna4-roofline-performance-rates-gfx1201
         :file: _templates/metrics_table.j2

Roofline plot points
--------------------

.. tab-set::
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      .. jinja:: rdna115x-roofline-plot-points-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      .. jinja:: rdna4-roofline-plot-points-gfx1201
         :file: _templates/metrics_table.j2

WGP block metrics
=================

WGP utilization
---------------

.. tab-set::
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      .. jinja:: rdna115x-wgp-utilization-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      .. jinja:: rdna4-wgp-utilization-gfx1201
         :file: _templates/metrics_table.j2

Wavefront launch stats
----------------------

.. tab-set::
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      .. jinja:: rdna115x-wavefront-launch-stats-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      .. jinja:: rdna4-wavefront-launch-stats-gfx1201
         :file: _templates/metrics_table.j2

Wave dispatch
-------------

.. tab-set::
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      .. jinja:: rdna115x-wave-dispatch-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      .. jinja:: rdna4-wave-dispatch-gfx1201
         :file: _templates/metrics_table.j2

Wave life
---------

.. tab-set::
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      .. jinja:: rdna115x-wave-life-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      .. jinja:: rdna4-wave-life-gfx1201
         :file: _templates/metrics_table.j2

Wave instruction mix
--------------------

.. tab-set::
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      .. jinja:: rdna115x-wave-instruction-mix-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      .. jinja:: rdna4-wave-instruction-mix-gfx1201
         :file: _templates/metrics_table.j2

.. note::

   The **Instructions - Dual VALU (VOPD)** row counts ``SQ_INSTS_DUAL_VALU_WAVE32`` —
   the number of VOPD-encoded wave32 instructions issued per normalization unit.
   Each unit corresponds to one dual-issue VALU slot consumed; kernels that use VOPD
   heavily typically show a large value relative to the **Instructions - VALU** row.
   The exact ratio depends on ISA counting semantics for paired operations, so treat
   the two rows as complementary signals rather than a fixed ceiling relationship.
   For peak-rate theory and FLOPs ceilings see :ref:`rdna-dual-issue-valu`.

VALU instruction mix
--------------------

Only gfx1201 has this table.

.. jinja:: rdna4-valu-instruction-mix-gfx1201
   :file: _templates/metrics_table.j2

WMMA instruction counting
-------------------------

Only gfx1201 has this table.

.. jinja:: rdna4-wmma-instruction-counting-gfx1201
   :file: _templates/metrics_table.j2

VMEM instruction mix
--------------------

.. tab-set::
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      .. jinja:: rdna115x-vmem-instruction-mix-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      .. jinja:: rdna4-vmem-instruction-mix-gfx1201
         :file: _templates/metrics_table.j2

LDS instruction mix
-------------------

.. tab-set::
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      .. jinja:: rdna115x-lds-instruction-mix-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      .. jinja:: rdna4-lds-instruction-mix-gfx1201
         :file: _templates/metrics_table.j2

Wait state analysis
-------------------

.. tab-set::
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      .. jinja:: rdna115x-wait-state-analysis-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      gfx1201 has no wait-state table.

WGP instruction cache
---------------------

.. tab-set::
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      .. jinja:: rdna115x-wgp-instruction-cache-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      .. jinja:: rdna4-wgp-instruction-cache-gfx1201
         :file: _templates/metrics_table.j2

WGP scalar data cache
---------------------

.. tab-set::
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      .. jinja:: rdna115x-wgp-scalar-data-cache-gfx115x
         :file: _templates/metrics_table.j2

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      .. jinja:: rdna4-wgp-scalar-data-cache-gfx1201
         :file: _templates/metrics_table.j2

Memory chart: WGP
=================

The following memory chart table covers the WGP launch summary on gfx1201. The
instruction cache and scalar data cache memory chart tables are on the
:doc:`gl0-cache` page.

Memory chart - Workgroup processors (WGPs)
------------------------------------------

.. tab-set::
   :sync-group: rdna-arch

   .. tab-item:: RDNA3.5 (gfx115x)
      :sync: gfx115x
      :selected:

      gfx115x does not publish this Memory Chart table on the WGP page.

   .. tab-item:: RDNA4 (gfx1201)
      :sync: gfx1201

      .. jinja:: rdna4-memory-chart-wgps-gfx1201
         :file: _templates/metrics_table.j2
