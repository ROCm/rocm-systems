.. meta::
   :description: How compute and memory partition modes affect ROCm Compute Profiler analysis metrics on MI300 and MI350 GPUs.
   :keywords: ROCm Compute Profiler, CDNA, partition, CPX, SPX, DPX, QPX, NPS, MI300, MI350, analyze

.. _compute-memory-partition:

**********************************************
Compute and memory partition modes in analysis
**********************************************

AMD Instinct™ MI300 and MI350 series GPUs support **compute partition modes**
(SPX, DPX, QPX, CPX) and **memory partition modes** (NPS1, NPS4, and others,
depending on the product). These modes expose the same physical GPU as
one or more logical devices and change how memory is grouped into NUMA domains.

ROCm Compute Profiler detects the active modes through AMD System Management
Interface (AMD-SMI) and adjusts metric normalization accordingly. **You do not
need manual rescaling** when profiling a logical partition: analysis reports
metrics for the partition you ran on.

For hardware background, configuration, and programming guidance, see:

.. seealso::

   * :doc:`cdna-performance-model`: CDNA3/CDNA4 architecture overview
   * `AMD Instinct MI300X GPU Partitioning Overview <https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/mi300x/overview.html>`_
   * `Deep dive into MI300 compute and memory partition modes (ROCm blog) <https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html>`_

Supported platforms
===================

Partition-aware metric normalization applies only to GPUs that support compute
and memory partitioning:

* **CDNA3:** MI300A, MI300X, MI325X
* **CDNA4:** MI350X, MI355X

Pre-MI300 Instinct GPUs (MI100, MI200 series) do not use these modes. ROCm
Compute Profiler treats them as a single compute partition with one XCD
equivalent.

What the tool reports
=====================

Profile mode
------------

On partition-capable GPUs, profile mode emits a warning when a compute or
memory partition is active. The message states that analysis will derive
logical XCD counts, L2 channels, and HBM channels from those modes.

Analyze mode
------------

The **System Info** section lists:

* **Compute Partition**: active compute mode (for example, SPX or CPX)
* **Memory Partition**: active memory mode (for example, NPS1 or NPS4)
* **Num XCDs**: logical accelerator complex dies (XCDs) in the compute
  partition
* **Total L2 Channels**: L2 cache channels used for bandwidth and utilization
  peaks in that partition

Use these fields together with the sections below to interpret Speed-of-Light
and memory-chart metrics.

How compute partition affects metrics
=====================================

Compute partition mode determines how many **logical XCDs** are in the scope of
the profiled device. On MI300X (``gfx942``), the whole chip has eight XCDs in
SPX mode. Other modes divide that count:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Mode
     - Logical XCDs (MI300X)
     - Summary
   * - SPX
     - 8
     - Single logical GPU; default mode
   * - DPX
     - 4
     - Two logical GPUs with four XCDs each
   * - QPX
     - 2
     - Four logical GPUs with two XCDs each
   * - CPX
     - 1
     - Eight logical GPUs; **each XCD is one logical GPU**

Exact counts vary by GPU model; see the product-specific tables in the
`MI300X partitioning overview <https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/mi300x/overview.html>`_.

Logical XCD count drives:

#. **Kernel active time normalization**: counters such as ``GRBM_GUI_ACTIVE``
   (GUI-active cycles) are normalized on a per-logical-XCD basis.
#. **Total L2 channels**: ``Total L2 Channels = L2 banks per XCD × logical
   XCDs``. This value feeds L2 bandwidth, busy, and stall metrics and their
   percent-of-peak calculations.

Metrics tied to a single logical partition (compute units, LDS, per-partition
L2) reflect **only the XCDs visible to the device you profiled**, not the
full package.

How memory partition affects metrics
====================================

Memory partition mode (NPS, NUMA Per Socket) controls how HBM stacks are
exposed as NUMA domains. It affects **theoretical HBM bandwidth**, which in
turn drives percent-of-peak metrics such as **L2-Fabric Read BW** and **L2-Fabric
Write BW**.

On MI300X (``gfx942``), the tool derives the number of 32-byte-per-clock HBM
channels from the memory mode. A common formulation is:

.. code-block:: text

   num_hbm_channels = 128 / NPS_denominator

where ``NPS_denominator`` is ``4`` for NPS4 and ``8`` for NPS8 on supported
parts. Theoretical HBM bandwidth is then:

.. code-block:: text

   peak_hbm_bw = (max_memory_clock_MHz / 1000) × 32 × num_hbm_channels

.. note::

   The two channel counts in **System Info** are scaled differently:

   * **Total L2 Channels** (``total_l2_chan``) follows the compute partition. It
     shrinks as the partition gets smaller, and it sets the peaks for L2
     bandwidth, busy, and stall metrics.
   * **Memory Channels** (``num_memory_channels``) ignores the compute partition.
     HBM is interleaved across the whole chip, so this count always starts from
     the full-chip value and is divided only by the NPS denominator. It sets the
     peak for L2-Fabric Read and Write BW.

Valid compute and memory mode pairings depend on the GPU model. Refer to the
`ROCm blog compatibility matrix <https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html>`_
and `MI300X partitioning overview <https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/mi300x/overview.html>`_
for supported combinations.

Example: CPX on MI300X
======================

.. note::

   CPX is fully supported. Metrics are reported for the **logical GPU**
   (one XCD) you profiled.

**Setup:**

.. code-block:: shell

   sudo amd-smi set -C CPX -g all
   # Memory mode depends on your platform; CPX is often paired with NPS4.
   export HIP_VISIBLE_DEVICES=0   # one logical GPU
   rocprof-compute profile -n vcopy -- ./vcopy

**Expected System Info (analyze):**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Field
     - Example (CPX)
   * - Compute Partition
     - CPX
   * - Memory Partition
     - NPS4 (if configured)
   * - Num XCDs
     - 1
   * - Total L2 Channels
     - L2 banks per XCD × 1

**How to read results:**

* **Active CUs**, VALU utilization, and L2 metrics describe **one XCD** (38 CUs
  on MI300X), not all 304 CUs on the package.
* **L2 Cache BW** peaks use the **partition** L2 channel count (one XCD), not
  the full eight-XCD total.
* **L2-Fabric Read/Write BW** percent-of-peak uses the **memory partition**
  HBM channel count (for example, NPS4 on MI300X), not the unified NPS1 view.

To compare against an SPX baseline, profile the same workload in SPX on the
same node. The SPX report covers the full GPU and the CPX report covers one
logical GPU, so read each report against its own **System Info** fields.

Cross-partition and fabric metrics
==================================

Most Speed-of-Light metrics are normalized to the profiled logical partition.
Some resources (for example, Infinity Fabric / XGMI paths shared across the
package) may reflect behavior outside a single XCD. When in doubt, check
**System Info** for the active partition fields and treat per-partition metrics
as authoritative for the device index you selected.

Further reading
===============

* :doc:`../../how-to/analyze/mode`: analyze mode overview and System Info
* :doc:`../../how-to/profile/mode`: profile mode, including partition warnings
* :doc:`system-speed-of-light`: metric definitions affected by partition-derived peaks
