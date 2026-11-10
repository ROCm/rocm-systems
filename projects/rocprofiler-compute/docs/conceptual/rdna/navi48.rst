.. meta::
   :description: Navi48 gfx1201 GPU metrics, Memory Chart, profiling prerequisites, and counter limitations in ROCm Compute Profiler.
   :keywords: ROCm Compute Profiler, RDNA4, Navi48, gfx1201, Radeon RX 9070, Memory Chart, Infinity Cache, GDDR6

.. _navi48-profiling:

****************
Navi48 (gfx1201)
****************

ROCm Compute Profiler supports GPU metrics, the Memory Chart, and the roofline
with its benchmark on Linux for Navi48 (gfx1201), including AMD Radeon RX 9070.
This support starts with ROCm Compute Profiler 3.11 in ROCm 10.3. Navi44
(gfx1200), Windows, and PC sampling are outside this support scope.

Profiling prerequisite
======================

Identify the Navi48 GPU with ``amd-smi list`` and ``rocminfo``. Before collecting
performance counters, set that GPU's performance level to ``STABLE_STD``. For
example, if Navi48 is GPU 0:

.. code-block:: bash

   sudo /opt/rocm/bin/amd-smi set --gpu 0 --perf-level STABLE_STD

Omitting ``--gpu`` sets the performance level for all GPUs. This setting changes
the clock policy; record it and use the same performance state when comparing
profiles. ROCm Compute Profiler does not run this privileged command for you.
Use a ROCm stack with gfx1201 counter support. A successful ``rocminfo`` query
alone does not establish that the installed profiler backend exposes those
counters.

Collect the System Speed-of-Light and Memory Chart panels for a short workload:

.. code-block:: bash

   HIP_VISIBLE_DEVICES=0 rocprof-compute profile --name vcopy-navi48 \
       -b sol memchart --no-roof -- ./vcopy -n 1048576 -b 256

``HIP_VISIBLE_DEVICES`` selects the GPU used by the HIP application. Use the
index of your Navi48 GPU. To discover available block aliases and metrics, run:

.. code-block:: bash

   rocprof-compute --list-metrics-gfx-id gfx1201

Analyze the output directory that contains ``profiling_config.yaml`` and select
one kernel from ``--list-stats``:

.. code-block:: bash

   rocprof-compute analyze --path <workload_directory> --list-stats
   rocprof-compute analyze --path <workload_directory> -k <kernel_id> \
       -b sol memchart

For installation and general profiling options, see
:doc:`/install/core-install` and :doc:`/how-to/profile/mode`.

Reading GPU metrics
===================

``VALU Issue Rate`` and ``FP64 VALU Issue Rate`` report instruction issues in
GInstr/s. The aggregate VALU counter mixes floating-point and integer
instructions, and neither row accounts for operations per instruction or active
lanes. These rows have no FLOP peak or percentage of a floating-point peak.

WGP ``SQ_INST_CYCLES_VALU`` reports the raw SDK aggregate of SIMD execution
cycles. Overlapping V_OP32_1 and V_OP32_T instructions count separately.
Its counted instance and wave population still need qualification. Do not
interpret it as elapsed GPU time, a FLOP count or a utilization percentage.

``WMMA FLOPs (F16->F32 16x16x16)`` assumes FP16 inputs and FP32 accumulation for
that matrix shape. The raw WMMA counter counts issues per vec32 and does not
distinguish FP16, BF16, or FP8. Its FLOP conversion and peak comparison are
approximate and require validation for the kernel's precision and wave size.
Do not treat a percentage from a mixed-precision workload as a validated FP16
utilization figure. A fixed wave32 FP16-to-FP32 16x16x16 diagnostic supports
WMMA inclusion in general VALU instruction counts for that instruction.
Counted wave population and overlap for other variants remain unqualified.
Do not combine these counts into an independent instruction total or subtract
WMMA issues to infer FP32 FLOPs from the Speed-of-Light rows. The roofline rows
subtract them by design, as an estimate described in :ref:`navi48-roofline`.

``GL2-Fabric Read BW`` and ``GL2-Fabric Write BW`` describe the L2 interface.
Infinity Cache can serve or absorb this traffic before it reaches GDDR6. These
rows have no percentage of the GDDR6 bandwidth peak.

These EA rates use the configured 32, 64, 128, and 256-byte request categories.
Separate compressed request and data counters are not included. Their overlap
and byte coverage still require validation, so these rows do not establish
total traffic under compression.

``GL2 Cache BW`` measures uncompressed read requests at the GL2 output FIFOs.
It does not measure all L2 traffic, and an applicable interface peak remains
unconfirmed. The row reports a rate without a percentage of peak.

Wave-count normalization still requires validation on Navi48. Measured wave
counts can exceed the application launch geometry and do not yet establish
exact application wave counts.

``Wavefront Occupancy`` and WGP ``Wave Life`` remain provisional estimates.
The living-wave counter units and instance reduction used by those rows also
require validation. Do not infer exact resident-wave counts, shader cycles,
or occupancy limits from these values.

``Wave Dependency Wait`` and ``Wave Issue Wait`` display ``N/A`` while
wait-counter units and instance reduction are validated. The CLI omits the WGP
Instruction Issue Stalls table, including ``Wait for Any``, because all its
values are unavailable. The ``wave_wait`` profiling shortcut is unavailable
on Navi48. Restoration of these metrics is tracked under AIPROFCOMP-907;
they cannot yet quantify scheduler stalls.

If a requested WMMA counter is missing, profiling diagnostics identify it.
A diagnostic can suggest the corresponding instruction or cycle counter only
when that peer counter was requested and appeared in the same profiling pass.
The two raw counters have different units and cannot replace each other.

An ``N/A`` value means the metric is unavailable, not zero. For example, GL1
is a write-combining stage on RDNA4 and has no hit-rate metric in this
configuration. Some bandwidth rows also lack counters that describe request
sizes. Counter availability depends on the installed ROCm profiler backend;
check profiling diagnostics when a requested counter is absent.

System information
-------------------

The saved workload and System Info output include the physical GPU UUID,
PCI BDF identifier, and topology node identifier when available. These fields
help distinguish GPUs with the same product name. They do not establish the
application's visible GPU index; use the visibility setting used for that
application when selecting its GPU.

System Speed-of-Light
---------------------

.. jinja:: sys-sol-gfx1201
   :file: _templates/metrics_table.j2

Reading the Memory Chart
=========================

The chart follows instruction and scalar data caches, vector GL0, LDS, GL1,
and GL2. GL1 combines writes between the WGPs and L2. EA to DRAM
shows DRAM read and write requests and bandwidth; Infinity Cache and GDDR6 appear
after that interface.

``DRAM Read Bandwidth`` and ``DRAM Write Bandwidth``
count requests that L2 sends toward memory, including requests served or
absorbed by Infinity Cache. They do not measure the bytes that physically reach
GDDR6. In particular, a value above the GDDR6 peak does not by itself indicate
an invalid counter or an impossible memory transfer rate.

``Infinity Cache Hit Rate``, ``GDDR6 Read Bandwidth``, and
``GDDR6 Write Bandwidth`` display ``N/A`` because this configuration does not
collect the counters needed to measure them. Do not infer those values from
memory-destined traffic. Other unavailable fields also remain ``N/A``.

Instruction Cache
------------------

.. jinja:: navi48-memory-chart-instruction-cache-gfx1201
   :file: _templates/metrics_table.j2

Scalar Data Cache
------------------

.. jinja:: navi48-memory-chart-scalar-data-cache-gfx1201
   :file: _templates/metrics_table.j2

GL0 Cache (Vector L0)
---------------------

.. jinja:: navi48-memory-chart-gl0-gfx1201
   :file: _templates/metrics_table.j2

LDS (Local Data Share)
----------------------

.. jinja:: navi48-memory-chart-lds-gfx1201
   :file: _templates/metrics_table.j2

GL0-GL1 Interface
-----------------

.. jinja:: navi48-memory-chart-gl0-gl1-gfx1201
   :file: _templates/metrics_table.j2

GL1 Cache (L1)
--------------

GL1 is a write-combining stage between the WGPs and L2. It has utilization
and stall metrics, and no hit rate.

.. jinja:: navi48-memory-chart-gl1-gfx1201
   :file: _templates/metrics_table.j2

GL1-GL2 Interface
------------------

.. jinja:: navi48-memory-chart-gl1-gl2-gfx1201
   :file: _templates/metrics_table.j2

GL2 Cache (L2)
---------------

.. jinja:: navi48-memory-chart-gl2-gfx1201
   :file: _templates/metrics_table.j2

EA to DRAM
----------

.. jinja:: navi48-memory-chart-ea-to-dram-gfx1201
   :file: _templates/metrics_table.j2

Infinity Cache and GDDR6
------------------------

.. jinja:: navi48-memory-chart-infinity-cache-gfx1201
   :file: _templates/metrics_table.j2

Workgroup Processors (WGPs)
---------------------------

.. jinja:: navi48-memory-chart-wgp-gfx1201
   :file: _templates/metrics_table.j2

L2 cache metrics
=================

GL2 Cache Request Statistics
----------------------------

.. jinja:: navi48-gl2-utilization-gfx1201
   :file: _templates/metrics_table.j2

GL2 Cache Performance
---------------------

.. jinja:: navi48-gl2-lookup-gfx1201
   :file: _templates/metrics_table.j2

GL2 Cache Bandwidth
-------------------

``GL2 Cache Read BW`` and ``GL2 Cache Write BW`` report size-weighted requests
at the GL2 output FIFOs. Reads cover uncompressed request categories; writes
include recompressed requests. These rates measure a different boundary from
the GL2/EA interface and do not represent all L2 traffic or physical GDDR6
transfers. Separate compressed-read categories are excluded. Compressed-write
overlap, complete byte coverage and request-size normalization still require
Navi48 qualification.

.. jinja:: navi48-gl2-bandwidth-gfx1201
   :file: _templates/metrics_table.j2

External address interface
===========================

The EA panels describe requests and stalls between L2 and the fabric. A counter
whose name mentions DRAM still describes this interface; it does not establish
the number of transfers that reach physical GDDR6.

EA Utilization
---------------

.. jinja:: navi48-ea-utilization-gfx1201
   :file: _templates/metrics_table.j2

EA Traffic Bandwidth
---------------------

.. jinja:: navi48-ea-traffic-gfx1201
   :file: _templates/metrics_table.j2

EA-SDP Interface - Outstanding Requests
---------------------------------------

.. jinja:: navi48-ea-outstanding-requests-gfx1201
   :file: _templates/metrics_table.j2

EA-SDP Interface - Stalls
-------------------------

.. jinja:: navi48-ea-stalls-gfx1201
   :file: _templates/metrics_table.j2

.. _navi48-roofline:

Roofline
========

Profile mode runs the roofline benchmark unless you pass ``--no-roof``. To
collect only the roofline, or to rerun the benchmark in an existing workload
directory, see :doc:`/how-to/profile/mode`. Set ``STABLE_STD`` first and use the
same performance state for every run you compare. The benchmark measures the
roofs in the performance state that is active, and ``STABLE_STD`` fixes lower
clocks than the default state, so its roofs are lower. Do not compare
``roofline.csv`` files or roofline percentages across performance states.

The Navi48 roofline has these measured roofs, in ``roofline.csv``:

* ``DRAM`` (column ``DRAMBw``) is the measured device-memory bandwidth of the
  256-bit, 20 Gbps GDDR6 interface. The benchmark streams a dataset far larger
  than the 64 MB Infinity Cache.
* ``MALL`` is the measured bandwidth of the 64 MB Infinity Cache, from a buffer
  that misses in the 8 MB L2 and fits in the Infinity Cache.
* ``L2``, ``L1`` and ``L0`` are measured with working sets that stay resident
  in GL2 and GL0. GL1 has no tags, so the ``L1`` roof is the GL2-hit return
  path for a set that misses GL0.
* ``LDS``, VALU (FP16, FP32, FP64, I8, I32, I64) and WMMA (FP8, BF8, FP16, BF16).

The ``AI DRAM`` point counts 32-byte GL2C/EA requests destined for memory.
Infinity Cache can serve those requests, and no counter separates its hits from
traffic to physical GDDR6. The plot marks the point once, at the ``DRAM`` level.
Compare its performance with both the ``DRAM`` and ``MALL`` roofs. A working set
that fits in the Infinity Cache can sit above the ``DRAM`` roof and below the
``MALL`` roof, so such a kernel is not limited by GDDR6. The "Limited by" mode
and the hover text consider the ``DRAM`` level only, and never name ``MALL``.
This is the same boundary as the Memory Chart's memory-destined bandwidth, and
neither measures physical GDDR6. The analysis database stores this point in its
``hbm_cache_data`` column, which MI architectures fill with ``AI HBM``.

The VALU rows subtract the WMMA instruction count from the aggregate VALU
instruction count and multiply the result by the wave size. That subtraction is
validated only for the fixed wave32 FP16 16x16x16 diagnostic. Integer and
floating-point instructions both count, and an FMA counts as one issue while the
measured FP32 peak counts it as two FLOPs. These rows and the arithmetic
intensity built from them are therefore estimates and not precision-specific
FLOP counts. They can read low for FMA-only loops and high when index and loop
arithmetic dominate. In a deterministic tiled FP32 matrix-multiply test, the
VALU row was about twice the true FLOP rate. One WMMA counter covers FP16,
BF16, and FP8 and counts per vec32, so the WMMA row assumes 16x16x16 FP16
inputs with FP32 accumulation and compares with the FP16 WMMA peak. Its
interpretation needs validation for other shapes and types.

The GL0 bandwidth uses 128 bytes per ``TCP_REQ`` request as an upper bound, so
``AI L0`` reads low. The ``L1`` roof is measured on the same GL2-hit return path
as the ``L2`` roof, so the two can differ by a few percent.

The WMMA counter needs a ROCprofiler-SDK and aqlprofile installation that
supports SQ event 233; check profiling diagnostics before relying on WMMA
points.
