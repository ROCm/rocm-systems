.. meta::
   :description: ROCm Compute Profiler: using Memory Bandwidth Analysis
   :keywords: ROCm Compute Profiler, Memory Bandwidth Analysis,
              guided analysis, bottleneck detection

************************************************************
Using memory bandwidth analysis
************************************************************

.. warning::

   Memory bandwidth analysis is an experimental feature. Pass
   ``--experimental --membw-analysis`` during ``profile`` to collect the
   required counters. Analyze mode detects the collected data
   automatically. Guidance text, feature behavior, and command-line
   options might change in future releases.

Memory bandwidth analysis identifies bottlenecks in the GPU memory
subsystem. It evaluates stall metrics collected from the
:doc:`L1 cache (GL1) </conceptual/cdna/vector-l1-cache>`,
:doc:`L2 cache (GL2) </conceptual/cdna/l2-cache>`, and Efficiency
Arbiter (EA) levels, and then reports which components are under pressure
and why.

When bottlenecks are detected, the analysis overlays stall annotations
on the memory chart and renders a guidance panel below it with
per-bottleneck details.

Supported hardware
==================

The memory bandwidth analysis feature is currently available for AMD Instinct MI350 Series GPUs (gfx950).

Profiling
=========

Collect memory bandwidth counters by adding ``--membw-analysis`` to
the ``profile`` command. Adding ``--membw-analysis`` enables Block 30, which contains the
stall and pressure metrics used by the analysis.

.. code-block:: shell

   $ rocprof-compute profile --experimental --membw-analysis -n my_workload -- ./my_app

Block 30 counters are collected alongside the standard profiling
counters. No other profiling options are needed.

Analysis
========

When the workload was profiled with ``--membw-analysis``, analyze
detects the collected data and enables bottleneck detection
automatically:

.. code-block:: shell

   $ rocprof-compute analyze -p workloads/my_workload/MI350/

To view only the memory chart and memory bandwidth analysis tables,
use the block filter:

.. code-block:: shell

   $ rocprof-compute analyze -p workloads/my_workload/MI350/ -b 3

``-b 3`` selects the memory chart. Block 30 is included automatically
when block 30 counters were collected during profiling.

.. code-block:: text

   3. Memory Chart (Normalization: per_kernel)
                                                                                                          ╭──────────────────────╮
                                                                                                          │ xGMI (to Peer GPU)   │
                                                                                                          ╰──────────────────────╯
                                                                                                             ||  Read BW    0.0 B/s
                                                                                                             ||  Write BW   0.0 B/s
                                                                                                             ||  Atomic BW  0.0 B/s
   |--------------------------------------------- GPU (XCD) ----------------------------------------------|---------------------------------- Fabric / Memory -----------------------------------|

   ╭── Kernel ──╮Non-buffer Request╭─────── VL1D ───────╮Read BW     ╭──────── L2 ────────╮               ╭─── Data Fabric ────╮╭─────── MALL ───────╮╭─────── UMC ────────╮╭─────── HBM ────────╮
   │            │Read   : 1.64e+04 │ Hit 62.0%          │1.2 TB/s    │ Hit 34.0%          │               │ [!] EA write stall ││                    ││                    ││ Read BW            │
   │            │<---------------  │ ██████░░░░         │<-----------│ ███░░░░░░░         │               │ 10.9%              ││                    ││                    ││ 1.2 TB/s           │
   │            │Write  : 1.64e+04 │                    │            │                    │               │                    ││                    ││                    ││                    │
   │            │--------------->  │ [!] TCP<-UTCL2     │Write BW    │ [!] HBM wr stall   │               │                    ││                    ││                    ││ Write BW           │
   │            │Atomic :       0  │ 23.3%              │1.2 TB/s    │ 11.3%              │               │                    ││                    ││                    ││ 1.2 TB/s           │
   │            │<-------------->  │                    │----------->│                    │               │                    ││                    ││                    ││                    │
   │            │Buffer Request    │                    │            │ [!] L2 low hit     │               │                    ││                    ││                    ││ Atomic BW          │
   │            │Read   :       0  │                    │Atomic BW   │ rate 34.2%         │               │                    ││                    ││                    ││ 0.0 B/s            │
   │            │<---------------  │                    │0.0 B/s     │                    │               │                    ││                    ││                    ││                    │
   │            │Write  :       0  │                    │<---------->│                    │               │                    ││                    ││                    ││                    │
   │            │--------------->  │                    │            │                    │               │                    ││                    ││                    ││                    │
   │            │Atomic :       0  │                    │            │                    │               │                    ││                    ││                    ││                    │
   │            │<-------------->  │                    │            │                    │Read BW        │                    ││                    ││                    ││                    │
   │ Shader     │                  │                    │            │                    │1.2 TB/s       │                    ││                    ││                    ││                    │
   │ Core       │                  ╰────────────────────╯            │                    │<-----------   │                    ││                    ││                    ││                    │
   │ Wave       │Read   :       0  ╭─────── LDS ────────╮            │                    │               │                    ││                    ││                    ││                    │
   │ Execution  │<---------------  │ Util 0.0%          │            │                    │Write/Atomic BW│                    ││                    ││                    ││                    │
   │            │Write  :       0  │ ░░░░░░░░░░         │            │                    │1.2 TB/s       │                    ││                    ││                    ││                    │
   │            │--------------->  │                    │            │                    │----------->   │                    ││                    ││                    ││                    │
   │            │Atomic :       0  │                    │            │                    │               │                    ││                    ││                    ││                    │
   │            │<-------------->  │                    │            │                    │               │                    ││                    ││                    ││                    │
   │            │Instr  :       0  │                    │            │                    │               │                    ││                    ││                    ││                    │
   │            │<-------------->  │                    │            │                    │               │                    ││                    ││                    ││                    │
   │            │                  │                    │            │                    │               │                    ││                    ││                    ││                    │
   │            │                  ╰────────────────────╯            │                    │               │                    ││                    ││                    ││                    │
   │            │SMEM              ╭─────── sL1D ───────╮Read BW     │                    │               │                    ││                    ││                    ││                    │
   │            │Read   : 6.55e+04 │ Hit 93.0%          │1.4 GB/s    │                    │               │                    ││                    ││                    ││                    │
   │            │<---------------  │ █████████░         │<-----------│                    │               │                    ││                    ││                    ││                    │
   │            │                  ╰────────────────────╯            │                    │               │                    ││                    ││                    ││                    │
   │            │ICACHE            ╭─────── L1I ────────╮Read BW     │                    │               │                    ││                    ││                    ││                    │
   │            │Read   : 6.55e+04 │ Hit 91.0%          │24.4 GB/s   │                    │               │                    ││                    ││                    ││                    │
   │            │<---------------  │ █████████░         │<-----------│                    │               │                    ││                    ││                    ││                    │
   ╰────────────╯                  ╰────────────────────╯            ╰────────────────────╯               ╰────────────────────╯╰────────────────────╯╰────────────────────╯╰────────────────────╯

                                                                                                             ||  Read BW    0.0 B/s
                                                                                                             ||  Write BW   0.0 B/s
                                                                                                             ||  Atomic BW  0.0 B/s
                                                                                               ╭────────────────────────────────────────────╮
                                                                                               │ PCIe (to CPU or Non-xGMI connected GPU)    │
                                                                                               ╰────────────────────────────────────────────╯

   Legend: <---- Read  ----> Write  <---> Atomic  █ Util  █ Hit%  █ Stall

   ╭───────────────────────────────────────────────────────────────────────────── Memory Bandwidth Guided Analysis ──────────────────────────────────────────────────────────────────────────────╮
   │                                                                                                                                                                                             │
   │  [GL1] TCP stalled by UTCL2                                                                    [GL2] HBM write bandwidth bound                                                              │
   │    Condition : TCP stalled by UTCL2 >= 10%                                                       Condition : HBM write credit stall >= 10%                                                  │
   │    Measured  : 23.3% (threshold: 10%)                                                            Measured  : 11.3% (threshold: 10%)                                                         │
   │    Impact    : TLB latency -- UTCL1 LFIFO resources exhausted waiting on UTCL2 lookups           Impact    : L2 stalled waiting for HBM write credits -- memory write bandwidth saturated   │
   │                                                                                                                                                                                             │
   │  [GL2] Low L2 cache efficiency                                                                 [EA] Write backpressure                                                                      │
   │    Condition : L2 hit rate <= 50%                                                                Condition : EA write stall >= 10%                                                          │
   │    Measured  : 34.2% (threshold: 50%)                                                            Measured  : 10.9% (threshold: 10%)                                                         │
   │    Impact    : Workload exceeds L2 cache capacity -- most accesses miss to HBM                   Impact    : EA write path congested -- HBM write saturation, write-heavy workload, or      │
   │                                                                                                atomic contention                                                                            │
   │                                                                                                                                                                                             │
   ╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Reading the output
==================

When active bottlenecks are found, the output includes:

Stall annotations on the memory chart
--------------------------------------

Active bottlenecks appear as ``[!]`` rows inside the affected cache
panel, showing the metric label and its measured value. Panels with
active stalls are highlighted with a red border, and a "Stall" entry
is added to the chart legend.

Guidance panel
--------------

A guidance panel appears below the memory chart. Each entry describes
one bottleneck:

* **Condition**: What was checked (for example,
  "TCP stalled by UTCL2 >= 10%")
* **Measured**: The actual value from the profiled workload and the
  threshold it was compared against
* **Impact**: A brief explanation of what this stall means for your
  workload

When bottlenecks are not found, a single status line is shown instead
(for example, "Memory Bandwidth Analysis: No bottlenecks detected").

Understanding the results
=========================

The analysis checks three levels of the memory hierarchy:

* **GL1 (L1 cache)**: stall sources within the L1 cache — address
  translation (UTCL1/UTCL2), texture data return (TD), L2
  backpressure, and shader core pressure (VMEM).
* **GL2 (L2 cache)**: HBM bandwidth pressure (read, write, or
  balanced), internal resource exhaustion (latency and source FIFOs),
  cache efficiency, and GMI remote access pressure.
* **EA (Efficiency Arbiter)**: PCIe/IO path pressure, write
  backpressure, and HBM atomic contention.

A bottleneck is reported when a stall metric exceeds its threshold
(typically 10% of busy time). When a parent metric exceeds the
threshold but no specific child does, a "balanced" or "other" entry
explains that the pressure is distributed rather than concentrated in
one path. Multiple bottlenecks can appear together when several stall
metrics exceed the threshold simultaneously.

.. note::

   All memory bandwidth analysis metrics are stall-cycle ratios (for
   example, ``100 * SUM(stall_cycles) / SUM(busy_cycles)``). These
   percentages are not affected by the normalization mode shown in the
   memory chart title (per_kernel, per_wave, and so on).

Further resources
=================

For deeper analysis beyond the guided output:

* **Block 30 raw metrics**: run ``-b 30`` to see the full set of
  Memory Bandwidth Analysis metric tables (L1 cache, L2 bottleneck
  indicators, EA indicators).
* **Memory chart**: the :doc:`CLI analysis documentation </how-to/analyze/cli>`
  covers the memory chart layout in detail.
* **CDNA performance model**: the :doc:`L2 cache
  </conceptual/cdna/l2-cache>` and :doc:`Vector L1 cache
  </conceptual/cdna/vector-l1-cache>` conceptual pages describe the
  cache hierarchy and how data moves through it.
* **Design document**: `hld-membw-guided-analysis-in-memchart.md <https://github.com/ROCm/rocm-systems/tree/rocprofiler-compute-develop/projects/rocprofiler-compute/docs/design/hld-membw-guided-analysis-in-memchart.md>`_
  describes the bottleneck tree structure and threshold semantics.

Terminal width
==============

The memory chart requires a terminal width of at least **240 columns**
to display properly. The guidance panel matches the chart width.
Narrower terminals cause line wrapping that reduces readability.

Check your terminal width with:

.. code-block:: shell

   $ tput cols

.. note::

   The analysis evaluates per-dispatch averages. Bottlenecks that occur in only a subset of dispatches may not be visible.
