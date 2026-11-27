.. meta::
   :description: ROCm Compute Profiler: using Memory Bandwidth Analysis
   :keywords: ROCm Compute Profiler, Memory Bandwidth Analysis,
              guided analysis, bottleneck detection

************************************************************
Using memory bandwidth analysis
************************************************************

.. warning::

   Memory bandwidth analysis is an experimental feature. To enable it,
   pass ``--experimental --membw-analysis`` at ``profile`` time. Analyze
   mode detects the data automatically from the profiling configuration.
   Guidance text, feature behavior, and command-line options might change in future releases.

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

Add ``--membw-analysis`` to the ``profile`` command to collect
Block 30, which contains the stall and pressure metrics used by the
analysis. Use the block filter ``-b 3`` to collect the memory chart
without unrelated blocks:

.. code-block:: shell

   $ rocprof-compute profile --experimental --membw-analysis -b 3 -n my_workload -- ./my_app

In this command, ``-b 3`` selects the memory chart and ``--membw-analysis``
adds Block 30 to the block filter.

Analysis
========

Analyze mode auto-detects that memory bandwidth analysis data was
collected and runs bottleneck detection automatically:

.. code-block:: shell

   $ rocprof-compute analyze -p workloads/my_workload/MI350/

Analyze mode applies the block filter used at ``profile`` time, so this
command shows the memory chart and the memory bandwidth analysis panel.
When bottlenecks are detected, the guidance annotations appear on the
memory chart and a guidance panel is rendered below it.

.. code-block:: text

   3. Memory Chart (Normalization: per_kernel)
                                                                                                                                                  ╭────────────────────╮
                                                                                                                                                  │ xGMI (to Peer GPU) │
                                                                                                                                                  ╰────────────────────╯
                                                                                                                                                     ||  Read BW    0.000 GB/s
                                                                                                                                                     ||  Write BW   0.000 GB/s
                                                                                                                                                     ||  Atomic BW  0.000 GB/s
   |----------------------------------------------------------------- GPU (XCD) ------------------------------------------------------------------|---------------------------------------------------------- Fabric / Memory -----------------------------------------------------------|

   ╭─────── Compute Units ────────╮                        ╭─────────── vL1D ───────────╮           ╭──────────── L2 ────────────╮                ╭─────── Data Fabric ────────╮                ╭─────────── MALL ───────────╮╭─────────── UMC ────────────╮╭─────────── HBM ────────────╮
   │ Wavefront Occupancy 0.0%     │Non-buffer Request      │ Hit 42.0%                  │Read BW    │ Hit 39.4%                  │                │                            │                │                            ││                            ││ Read BW 1.107 GB/s         │
   │ ░░░░░░░░░░                   │Read   :        6 instr │ ████░░░░░░                 │0.048 GB/s │ ███░░░░░░░                 │                │                            │                │                            ││                            ││                            │
   │ VGPRs 13                     │<---------------------- │                            │<--------- │                            │                │                            │                │                            ││                            ││ Write BW 0.008 GB/s        │
   │ SGPRs 28                     │Write  :        6 instr │ Coalesce 89.3%             │           │ [!] L2 low hit rate 39.4%  │                │                            │                │                            ││                            ││                            │
   │ AGPRs 1                      │----------------------> │ ████████░░                 │Write BW   │ ███░░░░░░░                 │                │                            │                │                            ││                            ││ Atomic BW 0.000 GB/s       │
   │ Scratch Allocation 0 B       │Atomic :        0 instr │                            │0.143 GB/s │                            │                │                            │                │                            ││                            ││                            │
   │ LDS Allocation 0 B           │<---------------------> │                            │---------> │                            │                │                            │                │                            ││                            ││                            │
   │ Workgroups per CU N/A        │Buffer Request          │                            │           │                            │                │                            │                │                            ││                            ││                            │
   │                              │Read   :        0 instr │                            │Atomic BW  │                            │                │                            │                │                            ││                            ││                            │
   │                              │<---------------------- │                            │0.000 GB/s │                            │                │                            │                │                            ││                            ││                            │
   │                              │Write  :        0 instr │                            │<--------> │                            │                │                            │                │                            ││                            ││                            │
   │                              │----------------------> │                            │           │                            │                │                            │                │                            ││                            ││                            │
   │                              │Atomic :        0 instr │                            │           │                            │                │                            │                │                            ││                            ││                            │
   │                              │<---------------------> │                            │           │                            │Read BW         │                            │Read BW         │                            ││                            ││                            │
   │                              │                        ╰────────────────────────────╯           │                            │1.441 GB/s      │                            │1.107 GB/s      │                            ││                            ││                            │
   │                              │                        ╭─────────── LDS ────────────╮           │                            │<-------------- │                            │<-------------- │                            ││                            ││                            │
   │                              │Read   :        0 instr │ Util 0.0%                  │           │                            │                │                            │                │                            ││                            ││                            │
   │                              │<---------------------- │ ░░░░░░░░░░                 │           │                            │Write/Atomic BW │                            │Write/Atomic BW │                            ││                            ││                            │
   │                              │Write  :        0 instr │                            │           │                            │0.012 GB/s      │                            │0.009 GB/s      │                            ││                            ││                            │
   │                              │----------------------> │                            │           │                            │--------------> │                            │--------------> │                            ││                            ││                            │
   │                              │Atomic :        0 instr │                            │           │                            │                │                            │                │                            ││                            ││                            │
   │                              │<---------------------> │                            │           │                            │                │                            │                │                            ││                            ││                            │
   │                              │Instr  :        0 instr │                            │           │                            │                │                            │                │                            ││                            ││                            │
   │                              │<---------------------> │                            │           │                            │                │                            │                │                            ││                            ││                            │
   │                              │                        ╰────────────────────────────╯           │                            │                │                            │                │                            ││                            ││                            │
   │                              │                        ╭─────────── sL1D ───────────╮           │                            │                │                            │                │                            ││                            ││                            │
   │                              │SMEM                    │ Hit 50.0%                  │Read BW    │                            │                │                            │                │                            ││                            ││                            │
   │                              │Read   :          1 req │ █████░░░░░                 │0.006 GB/s │                            │                │                            │                │                            ││                            ││                            │
   │                              │<---------------------- │                            │<--------- │                            │                │                            │                │                            ││                            ││                            │
   │                              │                        ╰────────────────────────────╯           │                            │                │                            │                │                            ││                            ││                            │
   │                              │                        ╭─────────── L1I ────────────╮           │                            │                │                            │                │                            ││                            ││                            │
   │                              │ICACHE                  │ Hit 86.6%                  │Read BW    │                            │                │                            │                │                            ││                            ││                            │
   │                              │Read   :         54 req │ ████████░░                 │0.487 GB/s │                            │                │                            │                │                            ││                            ││                            │
   │                              │<---------------------- │                            │<--------- │                            │                │                            │                │                            ││                            ││                            │
   ╰──────────────────────────────╯                        ╰────────────────────────────╯           ╰────────────────────────────╯                ╰────────────────────────────╯                ╰────────────────────────────╯╰────────────────────────────╯╰────────────────────────────╯

                                                                                                                                                     ||  Read BW    0.024 GB/s
                                                                                                                                                     ||  Write BW   0.003 GB/s
                                                                                                                                                     ||  Atomic BW  0.000 GB/s
                                                                                                                                                  ╭─────────────────────────────────────────╮
                                                                                                                                                  │ PCIe (to CPU or Non-xGMI connected GPU) │
                                                                                                                                                  ╰─────────────────────────────────────────╯

   Legend: <---- Read  ----> Write  <---> Atomic  █ Util %  █ Hit %  █ Stall %

   ╭───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── Memory Bandwidth Guided Analysis ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
   │                                                                                                                                                                                                                                                                                     │
   │  [GL2] Low L2 cache efficiency                                                                                                                                                                                                                                                      │
   │    Condition : L2 hit rate <= 50%                                                                                                                                                                                                                                                   │
   │    Measured  : 39.4% (threshold: 50%)                                                                                                                                                                                                                                               │
   │    Impact    : Workload exceeds L2 cache capacity -- most accesses miss to HBM                                                                                                                                                                                                      │
   │                                                                                                                                                                                                                                                                                     │
   ╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

Reading the output
==================

When active bottlenecks are found, the output includes:

Stall annotations on the memory chart
--------------------------------------

Active bottlenecks appear as ``[!]`` rows inside the affected cache
panel, showing the metric label and its measured value, with a bar. A
"Stall %" entry is added to the chart legend.

In the example above, the L2 panel shows ``[!] L2 low hit rate 39.4%``
because the L2 cache hit rate fell below the 50% threshold.

Guidance panel
--------------

A guidance panel appears below the memory chart. Each entry describes
one bottleneck:

* **Condition**: What was checked (for example,
  "L2 hit rate <= 50%")
* **Measured**: The actual value from the profiled workload and the
  threshold it was compared against
* **Impact**: A brief explanation of what this stall means for your
  workload

In the example above, the guidance panel reports:

.. code-block:: text

   [GL2] Low L2 cache efficiency
     Condition : L2 hit rate <= 50%
     Measured  : 39.4% (threshold: 50%)
     Impact    : Workload exceeds L2 cache capacity -- most accesses miss to HBM

When bottlenecks are not found, a single status line is shown instead
(for example, "Memory Bandwidth Analysis: No bottlenecks detected").

Understanding the results
=========================

The analysis checks three levels of the memory hierarchy:

* **GL1 (L1 cache)**: stall sources within the L1 cache, address
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

* **Block 30 raw metrics**: run ``-b 30`` to see the metrics the guided
  analysis evaluates (L1 cache stalls, L2 bottleneck indicators, EA
  bottleneck indicators).
* **Memory chart**: the :doc:`CLI analysis documentation </how-to/analyze/cli>`
  covers the memory chart layout in detail.
* **CDNA performance model**: the :doc:`L2 cache
  </conceptual/cdna/l2-cache>` and :doc:`Vector L1 cache
  </conceptual/cdna/vector-l1-cache>` conceptual pages describe the
  cache hierarchy and how data moves through it.
* **Design document**: `hld-membw-guided-analysis-in-memchart.md <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-compute/docs/design/hld-membw-guided-analysis-in-memchart.md>`_
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

   * The analysis evaluates per-dispatch averages. Bottlenecks that occur in only a subset of dispatches may not be visible.
   * Guidance output is capped at 5 blocks. Additional bottlenecks beyond the cap are noted but not expanded.
   * When Block 30 counters are present but incomplete, the guidance panel shows "Unavailable", "Partial data", or "Inconclusive" status lines instead of bottleneck details. When Block 30 is excluded by a ``-b`` filter, the guidance panel is omitted entirely.
