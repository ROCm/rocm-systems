.. meta::
   :description: ROCm Systems Profiler metrics glossary and interpretation guide
   :keywords: rocprof-sys, rocprofiler-systems, ROCm, metrics, glossary, performance, counters, CPU, GPU

*****************
Metrics Glossary
*****************

This glossary explains the key performance metrics collected by ROCm Systems Profiler
and how to interpret them for performance analysis. For hardware-specific counter details,
see the :doc:`Hardware Counters Reference <hardware-counters-reference>`.

.. contents:: Table of Contents
   :local:
   :backlinks: none
   :depth: 2

=======================
Understanding Metrics
=======================

Metrics are measurements that help you understand how your application uses system resources.
ROCm Systems Profiler collects metrics at different levels:

- **Function-level**: Time spent in specific functions
- **Thread-level**: Per-thread resource usage
- **Process-level**: Overall application resource usage
- **System-level**: Hardware utilization and environmental conditions

Reading Metric Tables
======================

When you view profiling output, you'll see tables with these common columns:

- **Count**: Number of times a function was called or a measurement was taken
- **Sum**: Total accumulated value (e.g., total time spent)
- **Mean**: Average value per occurrence
- **Min/Max**: Minimum and maximum observed values
- **Stddev**: Standard deviation (how much values vary)
- **% of Total**: Percentage of the total runtime/value

**What "Good" Looks Like:**

- Low stddev indicates consistent performance
- High % of total identifies hotspots (areas to optimize)
- High count with low individual time suggests inlining opportunities

==============
Timing Metrics
==============

Wall Time
=========

**What it is:** Real-world clock time elapsed

**Units:** Seconds, milliseconds, or microseconds

**When to use it:**

- Measuring end-to-end application runtime
- Understanding user-perceived performance
- Identifying I/O-bound operations

**Interpretation:**

- High wall time with low CPU time → Application is waiting (I/O, locks, GPU)
- Wall time ≈ CPU time → CPU-bound computation
- Wall time < CPU time (multi-threaded) → Parallel execution

.. code-block:: text

   Example output:
   |------------|-------|---------|----------|
   | FUNCTION   | COUNT | SUM(s)  | % TOTAL  |
   |------------|-------|---------|----------|
   | main       | 1     | 10.542  | 100.0%   |
   | compute    | 1000  | 8.234   | 78.1%    |  ← Hotspot!
   | io_write   | 100   | 1.892   | 17.9%    |
   |------------|-------|---------|----------|

CPU Time
========

**What it is:** Time the CPU was actively executing code (excludes waiting)

**Units:** Seconds, milliseconds, or microseconds

**When to use it:**

- Identifying CPU-intensive functions
- Measuring actual computation cost
- Comparing algorithm efficiency

**Interpretation:**

- High CPU time → Function is compute-intensive
- Low CPU utilization (CPU time / wall time) → Optimization opportunity
- Per-thread CPU time helps identify load imbalance

**Related metrics:**

- **User CPU time**: Time in user-space code
- **System CPU time**: Time in kernel/system calls

GPU Time
========

**What it is:** Time spent executing on GPU (kernel execution)

**Units:** Milliseconds or microseconds

**When to use it:**

- Profiling HIP/GPU-accelerated applications
- Measuring kernel efficiency
- Identifying GPU bottlenecks

**Interpretation:**

- Long GPU kernel time → GPU-bound application
- Short GPU kernels with many launches → Launch overhead may dominate
- GPU time < CPU time → Check for data transfer bottlenecks

===============
Memory Metrics
===============

Peak Memory Usage (High-Water Mark)
====================================

**What it is:** Maximum memory used by the application at any point

**Units:** Bytes, KB, MB, or GB

**When to use it:**

- Ensuring application fits in available RAM
- Identifying memory leaks
- Optimizing memory footprint

**Interpretation:**

- Gradually increasing → Potential memory leak
- Sudden spikes → Large allocations (check if necessary)
- Near system limit → Risk of OOM (Out Of Memory) errors

**What "Good" looks like:**

- Stable or predictable memory usage
- Memory usage proportional to problem size
- No unexpected growth over time

Page Faults
===========

**What it is:** Number of times the OS had to load memory pages from disk

**Types:**

- **Minor faults**: Page was in memory but not in process's page table
- **Major faults**: Page had to be loaded from disk (expensive!)

**When to use it:**

- Diagnosing slow application startup
- Identifying memory access patterns
- Detecting when working set exceeds physical RAM

**Interpretation:**

- High major faults → Not enough RAM or poor memory access patterns
- Many minor faults at startup → Normal (loading code)
- Continuous major faults → Thrashing (add RAM or reduce working set)

Cache Misses
============

**What it is:** Number of times data wasn't found in CPU cache

**Levels:**

- **L1 cache miss**: Data not in fastest cache (~1ns access)
- **L2 cache miss**: Data not in mid-level cache (~10ns access)
- **L3 cache miss**: Data not in last-level cache (~40ns access)
- **Main memory access**: ~100ns access time

**When to use it:**

- Optimizing memory access patterns
- Improving data locality
- Tuning algorithms for cache efficiency

**Interpretation:**

- High cache miss rate → Poor data locality
- Sequential access patterns → Better cache performance
- Random access patterns → More cache misses

**What "Good" looks like:**

- L1 miss rate < 5%
- L2 miss rate < 1%
- Consistent cache behavior across runs

GPU Memory Usage
================

**What it is:** Memory allocated on GPU device

**Units:** MB or GB

**When to use it:**

- Ensuring kernels fit in GPU memory
- Optimizing data transfers
- Managing multiple GPU workloads

**Interpretation:**

- Near device limit → May cause allocation failures
- Frequent allocations → Pool or pre-allocate memory
- High memory with low utilization → Opportunity to reduce

===========
CPU Metrics
===========

CPU Utilization
===============

**What it is:** Percentage of time CPU cores are active

**Range:** 0-100% per core (can exceed 100% for multi-core)

**When to use it:**

- Identifying underutilized resources
- Measuring parallelization effectiveness
- Detecting load imbalance

**Interpretation:**

- Low utilization (< 50%) → Application may be I/O bound or poorly parallelized
- High utilization (> 90%) → CPU-bound, good for compute workloads
- Uneven utilization across cores → Load imbalance

**What "Good" looks like:**

- Compute-intensive apps: 80-100% utilization
- I/O-intensive apps: Lower utilization is expected
- Parallel apps: Even utilization across cores

Instructions Per Cycle (IPC)
=============================

**What it is:** Average number of instructions executed per CPU clock cycle

**Range:** 0-4 (typically) depending on CPU architecture

**When to use it:**

- Measuring CPU efficiency
- Identifying pipeline stalls
- Comparing algorithm implementations

**Interpretation:**

- Low IPC (< 1.0) → CPU pipeline stalls (cache misses, branch mispredictions)
- Medium IPC (1.0-2.0) → Normal for most code
- High IPC (> 2.5) → Well-optimized, vectorized code

**How to improve:**

- Reduce branch mispredictions
- Improve data locality
- Use SIMD/vectorization
- Minimize dependencies between instructions

Branch Mispredictions
=====================

**What it is:** Number of times CPU incorrectly predicted branch direction

**When to use it:**

- Optimizing conditional code
- Understanding pipeline efficiency
- Tuning hot loops

**Interpretation:**

- High misprediction rate → Unpredictable branches hurt performance
- Predictable patterns → Good branch prediction
- Inner loop branches → Critical to optimize

**What "Good" looks like:**

- Misprediction rate < 5%
- Consistent branch patterns
- Minimal branches in hot paths

Cache Hit Rate
==============

**What it is:** Percentage of memory accesses satisfied by cache

**Range:** 0-100%

**When to use it:**

- Measuring memory system efficiency
- Optimizing data structures
- Tuning blocking factors for algorithms

**Interpretation:**

- L1 hit rate > 95% → Excellent data locality
- L2 hit rate > 90% → Good memory access patterns
- LLC (L3) hit rate > 80% → Acceptable working set size

===========
GPU Metrics
===========

GPU Utilization
===============

**What it is:** Percentage of time GPU is actively executing kernels

**Range:** 0-100%

**When to use it:**

- Identifying GPU underutilization
- Measuring kernel efficiency
- Detecting CPU-GPU synchronization issues

**Interpretation:**

- Low GPU utilization (< 30%) → CPU bound or launch overhead
- Medium utilization (30-70%) → Check for data transfer bottlenecks
- High utilization (> 80%) → GPU-bound, good for GPU workloads

**Common issues:**

- Many small kernels → Launch overhead
- Frequent CPU-GPU sync → Use async operations
- Small workloads → Increase batch size

Memory Bandwidth
================

**What it is:** Rate of data transfer to/from GPU memory

**Units:** GB/s

**When to use it:**

- Measuring memory-bound kernels
- Optimizing data transfers
- Understanding kernel efficiency

**Interpretation:**

- Low bandwidth utilization → Kernel is compute-bound or has poor memory patterns
- High bandwidth utilization → Memory-bound kernel
- Approaching peak bandwidth → Well-optimized memory access

**GPU-specific considerations:**

- HBM (MI200/MI300): ~1.6-2.0 TB/s peak
- GDDR (consumer GPUs): ~500-900 GB/s peak

Kernel Occupancy
================

**What it is:** Ratio of active wavefronts to maximum possible wavefronts

**Range:** 0-100%

**When to use it:**

- Optimizing kernel launch parameters
- Understanding resource constraints
- Tuning workgroup sizes

**Interpretation:**

- Low occupancy (< 25%) → Resource constraints (registers, LDS)
- Medium occupancy (25-50%) → May be optimal for some kernels
- High occupancy (> 75%) → Good resource utilization

**Important:** High occupancy doesn't always mean better performance. Balance with:

- Memory bandwidth utilization
- Instruction throughput
- Register pressure

Wavefront/Warp Execution
=========================

**What it is:** Number of SIMD execution units active

**Key concepts:**

- **Wavefront**: Group of 64 threads (AMD) executing in lockstep
- **Warp**: Group of 32 threads (NVIDIA)
- **Divergence**: When threads in a wavefront take different paths

**When to use it:**

- Identifying SIMD efficiency issues
- Optimizing branch divergence
- Understanding kernel parallelism

**Interpretation:**

- High divergence → Poor SIMD efficiency
- Low active wavefronts → Occupancy issues
- Uneven workload → Load imbalance

==============
System Metrics
==============

Temperature
===========

**What it is:** Hardware component temperature

**Units:** Degrees Celsius

**When to use it:**

- Monitoring thermal throttling
- Ensuring system stability
- Planning cooling requirements

**Interpretation:**

- Rising temperature → May lead to throttling
- Thermal throttling → Performance degradation
- High baseline temperature → Poor cooling

**Safe ranges (typical):**

- CPU: < 85°C under load
- GPU: < 95°C under load (depends on model)

Power Usage
===========

**What it is:** Electrical power consumed by components

**Units:** Watts (W)

**When to use it:**

- Measuring energy efficiency
- Planning power budgets
- Optimizing for power-constrained systems

**Interpretation:**

- High power with low utilization → Inefficient code
- Power proportional to utilization → Expected behavior
- Power spikes → Check for inefficient operations

Network I/O
===========

**What it is:** Data sent/received over network

**Units:** Bytes/s or MB/s

**When to use it:**

- Profiling distributed applications
- Identifying communication bottlenecks
- Optimizing MPI applications

**Interpretation:**

- High network I/O → Communication-bound application
- Bursty network traffic → Poor overlap of computation/communication
- Low bandwidth utilization → Consider batching messages

Disk I/O
========

**What it is:** Data read/written to storage

**Units:** Bytes/s or MB/s

**When to use it:**

- Identifying I/O-bound applications
- Optimizing file access patterns
- Reducing I/O overhead

**Interpretation:**

- High I/O wait time → I/O-bound application
- Sequential I/O → Faster than random I/O
- Many small I/O operations → Consider buffering

===============================
Common Metric Relationships
===============================

Understanding how metrics relate to each other helps identify bottlenecks:

CPU-Bound Application
======================

**Indicators:**

- High CPU utilization (> 80%)
- CPU time ≈ Wall time
- Low I/O wait time
- High IPC or instruction count

**Optimization strategies:**

- Algorithmic improvements
- Vectorization/SIMD
- Better cache utilization
- Parallelization

Memory-Bound Application
=========================

**Indicators:**

- High cache miss rates
- Low IPC (< 1.0)
- CPU utilization varies
- High memory bandwidth usage

**Optimization strategies:**

- Improve data locality
- Cache blocking
- Prefetching
- Reduce memory allocations

I/O-Bound Application
=====================

**Indicators:**

- Low CPU utilization (< 30%)
- High I/O wait time
- Wall time >> CPU time
- High disk/network I/O

**Optimization strategies:**

- Async I/O
- Buffering
- Reduce I/O operations
- Use faster storage

GPU-Bound Application
======================

**Indicators:**

- High GPU utilization
- CPU waiting for GPU
- High GPU memory bandwidth
- Long kernel execution times

**Optimization strategies:**

- Kernel optimization
- Reduce memory transfers
- Overlap computation/transfers
- Increase arithmetic intensity

====================================
Interpreting Metric Combinations
====================================

Low CPU + Low GPU = Synchronization Issues
===========================================

**What to check:**

- Excessive CPU-GPU synchronization
- Small kernel launches
- Inefficient data transfers

High Memory Usage + Many Page Faults
======================================

**What to check:**

- Working set exceeds physical RAM
- Memory leaks
- Excessive allocations

Low IPC + High Cache Misses
============================

**What to check:**

- Poor data locality
- Random memory access
- Working set exceeds cache size

High Function Count + Low Individual Time
==========================================

**What to check:**

- Function call overhead
- Opportunities for inlining
- Unnecessary abstraction

==================
Additional Resources
==================

- :doc:`Hardware Counters Reference <hardware-counters-reference>` - Detailed hardware counter descriptions
- :doc:`ROCm Systems Profiler Glossary <rocprof-sys-glossary>` - General terminology
- :doc:`Understanding Output <../how-to/understanding-rocprof-sys-output>` - How to read profiling reports
- :doc:`Configuring Runtime Options <../how-to/configuring-runtime-options>` - Enabling specific metrics
- `AMD GPU Performance Counters <https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html>`_ - GPU-specific details

