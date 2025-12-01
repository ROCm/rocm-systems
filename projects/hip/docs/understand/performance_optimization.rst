.. meta::
  :description: This chapter describes performance optimization concepts and metrics for AMD GPUs
  :keywords: AMD, ROCm, HIP, performance, optimization, roofline, occupancy, bandwidth, arithmetic intensity

.. _performance_optimization:

*******************************************************************************
Performance optimization
*******************************************************************************

This chapter describes key performance concepts and optimization strategies for 
AMD GPUs. Understanding these concepts is essential for achieving optimal 
performance in GPU applications.

Performance bottlenecks
=======================

A performance bottleneck is the limiting factor that prevents a GPU kernel from 
achieving higher performance. Identifying and addressing bottlenecks is crucial 
for optimization. The two primary categories of bottlenecks are:

* **Compute-bound**: The kernel is limited by arithmetic throughput
* **Memory-bound**: The kernel is limited by memory bandwidth

Understanding which bottleneck affects your kernel helps determine the appropriate 
optimization strategy.

.. _roofline_model:

Roofline model
==============

The roofline model is a visual performance analysis framework that relates 
achievable performance to hardware limits based on arithmetic intensity. It helps 
identify whether a kernel is compute-bound or memory-bound.

The model plots performance (FLOPS) against arithmetic intensity (FLOPS/byte) 
with two limiting factors:

1. **Memory bandwidth ceiling**: A sloped line representing peak memory bandwidth
2. **Compute ceiling**: A horizontal line representing peak arithmetic throughput

The intersection point determines the transition between memory-bound and 
compute-bound regions. Kernels below and to the left of the intersection are 
memory-bound, while those to the right are compute-bound.

Key characteristics:

* The roofline creates an upper bound on achievable performance
* Real applications typically achieve a significant portion of the theoretical limit
* The model helps guide optimization efforts based on kernel characteristics

.. _compute_bound:

Compute-bound kernels
=====================

A kernel is compute-bound when its performance is limited by the GPU's arithmetic 
throughput rather than memory bandwidth. These kernels have high arithmetic 
intensity and spend most cycles executing arithmetic operations.

Characteristics of compute-bound kernels:

* High ratio of arithmetic operations to memory accesses
* Performance scales with GPU compute capacity
* Limited benefit from memory optimization
* Can often achieve a high percentage of peak theoretical FLOPS

Optimization strategies:

* Increase occupancy to hide arithmetic latency
* Use specialized units (matrix cores, SFUs) when applicable
* Optimize instruction mix and scheduling
* Consider mixed-precision computation for higher throughput

.. _memory_bound:

Memory-bound kernels
====================

A kernel is memory-bound when its performance is limited by memory bandwidth 
rather than compute capacity. These kernels have low arithmetic intensity and 
spend significant time waiting for memory operations.

Characteristics of memory-bound kernels:

* Low ratio of arithmetic operations to memory accesses
* Performance scales with memory bandwidth
* Sensitive to memory access patterns
* Typically achieve lower percentage of peak FLOPS

Optimization strategies:

* Improve memory coalescing to reduce transactions
* Use shared memory (LDS) for data reuse
* Optimize data layout for access patterns
* Reduce memory traffic through compression or precision reduction

.. _arithmetic_intensity:

Arithmetic intensity
====================

Arithmetic intensity is the ratio of floating-point operations (FLOPs) to memory 
traffic (bytes) for a given kernel or algorithm. It determines whether a kernel 
is compute-bound or memory-bound.

.. math::

   \text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Transferred}}

Key points:

* Higher arithmetic intensity indicates more computation per byte transferred
* The balance point depends on the GPU's compute-to-bandwidth ratio
* Can be calculated theoretically or measured empirically
* Different precision types affect both FLOPs and bytes

For modern AMD GPUs:

* The compute-to-bandwidth ratio varies by GPU generation
* Higher-end models have higher ratios
* Kernels above the GPU's specific ratio are compute-bound

.. _latency_hiding:

Latency hiding
==============

GPUs hide memory and instruction latency through massive hardware multithreading 
rather than complex CPU techniques like out-of-order execution. This is achieved 
by rapidly switching between wavefronts when one stalls.

Mechanisms for latency hiding:

* **Wavefront switching**: Context switches occur every cycle with zero overhead
* **Multiple wavefronts per CU**: Many concurrent wavefronts supported
* **Instruction-level parallelism**: Multiple independent instructions in flight

Requirements for effective latency hiding:

* Sufficient occupancy (active wavefronts)
* Independent instructions to overlap
* Balanced resource usage
* Minimal divergence

The hardware can completely hide memory latency if there are enough active 
wavefronts with independent work. The number of instructions needed from other 
wavefronts to hide latency depends on the specific memory latency and instruction 
throughput characteristics of the GPU.

.. _wavefront_execution:

Wavefront execution states
==========================

Understanding wavefront states helps optimize GPU utilization. A wavefront can 
be in one of several states:

* **Active**: Currently executing on a SIMD unit
* **Ready**: Eligible for execution, waiting for scheduling
* **Stalled**: Waiting for a dependency (memory, synchronization)
* **Sleeping**: Blocked on a barrier or synchronization primitive

Key metrics:

* **Active cycles**: Percentage of cycles with at least one instruction executing
* **Stall cycles**: Percentage of cycles waiting for resources
* **Idle cycles**: No wavefronts available to execute

Optimization goals:

* Maximize active cycles
* Minimize stall cycles through latency hiding
* Reduce idle cycles by increasing occupancy

.. _occupancy:

Occupancy
=========

Occupancy measures the ratio of active wavefronts to the maximum possible 
wavefronts on a compute unit. Higher occupancy generally improves latency hiding 
but is limited by resource constraints.

.. math::

   \text{Occupancy} = \frac{\text{Active Wavefronts}}{\text{Max Wavefronts per CU}}

Limiting factors:

* **Register usage**: VGPRs and SGPRs per thread
* **Shared memory (LDS)**: Allocation per block
* **Wavefront slots**: Hardware limit on concurrent wavefronts
* **Block size**: Small blocks may waste resources

Trade-offs:

* Higher occupancy improves latency hiding
* Lower occupancy allows more resources per thread
* Optimal occupancy depends on kernel characteristics
* Memory-bound kernels benefit more from high occupancy

Tools like ``rocprofv3`` can measure achieved occupancy and identify limiting 
factors.

.. _memory_optimization:

Memory access optimization
==========================

Efficient memory access patterns are crucial for GPU performance. Key 
optimization techniques include:

Memory coalescing
-----------------

Memory coalescing combines memory accesses from multiple threads into fewer 
transactions. When consecutive threads access consecutive memory addresses, the 
hardware can merge requests into efficient cache line accesses.

**Coalesced access pattern**:

* Consecutive threads access consecutive memory addresses
* Results in minimal cache line requests (optimal)
* Can achieve a high percentage of peak bandwidth

**Non-coalesced pattern**:

* Threads access random or strided addresses
* Results in many separate memory transactions
* May achieve only a small fraction of peak bandwidth

Best practices:

* Ensure consecutive threads access consecutive addresses
* Use structure-of-arrays rather than array-of-structures
* Align data to cache line boundaries
* Consider padding to avoid conflicts

.. _bank_conflicts:

Bank conflicts
--------------

Shared memory (LDS) is organized into banks that can be accessed independently. 
Bank conflicts occur when multiple threads access different addresses in the 
same bank, causing serialization.

LDS organization:

* Multiple memory banks of fixed width
* Banks can be accessed independently each cycle
* Conflicts serialize accesses, reducing throughput

Common patterns:

* **No conflict**: Each thread accesses a different bank
* **Broadcast**: Multiple threads read the same address (no conflict)
* **Two-way conflict**: Two threads access the same bank (significant slowdown)
* **N-way conflict**: N threads access the same bank (proportional slowdown)

Avoiding conflicts:

* Pad arrays to avoid power-of-two strides
* Use odd strides when possible
* Reorganize data layout
* Use different indexing schemes

.. _register_pressure:

Register pressure
=================

Register pressure occurs when a kernel requires more registers than optimal for 
the target occupancy. This can limit the number of concurrent wavefronts and 
reduce performance.

Effects of high register pressure:

* Reduced occupancy due to register limitations
* Potential register spilling to memory
* Decreased ability to hide latency
* Lower overall throughput

Management strategies:

* Minimize live variables
* Recompute values instead of storing
* Use shared memory for temporary storage
* Split complex kernels
* Adjust launch bounds to guide compiler

The compiler reports register usage, and tools like ``rocm-smi`` can help 
analyze register-limited occupancy.

.. _performance_metrics:

Key performance metrics
=======================

Understanding performance metrics helps identify optimization opportunities:

Peak rate
---------

The theoretical maximum performance of a GPU, typically measured in FLOPS or 
bandwidth:

* **Peak FLOPS**: Maximum floating-point operations per second
* **Peak bandwidth**: Maximum memory throughput
* **Peak instruction rate**: Maximum instructions per cycle

Actual performance is always below peak due to various inefficiencies.

Pipe utilization
----------------

The percentage of execution cycles where the pipeline is actively processing 
instructions. Low utilization indicates stalls or insufficient work.

Issue efficiency
----------------

The ratio of issued instructions to the maximum possible. Low efficiency can 
indicate:

* Instruction cache misses
* Scheduling inefficiencies
* Resource conflicts

CU utilization
--------------

The percentage of compute units actively executing work. Low utilization suggests:

* Insufficient parallelism
* Load imbalance
* Synchronization overhead

Branch efficiency
-----------------

The ratio of non-divergent to total branches. Low efficiency indicates significant 
divergence overhead.

.. _optimization_workflow:

Optimization workflow
=====================

A systematic approach to GPU optimization:

1. **Profile and measure**: Use tools like ``rocprofv3`` to identify bottlenecks
2. **Analyze metrics**: Determine if kernel is compute or memory bound
3. **Apply optimizations**: Target the identified bottleneck
4. **Verify improvements**: Re-profile to confirm gains
5. **Iterate**: Repeat until performance goals are met

Common optimization priorities:

1. Ensure correct algorithm implementation
2. Optimize memory access patterns
3. Improve occupancy if latency-limited
4. Reduce divergence
5. Use specialized hardware features
6. Fine-tune resource usage

Summary
=======

Effective GPU optimization requires understanding both hardware capabilities and 
kernel characteristics. Key concepts include:

* The roofline model for classifying bottlenecks
* Arithmetic intensity as a performance predictor
* Occupancy and latency hiding strategies
* Memory optimization techniques
* Performance metrics for analysis

Success comes from systematically identifying and addressing the limiting factors 
in your specific kernels. For additional optimization guidance, see 
:doc:`../how-to/performance_guidelines` and the ROCm profiling tools documentation.
