.. meta::
   :description: ROCm Systems Profiler hardware counters reference
   :keywords: rocprof-sys, rocprofiler-systems, ROCm, hardware counters, PAPI, performance, CPU, GPU, MI200, MI300

*****************************
Hardware Counters Reference
*****************************

This reference provides detailed information about hardware performance counters available in
ROCm Systems Profiler. For user-friendly metric interpretations, see the :doc:`Metrics Glossary <metrics-glossary>`.

.. contents:: Table of Contents
   :local:
   :backlinks: none
   :depth: 2

.. warning::
   Collecting hardware counters adds overhead to profiling. Be selective about which counters you enable,
   and be aware that not all counters can be collected simultaneously.

==================================
Overview of Hardware Counters
==================================

Hardware performance counters are special-purpose registers built into CPUs and GPUs that count
specific hardware events. They provide low-level insights into how hardware resources are utilized.

Types of Counters
=================

**Basic Counters (Raw Counters)**

- Direct hardware event counts
- Examples: cache misses, branch mispredictions, instructions executed
- Can be collected with minimal interpretation
- Architecture-specific

**Derived Metrics**

- Calculated from basic counters using formulas
- Examples: IPC (instructions/cycles), cache hit rate, bandwidth utilization
- More intuitive but require multiple counter reads
- May have higher overhead

Discovering Available Counters
===============================

To see all hardware counters available on your system:

.. code-block:: bash

   # List all hardware counters with descriptions
   rocprof-sys-avail --hw-counters --description

   # CPU counters only (PAPI)
   rocprof-sys-avail --hw-counters -c CPU --description

   # GPU counters only (ROCm)
   rocprof-sys-avail --hw-counters -c GPU --description

   # Brief list without descriptions
   rocprof-sys-avail -H -b

====================
CPU Hardware Counters
====================

ROCm Systems Profiler uses PAPI (Performance API) to access CPU hardware counters.
PAPI provides a portable interface to CPU performance counters across different architectures.

Enabling CPU Counters
=====================

CPU hardware counters are configured using ``ROCPROFSYS_PAPI_EVENTS``:

.. code-block:: bash

   # Using PAPI identifiers
   export ROCPROFSYS_PAPI_EVENTS="PAPI_TOT_CYC PAPI_TOT_INS PAPI_L1_DCM"

   # Using perf identifiers (Linux-specific)
   export ROCPROFSYS_PAPI_EVENTS="perf::INSTRUCTIONS perf::CACHE-REFERENCES perf::CACHE-MISSES"

.. note::
   To collect most hardware counters via PAPI, ensure ``/proc/sys/kernel/perf_event_paranoid``
   has a value <= 2. With sudo access:

   .. code-block:: bash

      sudo sysctl -w kernel.perf_event_paranoid=2

Common CPU Counters
===================

Cache Counters
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Counter
     - Description
   * - ``PAPI_L1_DCM``
     - Level 1 data cache misses
   * - ``PAPI_L1_ICM``
     - Level 1 instruction cache misses
   * - ``PAPI_L1_TCM``
     - Level 1 total cache misses
   * - ``PAPI_L2_DCM``
     - Level 2 data cache misses
   * - ``PAPI_L2_ICM``
     - Level 2 instruction cache misses
   * - ``PAPI_L2_TCM``
     - Level 2 total cache misses
   * - ``PAPI_L3_DCM``
     - Level 3 data cache misses
   * - ``PAPI_L3_ICM``
     - Level 3 instruction cache misses
   * - ``PAPI_L3_TCM``
     - Level 3 total cache misses
   * - ``PAPI_L1_DCA``
     - Level 1 data cache accesses
   * - ``PAPI_L2_DCA``
     - Level 2 data cache accesses
   * - ``PAPI_L3_DCA``
     - Level 3 data cache accesses

**Cache hit rate calculation:**

.. code-block:: text

   L1 hit rate = (PAPI_L1_DCA - PAPI_L1_DCM) / PAPI_L1_DCA * 100%
   L2 hit rate = (PAPI_L2_DCA - PAPI_L2_DCM) / PAPI_L2_DCA * 100%

Instruction Counters
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Counter
     - Description
   * - ``PAPI_TOT_INS``
     - Total instructions executed
   * - ``PAPI_TOT_CYC``
     - Total CPU cycles
   * - ``PAPI_FP_INS``
     - Floating point instructions
   * - ``PAPI_FP_OPS``
     - Floating point operations (may include fused ops)
   * - ``PAPI_INT_INS``
     - Integer instructions
   * - ``PAPI_VEC_INS``
     - Vector/SIMD instructions
   * - ``PAPI_LD_INS``
     - Load instructions
   * - ``PAPI_SR_INS``
     - Store instructions

**IPC calculation:**

.. code-block:: text

   IPC = PAPI_TOT_INS / PAPI_TOT_CYC

Branch Counters
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Counter
     - Description
   * - ``PAPI_BR_INS``
     - Branch instructions executed
   * - ``PAPI_BR_MSP``
     - Branch mispredictions
   * - ``PAPI_BR_TKN``
     - Conditional branches taken
   * - ``PAPI_BR_NTK``
     - Conditional branches not taken

**Branch misprediction rate:**

.. code-block:: text

   Misprediction rate = PAPI_BR_MSP / PAPI_BR_INS * 100%

TLB Counters
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Counter
     - Description
   * - ``PAPI_TLB_DM``
     - Data TLB misses
   * - ``PAPI_TLB_IM``
     - Instruction TLB misses

Memory Counters
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Counter
     - Description
   * - ``PAPI_MEM_RD``
     - Memory read operations
   * - ``PAPI_MEM_WR``
     - Memory write operations

x86-Specific Counters (via perf)
=================================

On x86 architectures, you can access additional counters through the ``perf`` interface:

Common x86 Events
-----------------

.. code-block:: bash

   # Basic events
   export ROCPROFSYS_PAPI_EVENTS="perf::INSTRUCTIONS perf::CYCLES perf::CACHE-REFERENCES perf::CACHE-MISSES"

   # Branch events
   export ROCPROFSYS_PAPI_EVENTS="perf::BRANCHES perf::BRANCH-MISSES"

   # Memory events
   export ROCPROFSYS_PAPI_EVENTS="perf::DRAM-READS perf::DRAM-WRITES"

   # Stall events
   export ROCPROFSYS_PAPI_EVENTS="perf::STALLED-CYCLES-FRONTEND perf::STALLED-CYCLES-BACKEND"

.. note::
   Available ``perf`` events vary by CPU microarchitecture. Use ``perf list`` to see all available events
   on your system.

Counter Collection Limitations
===============================

**Multiplexing**

When you request more counters than can be collected simultaneously, the kernel multiplexes them:

- Counters are collected in groups
- Each group is sampled for a time slice
- Results are scaled based on time sampled
- Accuracy decreases with more counters

**Typical limits:**

- 4-8 counters simultaneously (varies by CPU)
- Fixed-function counters (e.g., cycles, instructions) don't count against limit
- Derived metrics may require multiple counters

====================
GPU Hardware Counters
====================

GPU hardware counters provide insights into how GPU resources are utilized. ROCm Systems Profiler
uses ROCProfiler to access these counters.

Enabling GPU Counters
=====================

GPU hardware counters are configured using ``ROCPROFSYS_ROCM_EVENTS``:

.. code-block:: bash

   # Enable ROCProfiler
   export ROCPROFSYS_USE_ROCPROFILER=ON

   # Specify GPU events
   export ROCPROFSYS_ROCM_EVENTS="SQ_WAVES SQ_INSTS_VALU GPUBusy"

Alternatively, use command-line options:

.. code-block:: bash

   rocprof-sys-run --rocm-events=SQ_WAVES,SQ_INSTS_VALU -- ./app.inst

AMD GPU Architecture Overview
==============================

Understanding AMD GPU architecture helps interpret hardware counters:

**Compute Unit (CU)**

- Basic execution unit containing multiple SIMD processors
- Contains L1 cache, LDS (Local Data Share), and schedulers
- MI200: 104-110 CUs per GPU
- MI300: Up to 304 CUs per GPU

**Wavefront**

- Group of 64 work-items (threads) executing in lockstep
- SIMD execution on a single SIMD unit
- Maximum wavefronts per CU varies by architecture

**Memory Hierarchy**

- Register file (per wavefront)
- LDS (Local Data Share) - 64KB per CU
- L1 cache - 16KB per CU
- L2 cache - Shared across CUs
- HBM (High Bandwidth Memory) - Main GPU memory

Common GPU Counters (MI200/MI300)
==================================

Compute Counters
----------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Counter
     - Description
   * - ``SQ_WAVES``
     - Number of wavefronts launched
   * - ``SQ_INSTS_VALU``
     - Vector ALU instructions executed
   * - ``SQ_INSTS_SALU``
     - Scalar ALU instructions executed
   * - ``SQ_INSTS_VMEM``
     - Vector memory instructions
   * - ``SQ_INSTS_SMEM``
     - Scalar memory instructions
   * - ``SQ_INSTS_FLAT``
     - Flat memory instructions
   * - ``SQ_INSTS_LDS``
     - LDS (Local Data Share) instructions
   * - ``SQ_INSTS_GDS``
     - GDS (Global Data Share) instructions
   * - ``SQ_WAVE_CYCLES``
     - Cycles spent by wavefronts

**Arithmetic intensity:**

.. code-block:: text

   Arithmetic intensity = SQ_INSTS_VALU / (SQ_INSTS_VMEM + SQ_INSTS_FLAT)

Occupancy Counters
------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Counter
     - Description
   * - ``SQ_ACTIVE_INST_VALU``
     - Cycles VALU was active
   * - ``SQ_INST_CYCLES_VALU``
     - Cycles VALU was stalled
   * - ``SQ_THREAD_CYCLES_VALU``
     - Thread-cycles for VALU
   * - ``SQ_LDS_BANK_CONFLICT``
     - LDS bank conflicts

Memory Counters
---------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Counter
     - Description
   * - ``TCP_TA_DATA_STALL_CYCLES``
     - Cycles stalled waiting for data
   * - ``TCC_HIT``
     - L2 cache hits
   * - ``TCC_MISS``
     - L2 cache misses
   * - ``TCC_EA_RDREQ``
     - Memory read requests
   * - ``TCC_EA_WRREQ``
     - Memory write requests
   * - ``TCC_EA_RDREQ_32B``
     - 32-byte memory reads
   * - ``TCC_EA_WRREQ_32B``
     - 32-byte memory writes

**L2 cache hit rate:**

.. code-block:: text

   L2 hit rate = TCC_HIT / (TCC_HIT + TCC_MISS) * 100%

**Memory bandwidth (approximate):**

.. code-block:: text

   Bandwidth = (TCC_EA_RDREQ_32B + TCC_EA_WRREQ_32B) * 32 bytes / time

Utilization Counters
--------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Counter
     - Description
   * - ``GPUBusy``
     - Percentage of time GPU was active
   * - ``VALUBusy``
     - Percentage of time vector ALU was busy
   * - ``SALUBusy``
     - Percentage of time scalar ALU was busy
   * - ``MemUnitBusy``
     - Percentage of time memory unit was busy
   * - ``TexUnitBusy``
     - Percentage of time texture unit was busy

Branch Counters
---------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Counter
     - Description
   * - ``SQ_INSTS_BRANCH``
     - Branch instructions
   * - ``SQ_WAVES_RESTORED``
     - Wavefronts restored after branch

Architecture-Specific Counter Sets
===================================

MI200 (CDNA2) Counter Examples
-------------------------------

.. code-block:: bash

   # Memory-focused profiling
   export ROCPROFSYS_ROCM_EVENTS="TCC_HIT,TCC_MISS,TCC_EA_RDREQ,TCC_EA_WRREQ"

   # Compute-focused profiling
   export ROCPROFSYS_ROCM_EVENTS="SQ_WAVES,SQ_INSTS_VALU,SQ_ACTIVE_INST_VALU,VALUBusy"

   # Occupancy analysis
   export ROCPROFSYS_ROCM_EVENTS="SQ_WAVES,SQ_WAVE_CYCLES,SQ_LDS_BANK_CONFLICT"

MI300 (CDNA3) Counter Examples
-------------------------------

.. code-block:: bash

   # Enhanced memory profiling
   export ROCPROFSYS_ROCM_EVENTS="TCC_HIT,TCC_MISS,TCC_ATOMIC,TCC_HBM_TRAFFIC"

   # Matrix instruction profiling (MFMA)
   export ROCPROFSYS_ROCM_EVENTS="SQ_INSTS_MFMA,SQ_ACTIVE_INST_MFMA"

Counter Collection Best Practices
==================================

Selecting Counter Sets
-----------------------

**Start with derived metrics:**

.. code-block:: bash

   # High-level GPU metrics
   export ROCPROFSYS_ROCM_EVENTS="GPUBusy,VALUBusy,MemUnitBusy"

**Drill down based on findings:**

- High VALUBusy → Check compute intensity
- High MemUnitBusy → Check memory bandwidth
- Low GPUBusy → Check kernel launch overhead

Minimizing Overhead
-------------------

1. **Collect counters in passes:**

   .. code-block:: bash

      # Pass 1: Utilization
      rocprof-sys-run --rocm-events=GPUBusy,VALUBusy -- ./app

      # Pass 2: Memory (if MemUnitBusy was high)
      rocprof-sys-run --rocm-events=TCC_HIT,TCC_MISS -- ./app

2. **Use sampling instead of instrumentation when possible**

3. **Profile representative sections, not entire runs**

Interpreting Results
--------------------

**Counter values are cumulative across all kernels**

- Use per-kernel breakdowns when available
- Normalize by number of wavefronts or cycles
- Compare across different implementations

**Watch for counter overflow**

- Some counters are 32-bit or 48-bit
- Long-running kernels may overflow
- Results will appear incorrect if overflow occurs

====================
Derived Metrics
====================

Derived metrics are calculated from basic hardware counters. They provide more intuitive
performance indicators.

Common CPU Derived Metrics
===========================

.. code-block:: text

   IPC = PAPI_TOT_INS / PAPI_TOT_CYC

   L1 Cache Miss Rate = PAPI_L1_DCM / PAPI_L1_DCA * 100%

   Branch Misprediction Rate = PAPI_BR_MSP / PAPI_BR_INS * 100%

   FLOPs per Cycle = PAPI_FP_OPS / PAPI_TOT_CYC

   Memory Bandwidth = (PAPI_MEM_RD + PAPI_MEM_WR) * cache_line_size / time

Common GPU Derived Metrics
===========================

.. code-block:: text

   Average Occupancy = SQ_WAVE_CYCLES / (num_CUs * cycles)

   VALU Utilization = SQ_ACTIVE_INST_VALU / SQ_WAVE_CYCLES * 100%

   L2 Hit Rate = TCC_HIT / (TCC_HIT + TCC_MISS) * 100%

   Memory Bandwidth = (TCC_EA_RDREQ_32B + TCC_EA_WRREQ_32B) * 32 / time

   Arithmetic Intensity = VALU_instructions / memory_instructions

===================
Usage Examples
===================

Example 1: Cache Analysis
==========================

.. code-block:: bash

   # Collect cache-related counters
   export ROCPROFSYS_PAPI_EVENTS="PAPI_L1_DCM PAPI_L1_DCA PAPI_L2_DCM PAPI_L2_DCA PAPI_L3_TCM"
   rocprof-sys-sample -- ./my_app

   # Analyze results
   cat rocprof-sys-output/wall_clock.txt

Example 2: GPU Memory Bottleneck
=================================

.. code-block:: bash

   # Check if memory-bound
   export ROCPROFSYS_USE_ROCPROFILER=ON
   export ROCPROFSYS_ROCM_EVENTS="TCC_HIT,TCC_MISS,TCC_EA_RDREQ,MemUnitBusy"
   rocprof-sys-run --hip-trace -- ./gpu_app.inst

Example 3: Instruction Mix Analysis
====================================

.. code-block:: bash

   # Understand instruction distribution
   export ROCPROFSYS_PAPI_EVENTS="PAPI_TOT_INS PAPI_FP_INS PAPI_INT_INS PAPI_LD_INS PAPI_SR_INS"
   rocprof-sys-sample -- ./compute_app

Example 4: Branch Prediction Analysis
======================================

.. code-block:: bash

   # Check branch prediction efficiency
   export ROCPROFSYS_PAPI_EVENTS="PAPI_BR_INS PAPI_BR_MSP PAPI_BR_TKN PAPI_BR_NTK"
   rocprof-sys-sample -- ./branch_heavy_app

==================
Troubleshooting
==================

Counter Not Available
=====================

**Problem:** Requested counter is not available

**Solutions:**

1. Check if counter exists on your hardware:

   .. code-block:: bash

      rocprof-sys-avail -H | grep COUNTER_NAME

2. Try alternative counters with similar meaning

3. For GPU counters, verify ROCm version compatibility

Permission Denied
=================

**Problem:** Cannot access hardware counters (CPU)

**Solution:**

.. code-block:: bash

   # Check current setting
   cat /proc/sys/kernel/perf_event_paranoid

   # Allow non-root access (requires sudo)
   sudo sysctl -w kernel.perf_event_paranoid=2

   # Make permanent
   echo "kernel.perf_event_paranoid=2" | sudo tee -a /etc/sysctl.conf

Unexpected Counter Values
==========================

**Problem:** Counter values seem incorrect

**Possible causes:**

1. **Counter overflow** - Use shorter profiling runs
2. **Multiplexing** - Requesting too many counters simultaneously
3. **Sample skew** - Sampling may miss events
4. **ROCProfiler limitations** - Some counters may not work together

==================
Additional Resources
==================

- :doc:`Metrics Glossary <metrics-glossary>` - User-friendly metric interpretations
- :doc:`Configuring Runtime Options <../how-to/configuring-runtime-options>` - How to enable counters
- `PAPI Documentation <http://icl.utk.edu/papi/>`_ - Detailed PAPI counter information
- `AMD GPU Performance Counters <https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html>`_ - Official AMD GPU counter documentation
- `Linux perf Events <https://perf.wiki.kernel.org/index.php/Tutorial>`_ - Linux perf tutorial

.. note::
   Hardware counter availability and behavior may change between hardware generations and driver versions.
   Always verify counter availability on your specific system.

