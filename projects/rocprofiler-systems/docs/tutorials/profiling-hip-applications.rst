.. _profiling-hip-applications:

============================
Profiling HIP Applications
============================

This tutorial walks you through profiling GPU-accelerated HIP applications with `rocprofiler-systems`.

.. contents:: Contents
   :local:
   :depth: 2

Introduction
============

HIP (Heterogeneous-compute Interface for Portability) is AMD's GPU programming model. Profiling HIP applications requires capturing:

- **Host-side API calls** - HIP runtime invocations (memory allocation, kernel launches)
- **GPU kernel execution** - Actual GPU compute time and parallelism
- **Memory transfers** - Host-to-device and device-to-host data movement
- **GPU utilization** - Hardware efficiency metrics
- **Synchronization** - GPU-CPU coordination overhead

Prerequisites
=============

Before starting, ensure you have:

1. ROCm installed (6.0 or later recommended)
2. `rocprofiler-systems` built and installed
3. A HIP-enabled GPU (check with ``rocm-smi``)
4. Basic familiarity with HIP programming

If you need sample HIP workloads, see :doc:`../examples/hip-quickstart/README` for minimal test applications.

Quick Start: Profile a HIP Application
=======================================

The fastest way to profile a HIP application:

.. code-block:: bash

   # For AI/ML workloads (PyTorch, TensorFlow, etc.)
   rocprof-sys-sample --trace-ai -- python train.py

   # For general HIP applications
   rocprof-sys-sample --quick --hip-trace -- ./hip_app

   # For compute-intensive HIP kernels
   rocprof-sys-sample --trace-hpc -- ./compute_app

What gets collected:

- HIP API calls (``hipMalloc``, ``hipMemcpy``, ``hipLaunchKernel``)
- GPU kernel dispatch and execution times
- Memory transfer operations
- GPU utilization, temperature, power consumption
- Timeline visualization in Perfetto format

Profiling Workflow
==================

Step 1: Initial Profiling
--------------------------

Start with a quick profile to identify GPU hotspots:

.. code-block:: bash

   rocprof-sys-sample --quick --hip-trace -o ./profile1 -- ./hip_app

**What to look for in results:**

Open ``profile1/perfetto-trace.proto`` in the Perfetto UI (`ui.perfetto.dev <https://ui.perfetto.dev>`_):

1. **Long-running kernels** - Kernels consuming most GPU time
2. **Memory transfer bottlenecks** - Large ``hipMemcpy`` operations
3. **GPU idle periods** - Gaps where GPU is waiting
4. **CPU-GPU synchronization** - Excessive ``hipDeviceSynchronize`` calls

Step 2: Detailed Kernel Analysis
---------------------------------

Once you identify problem kernels, profile with hardware counters:

.. code-block:: bash

   # Profile with GPU wave execution metrics
   rocprof-sys-run --hip-trace --rocm-events=SQ_WAVES,SQ_INSTS_VALU -- ./hip_app

   # Profile with memory access patterns
   rocprof-sys-run --hip-trace --rocm-events=TCC_HIT,TCC_MISS -- ./hip_app

**Available hardware counters** (check with ``rocprof --list-metrics``):

- **Compute**: ``SQ_WAVES``, ``SQ_INSTS_VALU``, ``SQ_INSTS_SALU``
- **Memory**: ``TCC_HIT``, ``TCC_MISS``, ``TCP_TCC_READ_REQ``
- **Occupancy**: ``SQ_LDS_BANK_CONFLICT``, ``SQ_WAIT_INST_ANY``

Step 3: Optimization Iteration
-------------------------------

Profile repeatedly as you optimize:

.. code-block:: bash

   # Baseline measurement
   rocprof-sys-sample --hip-trace -o ./baseline -- ./hip_app

   # After optimization
   rocprof-sys-sample --hip-trace -o ./optimized -- ./hip_app

   # Compare results
   ls -lh baseline/perfetto-trace.proto optimized/perfetto-trace.proto

Example Workflow: Matrix Multiplication
========================================

Let's profile a real HIP application - matrix multiplication.

Build the Example
-----------------

.. code-block:: bash

   cd examples/hip-quickstart
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make matrix_multiply

Initial Profile
---------------

.. code-block:: bash

   rocprof-sys-sample --quick --hip-trace -o ./profile_matrix -- ./matrix_multiply

**Analyze the output:**

.. code-block:: bash

   # View text summary
   cat profile_matrix/wall_clock.txt | grep -A 20 "REAL-CLOCK TIMER"

   # Open trace in browser
   # Visit https://ui.perfetto.dev and load profile_matrix/perfetto-trace.proto

**What you'll see:**

1. Host functions: ``main``, memory allocation, kernel launch overhead
2. GPU kernels: ``matrixMultiply`` kernel execution time
3. Memory transfers: ``hipMemcpy`` H2D (input), D2H (output)
4. Total execution breakdown

Common Optimization Scenarios
==============================

Scenario 1: Kernel is Too Slow
-------------------------------

**Symptoms:**

- Single kernel dominates GPU time
- Low GPU utilization percentage

**Diagnosis:**

.. code-block:: bash

   # Check wave occupancy
   rocprof-sys-run --hip-trace --rocm-events=SQ_WAVES -- ./hip_app

**Possible fixes:**

- Increase block size (more threads per block)
- Reduce register usage (check with ``--ptxas-options=-v``)
- Use shared memory for data reuse

Scenario 2: Memory Transfer Bottleneck
---------------------------------------

**Symptoms:**

- ``hipMemcpy`` calls take significant time
- Large gaps between kernel executions

**Diagnosis:**

.. code-block:: bash

   # Trace memory operations
   rocprof-sys-sample --trace-ai -o ./mem_profile -- ./hip_app
   # Check Perfetto trace for hipMemcpy durations

**Possible fixes:**

- Use ``hipMemcpyAsync`` with streams
- Overlap compute and transfers
- Reduce data movement (compute on GPU)
- Use pinned memory (``hipHostMalloc``)

Scenario 3: GPU Underutilization
---------------------------------

**Symptoms:**

- GPU busy percentage < 50%
- Many small kernels with gaps

**Diagnosis:**

.. code-block:: bash

   # Check GPU metrics over time
   rocprof-sys-sample --sampling-gpus=all --trace -o ./util_profile -- ./hip_app
   # Look at AMD_SMI metrics in Perfetto

**Possible fixes:**

- Batch small operations
- Use HIP streams for concurrency
- Remove unnecessary ``hipDeviceSynchronize``
- Increase grid size for better parallelism

Scenario 4: Multi-GPU Inefficiency
-----------------------------------

**Symptoms:**

- Only one GPU showing activity
- Load imbalance across GPUs

**Diagnosis:**

.. code-block:: bash

   # Profile all GPUs
   rocprof-sys-sample --sampling-gpus=0,1,2,3 --hip-trace -o ./multi_gpu -- ./multi_gpu_app

**Possible fixes:**

- Use RCCL for efficient GPU communication
- Balance work distribution
- Overlap communication and computation
- Profile with ``--trace-ai`` for RCCL visibility

Advanced Profiling Techniques
==============================

Profiling with Binary Instrumentation
--------------------------------------

For more detailed CPU-side call stacks:

.. code-block:: bash

   # Instrument the binary
   rocprof-sys-instrument --trace-ai -o hip_app.inst -- ./hip_app

   # Run with detailed profiling
   rocprof-sys-run --hip-trace --trace -- ./hip_app.inst

**Benefits:**

- Complete CPU call stacks (not just samples)
- User-defined region timing
- Better Python framework integration

Profiling Python + HIP (PyTorch/TensorFlow)
--------------------------------------------

For AI/ML frameworks:

.. code-block:: bash

   # Profile PyTorch training
   rocprof-sys-sample --trace-ai -- python train.py

   # Profile with Python line-level tracing
   rocprof-sys-python -b -- train.py

**What gets captured:**

- Python function calls
- PyTorch operations (``torch.nn.Module.forward``)
- HIP kernel launches from framework
- RCCL collective operations (multi-GPU)
- Memory allocation patterns

Profiling MPI + HIP Applications
---------------------------------

For multi-process, multi-GPU applications:

.. code-block:: bash

   # Profile each MPI rank
   mpirun -n 4 rocprof-sys-sample --trace-hpc -o ./mpi_profile_${PMIX_RANK} -- ./mpi_hip_app

   # Or use process sampling for all ranks
   mpirun -n 4 rocprof-sys-sample --trace-hpc --mpi -- ./mpi_hip_app

**Analysis tips:**

- Check for load imbalance between ranks
- Identify MPI communication overhead
- Verify GPU utilization per rank
- Look for GPU synchronization issues

Collecting Hardware Counters
-----------------------------

For in-depth kernel analysis:

.. code-block:: bash

   # List available counters
   rocprof --list-metrics

   # Collect specific counters
   rocprof-sys-run --hip-trace --rocm-events=GRBM_GUI_ACTIVE,SQ_WAVES,SQ_INSTS_VALU -- ./hip_app

**Key counter groups:**

- **Compute utilization**: ``GRBM_GUI_ACTIVE``, ``SQ_WAVES``
- **Memory bandwidth**: ``TCC_EA_RDREQ_sum``, ``TCC_EA_WRREQ_sum``
- **Cache efficiency**: ``TCC_HIT_sum``, ``TCC_MISS_sum``
- **Occupancy**: ``SQ_LDS_BANK_CONFLICT``, ``SQ_WAVE_CYCLES``

Interpreting Results
=====================

Perfetto Trace Visualization
-----------------------------

Open ``perfetto-trace.proto`` at `ui.perfetto.dev <https://ui.perfetto.dev>`_:

**Timeline tracks:**

1. **Process tracks** - CPU threads and functions
2. **GPU tracks** - HIP API calls, kernel execution
3. **GPU hw queue** - Actual kernel dispatch timeline
4. **Counter tracks** - GPU utilization, memory, power

**Key metrics to check:**

- **Kernel duration**: Individual GPU kernel execution time
- **API overhead**: Gap between host call and kernel start
- **Concurrency**: Overlap between kernels (multi-stream)
- **Memory bound**: Long memory operations vs compute

Text Profile Analysis
---------------------

The ``wall_clock.txt`` file shows:

.. code-block:: text

   |-> 95.2% _Z15matrixMultiplyPfS_S_ii (GPU kernel)
   |-> 3.1%  hipMemcpy (H2D transfers)
   |-> 1.2%  hipMemcpy (D2H transfers)
   |-> 0.5%  hipMalloc

**How to interpret:**

- **>80% in compute kernel**: Likely compute-bound, optimize kernel
- **>20% in memcpy**: Memory-bound, reduce transfers or use async
- **>10% in HIP API**: High overhead, batch operations

Performance Metrics Reference
==============================

Wall Clock Time
---------------

- **Definition**: Total elapsed time (real time)
- **Use**: Overall application performance
- **Target**: Minimize for production workloads

GPU Time
--------

- **Definition**: Sum of GPU kernel execution times
- **Use**: Isolate GPU compute performance
- **Target**: High GPU time / Wall time ratio indicates good GPU utilization

GPU Utilization %
-----------------

- **Definition**: Percentage of time GPU is actively computing
- **Use**: Hardware efficiency indicator
- **Target**: >80% for compute-bound, >50% for memory-bound

Memory Bandwidth
----------------

- **Definition**: GB/s transferred to/from GPU memory
- **Use**: Identify memory-bound kernels
- **Target**: Close to peak HBM bandwidth (>1000 GB/s for MI250X)

Kernel Occupancy
----------------

- **Definition**: Ratio of active wavefronts to maximum possible
- **Use**: Indicates resource utilization (registers, LDS, VGPRs)
- **Target**: >50% for most kernels

Troubleshooting
===============

No GPU Data in Trace
---------------------

**Problem**: Perfetto trace shows no GPU activity

**Solutions:**

.. code-block:: bash

   # Ensure HIP tracing is enabled
   rocprof-sys-sample --hip-trace --trace -- ./hip_app

   # Or use AI preset
   rocprof-sys-sample --trace-ai -- ./hip_app

   # Check GPU visibility
   rocm-smi

Incomplete Kernel Data
----------------------

**Problem**: Kernel names show as ``<unknown>`` or are missing

**Solutions:**

.. code-block:: bash

   # Build with debug symbols
   hipcc -g -O2 app.cpp -o app

   # Use ROCgdb for symbol resolution
   export ROCPROFSYS_USE_ROCGDB=ON

Perfetto Trace Too Large
-------------------------

**Problem**: ``perfetto-trace.proto`` is >1GB, hard to analyze

**Solutions:**

.. code-block:: bash

   # Reduce sampling frequency
   rocprof-sys-sample -f 10 --hip-trace -- ./hip_app

   # Profile shorter runs
   timeout 30s rocprof-sys-sample --trace-ai -- ./long_running_app

   # Use selective instrumentation
   rocprof-sys-instrument -R '^important_func' -o app.inst -- ./app

Overhead Too High
-----------------

**Problem**: Profiling slows down application significantly (>20%)

**Solutions:**

.. code-block:: bash

   # Use sampling mode (lower overhead)
   rocprof-sys-sample -f 50 -- ./hip_app

   # Disable CPU sampling for GPU-focused profiling
   export ROCPROFSYS_SAMPLING_CPUS=none
   rocprof-sys-sample --hip-trace -- ./hip_app

   # Or use the AI preset (already configured)
   rocprof-sys-sample --trace-ai -- ./hip_app

Next Steps
==========

Now that you can profile HIP applications:

1. **Optimize your kernels** - Apply insights from profiling
2. **Learn hardware counters** - Read :doc:`../reference/hardware-counters-reference`
3. **Explore metrics** - Understand :doc:`../reference/metrics-glossary`
4. **Try examples** - Profile the :doc:`../examples/hip-quickstart/README` workloads
5. **Advanced techniques** - Check :doc:`../how-to/instrumenting-rewriting-binary-application`

Additional Resources
====================

- `ROCm Documentation <https://rocm.docs.amd.com/>`_
- `HIP Programming Guide <https://rocm.docs.amd.com/projects/HIP/>`_
- `Perfetto Trace Viewer <https://ui.perfetto.dev>`_
- `AMD GPU ISA Documentation <https://gpuopen.com/amd-isa-documentation/>`_
- `rocprofiler-systems GitHub <https://github.com/ROCm/rocprofiler-systems>`_

