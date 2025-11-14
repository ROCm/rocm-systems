.. meta::
   :description: ROCm Systems Profiler quickstart guide
   :keywords: rocprof-sys, rocprofiler-systems, Omnitrace, ROCm, quickstart, tutorial, profiler, beginner, getting started

**********************************
ROCm Systems Profiler Quickstart
**********************************

This quickstart guide provides everything you need to start profiling applications with ROCm Systems Profiler.
Whether you're new to profiling or an experienced developer, this guide will help you get meaningful performance
insights quickly.

.. contents:: Table of Contents
   :local:
   :backlinks: none
   :depth: 2

.. _quickstart-for-beginners:

==========================
For Complete Beginners
==========================

New to profiling? Start here for a gentle introduction.

Prerequisites
=============

Before you begin, ensure you have:

1. ROCm Systems Profiler installed (see :doc:`installation guide <../install/install>`)
2. A sample application to profile (we'll provide examples)
3. ROCm Systems Profiler environment set up

Setting Up Your Environment
============================

Choose one of these methods to set up your environment:

**Option 1: Source the setup script (Recommended)**

.. code-block:: bash

   source /opt/rocprofiler-systems/share/rocprofiler-systems/setup-env.sh

**Option 2: Manual PATH setup**

.. code-block:: bash

   export PATH=/opt/rocprofiler-systems/bin:${PATH}
   export LD_LIBRARY_PATH=/opt/rocprofiler-systems/lib:${LD_LIBRARY_PATH}

Verify the installation:

.. code-block:: bash

   rocprof-sys-sample --version

5-Minute Quickstart
===================

Let's profile a simple application in just a few steps.

Step 1: Profile Your Application
---------------------------------

The simplest way to profile any application is using ``rocprof-sys-sample`` with the ``--quick`` preset:

.. code-block:: bash

   # Profile your application with sensible defaults
   rocprof-sys-sample --quick -- ./your_app

   # For HIP/GPU applications
   rocprof-sys-sample --quick --hip-trace -- ./your_hip_app

This command:

- Samples your application's call stack at 50Hz
- Generates both a trace and a profile
- Collects process-level metrics (CPU, memory, etc.)
- Creates output in the ``rocprof-sys-output/`` directory

**Workload-Specific Presets:**

For HPC applications (MPI, OpenMP, compute-intensive):

.. code-block:: bash

   # Optimized for HPC with hardware counters
   rocprof-sys-sample --trace-hpc -- ./hpc_app
   mpirun -n 4 rocprof-sys-sample --trace-hpc -- ./mpi_app

The ``--trace-hpc`` preset configures:

- Full trace and profile collection
- OpenMP (OMPT) and Kokkos instrumentation
- MPI profiling (MPIP)
- CPU hardware counters (PAPI): Instructions, Cycles, L3 Cache Misses
- GPU metrics: Utilization, Temperature, Power, Memory Usage
- ROCm API tracing: HIP runtime, kernels, memory operations

For AI/ML applications (PyTorch, TensorFlow, JAX):

.. code-block:: bash

   # Optimized for AI/ML with GPU tracing
   rocprof-sys-sample --trace-ai -- python train.py

The ``--trace-ai`` preset configures:

- Full trace and profile collection
- GPU-focused monitoring (no CPU sampling overhead)
- ROCm GPU tracing: HIP API calls, kernel dispatch, memory operations
- ROCm Communications Collective Library (RCCL) for distributed training
- ROCm Performance Delta (ROCPD) for kernel analysis
- GPU metrics: Utilization, Temperature, Power, Memory Usage
- Large Perfetto buffer (2GB) for long traces

Step 2: View the Results
-------------------------

After profiling completes, you'll see output files in ``rocprof-sys-output/``:

.. code-block:: bash

   $ ls rocprof-sys-output/
   rocpd-<process-id>.db     # SQLite3 database
   perfetto-trace.proto       # Detailed trace for visualization
   wall_clock.txt            # Text-based profile summary
   wall_clock.json           # JSON profile data
   ...

**View the trace visualization:**

1. Open `ui.perfetto.dev <https://ui.perfetto.dev>`_ in your web browser
2. Click "Open trace file"
3. Select ``rocprof-sys-output/perfetto-trace.proto``

**View the text profile:**

.. code-block:: bash

   cat rocprof-sys-output/wall_clock.txt

Step 3: Understand the Output
------------------------------

The text profile shows:

- **Function names**: Which functions were executing
- **Count**: How many times each function was called
- **Sum**: Total time spent in the function (including callees)
- **Mean/Min/Max**: Statistical information about call duration
- **% of total**: What percentage of runtime each function consumed

Look for functions with:

- High **% of total** - These consume the most time
- High **Count** with low individual time - Might benefit from inlining
- High **Stddev** - Performance varies significantly between calls

The Perfetto trace shows:

- Timeline of function execution
- Thread activity and parallelism
- GPU kernel execution (for HIP applications)
- System metrics over time

Common First Steps
==================

**Find the hottest functions:**

.. code-block:: bash

   # Generate a flat profile for easy identification of hotspots
   rocprof-sys-sample -F -- ./your_app
   
   # Look at the output
   head -30 rocprof-sys-output/sampling_wall_clock.txt

**Profile with different sampling rates:**

.. code-block:: bash

   # Higher frequency for more detailed sampling (higher overhead)
   rocprof-sys-sample -f 1000 -- ./your_app
   
   # Lower frequency for less overhead
   rocprof-sys-sample -f 10 -- ./your_app

**Profile only specific parts of your application:**

.. code-block:: bash

   # Sample for only the first 30 seconds
   rocprof-sys-sample --sampling-duration=30 -- ./your_app

**Exclude system libraries from profiling:**

.. code-block:: bash

   # Focus on your application code
   rocprof-sys-sample --quick -ME '^(lib|/usr)' -- ./your_app

Next Steps for Beginners
=========================

Once you're comfortable with basic profiling:

1. Explore :doc:`instrumenting applications <../how-to/instrumenting-rewriting-binary-application>` for more detailed traces
2. Learn about :doc:`profiling Python <../how-to/profiling-python-scripts>` if you use Python
3. Try :doc:`profiling HIP applications <profiling-hip-applications>` with our sample workloads
4. Read about :doc:`interpreting metrics <../reference/metrics-glossary>` for deeper analysis

.. _quickstart-for-experienced:

====================================
For Experienced Developers
====================================

Quick reference for developers familiar with profiling tools.

Quick Reference Table
=====================

.. list-table:: Profiling Modes Quick Reference
   :header-rows: 1
   :widths: 20 30 25 25

   * - Mode
     - Use Case
     - Command
     - Overhead
   * - Quick
     - General profiling
     - ``rocprof-sys-sample --quick``
     - Very Low (2-5%)
   * - HPC
     - MPI/OpenMP apps
     - ``rocprof-sys-sample --trace-hpc``
     - Low (3-8%)
   * - AI/ML
     - PyTorch/TensorFlow
     - ``rocprof-sys-sample --trace-ai``
     - Low (3-8%)
   * - Binary Rewrite
     - Precise function tracing
     - ``rocprof-sys-instrument -o app.inst``
     - Low (5-10%)
   * - Runtime Instr.
     - Quick library tracing
     - ``rocprof-sys-instrument --quick``
     - Medium (10-30%)
   * - Hybrid
     - Comprehensive analysis
     - ``rocprof-sys-run --sample``
     - Low-Medium

Command Cheat Sheet
===================

**Sampling**

.. code-block:: bash

   # Quick profiling (basic sampling + tracing)
   rocprof-sys-sample --quick -- ./app

   # HPC workload (OMPT, MPI, PAPI hardware counters, GPU metrics)
   rocprof-sys-sample --trace-hpc -- ./hpc_app
   mpirun -n 4 rocprof-sys-sample --trace-hpc -- ./mpi_app

   # AI/ML workload (HIP tracing, RCCL, ROCPD, 2GB buffer)
   rocprof-sys-sample --trace-ai -- python train.py

   # Flat profile only (minimal overhead)
   rocprof-sys-sample --simple -- ./app
   
   # Full trace + metrics
   rocprof-sys-sample --detailed -- ./app
   
   # Custom frequency and output
   rocprof-sys-sample -f 100 -o ./results -- ./app

**Binary Instrumentation**

.. code-block:: bash

   # Binary rewrite (recommended for production)
   rocprof-sys-instrument -o app.inst -- ./app
   rocprof-sys-run -- ./app.inst
   
   # Runtime instrumentation (quick testing)
   rocprof-sys-instrument --quick -- ./app
   
   # Profile-optimized rewrite
   rocprof-sys-instrument --profile-only -o app.inst -- ./app
   
   # Instrument only specific functions
   rocprof-sys-instrument -R '^compute_' -o app.inst -- ./app

**Python**

.. code-block:: bash

   # Profile Python script
   rocprof-sys-python -- script.py
   
   # With decorator-based profiling
   rocprof-sys-python -b -- script.py

**MPI Applications**

.. code-block:: bash

   # Sampling with MPI
   mpirun -n 4 rocprof-sys-sample -- ./mpi_app
   
   # Instrumented MPI (binary rewrite required)
   rocprof-sys-instrument -o mpi_app.inst -- ./mpi_app
   mpirun -n 4 rocprof-sys-run -- ./mpi_app.inst

Configuration File Template
===========================

Create ``~/.rocprof-sys.cfg`` for your default preferences:

.. code-block:: ini

   # Generate comprehensive output
   ROCPROFSYS_TRACE                = true
   ROCPROFSYS_PROFILE              = true
   ROCPROFSYS_USE_SAMPLING         = true
   ROCPROFSYS_USE_PROCESS_SAMPLING = true
   
   # Sampling configuration
   ROCPROFSYS_SAMPLING_FREQ        = 50
   ROCPROFSYS_SAMPLING_CPUS        = all
   ROCPROFSYS_SAMPLING_GPUS        = $env:HIP_VISIBLE_DEVICES
   
   # Output configuration
   ROCPROFSYS_OUTPUT_PATH          = ./rocprof-sys-output
   ROCPROFSYS_TIME_OUTPUT          = true
   
   # Perfetto trace configuration
   ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB = 1024000

Generate a full configuration file with all options:

.. code-block:: bash

   # Generate config with descriptions
   rocprof-sys-avail -G rocprof-sys.cfg --all

Advanced Workflow Patterns
===========================

**Iterative Performance Optimization**

.. code-block:: bash

   # 1. Identify hotspots with sampling
   rocprof-sys-sample -F -- ./app
   
   # 2. Detailed trace of hotspot functions
   rocprof-sys-instrument -R '^hotspot_func' -o app.inst -- ./app
   rocprof-sys-run --trace -- ./app.inst
   
   # 3. Analyze with hardware counters
   rocprof-sys-run --hip-trace --rocm-events=SQ_WAVES -- ./app.inst

**Multi-Process Analysis**

.. code-block:: bash

   # Use perfetto system backend for multi-process tracing
   pkill traced
   traced --background
   perfetto --out trace.proto --txt -c perfetto.cfg --background
   
   # Run multiple processes
   rocprof-sys-run --perfetto-backend=system -- ./app &
   rocprof-sys-run --perfetto-backend=system -- ./app2 &
   wait

**Overhead-Conscious Profiling**

.. code-block:: bash

   # Minimal overhead: sampling only, no trace
   rocprof-sys-sample -F -f 20 -- ./app
   
   # Low overhead: trace with selective instrumentation
   rocprof-sys-instrument -R '^(main|critical)' -o app.inst -- ./app
   
   # Accept higher overhead for comprehensive data
   rocprof-sys-instrument --detailed -o app.inst -- ./app

**GPU Profiling Strategies**

.. code-block:: bash

   # Basic GPU API and kernel tracing
   rocprof-sys-sample --hip-trace --trace -- ./hip_app
   
   # GPU tracing with hardware counters
   rocprof-sys-run --hip-trace --rocm-events=SQ_WAVES,SQ_INSTS_VALU -- ./app.inst
   
   # GPU metrics sampling
   rocprof-sys-sample --sampling-gpus=0,1 --trace -- ./multi_gpu_app

==================================
Preset Configuration Reference
==================================

This section details the exact environment variables configured by each workload preset.

--trace-hpc Preset
==================

Optimized for High-Performance Computing applications with MPI, OpenMP, and compute-intensive kernels.

**Environment Variables:**

.. code-block:: bash

   ROCPROFSYS_TRACE=ON                    # Enable tracing
   ROCPROFSYS_PROFILE=ON                  # Enable profiling
   ROCPROFSYS_USE_SAMPLING=OFF            # Disable statistical sampling (use instrumentation)
   ROCPROFSYS_SAMPLING_FREQ=100           # Backup sampling frequency
   ROCPROFSYS_USE_PROCESS_SAMPLING=ON     # Collect process-level metrics
   ROCPROFSYS_USE_OMPT=ON                 # OpenMP instrumentation
   ROCPROFSYS_USE_KOKKOSP=ON              # Kokkos instrumentation
   ROCPROFSYS_USE_MPIP=true               # MPI profiling
   ROCPROFSYS_SAMPLING_CPUS=none          # No CPU sampling (instrumentation-based)
   ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,marker_api,kernel_dispatch,memory_copy,scratch_memory
   ROCPROFSYS_AMD_SMI_METRICS=busy,temp,power,mem_usage  # GPU monitoring
   ROCPROFSYS_PAPI_EVENTS=PAPI_TOT_INS,PAPI_TOT_CYC,PAPI_L3_TCM  # Hardware counters

**Use Cases:**

- MPI applications (message passing analysis)
- OpenMP parallel regions (threading analysis)
- Kokkos performance portability applications
- CPU-bound compute kernels (instruction analysis)
- Cache performance analysis (L3 miss tracking)

--trace-ai Preset
=================

Optimized for AI/ML workloads with GPU-accelerated frameworks like PyTorch, TensorFlow, and JAX.

**Environment Variables:**

.. code-block:: bash

   ROCPROFSYS_TRACE=ON                    # Enable tracing
   ROCPROFSYS_PROFILE=ON                  # Enable profiling
   ROCPROFSYS_USE_SAMPLING=OFF            # Disable CPU sampling (focus on GPU)
   ROCPROFSYS_SAMPLING_FREQ=50            # Backup sampling frequency
   ROCPROFSYS_USE_PROCESS_SAMPLING=ON     # Collect process-level metrics
   ROCPROFSYS_USE_MPIP=true               # MPI support for distributed training
   ROCPROFSYS_SAMPLING_CPUS=none          # No CPU sampling overhead
   ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,marker_api,kernel_dispatch,memory_copy,scratch_memory
   ROCPROFSYS_AMD_SMI_METRICS=busy,temp,power,mem_usage  # GPU monitoring
   ROCPROFSYS_SAMPLING_GPUS=$env:HIP_VISIBLE_DEVICES  # Sample visible GPUs
   ROCPROFSYS_USE_ROCTRACER=ON            # ROCm runtime tracing
   ROCPROFSYS_TRACE_HIP_API=ON            # HIP API call tracing
   ROCPROFSYS_TRACE_HIP_ACTIVITY=ON       # HIP kernel activity tracing
   ROCPROFSYS_USE_RCCL=ON                 # RCCL collective operations
   ROCPROFSYS_USE_ROCPD=ON                # Performance Delta analysis
   ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB=2048000  # 2GB buffer for long traces

**Use Cases:**

- PyTorch model training and inference
- TensorFlow computational graphs
- JAX just-in-time compilation
- Multi-GPU distributed training (RCCL)
- GPU kernel performance analysis
- Memory transfer optimization
- Long-running training jobs (large buffer)

==================================
Common Issues and Solutions
==================================

Output Directory Issues
=======================

**Problem:** ``rocprof-sys-output`` directory contains files from previous runs

**Solution:** Use timestamped output or custom output path:

.. code-block:: bash

   # Timestamped output (automatic)
   export ROCPROFSYS_TIME_OUTPUT=ON
   rocprof-sys-sample -- ./app
   
   # Custom output path
   rocprof-sys-sample -o ./run1 myrun -- ./app

Sampling Returns No Data
=========================

**Problem:** Sampling completes but produces no profile data

**Solutions:**

1. Check if application runs long enough to be sampled:

.. code-block:: bash

   # Increase sampling frequency for short applications
   rocprof-sys-sample -f 1000 -- ./short_app

2. Verify environment setup:

.. code-block:: bash

   # Check if rocprof-sys libraries are accessible
   ldd $(which rocprof-sys-sample)

Instrumentation Hangs or Crashes
=================================

**Problem:** ``rocprof-sys-instrument`` hangs when processing large binaries

**Solutions:**

1. Use sampling mode instead (no instrumentation needed):

.. code-block:: bash

   rocprof-sys-sample -- ./large_app

2. Reduce instrumentation scope:

.. code-block:: bash

   # Instrument only main function for sampling support
   rocprof-sys-instrument -M sampling -o app.inst -- ./app

MPI Compatibility
=================

**Problem:** MPI job fails when using runtime instrumentation

**Solution:** Use binary rewrite instead:

.. code-block:: bash

   # Don't: Runtime instrumentation incompatible with some MPI
   # mpirun -n 4 rocprof-sys-instrument -- ./mpi_app
   
   # Do: Binary rewrite first
   rocprof-sys-instrument -o mpi_app.inst -- ./mpi_app
   mpirun -n 4 rocprof-sys-run -- ./mpi_app.inst
   
   # Or: Use sampling (no instrumentation)
   mpirun -n 4 rocprof-sys-sample -- ./mpi_app

=======================
Additional Resources
=======================

- :doc:`General profiling tips <../how-to/general-tips-using-rocprof-sys>`
- :doc:`Understanding output <../how-to/understanding-rocprof-sys-output>`
- :doc:`Configuring runtime options <../how-to/configuring-runtime-options>`
- :doc:`Metrics glossary <../reference/metrics-glossary>`
- :doc:`HIP profiling tutorial <profiling-hip-applications>`
- `Example code on GitHub <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-systems/examples>`_

For more help, see the `ROCm Systems Profiler documentation <https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/>`_.

