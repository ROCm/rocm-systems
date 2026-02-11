.. meta::
   :description: ROCprofiler-Systems preset profiles guide
   :keywords: ROCm, profiling, presets, HPC, AI, ML, GPU, tracing

******************************************************************************
Using Preset Profiles
******************************************************************************

ROCprofiler-Systems provides preset profiles that configure the profiler for common workload scenarios. Instead of manually setting numerous environment variables and command-line options, presets offer optimized configurations for specific use cases.

Overview
========

Presets are command-line options that automatically configure profiling settings for different workload types. They provide:

* **Simplified usage** - Single flag instead of multiple configuration options
* **Optimized settings** - Pre-tuned configurations based on real-world usage
* **Reduced overhead** - Settings tailored to minimize performance impact
* **Consistent behavior** - Standardized profiling across different scenarios

To see detailed information about active preset configuration, use the ``-v`` or ``--verbose`` flag.

Available Presets
==================

General Purpose Presets
------------------------

--balanced
~~~~~~~~~~

**Purpose:** Balanced profiling with moderate overhead and comprehensive data

**Best for:** Most profiling scenarios, recommended starting point

**Configuration:**

* Tracing: ON (Perfetto timeline)
* Profiling: ON (call-stack based)
* CPU Sampling: ON @ 50 Hz
* Process Metrics: ON (CPU freq, memory)

**Example:**

.. code-block:: shell

   rocprof-sys-sample --balanced -- ./myapp
   rocprof-sys-run --balanced -- ./myapp.inst

**When to use:** First-time profiling, getting an overview of application behavior, general-purpose profiling

--profile-only
~~~~~~~~~~~~~~

**Purpose:** Profiling-only mode without tracing (flat profile)

**Best for:** Production environments, minimal overhead profiling

**Configuration:**

* Tracing: OFF
* Profiling: ON (flat profile)
* CPU Sampling: ON @ 100 Hz
* Process Metrics: OFF

**Example:**

.. code-block:: shell

   rocprof-sys-sample --profile-only -- ./production_app

**When to use:** Profiling production workloads where tracing overhead is unacceptable

--detailed
~~~~~~~~~~

**Purpose:** Comprehensive profiling with full system metrics

**Best for:** In-depth performance analysis, identifying bottlenecks

**Configuration:**

* Tracing: ON (Perfetto timeline)
* Profiling: ON (call-stack based)
* CPU Sampling: ON @ 100 Hz (all CPUs)
* Process Metrics: ON (CPU freq, memory)

**Example:**

.. code-block:: shell

   rocprof-sys-sample --detailed -- ./complex_app

**When to use:** Detailed performance investigation, comprehensive analysis

Workload-Specific Presets
--------------------------

--trace-hpc
~~~~~~~~~~~

**Purpose:** Optimized for HPC/MPI/OpenMP applications

**Best for:** High-Performance Computing workloads, MPI applications, OpenMP codes

**Configuration:**

* Tracing: ON (Perfetto timeline)
* Profiling: ON (call-stack based)
* CPU Sampling: OFF (reduced overhead)
* Process Metrics: ON
* OpenMP (OMPT): ON
* MPI (MPIP): ON
* Kokkos: ON
* RCCL: ON
* PAPI Events: PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_L3_TCM
* ROCm Domains: HIP API, kernels, memory, scratch
* GPU Metrics: busy, temp, power, mem_usage

**Example:**

.. code-block:: shell

   mpirun -n 4 rocprof-sys-sample --trace-hpc -- ./mpi_app
   rocprof-sys-sample --trace-hpc -- ./openmp_offload_app

**When to use:** MPI applications, OpenMP offload, scientific computing codes

--workload-trace
~~~~~~~~~~~~~~~~

**Purpose:** General compute workloads (AI/ML, HPC, etc.)

**Best for:** AI/ML frameworks (ROCm supported AI/ML frameworks), GPU-intensive workloads

**Configuration:**

* Tracing: ON (Perfetto timeline)
* Profiling: ON (call-stack based)
* CPU Sampling: OFF (reduced overhead)
* Process Metrics: ON
* ROCtracer: ON
* HIP API Trace: ON
* HIP Activity: ON (kernel timing)
* RCCL: ON (collective comms)
* rocPD: ON (SQLite Database Output)
* MPI (MPIP): ON
* ROCm Domains: HIP API, kernels, memory, scratch
* GPU Metrics: busy, temp, power, mem_usage
* Buffer Size: 2 GB (for long traces)

**Example:**

.. code-block:: shell

   rocprof-sys-sample --workload-trace -- python train.py
   rocprof-sys-instrument --workload-trace -- python inference.py

**When to use:** AI/ML training and inference, GPU compute workloads, Python applications

--trace-gpu
~~~~~~~~~~~

**Purpose:** GPU workload analysis with host functions, MPI, and device activity

**Best for:** Understanding GPU utilization, kernel execution, memory transfers

**Configuration:**

* Tracing: ON (Perfetto timeline)
* Profiling: OFF (reduced overhead)
* ROCm: ON
* AMD SMI: ON (GPU metrics)
* CPU Sampling: Disabled (none)
* ROCm Domains: HIP runtime, ROCTx, kernels, memory, scratch

**Example:**

.. code-block:: shell

   rocprof-sys-sample --trace-gpu -- ./gpu_compute_app

**When to use:** GPU-focused performance analysis, identifying GPU bottlenecks

--trace-openmp
~~~~~~~~~~~~~~

**Purpose:** OpenMP offload workloads with HSA domains

**Best for:** OpenMP target offload to GPUs

**Configuration:**

* Tracing: ON (Perfetto timeline)
* Profiling: OFF (reduced overhead)
* ROCm: ON
* OMPT: ON (OpenMP tools interface)
* ROCm Domains: HIP runtime, ROCTx, kernels, memory, HSA API

**Example:**

.. code-block:: shell

   rocprof-sys-sample --trace-openmp -- ./openmp_target_app

**When to use:** OpenMP offload applications, analyzing host-device data transfers

--profile-mpi
~~~~~~~~~~~~~

**Purpose:** MPI communication latency profiling

**Best for:** Studying MPI performance, communication patterns

**Configuration:**

* Tracing: OFF
* Profiling: ON (flat profile)
* AMD SMI: OFF
* ROCm: OFF
* Focus: Wall-clock files per rank

**Example:**

.. code-block:: shell

   mpirun -n 16 rocprof-sys-sample --profile-mpi -- ./mpi_comm_app

**When to use:** MPI-only applications, analyzing communication overhead

--trace-hw-counters
~~~~~~~~~~~~~~~~~~~

**Purpose:** Hardware counter collection during execution

**Best for:** Understanding GPU performance metrics, VALU utilization

**Configuration:**

* Profiling: ON
* CPU Sampling: Disabled (none)
* ROCm Events: VALUUtilization, Occupancy

**Example:**

.. code-block:: shell

   rocprof-sys-sample --trace-hw-counters -- ./kernel_heavy_app

**When to use:** GPU kernel optimization, understanding hardware utilization

API Tracing Presets
-------------------

--sys-trace
~~~~~~~~~~~

**Purpose:** Comprehensive system API tracing

**Best for:** Complete API call tracing, debugging API usage

**Configuration:**

* Tracing: ON (Perfetto timeline)
* Profiling: ON (call-stack based)
* ROCm APIs: HIP API, HSA API
* Marker API: ROCTx
* RCCL: ON (collective communications)
* Decode/JPEG: rocDecode, rocJPEG
* Memory Ops: copies, scratch, allocations
* Kernel Dispatch: ON

**Example:**

.. code-block:: shell

   rocprof-sys-sample --sys-trace -- ./my_rocm_app

**When to use:** Tracing all ROCm API calls including low-level HSA

--runtime-trace
~~~~~~~~~~~~~~~

**Purpose:** Runtime API tracing (excludes compiler and low-level HSA)

**Best for:** Application-level API tracing without low-level noise

**Configuration:**

* Tracing: ON (Perfetto timeline)
* Profiling: ON (call-stack based)
* HIP Runtime: ON (excludes compiler API)
* Marker API: ROCTx
* RCCL: ON (collective communications)
* Decode/JPEG: rocDecode, rocJPEG
* Memory Ops: copies, scratch, allocations
* Kernel Dispatch: ON

**Example:**

.. code-block:: shell

   rocprof-sys-sample --runtime-trace -- ./my_hip_app

**When to use:** Focusing on runtime API calls, excluding HIP compiler and HSA internals

Usage Examples
==============

Quick Start
-----------

Start with ``--balanced`` for an initial overview:

.. code-block:: shell

   rocprof-sys-sample --balanced -- ./myapp

This provides a balanced view of performance with moderate overhead.

Targeting Specific Workloads
-----------------------------

**MPI Application:**

.. code-block:: shell

   mpirun -n 4 rocprof-sys-sample --trace-hpc -v -- ./simulation

**OpenMP Offload:**

.. code-block:: shell

   rocprof-sys-sample --trace-openmp -v -- ./offload_compute

Combining with Other Options
-----------------------------

Presets can be combined with other command-line options:

.. code-block:: shell

   # Use preset with custom output directory
   rocprof-sys-sample --balanced -o ./my-results -- ./myapp

   # Use preset with additional instrumentation options
   rocprof-sys-instrument --trace-hpc -R '^compute_' -o app.inst -- ./app

Viewing Results
===============

After profiling with a preset, results are saved to ``rocprof-sys-output/`` (or custom directory specified with ``-o``):

**Text Profile:**

.. code-block:: shell

   cat rocprof-sys-output/wall_clock.txt

**Visual Timeline:**

Open ``rocprof-sys-output/perfetto-trace.proto`` in https://ui.perfetto.dev

**JSON Data:**

.. code-block:: shell

   cat rocprof-sys-output/wall_clock.json

Best Practices
==============

Choosing the Right Preset
--------------------------

1. **Start simple** - Begin with ``--balanced`` or ``--profile-only`` to minimize overhead
2. **Match your workload** - Use workload-specific presets for better insights
3. **Iterate** - Start with low overhead, increase detail as needed

Performance Considerations
--------------------------

* **CPU sampling** - Some presets disable sampling to reduce overhead
* **Buffer sizes** - ``--workload-trace`` uses larger buffers for long-running applications
* **ROCm domains** - API tracing presets focus on specific API layers

Preset Limitations
------------------

* **Mutual exclusion** - Only ONE preset can be used at a time
* **Override with env vars** - Environment variables can override preset settings if needed
* **No mixing** - Cannot combine multiple presets in a single invocation

Troubleshooting
===============

Preset Not Recognized
---------------------

Ensure you're using a valid preset name:

.. code-block:: shell

   rocprof-sys-sample --help | grep -A20 "PRESET"

Multiple Presets Error
-----------------------

If you see "Multiple preset modes specified":

.. code-block:: shell

   # Wrong: Multiple presets
   rocprof-sys-sample --balanced --detailed -- ./app

   # Correct: Single preset
   rocprof-sys-sample --balanced -- ./app

No Output with Preset
---------------------

Add ``-v`` flag to see preset configuration:

.. code-block:: shell

   rocprof-sys-sample --balanced -v 2 -- ./app

This shows which settings are active.

Advanced Usage
==============

Viewing Active Configuration
-----------------------------

Use verbose mode to see what the preset configures:

.. code-block:: shell

   rocprof-sys-sample --trace-hpc -v 2 -- ls

This displays the full preset configuration before execution.

Overriding Preset Settings
---------------------------

Environment variables can override preset defaults:

.. code-block:: shell

   # Use --balanced preset but customize sampling frequency
   ROCPROFSYS_SAMPLING_FREQ=200 rocprof-sys-sample --balanced -- ./app

Custom Configuration Files
---------------------------

For complex configurations beyond presets:

.. code-block:: shell

   rocprof-sys-sample -c custom-config.cfg -- ./app

See Also
========

* :doc:`sampling-call-stack` - Call-stack sampling basics
* :doc:`instrumenting-rewriting-binary-application` - Binary instrumentation
* :doc:`configuring-validating-environment` - Environment configuration

Additional Resources
====================

* `ROCprofiler-Systems Documentation <https://rocm.docs.amd.com/projects/rocprofiler-systems>`_
* `Perfetto UI <https://ui.perfetto.dev>`_ for trace visualization
* `ROCm Documentation <https://rocm.docs.amd.com>`_
