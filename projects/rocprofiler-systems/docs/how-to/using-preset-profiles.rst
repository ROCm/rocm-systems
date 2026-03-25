.. meta::
   :description: ROCprofiler-Systems preset profiles and domain flags guide
   :keywords: ROCm, profiling, presets, domain flags, HPC, AI, ML, GPU, tracing, configuration

******************************************************************************
Using preset profiles and domain flags
******************************************************************************

ROCm Systems Profiler provides preset profiles and domain flags that simplify
configuring the profiler for common workload scenarios. Instead of manually
setting numerous environment variables, use a single ``--preset`` flag or
combine domain flags for composable configuration.

Quick start
===========

.. code-block:: shell

   # Use a preset for balanced profiling
   rocprof-sys-run --preset=balanced -- ./myapp

   # List all available presets
   rocprof-sys-run --list-presets

   # See detailed info about a preset
   rocprof-sys-run --explain=balanced

   # Combine a preset with domain flags
   rocprof-sys-run --preset=balanced --gpu=temp,power -- ./myapp

   # Export configuration as reusable JSON
   rocprof-sys-run --preset=balanced --gpu --export-config=my-config.json

Topic-based help
================

The ``--help`` system is organized by topic to avoid overwhelming output:

.. code-block:: shell

   # Compact summary with essential options
   rocprof-sys-run --help

   # Help for a specific topic
   rocprof-sys-run --help=sampling
   rocprof-sys-run --help=gpu
   rocprof-sys-run --help=tracing

   # Full option listing
   rocprof-sys-run --help=all

Available help topics:

* **Group topics:** ``preset``, ``general``, ``tracing``, ``profiling``,
  ``sampling``, ``process``, ``counters``, ``backend``, ``debug``, ``misc``
* **Domain topics:** ``gpu``, ``cpu``, ``rocm``, ``parallel``

Available presets
==================

Use ``--list-presets`` to see all presets grouped by category, or
``--explain=<name>`` for detailed information about a specific preset.

General purpose presets
------------------------

--preset=balanced
~~~~~~~~~~~~~~~~~

**Purpose:** Balanced profiling with moderate overhead and comprehensive data

**Best for:** Most profiling scenarios, recommended starting point

**Configuration:**

* Tracing: ON (Perfetto timeline)
* Profiling: ON (call-stack based)
* CPU Sampling: ON @ 50 Hz
* GPU Metrics: ON (via AMD SMI)

.. code-block:: shell

   rocprof-sys-run --preset=balanced -- ./myapp
   rocprof-sys-sample --preset=balanced -- ./myapp

--preset=profile-only
~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Profiling-only mode without tracing (flat profile)

**Best for:** Production environments, minimal overhead profiling

**Configuration:**

* Tracing: OFF
* Profiling: ON (flat profile)
* CPU Sampling: OFF

.. code-block:: shell

   rocprof-sys-sample --preset=profile-only -- ./production_app

--preset=detailed
~~~~~~~~~~~~~~~~~

**Purpose:** Comprehensive profiling with full system metrics

**Best for:** In-depth performance analysis, identifying bottlenecks

**Configuration:**

* Tracing: ON (Perfetto timeline)
* Profiling: ON (call-stack based)
* CPU Sampling: ON (all CPUs)
* GPU Metrics: ON

.. code-block:: shell

   rocprof-sys-sample --preset=detailed -- ./complex_app

GPU and workload presets
--------------------------

--preset=trace-gpu
~~~~~~~~~~~~~~~~~~

**Purpose:** GPU workload analysis with device activity

**Configuration:** Tracing ON, Profiling OFF, ROCm domains ON, GPU metrics ON,
CPU Sampling OFF

.. code-block:: shell

   rocprof-sys-sample --preset=trace-gpu -- ./gpu_compute_app

--preset=workload-trace
~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Comprehensive tracing for AI/ML, HPC, and GPU workloads

**Configuration:** Tracing ON (2 GB buffer), Profiling ON, MPI ON, RCCL ON,
ROCm domains ON, GPU metrics ON, rocPD ON

.. code-block:: shell

   rocprof-sys-sample --preset=workload-trace -- python train.py

--preset=trace-hw-counters
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Hardware counter collection (VALUUtilization, Occupancy)

.. code-block:: shell

   rocprof-sys-sample --preset=trace-hw-counters -- ./kernel_heavy_app

HPC presets
-----------

--preset=trace-hpc
~~~~~~~~~~~~~~~~~~

**Purpose:** Optimized for MPI, OpenMP, and compute-intensive applications

**Configuration:** Tracing ON, Profiling ON, MPI ON, OpenMP ON, Kokkos ON,
RCCL ON, PAPI events ON, ROCm domains ON, GPU metrics ON

.. code-block:: shell

   mpirun -n 4 rocprof-sys-sample --preset=trace-hpc -- ./mpi_app

--preset=trace-openmp
~~~~~~~~~~~~~~~~~~~~~

**Purpose:** OpenMP offload with HSA domains and OMPT

.. code-block:: shell

   rocprof-sys-sample --preset=trace-openmp -- ./openmp_target_app

--preset=profile-mpi
~~~~~~~~~~~~~~~~~~~~

**Purpose:** MPI communication latency profiling (no tracing, no GPU)

.. code-block:: shell

   mpirun -n 16 rocprof-sys-sample --preset=profile-mpi -- ./mpi_comm_app

API tracing presets
-------------------

--preset=sys-trace
~~~~~~~~~~~~~~~~~~

**Purpose:** Comprehensive system API tracing (HIP, HSA, ROCTx, RCCL)

.. code-block:: shell

   rocprof-sys-sample --preset=sys-trace -- ./my_rocm_app

--preset=runtime-trace
~~~~~~~~~~~~~~~~~~~~~~

**Purpose:** Runtime API tracing (excludes compiler API and HSA internals)

.. code-block:: shell

   rocprof-sys-sample --preset=runtime-trace -- ./my_hip_app

Domain flags
============

Domain flags provide high-level control over specific profiling domains.
They can be used standalone or combined with presets.

--gpu
-----

Enable GPU metrics collection via AMD SMI. Optionally specify which metrics:

.. code-block:: shell

   # All default GPU metrics
   rocprof-sys-run --gpu -- ./myapp

   # Specific metrics only
   rocprof-sys-run --gpu=temp,power,busy -- ./myapp

Available metrics: ``temp``, ``power``, ``busy`` (utilization), ``mem_usage``

Shortcuts: ``temperature`` -> ``temp``, ``usage``/``utilization`` -> ``busy``,
``memory`` -> ``mem_usage``

--rocm
------

Enable ROCm API tracing. Optionally specify which API domains:

.. code-block:: shell

   # All default ROCm domains
   rocprof-sys-run --rocm -- ./myapp

   # Specific domains only
   rocprof-sys-run --rocm=hip,kernel,memory -- ./myapp

Available domains: ``hip_runtime_api``, ``marker_api``, ``kernel_dispatch``,
``memory_copy``, ``scratch_memory``, ``hsa_api``, ``rccl_api``

Shortcuts: ``hip`` -> ``hip_runtime_api``, ``kernel`` -> ``kernel_dispatch``,
``memory`` -> ``memory_copy``, ``hsa`` -> ``hsa_api``,
``marker`` -> ``marker_api``, ``rccl`` -> ``rccl_api``

--cpu
-----

Enable CPU call-stack sampling. Optionally specify frequency in Hz:

.. code-block:: shell

   # Default 100 Hz sampling
   rocprof-sys-run --cpu -- ./myapp

   # Custom frequency
   rocprof-sys-run --cpu=50 -- ./myapp

--parallel
----------

Enable parallel runtime profiling. Optionally specify which runtimes:

.. code-block:: shell

   # All runtimes (MPI, OpenMP, Kokkos, RCCL)
   rocprof-sys-run --parallel -- ./myapp

   # Specific runtimes
   rocprof-sys-run --parallel=mpi,openmp -- ./mpi_app

Combining presets with domain flags
------------------------------------

Domain flags override or extend preset settings:

.. code-block:: shell

   # Balanced preset with specific GPU metrics
   rocprof-sys-run --preset=balanced --gpu=temp,power -- ./myapp

   # HPC preset with custom ROCm domains
   rocprof-sys-sample --preset=trace-hpc --rocm=hip,kernel,rccl -- ./app

   # Profile-only preset with CPU sampling enabled
   rocprof-sys-run --preset=profile-only --cpu=100 -- ./myapp

Configuration export
====================

Export the resolved configuration (preset + domain flags + env overrides) as
a reusable JSON file:

.. code-block:: shell

   # Export to stdout
   rocprof-sys-run --preset=balanced --gpu --export-config

   # Export to file
   rocprof-sys-run --preset=balanced --gpu --export-config=team-config.json

   # Reuse the exported configuration
   rocprof-sys-run --preset=./team-config.json -- ./myapp

Custom configuration files
==========================

Custom JSON configuration files can be loaded using the ``--preset`` flag
with a file path:

.. code-block:: shell

   # Load from relative path
   rocprof-sys-run --preset=./my-config.json -- ./myapp

   # Load from absolute path
   rocprof-sys-run --preset=/path/to/config.json -- ./myapp

See the JSON schema file at ``share/rocprofiler-systems/presets/schema.json``
for the full configuration format.

Preset directory
================

Presets are loaded from the following locations (in order):

1. ``$ROCPROFSYS_PRESET_DIR`` (if set)
2. ``<install-prefix>/share/rocprofiler-systems/presets``
3. ``$ROCM_PATH/share/rocprofiler-systems/presets``

Adding custom presets
---------------------

Create a JSON file in the preset directory:

.. code-block:: json

   {
       "metadata": {
           "name": "my-preset",
           "description": "Custom profiling configuration",
           "use_case": "My specific workload",
           "category": "custom"
       },
       "tracing": { "enabled": true },
       "profiling": { "enabled": true },
       "sampling": {
           "enabled": true,
           "frequency_hz": { "value": 100 }
       }
   }

The preset will be automatically discovered and available via
``--preset=my-preset``.

Troubleshooting
===============

Preset not found
----------------

.. code-block:: shell

   # Check available presets
   rocprof-sys-run --list-presets

   # Set preset directory explicitly
   export ROCPROFSYS_PRESET_DIR=/opt/rocm/share/rocprofiler-systems/presets

ROCm not available warning
--------------------------

If you see "GPU tracing requested but ROCm is not available", ensure ROCm
is installed and ``/opt/rocm/bin/hipconfig`` is accessible, or set
``ROCM_PATH`` to your ROCm installation.

Viewing active configuration
-----------------------------

Use verbose mode to see what settings a preset configures:

.. code-block:: shell

   rocprof-sys-run --preset=balanced -v 2 -- ./myapp

See also
========

* :doc:`sampling-call-stack` - Call-stack sampling basics
* :doc:`instrumenting-rewriting-binary-application` - Binary instrumentation
* :doc:`configuring-validating-environment` - Environment configuration
