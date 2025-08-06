.. meta::
   :description: ROCm Systems Profiler feature set documentation and reference
   :keywords: rocprof-sys, rocprofiler-systems, Omnitrace, ROCm, profiler, feature set, use cases, tracking, visualization, tool, Instinct, accelerator, AMD

********************************************
ROCm Systems Profiler features and use cases
********************************************

`ROCm Systems Profiler <https://github.com/ROCm/rocprofiler-systems>`_ is designed to be highly extensible.
Internally, it leverages the `Timemory performance analysis toolkit <https://github.com/ROCm/timemory>`_
to manage extensions, resources, data, and other items. It supports the following features,
modes, metrics, and APIs.

Data collection modes
========================================

* Dynamic instrumentation

  * Runtime instrumentation: Instrument executables and shared libraries at runtime
  * Binary rewriting: Generate a new executable and/or library with instrumentation built-in

* Statistical sampling: Periodic software interrupts per-thread
* Process-level sampling: A background thread records process-, system- and device-level metrics while the application runs
* Causal profiling: Quantifies the potential impact of optimizations in parallel code

Data analysis
========================================

* High-level summary profiles with mean, min, max, and standard deviation statistics

  * Low overhead and memory efficient
  * Ideal for running at scale

* Comprehensive traces for every individual event and measurement
* Application speed-up predictions resulting from potential optimizations in functions and lines of code based on causal profiling

Parallelism API support
========================================

* HIP
* HSA
* Pthreads
* MPI
* Kokkos-Tools (KokkosP)
* OpenMP-Tools (OMPT)

GPU metrics
========================================

* GPU hardware counters
* HIP API tracing
* HIP kernel tracing
* HSA API tracing
* HSA operation tracing
* rocDecode API tracing
* rocJPEG API tracing
* System-level sampling (via AMD-SMI)

  * Memory usage
  * Power usage
  * Temperature
  * Utilization
  * VCN activity
  * JPEG activity
    Note: The availability of VCN and JPEG engine activity depends on device support for different ASICs. If unsupported, all values for VCN_ACTIVITY and JPEG_ACTIVITY will be reported as N/A in the output of amd-smi metric--usage.

CPU metrics
========================================

* CPU hardware counters sampling and profiles
* CPU frequency sampling
* Various timing metrics

  * Wall time
  * CPU time (process and thread)
  * CPU utilization (process and thread)
  * User CPU time
  * Kernel CPU time

* Various memory metrics

  * High-water mark (sampling and profiles)
  * Memory page allocation
  * Virtual memory usage

* Network statistics
* I/O metrics
* Many others

Third-party API support
========================================

* TAU
* LIKWID
* Caliper
* CrayPAT
* VTune
* NVTX
* ROCTX

ROCm Systems Profiler use cases
========================================

When analyzing the performance of an application, do NOT
assume you know where the performance bottlenecks are
and why they are happening. ROCm Systems Profiler is a tool for analyzing the entire
application and its performance. It is
ideal for characterizing where optimization would have the greatest impact
on an end-to-end run of the application and for
viewing what else is happening on the system during a performance bottleneck.

When GPUs are involved, there is a tendency to assume that
the quickest path to performance improvement is minimizing
the runtime of the GPU kernels. This is a highly flawed assumption.
If you optimize the runtime of a kernel from one millisecond
to 1 microsecond (1000x speed-up) but the original application never
spent time waiting for kernels to complete,
there would be no statistically significant reduction in the end-to-end
runtime of your application. In other words, it does not matter
how fast or slow the code on GPU is if the application has a
bottleneck on waiting on the GPU.

Use ROCm Systems Profiler to obtain a high-level view of the entire application. Use it
to determine where the performance bottlenecks are and
obtain clues to why these bottlenecks are happening. Rather than worrying about kernel
performance, start your investigation with ROCm Systems Profiler, which characterizes the
broad picture.

.. note::

   For insight into the execution of individual kernels on the GPU,
   use `ROCm Compute Profiler <https://github.com/rocm/rocprofiler-compute>`_.

In terms of CPU analysis, ROCm Systems Profiler does not target any specific vendor.
It works just as well on AMD and non-AMD CPUs.
With regard to the GPU, ROCm Systems Profiler is currently restricted to HIP and HSA APIs
and kernels running on AMD GPUs.
