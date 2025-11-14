.. meta::
   :description: ROCm Systems Profiler glossary and reference
   :keywords: rocprof-sys, rocprofiler-systems, Omnitrace, ROCm, glossary, terminology, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

********
Glossary
********

This topic explains the terminology necessary to use ROCm Systems Profiler.
The list below provides a basic glossary for those who
are new to binary instrumentation. It also clarifies ambiguities
when certain terms have different
contextual meanings, for example, the ROCm Systems Profiler meaning of the term "module"
when instrumenting Python.

Binary
  A file written in the Executable and Linkable Format (ELF). This is the standard file
  format for executable files, shared libraries, etc.

Binary instrumentation
  Inserting callbacks to instrumentation into an existing binary. This can be performed
  statically or dynamically.

Static binary instrumentation
  Loads an existing binary, determines instrumentation points, and generates a new binary
  with instrumentation directly embedded. It is applicable to executables and libraries but
  limited to only the functions defined in the binary. This is also known as **Binary rewrite**.

Dynamic binary instrumentation
  Loads an existing binary into memory, inserts instrumentation, and runs the binary.
  It is limited to executables but is capable of instrumenting linked libraries.
  This is also known as **Runtime instrumentation**.

Statistical sampling
  At periodic intervals, the application is paused and the current call-stack of the CPU
  is recorded along with various other metrics. It uses timers that measure either
  (A) real clock time or (B) the CPU time used by the current thread and the CPU time
  expended on behalf of the thread by the system. This is also known as simply **sampling**.

  Sampling rate
    * The period at which (A) or (B) are triggered (in units of ``# interrupts / second``)
    * Higher values increase the number of samples

  Sampling delay
    * How long to wait before (A) and (B) begin triggering at their designated rate

  Sampling duration
    * The amount of time (in real-time) after the start of the application to record samples.
    * After this time limit has been reached, no more samples are recorded.

Process sampling
  At periodic (real-time) intervals, a background thread records global metrics without
  interrupting the current process. These metrics include, but are not limited to:
  CPU frequency, CPU memory high-water mark (i.e. peak memory usage), GPU temperature,
  and GPU power usage.

  Sampling rate
    * The real-time period for recording metrics (in units of ``# measurements / second``)
    * Higher values increase the number of samples

  Sampling delay
    * How long to wait (in real-time) before recording samples

  Sampling duration
    * The amount of time (in real-time) after the start of the application to record samples.
    * After this time limit has been reached, no more samples are recorded.

Module
  With respect to binary instrumentation, a module is defined as either the filename
  (such as ``foo.c``) or library name (``libfoo.so``) which contains the definition
  of one or more functions.

  With respect to Python instrumentation, a module is defined as the **file** which contains
  the definition of one or more functions. The full path to this file typically contains the
  name of the "Python module".

Basic block
  A straight-line code sequence with no branches in (except for the entry) and
  no branches out (except for the exit).

Address range
  The instructions for a function in a binary start at certain address with the ELF file
  and end at a certain address. The range is ``end - start``.

  The address range is a decent approximation for the "cost" of a function.
  For example, a larger address range approximately equates to more instructions.

Instrumentation traps
  On the x86 architecture, because instructions are of variable size, an instruction
  might be too small for Dyninst to replace it with the normal code sequence
  used to call instrumentation. When instrumentation is placed at points other
  than subroutine entry, exit, or call points, traps may be used to ensure
  the instrumentation fits. (By default, ``rocprof-sys-instrument`` avoids instrumentation
  which requires a trap.)

Overlapping functions
  Due to language constructs or compiler optimizations, it might be possible for
  multiple functions to overlap (that is, share part of the same function body)
  or for a single function to have multiple entry points. In practice, it's
  impossible to determine the difference between multiple overlapping functions
  and a single function with multiple entry points. (By default, ``rocprof-sys-instrument``
  avoids instrumenting overlapping functions.)

Performance Metrics
===================

Wall time (Wall clock time)
  The actual elapsed time (real time) from the start to the end of a function or
  program execution. This includes time spent waiting for I/O, synchronization,
  and other blocking operations. Most useful for measuring end-to-end performance.

CPU time
  The amount of time the CPU spends actively executing a function or program,
  excluding time spent waiting for I/O or blocked. Useful for identifying
  CPU-bound operations.

GPU time
  The amount of time GPU kernels spend executing on the GPU hardware. This
  excludes host-side API overhead and data transfer times. Useful for measuring
  actual GPU compute performance.

Hardware counters
  Special registers in the CPU or GPU that count specific hardware events such as
  instructions executed, cache hits/misses, memory accesses, etc. See
  :doc:`hardware-counters-reference` for detailed information.

PAPI (Performance Application Programming Interface)
  A portable interface for accessing hardware performance counters on various
  CPU architectures. ROCm Systems Profiler uses PAPI for collecting CPU hardware
  counter data like instructions, cycles, cache misses, etc.

ROCm hardware counters
  GPU-specific performance counters for AMD GPUs accessible through the ROCm stack.
  These include metrics like wave execution, memory bandwidth, compute unit
  utilization, etc.

Flat profile
  A summary view showing the total time spent in each function, without call stack
  information. Fast to generate and useful for quickly identifying hotspots.

Call tree
  A hierarchical view showing the call relationships between functions and the
  time spent in each call path. Useful for understanding which callers contribute
  most to a function's execution time.

HIP (Heterogeneous-compute Interface for Portability)
  AMD's GPU programming model for writing portable GPU applications. ROCm Systems
  Profiler can trace HIP API calls and GPU kernel execution.

Kernel
  A function that executes on the GPU. In HIP/ROCm context, refers to GPU compute
  kernels launched from host code.

Memory transfer
  Data movement between host (CPU) memory and device (GPU) memory, typically
  via ``hipMemcpy`` or similar APIs. These transfers can be a significant
  performance bottleneck.

GPU utilization
  The percentage of time the GPU is actively executing compute work. High
  utilization (>80%) indicates the GPU is being used efficiently.

Perfetto
  An open-source trace visualization tool used by ROCm Systems Profiler to display
  timeline traces. Perfetto traces show function execution, GPU kernels, and
  system metrics over time.

Trace
  A detailed timeline recording of events during program execution, including
  function calls, GPU kernels, memory operations, and system metrics. More
  detailed than profiles but larger in size.

Profile
  A statistical summary of program execution showing where time is spent. Less
  detailed than traces but more compact and easier to analyze for identifying
  hotspots.

Hotspot
  A function or code region that consumes a large percentage of execution time.
  Optimizing hotspots typically provides the greatest performance improvements.

Overhead
  The performance cost of profiling itself. ROCm Systems Profiler aims to minimize
  overhead, typically 2-8% depending on the profiling mode used.

For detailed metric definitions and interpretation guidance, see :doc:`metrics-glossary`.
