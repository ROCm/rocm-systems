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
